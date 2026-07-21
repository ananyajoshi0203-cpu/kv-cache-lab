"""Iso-ratio comparison runner. Prefill a context, apply each method to the cache,
score a held-out continuation (perplexity), and account for KV memory.

Every method's hyperparameters are derived from one target compression ratio
(fraction of cache bytes removed, KVPress convention), so rows are comparable:
eviction budgets come from the token count, KIVI's bit width is chosen to land
nearest the target given its fp16 residual. The achieved size fraction is reported
per row because discrete knobs cannot always hit the target exactly. This is the
minimal harness for the reference methods; standard accuracy benchmarks run through
the KVPress backend (see kvlab.evals)."""

from __future__ import annotations

import csv
import json
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from . import memory
from .methods import build

logger = logging.getLogger(__name__)


@dataclass
class Row:
    method: str
    family: str
    kept_tokens: int
    bits: float
    kv_mb: float
    size_ratio: float
    perplexity: float
    delta_pct: float


# Non-repetitive on purpose: a repeated passage lets the model predict the
# continuation from any surviving copy, which flattens perplexity and hides the
# damage eviction does. Each paragraph covers distinct content so the score
# depends on the specific tokens the cache retains.
SAMPLE_TEXT = (
    "The key-value cache stores the key and value vectors of every past token so an "
    "autoregressive model never recomputes them. It makes generation fast, but grows "
    "linearly with context length. As windows stretch from thousands of tokens to "
    "millions, the cache outgrows a single accelerator and becomes the primary limit "
    "on scalable inference. "
    "The arithmetic is unforgiving. A seven billion parameter model with thirty-two "
    "layers and heads of dimension one hundred twenty-eight stores half a megabyte of "
    "cache for each token it has seen. At a context of one hundred twenty-eight "
    "thousand tokens that is sixty-four gigabytes, more than the largest widely "
    "deployed accelerator can spare once weights and activations are resident. "
    "Hardware trends make the squeeze worse rather than better. Compute has grown "
    "faster than memory bandwidth for a decade, and bandwidth has grown faster than "
    "capacity. Serving systems therefore hit the memory wall first: the accelerator "
    "idles while tensors stream in from high-bandwidth memory, and batch sizes shrink "
    "until utilization collapses. "
    "One family of responses evicts tokens judged unimportant, betting that attention "
    "is sparse enough for the model never to miss them. Another keeps every token but "
    "quantizes the stored numbers down to four, two, or even fewer bits, trading "
    "numerical precision for capacity. A third moves the cache to host memory or "
    "disk and pages the working set back on demand, paying in bandwidth instead of "
    "accuracy. The most radical line of work redesigns attention itself so that no "
    "per-token state accumulates at all. "
    "Evaluation is its own problem. Perplexity on held-out text barely moves under "
    "aggressive compression, yet the same model may fail to retrieve a name buried "
    "mid-document or silently ignore one instruction out of five. Benchmarks built "
    "around retrieval depth, multi-turn reuse, and instruction following expose "
    "failures that a single scalar score conceals. "
    "Deployment settings pull the design space in different directions. A phone "
    "assistant wants a tiny resident cache and tolerates approximation; a datacenter "
    "serving thousands of concurrent sessions cares about throughput and cache "
    "sharing; an agent reasoning over a repository for an hour needs its early "
    "conclusions intact at the end. No single method wins everywhere, which is why "
    "fair comparison at a matched budget matters more than any headline number."
)


def _score(model, cont_ids, past_key_values, start_pos):
    import torch
    from .model import tuples_to_cache
    pos = torch.arange(start_pos, start_pos + cont_ids.shape[1], device=cont_ids.device).unsqueeze(0)
    out = model(input_ids=cont_ids, past_key_values=tuples_to_cache(past_key_values),
                position_ids=pos, use_cache=False)
    logits, targets = out.logits[:, :-1, :], cont_ids[:, 1:]
    nll = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return float(math.exp(nll.item()))


KIVI_BIT_CANDIDATES = (2, 4, 8)


def derive_kwargs(key: str, prefill: int, ratio: float, recent: int, cfg) -> dict[str, int]:
    """Hyperparameters that bring a method nearest the target compression ratio
    (fraction of cache bytes removed). Eviction budgets follow directly from the
    token count; KIVI's bit width is picked by its true byte accounting, fp16
    residual included."""
    keep = 1.0 - ratio
    if key in ("h2o", "snapkv"):
        budget = max(1, round(prefill * keep))
        window = {"recent": recent} if key == "h2o" else {"window": recent}
        return {"budget": budget, **window}
    if key == "kivi":
        from .methods import FullCache, KIVIQuant
        full_bytes = FullCache().kv_bytes(prefill, cfg)
        best = min(KIVI_BIT_CANDIDATES,
                   key=lambda b: abs(KIVIQuant(bits=b).kv_bytes(prefill, cfg) / full_bytes - keep))
        return {"bits": best}
    return {}


def run(method_specs, model_name="distilgpt2", text=SAMPLE_TEXT,
        prefill=384, cont=64, ratio=0.75, recent=32):
    import torch
    from .model import cache_to_tuples, load_model

    model, tok, cfg = load_model(model_name)
    logger.info("model %s: layers=%d kv_heads=%d head_dim=%d", cfg.name, cfg.layers, cfg.n_kv_heads, cfg.head_dim)

    ids = tok(text, return_tensors="pt").input_ids
    prefill = min(prefill, ids.shape[1] - cont - 1)
    prefill_ids, cont_ids = ids[:, :prefill], ids[:, prefill: prefill + cont]

    with torch.no_grad():
        base = model(prefill_ids, use_cache=True, output_attentions=True)
    base_pkv, attns = cache_to_tuples(base.past_key_values), base.attentions
    if len(attns) != len(base_pkv):
        raise RuntimeError(
            f"got {len(attns)} attention tensors for {len(base_pkv)} cache layers; "
            "the model must run with attn_implementation='eager' to expose attentions")

    specs = [(s, {}) if isinstance(s, str) else s for s in method_specs]
    rows, baseline_ppl = [], None
    full_bytes = 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * 2.0 * prefill
    for key, kw in specs:
        method = build(key, **{**derive_kwargs(key, prefill, ratio, recent, cfg), **kw})
        logger.info("method %s (%s, lever=%s)", method.name, method.family, method.lever)
        pkv = tuple((k.clone(), v.clone()) for k, v in base_pkv)
        pkv = method.apply(pkv, attns)
        with torch.no_grad():
            ppl = _score(model, cont_ids, pkv, start_pos=prefill)

        kept = method.kept_len(prefill)
        kv_bytes = method.kv_bytes(prefill, cfg)
        if key == "full":
            baseline_ppl = ppl
        delta = 0.0 if baseline_ppl is None else (ppl - baseline_ppl) / baseline_ppl * 100
        rows.append(Row(method.name, method.family, kept, method.bits,
                        kv_bytes / memory.MB, kv_bytes / full_bytes, ppl, delta))
    return rows, cfg


def format_table(rows, cfg) -> str:
    header = f"{'method':<18}{'family':<12}{'kept':>7}{'bits':>6}{'KV MB':>9}{'size':>7}{'ppl':>9}{'d_ppl%':>8}"
    lines = [f"model: {cfg.name} (layers={cfg.layers}, kv_heads={cfg.n_kv_heads}, head_dim={cfg.head_dim})",
             "", header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r.method:<18}{r.family:<12}{r.kept_tokens:>7}{r.bits:>6.1f}"
                     f"{r.kv_mb:>9.3f}{r.size_ratio:>7.3f}{r.perplexity:>9.2f}{r.delta_pct:>+8.1f}")
    return "\n".join(lines)


def write_results(rows, path_stem, config: Mapping[str, object] | None = None):
    """CSV holds the rows for spreadsheet use; JSON embeds the run configuration
    so a results file identifies the model, ratio, and library versions that
    produced it."""
    with open(f"{path_stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in Row.__dataclass_fields__.values()])
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    with open(f"{path_stem}.json", "w") as f:
        json.dump({"config": dict(config or {}), "rows": [asdict(r) for r in rows]}, f, indent=2)
