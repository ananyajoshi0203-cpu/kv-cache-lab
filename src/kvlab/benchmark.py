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


SAMPLE_TEXT = (
    "The key-value cache stores the key and value vectors of every past token so an "
    "autoregressive model never recomputes them. It makes generation fast, but grows "
    "linearly with context length. As windows stretch from thousands of tokens to "
    "millions, the cache outgrows a single accelerator and becomes the primary limit "
    "on scalable inference. Responses fall into a few families: evicting unimportant "
    "tokens, quantizing the stored numbers, offloading the cache to slower memory, and "
    "redesigning attention so no per-token cache is needed. No single family wins "
    "everywhere; the right choice depends on context length, hardware, and how much "
    "accuracy the task can lose. "
) * 6


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


def write_results(rows, path_stem):
    with open(f"{path_stem}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[f.name for f in Row.__dataclass_fields__.values()])
        w.writeheader()
        w.writerows(asdict(r) for r in rows)
    with open(f"{path_stem}.json", "w") as f:
        json.dump([asdict(r) for r in rows], f, indent=2)
