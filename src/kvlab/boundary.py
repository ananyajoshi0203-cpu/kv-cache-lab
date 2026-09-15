"""Where score-free eviction stops being enough.

The question this runner exists to answer is not "which evictor wins". It is: at
what prompt length, generation length and cache budget does a scored evictor start
to beat one that keeps the prompt and then flips a coin (kvlab.methods
PromptProtectedRandom, arXiv:2609.03430)? Until that crossing is located, a win
over random cannot be attributed to the score.

Everything is swept against one controlled variable: **the retained fraction of the
full cache**. Every method in a row is held to the same final cache size, and the
nominal budget each one needs to get there is derived per method and recorded,
because the methods bound different things.

- H2O bounds the whole cache at every decode step, so its budget is the target.
- SnapKV, CAKE and OBCache compress the prompt once after prefill and then grow by
  one position per generated token, so their budget is the target minus that
  growth.
- Prompt-protected random takes a budget over the generated cache only, so its
  budget is the target minus the prompt it refuses to touch.

Handing all of them one number instead is how an "iso-budget" table comes to
compare three different cache sizes and call it controlled. Where a method cannot
reach the target the cell is skipped with a reason, never fudged -- and the most
important of those skips is a result in itself: below a retained fraction that
holds the prompt, a prompt-protected method has no feasible configuration at all,
which is the honest form of "the protected prompt makes total memory differ".

Two regimes, answering different questions.

**Published.** Every method runs exactly as its paper defines it, at the shared
target. The prompt/generated split in the accounting columns is what to read here:
at equal memory, how much of each method's cache went to the prompt.

**Prompt-protected.** Methods are wrapped so the prompt is untouchable and the
target is spent on the generated cache alone. A difference between methods here is
a difference in what the score bought, with prompt fragility controlled for.

Three of the six methods are *not* in the prompt-protected regime, and the reason
is a Phase 1 finding rather than an omission. SnapKV, CAKE and OBCache vote with an
observation window of prompt queries. Protect the prompt and that window is gone:
all that is left is the single query of one decode step, and a scorer reading one
query is a different scorer wearing the same name. Giving them a faithful
decode-time form is method design, which this runner deliberately does not do.

Budgets are carried in every framing at once and never converted implicitly; see
Budget. This is deliberately not routed through benchmark.derive_kwargs, whose
single compression ratio over a prefill cache has no room for a protected prompt.

Metrics: the task score is retrieval, reusing the passkey infrastructure in
kvlab.needle, never perplexity alone. One known limitation, stated rather than
hidden: on passkey retrieval the model answers within the first few generated
tokens, so a budget over the generated cache cannot move the score on this
workload. What makes the generated axis bite is a workload whose answer depends on
its own trace (reasoning, multi-instruction), and Workload is a two-function
interface so adding one is adding a Workload, not editing this runner.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from enum import Enum
from statistics import mean, pstdev

from . import needle
from .accounting import CacheLedger
from .memory import ModelConfig
from .methods import KVMethod, build

logger = logging.getLogger(__name__)

KV_BYTES_KIND = "analytical"
RECENT_SHARE = 4       # a method's recency window may claim at most 1/4 of its budget
TOKEN_TOLERANCE = 8    # how close a built prompt must land to its requested length

UNREACHABLE_BY_PRESS = (
    "an after-prefill press cannot reach the target: it compresses the prompt once and then "
    "accumulates every generated token, and that accumulation alone already overflows the target")
UNREACHABLE_BY_PROTECTION = (
    "a protected prompt does not fit the target: keeping the whole prompt already costs more "
    "than the target cache, so there is no generated budget left to state")

#: Why an after-prefill press has no faithful prompt-protected form, keyed by
#: method so a results file carries the reason next to the missing rows.
NO_PROMPT_PROTECTED_FORM = dict.fromkeys(
    ("snapkv", "cake", "obcache"),
    "votes with an observation window of prompt queries; once the prompt is protected only the "
    "single query of a decode step is left, which is a different scorer under the same name")

SCORED_KEYS = ("h2o", "snapkv", "cake", "obcache")
DEFAULT_METHODS = ("full", "random", *SCORED_KEYS)
_WINDOW_KWARG = {"h2o": "recent", "obcache": "recent", "snapkv": "window", "cake": "window"}


class Regime(str, Enum):
    PUBLISHED = "published"
    PROMPT_PROTECTED = "prompt_protected"


@dataclass(frozen=True)
class Budget:
    """One target cache size, carried in every framing a method might take.

    A generated-token budget and a total-token budget are different quantities, and
    reporting them in one column is how a prompt-protected method comes to look
    like it wins for free. Nothing here converts between them silently: a caller
    states the fraction of the cache to retain, and each method reads the framing
    it actually takes, or learns that it has none.
    """

    prompt_length: int
    generation_length: int
    retained_fraction: float

    @property
    def cached_generation_length(self) -> int:
        """Generated positions that actually enter the cache. The final generated
        token's KV never does: nothing attends to it."""
        return max(0, self.generation_length - 1)

    @property
    def full_length(self) -> int:
        return self.prompt_length + self.cached_generation_length

    @property
    def total_budget(self) -> int:
        """Cache positions every method in this cell is held to."""
        return max(1, round(self.retained_fraction * self.full_length))

    @property
    def generation_budget(self) -> int:
        """What is left for the generated cache once the prompt is protected.
        Negative means a protected prompt cannot fit the target at all."""
        return self.total_budget - self.prompt_length

    def nominal_for(self, key: str, regime: Regime) -> tuple[int | None, str]:
        """The budget this method needs to finish at total_budget, with the reason
        when it has none."""
        if key == "full":
            return 0, ""
        if key == "random" or regime is Regime.PROMPT_PROTECTED:
            budget = self.generation_budget
            return (budget, "") if budget >= 0 else (None, UNREACHABLE_BY_PROTECTION)
        if key == "h2o":
            return self.total_budget, ""
        budget = self.total_budget - self.cached_generation_length
        return (budget, "") if budget >= 1 else (None, UNREACHABLE_BY_PRESS)

    def window(self, nominal_budget: int) -> int:
        """A recency window large enough to swallow its own budget turns any scorer
        into plain sliding-window attention, silently. Capped, and the cap is
        reported per row so a degenerate setting is visible rather than inferred."""
        return max(1, nominal_budget // RECENT_SHARE)


@dataclass(frozen=True)
class Example:
    context: str
    question: str
    answer: str


@dataclass(frozen=True)
class Workload:
    """A task the sweep can run. Adding reasoning, multi-instruction or code means
    adding one of these; the runner below does not change."""

    key: str
    task_metric: str
    note: str
    build: Callable[..., Example]
    score: Callable[[Example, str], float]


def _needle_example(tokenizer, target_prompt_tokens: int, index: int, examples: int,
                    seed: int) -> Example:
    """A passkey buried at a controlled depth, with the filler grown or shrunk
    until the tokenized prompt lands near the requested length."""
    passkey = random.Random(seed * 1_000 + index).randint(10_000, 99_999)
    depth = index / max(1, examples - 1)
    sentences = max(1, target_prompt_tokens // 12)
    context, question = "", ""
    for _ in range(8):
        filler_rng = random.Random(seed * 1_000 + index + 500_000)
        context, question = needle.make_example(passkey, sentences, depth, filler_rng)
        length = len(tokenizer(context + question).input_ids)
        if abs(length - target_prompt_tokens) <= TOKEN_TOLERANCE or sentences <= 1:
            break
        sentences = max(1, round(sentences * target_prompt_tokens / length))
    return Example(context, question, str(passkey))


WORKLOADS = {
    "needle": Workload(
        key="needle",
        task_metric="passkey_em",
        note="passkey retrieval; the answer is emitted within the first generated tokens, so a "
             "budget over the generated cache cannot move this score",
        build=_needle_example,
        score=lambda example, generated: float(example.answer in generated)),
}


class PromptProtected(KVMethod):
    """Holds the prompt out of an inner method's reach and spends the budget on the
    generated cache alone.

    This is a modification of what it wraps and it carries its own key and name so
    it is never reported under a published one: the prompt stops being a candidate
    for eviction, which none of the wrapped papers say. The scorer itself is
    untouched. It sees the generated sub-cache and the attention columns over it and
    scores them exactly as it always does, which is why only a method that already
    scores during decoding can be wrapped: see NO_PROMPT_PROTECTED_FORM.
    """

    family, lever, phase = "Eviction", "context", "decoding"

    def __init__(self, inner: KVMethod, generation_budget: int):
        if inner.key in NO_PROMPT_PROTECTED_FORM:
            raise ValueError(f"{inner.name} {NO_PROMPT_PROTECTED_FORM[inner.key]}")
        self.inner = inner
        self.generation_budget = generation_budget
        self.key = f"pp-{inner.key}"
        self.name = f"{inner.name} (prompt-protected)"
        self.bits = inner.bits
        self.prompt_length: int | None = None

    def apply(self, past_key_values, attentions):
        self.prompt_length = past_key_values[0][0].shape[2]
        self.last_indices = None
        return past_key_values

    def step(self, past_key_values, attentions):
        import torch

        if self.prompt_length is None:
            raise RuntimeError("apply() must run on the prefilled cache before step()")
        prompt = self.prompt_length
        generated_kv = tuple((key[:, :, prompt:], value[:, :, prompt:])
                             for key, value in past_key_values)
        generated_attn = tuple(attn[..., prompt:] for attn in attentions)
        kept = self.inner.step(generated_kv, generated_attn)

        inner_indices = self.inner.last_indices
        merged, indices = [], []
        for layer, ((key, value), (kept_key, kept_value)) in enumerate(zip(past_key_values, kept)):
            merged.append((torch.cat([key[:, :, :prompt], kept_key], dim=2),
                           torch.cat([value[:, :, :prompt], kept_value], dim=2)))
            inner_idx = None if inner_indices is None else inner_indices[layer]
            if inner_idx is None:
                indices.append(None)
                continue
            batch, heads = inner_idx.shape[:2]
            protected = (torch.arange(prompt, device=inner_idx.device)
                         .view(1, 1, prompt).expand(batch, heads, prompt))
            indices.append(torch.cat([protected, inner_idx + prompt], dim=-1))
        self.last_indices = tuple(indices)
        return tuple(merged)

    def kept_len(self, orig_len: int) -> int:
        if self.prompt_length is None:
            return orig_len
        generated = max(0, orig_len - self.prompt_length)
        return self.prompt_length + min(self.generation_budget, generated)


def budget_metadata(regime: Regime, key: str, budget: Budget) -> dict[str, int]:
    """The budget facts that belong to a cell rather than to a decode.

    These are derived here and never returned from a memoized decode. A decode is
    reused across cells it cannot tell apart -- the full cache ignores the budget
    entirely, so it runs once and is reported at every retained fraction -- and a
    cached result that carried its own budget would stamp the first cell's numbers
    onto every later one, which is a silent bookkeeping error rather than a loud
    failure. So the reused part is the decode, and these are recomputed per row.
    """
    nominal, _ = budget.nominal_for(key, regime)
    nominal = 0 if nominal is None else nominal
    return {"total_budget": budget.total_budget,
            "generation_budget": budget.generation_budget,
            "nominal_budget": nominal,
            "window": budget.window(nominal) if key in SCORED_KEYS else 0}


def method_for(regime: Regime, key: str, budget: Budget,
               seed: int) -> tuple[KVMethod, int, int] | str:
    """The method, the nominal budget it was handed and the recency window it ended
    up with, or the reason it has no configuration that reaches this cell's target.
    Returning the budget alongside the method is the point: a row that does not
    record which budget a method was actually given cannot be compared with a row
    given another framing."""
    if regime is Regime.PROMPT_PROTECTED and key in NO_PROMPT_PROTECTED_FORM:
        return NO_PROMPT_PROTECTED_FORM[key]
    nominal, reason = budget.nominal_for(key, regime)
    if nominal is None:
        return reason
    if key == "full":
        return build("full"), 0, 0
    if key == "random":
        return build("random", generation_budget=nominal, seed=seed), nominal, 0
    if key not in SCORED_KEYS:
        raise KeyError(f"{key} is not a method this sweep knows how to budget")

    window = budget_metadata(regime, key, budget)["window"]
    inner = build(key, budget=nominal, **{_WINDOW_KWARG[key]: window})
    if regime is Regime.PUBLISHED:
        return inner, nominal, window
    return PromptProtected(inner, nominal), nominal, window


@dataclass(frozen=True)
class BoundaryRow:
    model: str
    regime: str
    method: str
    method_key: str
    seed: int
    example: int
    task: str
    task_metric: str
    metric_value: float
    requested_prompt_length: int
    prompt_length: int
    requested_generation_length: int
    actual_generation_length: int
    retained_fraction: float
    total_budget: int
    generation_budget: int
    nominal_budget: int
    window: int
    prompt_retained: float
    generated_retained: float
    total_retained: float
    kv_bytes: float
    kv_bytes_kind: str
    compression_ratio: float
    decode_wall_seconds: float


@dataclass(frozen=True)
class Cell:
    """One point of the sweep grid, before a method or a seed is chosen."""

    prompt_length: int
    generation_length: int
    retained_fraction: float


@dataclass(frozen=True)
class Skip:
    """A cell that was not run, and why. Recorded rather than dropped: a missing row
    in a sweep is a finding when it is explained and a bug when it is not."""

    regime: str
    method_key: str
    prompt_length: int
    generation_length: int
    retained_fraction: float
    reason: str


@dataclass(frozen=True)
class SweepResult:
    rows: list[BoundaryRow]
    skipped: list[Skip]
    cfg: ModelConfig


def grid(prompt_lengths: Iterable[int], generation_lengths: Iterable[int],
         retained_fractions: Iterable[float]) -> list[Cell]:
    return [Cell(prompt, generation, fraction)
            for prompt in prompt_lengths
            for generation in generation_lengths
            for fraction in retained_fractions]


def _memo_key(regime: Regime, method_key: str, cell: Cell, seed: int, index: int) -> tuple:
    """What a decode actually depends on, so nothing is run twice for a distinction
    it cannot see. The full cache ignores the budget entirely, and both it and
    random behave identically in either regime -- random is prompt-protected by
    construction -- so they are decoded once and reported under every cell that
    shares what they do depend on. Only the decode is shared: everything a cell
    knows and a decode does not comes from budget_metadata, per row."""
    if method_key == "full":
        return ("-", "full", cell.prompt_length, cell.generation_length, seed, index)
    return (regime.value if method_key in SCORED_KEYS else "-", method_key, cell, seed, index)


def sweep(model_name: str = "distilgpt2", *, workload_key: str = "needle",
          prompt_lengths: Sequence[int] = (128, 384),
          generation_lengths: Sequence[int] = (48,),
          retained_fractions: Sequence[float] = (0.25, 0.5, 0.75),
          seeds: Sequence[int] = (0, 1, 2, 3, 4),
          examples: int = 1,
          regimes: Sequence[Regime] = tuple(Regime),
          method_keys: Sequence[str] | None = None) -> SweepResult:
    from .decode import generate_stepwise
    from .model import load_model

    workload = WORKLOADS[workload_key]
    model, tokenizer, cfg = load_model(model_name)
    limit = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", 0)
    memo: dict[tuple, dict | str] = {}
    rows: list[BoundaryRow] = []
    skipped: list[Skip] = []

    def note(regime: Regime, key: str, cell: Cell, reason: str) -> None:
        skip = Skip(regime.value, key, cell.prompt_length, cell.generation_length,
                    cell.retained_fraction, reason)
        if skip not in skipped:
            skipped.append(skip)
            logger.info("skipping %s/%s at prompt=%d retain=%.2f: %s", regime.value, key,
                        cell.prompt_length, cell.retained_fraction, reason)

    for cell in grid(prompt_lengths, generation_lengths, retained_fractions):
        if limit and cell.prompt_length + cell.generation_length > limit:
            raise ValueError(
                f"prompt {cell.prompt_length} + generation {cell.generation_length} exceeds the "
                f"{limit}-position context of {model_name}")
        for regime in regimes:
            for key in (method_keys or DEFAULT_METHODS):
                for seed in seeds:
                    for index in range(examples):
                        memo_key = _memo_key(regime, key, cell, seed, index)
                        if memo_key not in memo:
                            memo[memo_key] = _one(model, tokenizer, cfg, workload, regime, key,
                                                  cell, seed, index, examples, generate_stepwise)
                        measured = memo[memo_key]
                        if isinstance(measured, str):
                            note(regime, key, cell, measured)
                            continue
                        budget = Budget(measured["prompt_length"], cell.generation_length,
                                        cell.retained_fraction)
                        rows.append(BoundaryRow(
                            model=cfg.name, regime=regime.value, seed=seed, example=index,
                            task=workload.key, task_metric=workload.task_metric,
                            requested_prompt_length=cell.prompt_length,
                            requested_generation_length=cell.generation_length,
                            retained_fraction=cell.retained_fraction,
                            kv_bytes_kind=KV_BYTES_KIND,
                            **budget_metadata(regime, key, budget), **measured))
                        logger.debug("%s", rows[-1])
    return SweepResult(rows=rows, skipped=skipped, cfg=cfg)


def _one(model, tokenizer, cfg, workload: Workload, regime: Regime, key: str, cell: Cell,
         seed: int, index: int, examples: int, generate_stepwise) -> dict | str:
    example = workload.build(tokenizer, cell.prompt_length, index, examples, seed)
    ids = tokenizer(example.context + example.question, return_tensors="pt").input_ids
    budget = Budget(ids.shape[1], cell.generation_length, cell.retained_fraction)
    chosen = method_for(regime, key, budget, seed)
    if isinstance(chosen, str):
        return chosen
    method, _nominal, _window = chosen

    ledger = CacheLedger()
    started = time.perf_counter()
    generated = generate_stepwise(model, ids, method,
                                  max_new_tokens=cell.generation_length, ledger=ledger)
    elapsed = time.perf_counter() - started
    text = tokenizer.decode(generated[0], skip_special_tokens=True)
    account = ledger.account(cfg)
    return {
        "method": method.name, "method_key": method.key,
        "metric_value": workload.score(example, text),
        "prompt_length": ids.shape[1],
        "actual_generation_length": int(generated.shape[1]),
        "prompt_retained": account.prompt_retained,
        "generated_retained": account.generated_retained,
        "total_retained": account.total_retained,
        "kv_bytes": account.total_kv_bytes,
        "compression_ratio": account.compression_ratio,
        "decode_wall_seconds": elapsed,
    }


@dataclass(frozen=True)
class Summary:
    """One grid point for one method, aggregated over seeds and nothing else.

    Every independent variable of the sweep is part of the grouping key, generation
    length included: a summary that averaged a 64-token generation together with a
    384-token one would be reporting a number that describes neither. The dispersion
    column is why seeds are swept at all, since a single lucky draw of a score-free
    method is not a result and neither is a single unlucky one."""

    regime: str
    method: str
    prompt_length: int
    generation_length: int
    retained_fraction: float
    seeds: int
    metric_mean: float
    metric_stdev: float
    prompt_retained: float
    total_retained: float
    kv_bytes: float


def summarize(rows: Iterable[BoundaryRow]) -> list[Summary]:
    grouped: dict[tuple, list[BoundaryRow]] = {}
    for row in rows:
        grouped.setdefault(
            (row.regime, row.method, row.requested_prompt_length,
             row.requested_generation_length, row.retained_fraction),
            []).append(row)
    summaries = []
    for (regime, method, prompt_length, generation_length, fraction), group in grouped.items():
        per_seed: dict[int, list[float]] = {}
        for row in group:
            per_seed.setdefault(row.seed, []).append(row.metric_value)
        seed_means = [mean(values) for values in per_seed.values()]
        summaries.append(Summary(
            regime=regime, method=method, prompt_length=prompt_length,
            generation_length=generation_length,
            retained_fraction=fraction, seeds=len(seed_means),
            metric_mean=mean(seed_means), metric_stdev=pstdev(seed_means),
            prompt_retained=mean(row.prompt_retained for row in group),
            total_retained=mean(row.total_retained for row in group),
            kv_bytes=mean(row.kv_bytes for row in group)))
    return summaries


def format_summary(summaries: Sequence[Summary], task_metric: str) -> str:
    header = (f"{'regime':<17}{'method':<32}{'prompt':>7}{'gen':>6}{'retain':>7}{'seeds':>6}"
              f"{task_metric:>12}{'stdev':>7}{'kept':>8}{'prompt_kept':>12}{'KV KB':>9}")
    lines = [header, "-" * len(header)]
    order = sorted(summaries, key=lambda s: (s.regime, s.prompt_length, s.generation_length,
                                             s.retained_fraction, s.method))
    for s in order:
        lines.append(f"{s.regime:<17}{s.method:<32}{s.prompt_length:>7}{s.generation_length:>6}"
                     f"{s.retained_fraction:>7.2f}{s.seeds:>6}{s.metric_mean:>12.2f}"
                     f"{s.metric_stdev:>7.2f}{s.total_retained:>8.1f}{s.prompt_retained:>12.1f}"
                     f"{s.kv_bytes / 1024:>9.1f}")
    return "\n".join(lines)


def write_rows(rows: Sequence[BoundaryRow], summaries: Sequence[Summary], path_stem: str,
               config: dict) -> None:
    """CSV of every example for a spreadsheet, JSONL of the same rows for streaming,
    and a JSON carrying the run configuration and the per-cell summary so a results
    file identifies what produced it."""
    fields = [f.name for f in BoundaryRow.__dataclass_fields__.values()]
    with open(f"{path_stem}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    with open(f"{path_stem}.jsonl", "w") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row)) + "\n")
    with open(f"{path_stem}.json", "w") as handle:
        json.dump({"config": config, "summary": [asdict(s) for s in summaries]}, handle, indent=2)
