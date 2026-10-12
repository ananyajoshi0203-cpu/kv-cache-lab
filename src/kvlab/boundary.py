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

Two workloads, because they fail differently and the study needs both.

- **needle** (kvlab.needle) probes prompt retention. Its answer is emitted within
  the first few generated tokens, so a budget over the generated cache cannot move
  its score. That is a limitation of the task, stated rather than hidden, and it is
  precisely why it cannot be the only one.
- **multistep** (kvlab.multistep) gives the generated cache something to do. Each
  step consumes the running total the model just wrote, so the natural way to answer
  is to read back what it already wrote. It does *not* follow that the operand exists
  only in the generated KV: every fact is in the prompt, so the trace is a shortcut
  the model may or may not lean on, and whether it does is measured by the trace
  ablation in kvlab.ablation rather than assumed. multistep also carries the
  redundancy variable: at a fixed prompt length and a fixed answer, its facts can be
  stated once or several times in different words while filler shrinks to compensate,
  which moves the share of prompt KV carrying information the model has seen elsewhere
  without moving the answer. That separates a redundancy account of the boundary from
  a context-fraction one.

Both are scored on the final answer only, never on the shape of the reasoning, and
never with perplexity alone. Workload is a two-function interface, so adding a third
is adding a Workload rather than editing this runner.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from enum import Enum
from statistics import mean, pstdev

from . import ablation, multistep, needle
from .accounting import CacheLedger
from .memory import ModelConfig
from .methods import KVMethod, build

logger = logging.getLogger(__name__)

KV_BYTES_KIND = "analytical"
RECENT_SHARE = 4       # a method's recency window may claim at most 1/4 of its budget
TOKEN_TOLERANCE = 8            # floor on how close a built prompt must land to its target
PROMPT_TOLERANCE_FRACTION = 0.05

PROMPT_TARGET_UNREACHABLE = (
    "the workload cannot build a prompt near this length: its evidence and instructions alone "
    "overflow the target, so the cell would silently run at a different prompt length than it "
    "claims and redundancy would be confounded with prompt length")

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

#: Diagnostics, not cache methods, and never to be reported as such. They keep the
#: prompt and remove generated KV by what the token was, to test whether the model's
#: computation actually routes through its own trace. See kvlab.ablation.
ABLATION_KEYS = tuple(f"ablate-{target}" for target in ablation.TARGETS)

#: Methods that protect the prompt by construction, so the prompt-protected regime
#: asks nothing new of them and their budget is always over the generated cache.
PROMPT_PROTECTING_KEYS = ("random", *ABLATION_KEYS)

#: Methods whose behaviour depends on an eviction draw. Everything else is a
#: deterministic function of the task, so it is decoded once and reported once
#: rather than once per eviction seed.
STOCHASTIC_KEYS = PROMPT_PROTECTING_KEYS

#: The eviction seed recorded for a deterministic method. Not a seed that was used:
#: a marker that the row has no eviction draw behind it, so nothing downstream can
#: average a deterministic method over draws it never made.
NO_EVICTION_SEED = -1
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
        if key in PROMPT_PROTECTING_KEYS or regime is Regime.PROMPT_PROTECTED:
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


NO_REDUNDANCY = "none"


@dataclass(frozen=True)
class Example:
    context: str
    question: str
    answer: str
    #: How much unique evidence this example carries and how many times it is
    #: stated. Recorded per row so redundancy is a measured quantity in the results
    #: and not just the name of the setting that produced them.
    facts: int = 0
    statements: int = 0


@dataclass(frozen=True)
class Workload:
    """A task the sweep can run. Adding multi-instruction or code means adding one of
    these; the runner below does not change."""

    key: str
    task_metric: str
    note: str
    build: Callable[..., Example]
    score: Callable[[Example, str], float]
    #: Redundancy levels this workload can vary. A workload with nothing to vary
    #: declares NO_REDUNDANCY, which keeps the column present and honest rather than
    #: blank for some rows and meaningful for others.
    redundancies: tuple[str, ...] = (NO_REDUNDANCY,)


def prompt_band(target_prompt_tokens: int) -> int:
    """How far a built prompt may land from its requested length before the cell stops
    being the cell that was asked for. Prompt length is an independent variable here,
    and for the multistep workload it is also the denominator of the redundancy
    control: if a high-redundancy prompt runs long because its evidence does not fit,
    redundancy and prompt length move together and neither can be read."""
    return max(TOKEN_TOLERANCE, round(PROMPT_TOLERANCE_FRACTION * target_prompt_tokens))


def _fit_filler(build_at, target_prompt_tokens: int, tokenizer):
    """Grow or shrink the filler until the tokenized prompt lands near its target.

    Every workload needs this and none of them should implement it twice: prompt
    length is an independent variable of the sweep, so an example that misses its
    target by a wide margin is a different cell than the one that was asked for.
    """
    def measure(built):
        return len(tokenizer(built.context + built.question).input_ids)

    def miss(length):
        return abs(length - target_prompt_tokens)

    band = prompt_band(target_prompt_tokens)
    sentences = max(1, target_prompt_tokens // 12)
    built = build_at(sentences)
    length = measure(built)
    for _ in range(8):
        scaled = max(0, round(sentences * target_prompt_tokens / length))
        if miss(length) <= band or scaled == sentences:
            break
        sentences = scaled
        built = build_at(sentences)
        length = measure(built)

    # Then walk one sentence at a time. Scaling cannot land inside the band when a
    # single filler sentence is a large share of the slack, which is precisely the
    # high-redundancy case: most of the prompt is evidence, few filler sentences are
    # left to trade, and the example would otherwise be refused for missing a target
    # it could have hit.
    step = 1 if length < target_prompt_tokens else -1
    for _ in range(12):
        if miss(length) <= band or sentences + step < 0:
            break
        candidate = build_at(sentences + step)
        candidate_length = measure(candidate)
        if miss(candidate_length) >= miss(length):
            break
        sentences += step
        built, length = candidate, candidate_length
    return built


def _needle_example(tokenizer, target_prompt_tokens: int, index: int, examples: int,
                    seed: int, redundancy: str) -> Example:
    """A passkey buried at a controlled depth. Redundancy does not apply: the task is
    one unique fact by construction, which is what makes it a prompt-retention probe
    in the first place."""
    passkey = random.Random(seed * 1_000 + index).randint(10_000, 99_999)
    depth = index / max(1, examples - 1)

    def at(sentences: int) -> Example:
        rng = random.Random(seed * 1_000 + index + 500_000)
        context, question = needle.make_example(passkey, sentences, depth, rng)
        return Example(context, question, str(passkey), facts=1, statements=1)

    return _fit_filler(at, target_prompt_tokens, tokenizer)


def _multistep_example(tokenizer, target_prompt_tokens: int, index: int, examples: int,
                       seed: int, redundancy: str) -> Example:
    """A fact chain whose later operands exist only in the model's own trace. The
    chain, and therefore the answer, is a function of the seed and the example index
    alone: redundancy changes how often the facts are stated and nothing else, so a
    row at high redundancy is the same question as its low-redundancy twin."""
    chain = multistep.build_chain(MULTISTEP_STEPS, random.Random(seed * 1_000 + index))
    statements = len(multistep.evidence_sentences(chain, redundancy))

    def at(sentences: int) -> Example:
        rng = random.Random(seed * 1_000 + index + 500_000)
        context = multistep.compose(chain, redundancy, sentences, rng)
        return Example(context, multistep.PROMPT_TAIL, str(chain.answer),
                       facts=len(chain.values), statements=statements)

    return _fit_filler(at, target_prompt_tokens, tokenizer)


MULTISTEP_STEPS = 4

WORKLOADS = {
    "needle": Workload(
        key="needle",
        task_metric="passkey_em",
        note="passkey retrieval, a prompt-retention probe; the answer is emitted within the "
             "first generated tokens, so a budget over the generated cache cannot move it",
        build=_needle_example,
        score=lambda example, generated: float(example.answer in generated)),
    "multistep": Workload(
        key="multistep",
        task_metric="final_answer_em",
        note="a fact chain whose later operands exist only in the generated trace, scored on "
             "the last integer written and on nothing about the reasoning",
        build=_multistep_example,
        score=lambda example, generated: multistep.score_answer(generated, int(example.answer)),
        redundancies=tuple(multistep.REDUNDANCY_LEVELS)),
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


def method_for(regime: Regime, key: str, budget: Budget, seed: int,
               numeric_tokens: frozenset[int] | None = None) -> tuple[KVMethod, int, int] | str:
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
    if key in ABLATION_KEYS:
        if numeric_tokens is None:
            raise ValueError(f"{key} needs the vocabulary's numeric token ids; they come "
                             "from the tokenizer that will run the sweep")
        return ablation.TraceAblation(
            generation_budget=nominal, target=key.removeprefix("ablate-"),
            numeric_tokens=numeric_tokens, seed=seed), nominal, 0
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
    #: The task instance: which facts, operations, filler and target this row ran on.
    task_seed: int
    #: The eviction draw, NO_EVICTION_SEED for a method that has none. Separated from
    #: the task seed because one number doing both jobs cannot say whether a random
    #: method's spread comes from the tasks it saw or from the draws it made.
    eviction_seed: int
    example: int
    task: str
    redundancy: str
    facts: int
    statements: int
    task_metric: str
    metric_value: float
    requested_prompt_length: int
    prompt_length: int
    requested_generation_length: int
    actual_generation_length: int
    cached_generation_length: int
    retained_fraction: float
    total_budget: int
    generation_budget: int
    nominal_budget: int
    window: int
    prompt_retained: float
    generated_retained: float
    total_retained: float
    generated_position_mean: float
    #: How many positions a trace ablation removed from the class it targets and from
    #: the other, so the two arms can be checked against each other rather than
    #: assumed matched. Zero for every method that is not a trace ablation.
    ablated_targeted: int
    ablated_other: int
    kv_bytes: float
    kv_bytes_kind: str
    compression_ratio: float
    decode_wall_seconds: float


@dataclass(frozen=True)
class Cell:
    """One point of the sweep grid, before a method or a seed is chosen."""

    workload: str
    redundancy: str
    prompt_length: int
    generation_length: int
    retained_fraction: float


@dataclass(frozen=True)
class Skip:
    """A cell that was not run, and why. Recorded rather than dropped: a missing row
    in a sweep is a finding when it is explained and a bug when it is not."""

    regime: str
    method_key: str
    workload: str
    redundancy: str
    prompt_length: int
    generation_length: int
    retained_fraction: float
    reason: str


@dataclass(frozen=True)
class SweepResult:
    rows: list[BoundaryRow]
    skipped: list[Skip]
    cfg: ModelConfig


Shape = tuple[int, int]     # (prompt length, generation length)


def grid(workload_keys: Iterable[str], shapes: Iterable[Shape],
         retained_fractions: Iterable[float],
         redundancies: Iterable[str] | None = None) -> list[Cell]:
    """Shapes are given as explicit (prompt, generation) pairs rather than crossed:
    the interesting matrix is not a full product, and a cross would spend most of its
    compute on shapes nobody asked about.

    `redundancies` None means each workload's own levels. Naming levels instead
    intersects with what a workload can vary, and a workload left with nothing is
    dropped loudly rather than quietly producing no rows."""
    cells = []
    for key in workload_keys:
        workload = WORKLOADS[key]
        levels = (tuple(workload.redundancies) if redundancies is None else
                  tuple(level for level in redundancies if level in workload.redundancies))
        if not levels:
            logger.warning("workload %s varies %s, none of which was requested; no cells for it",
                           key, ", ".join(workload.redundancies))
            continue
        cells.extend(Cell(key, level, prompt, generation, fraction)
                     for level in levels
                     for prompt, generation in shapes
                     for fraction in retained_fractions)
    return cells


def eviction_seeds_for(method_key: str, eviction_seeds: Sequence[int]) -> tuple[int, ...]:
    """A deterministic method gets one row, not one per eviction seed. Repeating its
    inference across draws it never makes would cost the same again and would also put
    duplicate rows into every mean and every bootstrap interval."""
    return tuple(eviction_seeds) if method_key in STOCHASTIC_KEYS else (NO_EVICTION_SEED,)


def _memo_key(regime: Regime, method_key: str, cell: Cell, task_seed: int,
              eviction_seed: int, index: int) -> tuple:
    """What a decode actually depends on, so nothing is run twice for a distinction
    it cannot see. The full cache ignores the budget entirely, and both it and
    random behave identically in either regime -- random is prompt-protected by
    construction -- so they are decoded once and reported under every cell that
    shares what they do depend on. Only the decode is shared: everything a cell
    knows and a decode does not comes from budget_metadata, per row."""
    draw = eviction_seed if method_key in STOCHASTIC_KEYS else NO_EVICTION_SEED
    if method_key == "full":
        return ("-", "full", cell.workload, cell.redundancy, cell.prompt_length,
                cell.generation_length, task_seed, index)
    return (regime.value if method_key in SCORED_KEYS else "-", method_key, cell,
            task_seed, index, draw)


#: Prompt-dominated through generation-dominated, as explicit pairs. Not every cell
#: is runnable by every workload: multistep needs room for its evidence, so its
#: shortest prompts are feasible at low redundancy only and the rest are skipped with
#: PROMPT_TARGET_UNREACHABLE rather than quietly run at the wrong prompt length.
DEFAULT_SHAPES: tuple[Shape, ...] = ((128, 64), (128, 256), (128, 512),
                                     (256, 64), (256, 256), (256, 512),
                                     (512, 64), (512, 256))
#: Enough points to show a transition rather than only its ends.
DEFAULT_FRACTIONS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 0.90)


def sweep(model_name: str = "distilgpt2", *,
          workload_keys: Sequence[str] = tuple(WORKLOADS),
          shapes: Sequence[Shape] = DEFAULT_SHAPES,
          retained_fractions: Sequence[float] = DEFAULT_FRACTIONS,
          redundancies: Sequence[str] | None = None,
          task_seeds: Sequence[int] = (0, 1, 2, 3, 4),
          eviction_seeds: Sequence[int] = (0, 1, 2, 3, 4),
          examples: int = 1,
          regimes: Sequence[Regime] = tuple(Regime),
          method_keys: Sequence[str] | None = None) -> SweepResult:
    from .decode import generate_stepwise
    from .model import load_model

    model, tokenizer, cfg = load_model(model_name)
    limit = getattr(model.config, "max_position_embeddings", None) or getattr(
        model.config, "n_positions", 0)
    memo: dict[tuple, dict | str] = {}
    rows: list[BoundaryRow] = []
    skipped: list[Skip] = []

    def note(regime: Regime, key: str, cell: Cell, reason: str) -> None:
        skip = Skip(regime.value, key, cell.workload, cell.redundancy, cell.prompt_length,
                    cell.generation_length, cell.retained_fraction, reason)
        if skip not in skipped:
            skipped.append(skip)
            logger.info("skipping %s/%s on %s at prompt=%d retain=%.2f: %s", regime.value, key,
                        cell.workload, cell.prompt_length, cell.retained_fraction, reason)

    for cell in grid(workload_keys, shapes, retained_fractions, redundancies):
        if limit and cell.prompt_length + cell.generation_length > limit:
            raise ValueError(
                f"prompt {cell.prompt_length} + generation {cell.generation_length} exceeds the "
                f"{limit}-position context of {model_name}")
        for regime in regimes:
            for key in (method_keys or DEFAULT_METHODS):
                for task_seed in task_seeds:
                    for index, draw in [(i, d) for i in range(examples)
                                        for d in eviction_seeds_for(key, eviction_seeds)]:
                        memo_key = _memo_key(regime, key, cell, task_seed, draw, index)
                        if memo_key not in memo:
                            memo[memo_key] = _one(model, tokenizer, cfg, regime, key, cell,
                                                  task_seed, draw, index, examples,
                                                  generate_stepwise)
                        measured = memo[memo_key]
                        if isinstance(measured, str):
                            note(regime, key, cell, measured)
                            continue
                        budget = Budget(measured["prompt_length"], cell.generation_length,
                                        cell.retained_fraction)
                        rows.append(BoundaryRow(
                            model=cfg.name, regime=regime.value, task_seed=task_seed,
                            eviction_seed=draw, example=index,
                            task=cell.workload, redundancy=cell.redundancy,
                            task_metric=WORKLOADS[cell.workload].task_metric,
                            requested_prompt_length=cell.prompt_length,
                            requested_generation_length=cell.generation_length,
                            retained_fraction=cell.retained_fraction,
                            kv_bytes_kind=KV_BYTES_KIND,
                            **budget_metadata(regime, key, budget), **measured))
                        logger.debug("%s", rows[-1])
    return SweepResult(rows=rows, skipped=skipped, cfg=cfg)


@lru_cache(maxsize=4)
def _numeric_tokens(tokenizer) -> frozenset[int]:
    """One scan of the vocabulary per run, not per decode."""
    return ablation.numeric_token_ids(tokenizer)


def _one(model, tokenizer, cfg, regime: Regime, key: str, cell: Cell, task_seed: int,
         eviction_seed: int, index: int, examples: int, generate_stepwise) -> dict | str:
    workload = WORKLOADS[cell.workload]
    example = workload.build(tokenizer, cell.prompt_length, index, examples, task_seed,
                             cell.redundancy)
    ids = tokenizer(example.context + example.question, return_tensors="pt").input_ids
    if abs(ids.shape[1] - cell.prompt_length) > prompt_band(cell.prompt_length):
        return PROMPT_TARGET_UNREACHABLE
    budget = Budget(ids.shape[1], cell.generation_length, cell.retained_fraction)
    chosen = method_for(regime, key, budget, eviction_seed,
                        numeric_tokens=_numeric_tokens(tokenizer) if key in ABLATION_KEYS
                        else None)
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
        "facts": example.facts, "statements": example.statements,
        "prompt_length": ids.shape[1],
        "actual_generation_length": int(generated.shape[1]),
        "cached_generation_length": ledger.generated_length,
        "prompt_retained": account.prompt_retained,
        "generated_retained": account.generated_retained,
        "total_retained": account.total_retained,
        "generated_position_mean": account.generated_position_mean,
        "ablated_targeted": getattr(method, "removed_targeted", 0),
        "ablated_other": getattr(method, "removed_other", 0),
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
    workload: str
    redundancy: str
    prompt_length: int
    generation_length: int
    retained_fraction: float
    task_seeds: int
    eviction_seeds: int
    metric_mean: float
    metric_stdev: float
    prompt_retained: float
    total_retained: float
    kv_bytes: float


def summarize(rows: Iterable[BoundaryRow]) -> list[Summary]:
    grouped: dict[tuple, list[BoundaryRow]] = {}
    for row in rows:
        grouped.setdefault(
            (row.regime, row.method, row.task, row.redundancy, row.requested_prompt_length,
             row.requested_generation_length, row.retained_fraction),
            []).append(row)
    summaries = []
    for key, group in grouped.items():
        regime, method, workload, redundancy, prompt_length, generation_length, fraction = key
        per_task: dict[tuple[int, int], list[float]] = {}
        for row in group:
            per_task.setdefault((row.task_seed, row.example), []).append(row.metric_value)
        # Eviction draws are replicates *within* a task instance, so they are averaged
        # there first. Pooling them with task instances would let a method with more
        # draws look more precisely measured than one with none.
        seed_means = [mean(values) for values in per_task.values()]
        summaries.append(Summary(
            regime=regime, method=method, workload=workload, redundancy=redundancy,
            prompt_length=prompt_length, generation_length=generation_length,
            retained_fraction=fraction, task_seeds=len({row.task_seed for row in group}),
            eviction_seeds=len({row.eviction_seed for row in group
                                if row.eviction_seed != NO_EVICTION_SEED}),
            metric_mean=mean(seed_means), metric_stdev=pstdev(seed_means),
            prompt_retained=mean(row.prompt_retained for row in group),
            total_retained=mean(row.total_retained for row in group),
            kv_bytes=mean(row.kv_bytes for row in group)))
    return summaries


def format_summary(summaries: Sequence[Summary], task_metric: str) -> str:
    header = (f"{'workload':<10}{'redun':<7}{'regime':<17}{'method':<30}{'prompt':>7}{'gen':>6}"
              f"{'retain':>7}{'tsd':>5}{'esd':>5}{task_metric:>16}{'stdev':>7}{'kept':>8}"
              f"{'prompt_kept':>12}{'KV KB':>9}")
    lines = [header, "-" * len(header)]
    order = sorted(summaries, key=lambda s: (s.workload, s.redundancy, s.regime, s.prompt_length,
                                             s.generation_length, s.retained_fraction, s.method))
    for s in order:
        lines.append(f"{s.workload:<10}{s.redundancy:<7}{s.regime:<17}{s.method:<30}"
                     f"{s.prompt_length:>7}{s.generation_length:>6}{s.retained_fraction:>7.2f}"
                     f"{s.task_seeds:>5}{s.eviction_seeds:>5}"
                     f"{s.metric_mean:>16.2f}{s.metric_stdev:>7.2f}"
                     f"{s.total_retained:>8.1f}{s.prompt_retained:>12.1f}{s.kv_bytes / 1024:>9.1f}")
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
