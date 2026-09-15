"""Budget arithmetic and prompt protection for the boundary sweep.

The sweep's whole claim to being controlled is that every method in a cell ends up
holding the same cache, reached by a budget derived separately for each because the
methods bound different things. These tests pin that derivation against a table
worked out by hand, pin the two ways a method can have no feasible configuration at
all, and pin the wrapper that holds a prompt out of a scorer's reach.

Nothing here loads a model: the arithmetic and the wrapper are exactly the parts
that must be right before a single GPU-second is spent. The end-to-end run is
covered in test_rope_integration. Run: `pytest tests/`.
"""

import os
import sys

import json

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab.boundary import (                                          # noqa: E402
    NO_EVICTION_SEED, NO_PROMPT_PROTECTED_FORM, PROMPT_TARGET_UNREACHABLE,
    UNREACHABLE_BY_PRESS, UNREACHABLE_BY_PROTECTION, BoundaryRow, Budget, PromptProtected,
    Regime, Summary, format_summary, method_for, summarize, sweep, write_rows,
)
from kvlab import boundary as boundary_module                        # noqa: E402
from kvlab.memory import MODELS                                       # noqa: E402
from kvlab.methods import H2O, SnapKV                                 # noqa: E402

PROMPT, GENERATION = 100, 51        # 50 generated positions ever reach the cache
FULL = PROMPT + GENERATION - 1
BATCH, HEADS, DIM = 1, 3, 2


@pytest.mark.parametrize("fraction, total, generation_budget, h2o, press, random_budget", [
    # fraction   total  gen_budget  h2o  press  random
    (1.00,        150,          50,  150,   100,     50),
    (0.80,        120,          20,  120,    70,     20),
    # The target no longer fits a protected prompt, but a press still reaches it
    # by compressing the prompt to 25.
    (0.50,         75,         -25,   75,    25,   None),
    # Now even compressing the prompt to nothing leaves the press over target,
    # because it accumulates all 50 generated positions regardless.
    (0.20,         30,         -70,   30,  None,   None),
])
def test_one_target_becomes_a_different_budget_for_every_method(
        fraction, total, generation_budget, h2o, press, random_budget):
    budget = Budget(PROMPT, GENERATION, fraction)
    assert budget.cached_generation_length == GENERATION - 1
    assert budget.full_length == FULL
    assert budget.total_budget == total
    assert budget.generation_budget == generation_budget

    assert budget.nominal_for("h2o", Regime.PUBLISHED)[0] == h2o
    assert budget.nominal_for("snapkv", Regime.PUBLISHED)[0] == press
    assert budget.nominal_for("random", Regime.PUBLISHED)[0] == random_budget
    # Under protection every method takes the generated-cache framing, h2o included.
    assert budget.nominal_for("h2o", Regime.PROMPT_PROTECTED)[0] == random_budget


@pytest.mark.parametrize("key, regime, fraction, reason", [
    ("random", Regime.PUBLISHED, 0.5, UNREACHABLE_BY_PROTECTION),
    ("h2o", Regime.PROMPT_PROTECTED, 0.5, UNREACHABLE_BY_PROTECTION),
    ("snapkv", Regime.PUBLISHED, 0.2, UNREACHABLE_BY_PRESS),
    ("snapkv", Regime.PROMPT_PROTECTED, 1.0, NO_PROMPT_PROTECTED_FORM["snapkv"]),
    ("cake", Regime.PROMPT_PROTECTED, 1.0, NO_PROMPT_PROTECTED_FORM["cake"]),
    ("obcache", Regime.PROMPT_PROTECTED, 1.0, NO_PROMPT_PROTECTED_FORM["obcache"]),
])
def test_an_unreachable_cell_gives_a_reason_not_a_method(key, regime, fraction, reason):
    chosen = method_for(regime, key, Budget(PROMPT, GENERATION, fraction), seed=0)
    assert chosen == reason


@pytest.mark.parametrize("key, regime, expected_key, nominal, window", [
    ("h2o", Regime.PUBLISHED, "h2o", 150, 37),
    ("h2o", Regime.PROMPT_PROTECTED, "pp-h2o", 50, 12),
    ("snapkv", Regime.PUBLISHED, "snapkv", 100, 25),
    ("random", Regime.PUBLISHED, "random", 50, 0),
    ("full", Regime.PUBLISHED, "full", 0, 0),
])
def test_the_budget_a_method_was_handed_is_reported_with_it(
        key, regime, expected_key, nominal, window):
    method, handed, recent = method_for(regime, key, Budget(PROMPT, GENERATION, 1.0), seed=0)
    assert (method.key, handed, recent) == (expected_key, nominal, window)


def coded(positions, layers=1, heads=HEADS):
    values = torch.tensor(list(positions), dtype=torch.float32)
    key = values.view(1, 1, -1, 1).expand(BATCH, heads, len(values), DIM).clone()
    return tuple((key.clone(), key.clone()) for _ in range(layers))


def attention_over(length, layers=1, heads=HEADS):
    """One decode query attending to `length` cached positions, weighted so the
    oldest generated positions score highest. A scorer that ignores the prompt
    slice would keep a different set, which is what makes this non-uniform."""
    weights = torch.linspace(1.0, 0.1, length).view(1, 1, 1, length).expand(BATCH, heads, 1, length)
    return tuple(weights.clone() for _ in range(layers))


def test_a_wrapped_scorer_never_reaches_the_prompt():
    prompt_length, generation_budget, steps = 6, 3, 9
    method = PromptProtected(H2O(budget=generation_budget, recent=1), generation_budget)
    cache = coded(range(prompt_length))
    method.apply(cache, None)
    for step in range(steps):
        cache = tuple((torch.cat([k, nk], dim=2), torch.cat([v, nv], dim=2))
                      for (k, v), (nk, nv) in zip(cache, coded([prompt_length + step])))
        cache = method.step(cache, attention_over(cache[0][0].shape[2]))

    for head in range(HEADS):
        kept = [int(p) for p in cache[0][0][0, head, :, 0].tolist()]
        assert kept[:prompt_length] == list(range(prompt_length))
        assert len(kept) == prompt_length + min(generation_budget, steps)
        assert kept == sorted(kept)
    assert torch.equal(cache[0][0], cache[0][1])
    assert method.kept_len(prompt_length + steps) == prompt_length + generation_budget


def test_published_indices_are_composed_through_the_protected_prefix():
    """The wrapper's inner method indexes the generated sub-cache. Published
    unshifted, those indices would point into the prompt and every provenance
    figure downstream would be wrong."""
    prompt_length, generation_budget = 4, 2
    method = PromptProtected(H2O(budget=generation_budget, recent=1), generation_budget)
    cache = coded(range(prompt_length))
    method.apply(cache, None)
    for step in range(4):
        cache = tuple((torch.cat([k, nk], dim=2), torch.cat([v, nv], dim=2))
                      for (k, v), (nk, nv) in zip(cache, coded([prompt_length + step])))
        cache = method.step(cache, attention_over(cache[0][0].shape[2]))

    published = method.last_indices[0]
    assert published.shape[-1] == cache[0][0].shape[2]
    assert published[0, 0, :prompt_length].tolist() == list(range(prompt_length))
    assert (published[0, 0, prompt_length:] >= prompt_length).all()


def test_an_after_prefill_press_cannot_be_wrapped():
    with pytest.raises(ValueError, match="observation window of prompt queries"):
        PromptProtected(SnapKV(budget=4, window=2), generation_budget=4)


def row(seed, metric_value, method="Prompt-protected random", regime="published",
        generation_length=48, retained_fraction=0.5, task="needle", redundancy="none",
        method_key="random", prompt_retained=130.0, generated_retained=0.0,
        eviction_seed=0):
    return BoundaryRow(
        model="distilgpt2", regime=regime, method=method, method_key=method_key,
        task_seed=seed, eviction_seed=eviction_seed,
        example=0, task=task, redundancy=redundancy, facts=1, statements=1,
        task_metric="passkey_em", metric_value=metric_value,
        requested_prompt_length=128, prompt_length=130,
        requested_generation_length=generation_length,
        actual_generation_length=generation_length,
        cached_generation_length=generation_length - 1, retained_fraction=retained_fraction,
        total_budget=88, generation_budget=-42, nominal_budget=0, window=0,
        prompt_retained=prompt_retained, generated_retained=generated_retained,
        total_retained=prompt_retained + generated_retained, generated_position_mean=0.0,
        ablated_targeted=0, ablated_other=0, kv_bytes=1.0, kv_bytes_kind="analytical", compression_ratio=0.25,
        decode_wall_seconds=0.1)


def test_summary_reports_the_spread_across_seeds_not_one_lucky_draw():
    hits = [1.0, 1.0, 1.0, 0.0, 0.0]
    summaries = summarize([row(seed, hit) for seed, hit in enumerate(hits)])
    assert len(summaries) == 1
    assert summaries[0].task_seeds == len(hits)
    assert summaries[0].metric_mean == pytest.approx(sum(hits) / len(hits))
    assert summaries[0].metric_stdev > 0

    split = summarize([row(0, 1.0), row(1, 1.0, method="H2O")])
    assert {s.method for s in split} == {"Prompt-protected random", "H2O"}
    assert all(s.task_seeds == 1 and s.metric_stdev == 0 for s in split)


def test_results_carry_every_column_a_reader_has_to_check(tmp_path):
    stem = str(tmp_path / "run")
    rows = [row(0, 1.0)]
    write_rows(rows, summarize(rows), stem, {"model": "distilgpt2"})

    import csv
    import json

    with open(f"{stem}.csv") as handle:
        header = next(csv.reader(handle))
    required = {"model", "method", "task_seed", "eviction_seed", "prompt_length",
                "requested_generation_length",
                "actual_generation_length", "generation_budget", "total_budget",
                "prompt_retained", "generated_retained", "total_retained", "kv_bytes",
                "kv_bytes_kind", "task", "task_metric", "metric_value"}
    assert required <= set(header)

    with open(f"{stem}.jsonl") as handle:
        streamed = [json.loads(line) for line in handle]
    assert [record["metric_value"] for record in streamed] == [r.metric_value for r in rows]

    with open(f"{stem}.json") as handle:
        saved = json.load(handle)
    assert saved["config"] == {"model": "distilgpt2"}
    assert [Summary(**s).method for s in saved["summary"]] == ["Prompt-protected random"]


# Two generation lengths and two retained fractions, with a metric value chosen per
# cell so that any pair averaged together lands on a value no cell reported.
GENERATION_LENGTHS = (64, 384)
RETAINED_FRACTIONS = (0.25, 0.75)
CELL_METRIC = {(64, 0.25): 1.0, (64, 0.75): 0.0, (384, 0.25): 0.0, (384, 0.75): 1.0}


def test_every_independent_variable_survives_summarizing():
    """Generation length is a primary axis of the sweep, so summarizing must not
    average across it. It did, which turned two cells that disagree completely into
    one row reporting a figure neither of them measured."""
    rows = [row(seed, metric, generation_length=generation, retained_fraction=fraction)
            for (generation, fraction), metric in CELL_METRIC.items()
            for seed in range(3)]
    summaries = summarize(rows)

    assert len(summaries) == len(CELL_METRIC)
    assert {(s.generation_length, s.retained_fraction): s.metric_mean
            for s in summaries} == CELL_METRIC
    assert all(s.task_seeds == 3 and s.metric_stdev == 0 for s in summaries)
    assert "gen" in format_summary(summaries, "passkey_em").splitlines()[0]


CFG = MODELS["distilgpt2"]
FAKE_PROMPT_TOKENS = 40


class FakeTokenizer:
    """Whitespace tokenisation, enough for the sweep to size a prompt and read one
    back. The sweep only ever asks for a length or a decoded string."""

    def __call__(self, text, return_tensors=None):
        words = text.split()
        if return_tensors is None:
            return type("Encoded", (), {"input_ids": words})()
        return type("Encoded", (), {
            "input_ids": torch.zeros(1, len(words), dtype=torch.long)})()

    def decode(self, ids, skip_special_tokens=True):
        return ""


def fake_cache(length, layers=CFG.layers, heads=CFG.n_kv_heads):
    key = torch.zeros(1, heads, length, 2)
    return tuple((key.clone(), key.clone()) for _ in range(layers))


DECODES = []


def fake_attentions(cache, queries):
    """Attention weights shaped like the real ones and fixed rather than random, so a
    scored method stays a deterministic function of the task. Earlier positions score
    higher, which gives the scorers something to prefer."""
    return tuple(torch.linspace(1.0, 0.1, key.shape[2])
                 .view(1, 1, 1, key.shape[2])
                 .expand(1, key.shape[1], queries, key.shape[2]).contiguous()
                 for key, _ in cache)


def fake_generate(model, ids, method, max_new_tokens=64, eos_token_id=None, ledger=None):
    """A decode with no model in it, driving the ledger exactly as the real loop
    does so the accounting a row reports is still real."""
    DECODES.append(method.key)
    cache = fake_cache(ids.shape[1])
    ledger.start(cache)
    cache = method.apply(cache, fake_attentions(cache, ids.shape[1]))
    ledger.compact(method, cache)
    for _ in range(max_new_tokens - 1):
        grown = torch.zeros(1, cache[0][0].shape[1], 1, 2)
        cache = tuple((torch.cat([k, grown], dim=2), torch.cat([v, grown], dim=2))
                      for k, v in cache)
        ledger.append()
        cache = method.step(cache, fake_attentions(cache, 1))
        ledger.compact(method, cache)
    return torch.zeros(1, max_new_tokens, dtype=torch.long)


@pytest.fixture
def modelless_sweep(monkeypatch):
    DECODES.clear()
    model = type("FakeModel", (), {"config": type("Config", (), {"n_positions": 1024})()})()
    monkeypatch.setattr("kvlab.model.load_model", lambda name, device="cpu": (
        model, FakeTokenizer(), CFG))
    monkeypatch.setattr("kvlab.decode.generate_stepwise", fake_generate)


def test_a_reused_decode_never_carries_another_cells_budget(modelless_sweep):
    """The full cache ignores the budget, so its decode is memoized across retained
    fractions. The budget columns are not part of that decode and must be recomputed
    per row: reporting them out of the cached result stamped the first fraction's
    numbers onto every later one, silently."""
    result = sweep("fake", workload_keys=("needle",),
                   shapes=tuple((FAKE_PROMPT_TOKENS, generation)
                                for generation in GENERATION_LENGTHS),
                   retained_fractions=RETAINED_FRACTIONS, task_seeds=(0,),
                   eviction_seeds=(0,), examples=1,
                   regimes=(Regime.PUBLISHED,), method_keys=("full",))

    assert len(result.rows) == len(GENERATION_LENGTHS) * len(RETAINED_FRACTIONS)
    by_cell = {(r.requested_generation_length, r.retained_fraction): r for r in result.rows}
    assert set(by_cell) == set(CELL_METRIC)

    for generation_length in GENERATION_LENGTHS:
        low = by_cell[(generation_length, min(RETAINED_FRACTIONS))]
        high = by_cell[(generation_length, max(RETAINED_FRACTIONS))]
        # Same decode, genuinely reused: identical prompt and identical wall clock.
        assert low.prompt_length == high.prompt_length
        assert low.decode_wall_seconds == high.decode_wall_seconds
        # Different cell, so different budget.
        assert low.total_budget < high.total_budget
        assert low.generation_budget < high.generation_budget


MULTISTEP_TOO_SHORT, MULTISTEP_ROOMY = 40, 300


def test_a_prompt_the_workload_cannot_build_is_skipped_not_relabelled(modelless_sweep):
    """multistep's evidence and instructions have a floor. A target below it would
    still produce rows, just at a longer prompt than the row claims, which would put
    redundancy and prompt length on the same axis and make both unreadable."""
    result = sweep("fake", workload_keys=("multistep",),
                   shapes=((MULTISTEP_TOO_SHORT, 12), (MULTISTEP_ROOMY, 12)),
                   retained_fractions=(0.5,), redundancies=("low",), task_seeds=(0,),
                   eviction_seeds=(0,), examples=1,
                   regimes=(Regime.PUBLISHED,), method_keys=("full",))

    assert {row.requested_prompt_length for row in result.rows} == {MULTISTEP_ROOMY}
    refused = [skip for skip in result.skipped if skip.reason == PROMPT_TARGET_UNREACHABLE]
    assert [skip.prompt_length for skip in refused] == [MULTISTEP_TOO_SHORT]
    for row in result.rows:
        assert abs(row.prompt_length - row.requested_prompt_length) <= \
            boundary_module.prompt_band(row.requested_prompt_length)
        assert (row.task, row.redundancy) == ("multistep", "low")


def test_a_configuration_file_is_read_and_the_command_line_still_wins():
    """The checked-in configurations are the reproducible part of the study, so a flag
    has to override one without a default silently counting as an override."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "run_boundary_sweep",
        os.path.join(os.path.dirname(__file__), "..", "examples", "run_boundary_sweep.py"))
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    config = os.path.join(os.path.dirname(__file__), "..", "experiments", "smoke.json")
    with open(config) as handle:
        saved = json.load(handle)

    from_file = cli.resolve(["--config", config])
    assert from_file.model == saved["model"]
    assert from_file.shapes == tuple(tuple(shape) for shape in saved["shapes"])
    assert from_file.task_seeds == tuple(saved["task_seeds"])
    assert from_file.eviction_seeds == tuple(saved["eviction_seeds"])
    assert from_file.examples == saved["examples"]

    overridden = cli.resolve(["--config", config, "--task-seeds", "9", "--examples", "3"])
    assert overridden.task_seeds == (9,) and overridden.examples == 3
    assert overridden.eviction_seeds == tuple(saved["eviction_seeds"])
    assert overridden.shapes == from_file.shapes, "unspecified flags keep the file's values"


TASK_SEEDS, EVICTION_SEEDS = (0, 1), (0, 1, 2)


def seed_separated_sweep(**overrides):
    return sweep("fake", workload_keys=("needle",), shapes=((FAKE_PROMPT_TOKENS, 12),),
                 retained_fractions=(0.9,), task_seeds=TASK_SEEDS,
                 eviction_seeds=EVICTION_SEEDS, examples=1,
                 regimes=(Regime.PUBLISHED,), **overrides)


def test_a_deterministic_method_is_not_decoded_once_per_eviction_draw(modelless_sweep):
    """Eviction seeds are draws a scored method never makes. Sweeping one over the
    other would pay for the same inference len(eviction_seeds) times and would also
    put duplicate rows into every mean and every bootstrap interval."""
    result = seed_separated_sweep(method_keys=("h2o", "random"))

    deterministic = [r for r in result.rows if r.method_key == "h2o"]
    stochastic = [r for r in result.rows if r.method_key == "random"]
    assert len(deterministic) == len(TASK_SEEDS)
    assert len(stochastic) == len(TASK_SEEDS) * len(EVICTION_SEEDS)
    assert {r.eviction_seed for r in deterministic} == {NO_EVICTION_SEED}
    assert {r.eviction_seed for r in stochastic} == set(EVICTION_SEEDS)
    assert DECODES.count("h2o") == len(TASK_SEEDS)
    assert DECODES.count("random") == len(TASK_SEEDS) * len(EVICTION_SEEDS)


def test_the_task_seed_moves_the_task_and_the_eviction_seed_moves_only_the_draw(
        modelless_sweep):
    """The confound this replaces: one seed doing both jobs means a random method's
    spread cannot be attributed to the tasks it saw or to the draws it made."""
    rows = {(r.task_seed, r.eviction_seed): r
            for r in seed_separated_sweep(method_keys=("random",)).rows}

    same_task = [rows[(0, draw)] for draw in EVICTION_SEEDS]
    assert len({r.prompt_length for r in same_task}) == 1, "the task must not move"
    assert len({r.generated_position_mean for r in same_task}) > 1, "the draw must move"

    same_draw = [rows[(task, 0)] for task in TASK_SEEDS]
    assert len({r.prompt_length for r in same_draw}) > 1, "the task must move"
