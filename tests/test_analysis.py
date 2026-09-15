"""Offline analysis: does it compute the quantity the study is about, and does it
refuse to quietly hand back a number that cannot be read.

The scoring advantage is a paired difference over (seed, example) units, so the
tests build units whose means are known by construction and check the difference
against them. The rest is the sanity machinery: a cell whose full cache cannot do
the task, a cell where the equal-memory comparison has broken, and a random baseline
that never actually varies all have to be flagged rather than averaged into the
table. Run: `pytest tests/`.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import analysis                                          # noqa: E402
from kvlab.boundary import BoundaryRow, summarize, write_rows       # noqa: E402

PROMPT, GENERATION = 120, 61        # 60 generated positions reach the cache
CACHED = GENERATION - 1
RETAINED = 130.0


def row(*, method_key, metric, seed=0, example=0, task="multistep", redundancy="low",
        generation_length=GENERATION, retained_fraction=0.5, total_retained=RETAINED,
        generated_retained=10.0, generated_position_mean=150.0, compression_ratio=0.5):
    names = {"full": "Full cache", "random": "Prompt-protected random", "h2o": "H2O",
             "snapkv": "SnapKV"}
    return BoundaryRow(
        model="stub", regime="published", method=names[method_key], method_key=method_key,
        seed=seed, example=example, task=task, redundancy=redundancy, facts=4, statements=4,
        task_metric="final_answer_em", metric_value=metric,
        requested_prompt_length=PROMPT, prompt_length=PROMPT,
        requested_generation_length=generation_length, actual_generation_length=generation_length,
        cached_generation_length=CACHED, retained_fraction=retained_fraction,
        total_budget=130, generation_budget=10, nominal_budget=130, window=32,
        prompt_retained=total_retained - generated_retained,
        generated_retained=generated_retained, total_retained=total_retained,
        generated_position_mean=generated_position_mean, kv_bytes=1000.0,
        kv_bytes_kind="analytical", compression_ratio=compression_ratio,
        decode_wall_seconds=0.1)


UNITS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]


def cell_rows(scored_metrics, baseline_metrics, *, full_metric=1.0, scored_retained=RETAINED,
              position_spread=5.0, **shared):
    """One cell: a full-cache control, a scored method and the random baseline, over
    the same units. The knobs are the things the sanity checks are supposed to catch,
    so a test names one instead of editing rows after the fact."""
    rows = [row(method_key="full", metric=full_metric, seed=s, example=e, **shared)
            for s, e in UNITS]
    for (seed, example), scored, base in zip(UNITS, scored_metrics, baseline_metrics):
        rows.append(row(method_key="h2o", metric=scored, seed=seed, example=example,
                        total_retained=scored_retained, **shared))
        rows.append(row(method_key="random", metric=base, seed=seed, example=example,
                        generated_position_mean=150.0 + position_spread * seed, **shared))
    return rows


def only(results, method_key):
    assert len(results) == 1
    return next(m for m in results[0].methods if m.method_key == method_key)


@pytest.mark.parametrize("scored, baseline, advantage", [
    ([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0], 1.0),
    ([0, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1], -1.0),
    ([1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0], 1 / 3),
    ([1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0], 0.0),
])
def test_scoring_advantage_is_the_paired_difference(scored, baseline, advantage):
    results = analysis.analyse(cell_rows(scored, baseline))
    h2o = only(results, "h2o")

    assert h2o.metric.mean == pytest.approx(sum(scored) / len(scored))
    assert h2o.metric.samples == len(UNITS)
    assert h2o.scoring_advantage.mean == pytest.approx(advantage)
    assert h2o.scoring_advantage.ci_low <= h2o.scoring_advantage.mean <= \
        h2o.scoring_advantage.ci_high
    # The baseline is reported, but it is not its own advantage.
    assert only(results, "random").scoring_advantage is None


def test_a_cell_with_no_feasible_baseline_reports_no_advantage_rather_than_zero():
    """The baseline being infeasible and the baseline tying are opposite findings and
    must not arrive as the same number."""
    rows = [r for r in cell_rows([1] * 6, [0] * 6) if r.method_key != "random"]
    assert only(analysis.analyse(rows), "h2o").scoring_advantage is None


def test_context_fractions_come_from_the_lengths_actually_run():
    cell = analysis.analyse(cell_rows([1] * 6, [0] * 6))[0]
    assert cell.prompt_fraction == pytest.approx(PROMPT / (PROMPT + CACHED))
    assert cell.generated_fraction == pytest.approx(CACHED / (PROMPT + CACHED))
    assert cell.prompt_fraction + cell.generated_fraction == pytest.approx(1.0)
    assert cell.cached_generation_length == CACHED


@pytest.mark.parametrize("varied", [
    {"generation_length": 256}, {"redundancy": "high"}, {"task": "needle"},
    {"retained_fraction": 0.9},
])
def test_every_independent_variable_keeps_its_own_cell(varied):
    """Generation length included: averaging two generation lengths together reports a
    figure neither of them measured."""
    rows = cell_rows([1] * 6, [0] * 6) + cell_rows([0] * 6, [0] * 6, **varied)
    results = analysis.analyse(rows)

    assert len(results) == 2
    assert {only([c], "h2o").metric.mean for c in results} == {1.0, 0.0}
    assert len(summarize(rows)) == 2 * 3     # three methods in each cell


@pytest.mark.parametrize("overrides, drop_control, expected", [
    ({"full_metric": 0.0}, False, "cannot do this task uncompressed"),
    ({}, True, "no full-cache control"),
    ({"scored_retained": RETAINED + 40}, False, "equal-memory comparison broken"),
    ({"position_spread": 0.0}, False, "baseline is not varying"),
    ({"compression_ratio": 0.9}, False, "of the cache against a requested"),
])
def test_a_cell_that_cannot_be_read_says_so(overrides, drop_control, expected):
    rows = cell_rows([1] * 6, [0] * 6, **overrides)
    if drop_control:
        rows = [r for r in rows if r.method_key != "full"]
    assert expected in " ".join(analysis.analyse(rows)[0].flags)


def test_a_healthy_cell_is_not_flagged():
    """The counterpart every check needs: it must fire on the fault and stay quiet
    otherwise, or it is not a check."""
    assert analysis.analyse(cell_rows([1] * 6, [0] * 6))[0].flags == ()


@pytest.mark.parametrize("suffix", [".csv", ".jsonl"])
def test_rows_survive_a_round_trip_with_their_workload_metadata(tmp_path, suffix):
    stem = str(tmp_path / "run")
    rows = cell_rows([1] * 6, [0] * 6, task="needle", redundancy="high")
    write_rows(rows, summarize(rows), stem, {})

    restored = analysis.load_rows(stem + suffix)
    assert [(r.task, r.redundancy, r.method_key, r.metric_value, r.seed, r.example)
            for r in restored] == [
        (r.task, r.redundancy, r.method_key, r.metric_value, r.seed, r.example) for r in rows]
    assert analysis.analyse(restored)[0].workload == "needle"
    assert analysis.analyse(restored)[0].redundancy == "high"


def test_every_figure_is_drawn_from_saved_records(tmp_path):
    plots = pytest.importorskip("kvlab.plots")
    pytest.importorskip("matplotlib")
    rows = cell_rows([1] * 6, [0] * 6) + cell_rows([0] * 6, [1] * 6, generation_length=256)
    written = plots.draw_all(analysis.flat_rows(analysis.analyse(rows)), str(tmp_path))

    assert len(written) == len(plots.PLOTS)
    assert all(os.path.getsize(path) > 0 for path in written)
    assert {os.path.basename(p)[0] for p in written} == {"A", "B", "C", "D"}
