"""Feasibility checked against the tokenizer that will run the grid, before inference.

The bug this replaces: the full study quoted prompt floors measured on GPT-2 while it
was configured for Qwen2.5. Those are not the same quantity -- the same evidence
segments differently, and high redundancy needs 265 tokens under one and 310 under the
other -- so a cell that looked feasible would have run its prompt long, which moves
redundancy and prompt length together and destroys the comparison the study is for.

Tokenizers here are stubs with deliberately different granularity, so the tests pin
that the floors follow the tokenizer rather than being baked in. Run: `pytest tests/`.
"""

import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import preflight                                          # noqa: E402
from kvlab.boundary import prompt_band                               # noqa: E402

LEVELS = ("low", "medium", "high")


class WordTokenizer:
    """One token per word."""

    def __call__(self, text, return_tensors=None):
        return type("Encoded", (), {"input_ids": text.split()})()


class SplitTokenizer(WordTokenizer):
    """Two tokens per word, so every floor is exactly twice the word tokenizer's. A
    real tokenizer difference is not this tidy, but it is the same kind of difference
    and it makes the dependence checkable."""

    def __call__(self, text, return_tensors=None):
        return type("Encoded", (), {"input_ids": text.split() * 2})()


def check(tokenizer=None, shapes=((400, 64),), workloads=("multistep",), limit=100_000,
          **kwargs):
    return preflight.check(tokenizer or WordTokenizer(), model="stub", context_limit=limit,
                           workload_keys=workloads, shapes=shapes, **kwargs)


def floors(report):
    return {(f.workload, f.redundancy): f.minimum_prompt_tokens for f in report.floors}


def test_the_floors_follow_the_tokenizer_they_were_measured_with():
    coarse, fine = floors(check(WordTokenizer())), floors(check(SplitTokenizer()))

    assert set(coarse) == {("multistep", level) for level in LEVELS}
    assert all(fine[key] == 2 * coarse[key] for key in coarse)
    # And the floor rises with redundancy, which is what makes short prompts infeasible.
    ordered = [coarse[("multistep", level)] for level in LEVELS]
    assert ordered == sorted(ordered) and len(set(ordered)) == len(LEVELS)


def test_cells_below_their_floor_are_named_before_any_inference_runs():
    high_floor = floors(check())[("multistep", "high")]
    short = high_floor // 3
    report = check(shapes=((short, 32), (high_floor * 2, 32)))

    by_shape = {(c.prompt_length, c.redundancy): c for c in report.cells}
    assert by_shape[(short, "high")].feasible is False
    assert "needs" in by_shape[(short, "high")].reason
    assert by_shape[(high_floor * 2, "high")].feasible is True
    assert len(report.feasible_cells) < len(report.cells)


def test_a_grid_that_cannot_compare_redundancy_at_all_is_refused():
    """The expensive mistake: a grid where no shape holds every level runs perfectly
    and answers nothing."""
    low_only = floors(check())[("multistep", "low")] + 1
    refused = check(shapes=((low_only, 32),))

    assert refused.comparable_shapes["multistep"] == ()
    assert not refused.ok
    assert "redundancy comparison this study is for cannot be made" in " ".join(refused.problems)

    roomy = floors(check())[("multistep", "high")] * 2
    allowed = check(shapes=((low_only, 32), (roomy, 32)))
    assert allowed.comparable_shapes["multistep"] == ((roomy, 32),)
    assert allowed.ok


@pytest.mark.parametrize("required, expected_ok", [(1, True), (2, False)])
def test_how_many_comparable_shapes_the_study_needs_is_the_study_s_choice(required, expected_ok):
    roomy = floors(check())[("multistep", "high")] * 2
    report = check(shapes=((roomy, 32),), require_comparable_shapes=required)
    assert report.ok is expected_ok


def test_a_shape_beyond_the_context_window_is_caught_here_not_mid_sweep():
    roomy = floors(check())[("multistep", "high")] * 2
    report = check(shapes=((roomy, 900),), limit=roomy + 100)

    overflowing = [c for c in report.cells if not c.feasible]
    assert overflowing and all(preflight.CONTEXT_OVERFLOW in c.reason for c in overflowing)
    assert not report.ok


def test_a_target_the_floor_overshoots_only_inside_the_band_is_still_allowed():
    """The band is the same one the runner enforces, so preflight and the sweep cannot
    disagree about which cells exist. A target the floor overshoots by less than the
    band is runnable; one it overshoots by more is not."""
    floor = floors(check())[("multistep", "medium")]
    inside = floor - 5                      # band is at least 8 tokens, so this fits
    outside = floor - 4 * prompt_band(floor)
    assert prompt_band(inside) >= 5 and prompt_band(outside) < floor - outside
    report = check(shapes=((floor, 32), (inside, 32), (outside, 32)), redundancies=("medium",))

    by_prompt = {c.prompt_length: c.feasible for c in report.cells}
    assert by_prompt[floor] is True
    assert by_prompt[inside] is True, "the band must be honoured, not just the exact target"
    assert by_prompt[outside] is False


def test_the_checked_in_study_report_matches_the_configuration_it_describes():
    """The committed Qwen report is evidence only while it describes the grid it
    claims to. A hand-edited or stale one is worse than none."""
    here = os.path.join(os.path.dirname(__file__), "..", "experiments")
    with open(os.path.join(here, "full_study.json")) as handle:
        study = json.load(handle)
    with open(os.path.join(here, "full_study_preflight.json")) as handle:
        report = json.load(handle)

    assert report["model"] == study["model"]
    assert report["ok"] is True and report["problems"] == []
    assert set(report["prompt_floors"]) == {"needle/none", "multistep/low", "multistep/medium",
                                            "multistep/high"}
    shapes = {tuple(shape) for shape in study["shapes"]}
    assert {tuple(cell["shape"]) for cell in report["infeasible_cells"]} <= shapes
    assert {tuple(shape) for shape in report["shapes_holding_every_redundancy_level"]["multistep"]} \
        <= shapes
