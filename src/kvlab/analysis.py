"""Offline analysis of a boundary sweep. Reads saved rows; never runs inference.

The central quantity is deliberately a difference and not a verdict:

    scoring_advantage = scored_method_metric - random_metric

reported as a continuous number with an interval, per cell. No threshold is applied
and no cell is declared "the boundary": where a difference stops mattering is a
judgement about the difference, and printing a cut-off would bury that judgement
inside the tool. The transition is something to look at in the plots, not something
this module decides.

Resampling unit: **one (seed, example) pair**. A seed fixes both the task instance
and a random method's draw, and an example index fixes which instance of the task it
is, so that pair is the smallest thing that could independently have come out
differently. Bootstrap resamples those units with replacement, and for an advantage
it resamples them *paired* -- the same unit contributes to the scored method and to
random in the same replicate -- because the two ran on the same task instance and
treating them as independent would overstate the interval. No p-values: an interval
that overlaps zero says what needs saying.

Four sanity checks run before any of it is worth reading, and each one flags rather
than silently drops:

- **solvable** -- the full cache's own score on this cell. Where the model cannot do
  the task uncompressed, nothing below it is evidence about compression.
- **equal memory** -- every compressed method in a cell should hold the same retained
  KV, since that is what the sweep controls. Checked to slot granularity.
- **random varies** -- once eviction is active, different seeds must retain visibly
  different parts of the trace, or the random baseline is not random in practice.
- **compression** -- requested retained fraction against the fraction actually
  achieved, both reported.
"""

from __future__ import annotations

import csv
import json
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, fields
from statistics import mean, pstdev

from .boundary import BoundaryRow

logger = logging.getLogger(__name__)

RESAMPLING_UNIT = "one (seed, example) pair"
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE = 0.95
#: Retained KV is counted in whole positions per layer per head, so two methods that
#: are meant to hold the same cache may legitimately differ by rounding within one
#: position. More than that is a real mismatch.
MEMORY_TOLERANCE_TOKENS = 1.0
BASELINE_KEY = "random"
CONTROL_KEY = "full"

_COERCE = {"int": int, "float": float, "str": str}


def load_rows(path: str) -> list[BoundaryRow]:
    """Rows back from a sweep's .csv or .jsonl, typed by the dataclass rather than by
    a second hand-written schema that could drift from it."""
    if path.endswith(".jsonl"):
        with open(path) as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(path) as handle:
            records = list(csv.DictReader(handle))
    coerce = {field.name: _COERCE[field.type] for field in fields(BoundaryRow)}
    return [BoundaryRow(**{name: cast(record[name]) for name, cast in coerce.items()})
            for record in records]


@dataclass(frozen=True)
class Estimate:
    mean: float
    stdev: float
    samples: int
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class MethodResult:
    method: str
    method_key: str
    metric: Estimate
    #: None when the baseline had no feasible configuration in this cell, which is a
    #: different statement from an advantage of zero and must not be read as one.
    scoring_advantage: Estimate | None
    prompt_retained: float
    generated_retained: float
    total_retained: float
    generated_position_mean: float
    actual_retained_fraction: float
    kv_bytes: float


@dataclass(frozen=True)
class CellResult:
    workload: str
    redundancy: str
    regime: str
    prompt_length: float
    generation_length: int
    cached_generation_length: float
    prompt_fraction: float
    generated_fraction: float
    requested_retained_fraction: float
    full_cache_metric: float | None
    solvable: bool
    methods: tuple[MethodResult, ...]
    flags: tuple[str, ...]


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))]


def _bootstrap(per_unit: Sequence[float], rng: random.Random) -> tuple[float, float]:
    if len(per_unit) < 2:
        return (per_unit[0], per_unit[0]) if per_unit else (0.0, 0.0)
    replicates = [mean(rng.choice(per_unit) for _ in per_unit)
                  for _ in range(BOOTSTRAP_REPLICATES)]
    tail = (1 - CONFIDENCE) / 2
    return _percentile(replicates, tail), _percentile(replicates, 1 - tail)


def _paired_bootstrap(units: Sequence[tuple[float, float]],
                      rng: random.Random) -> tuple[float, float]:
    """Resamples whole units, so a replicate always compares the two methods on the
    same task instances."""
    return _bootstrap([scored - baseline for scored, baseline in units], rng)


def _estimate(values: Sequence[float], rng: random.Random) -> Estimate:
    low, high = _bootstrap(values, rng)
    return Estimate(mean(values), pstdev(values), len(values), low, high)


def analyse(rows: Iterable[BoundaryRow], *, solvable_at: float = 0.5,
            seed: int = 0) -> list[CellResult]:
    cells: dict[tuple, list[BoundaryRow]] = {}
    for row in rows:
        cells.setdefault((row.task, row.redundancy, row.regime, row.requested_prompt_length,
                          row.requested_generation_length, row.retained_fraction),
                         []).append(row)

    results = []
    for key, group in sorted(cells.items(), key=lambda item: str(item[0])):
        workload, redundancy, regime, prompt_length, generation_length, fraction = key
        rng = random.Random(seed)
        by_method: dict[str, list[BoundaryRow]] = {}
        for row in group:
            by_method.setdefault(row.method_key, []).append(row)

        baseline_units = {(row.seed, row.example): row.metric_value
                          for row in by_method.get(BASELINE_KEY, [])}
        control = by_method.get(CONTROL_KEY, [])
        full_metric = mean(row.metric_value for row in control) if control else None
        solvable = full_metric is not None and full_metric >= solvable_at

        methods = []
        for method_key, method_rows in sorted(by_method.items()):
            if method_key == CONTROL_KEY:
                continue
            values = [row.metric_value for row in method_rows]
            paired = [(row.metric_value, baseline_units[(row.seed, row.example)])
                      for row in method_rows if (row.seed, row.example) in baseline_units]
            advantage = None
            if method_key != BASELINE_KEY and paired:
                low, high = _paired_bootstrap(paired, rng)
                differences = [scored - base for scored, base in paired]
                advantage = Estimate(mean(differences), pstdev(differences), len(differences),
                                     low, high)
            methods.append(MethodResult(
                method=method_rows[0].method, method_key=method_key,
                metric=_estimate(values, rng), scoring_advantage=advantage,
                prompt_retained=mean(row.prompt_retained for row in method_rows),
                generated_retained=mean(row.generated_retained for row in method_rows),
                total_retained=mean(row.total_retained for row in method_rows),
                generated_position_mean=mean(row.generated_position_mean
                                             for row in method_rows),
                actual_retained_fraction=mean(1 - row.compression_ratio for row in method_rows),
                kv_bytes=mean(row.kv_bytes for row in method_rows)))

        measured_prompt = mean(row.prompt_length for row in group)
        cached = mean(row.cached_generation_length for row in group)
        context = measured_prompt + cached
        results.append(CellResult(
            workload=workload, redundancy=redundancy, regime=regime,
            prompt_length=measured_prompt, generation_length=generation_length,
            cached_generation_length=cached,
            prompt_fraction=measured_prompt / context if context else 0.0,
            generated_fraction=cached / context if context else 0.0,
            requested_retained_fraction=fraction, full_cache_metric=full_metric,
            solvable=solvable, methods=tuple(methods),
            flags=tuple(check_cell(group, methods, fraction, solvable, full_metric))))
    return results


def check_cell(group: Sequence[BoundaryRow], methods: Sequence[MethodResult],
               requested_fraction: float, solvable: bool,
               full_metric: float | None) -> list[str]:
    """Everything that would make this cell's numbers uninterpretable, named."""
    flags = []
    if full_metric is None:
        flags.append("no full-cache control in this cell")
    elif not solvable:
        flags.append(f"full cache scores {full_metric:.2f}: the model cannot do this task "
                     "uncompressed here, so nothing below it is evidence about compression")

    compressed = [m for m in methods if m.method_key != CONTROL_KEY]
    if len(compressed) > 1:
        retained = [m.total_retained for m in compressed]
        if max(retained) - min(retained) > MEMORY_TOLERANCE_TOKENS:
            flags.append(f"equal-memory comparison broken: retained KV spans "
                         f"{min(retained):.1f} to {max(retained):.1f} tokens per layer")

    for method in compressed:
        drift = abs(method.actual_retained_fraction - requested_fraction)
        if drift > MEMORY_TOLERANCE_TOKENS / max(1.0, method.total_retained) + 0.02:
            flags.append(f"{method.method_key} retained {method.actual_retained_fraction:.3f} "
                         f"of the cache against a requested {requested_fraction:.2f}")

    baseline = [row for row in group if row.method_key == BASELINE_KEY]
    if baseline:
        evicting = any(row.generated_retained < row.cached_generation_length
                       for row in baseline)
        positions = {row.seed: row.generated_position_mean for row in baseline}
        if evicting and len(positions) > 1 and pstdev(positions.values()) == 0:
            flags.append("random evicted but every seed retained the same positions: the "
                         "baseline is not varying and cannot be read as a random draw")
    return flags


def flat_rows(results: Iterable[CellResult]) -> list[dict]:
    """One record per (cell, method), which is the shape a spreadsheet or a plot
    wants and the shape CellResult is not."""
    records = []
    for cell in results:
        for method in cell.methods:
            advantage = method.scoring_advantage
            records.append({
                "workload": cell.workload, "redundancy": cell.redundancy, "regime": cell.regime,
                "prompt_length": cell.prompt_length, "generation_length": cell.generation_length,
                "cached_generation_length": cell.cached_generation_length,
                "prompt_fraction": cell.prompt_fraction,
                "generated_fraction": cell.generated_fraction,
                "requested_retained_fraction": cell.requested_retained_fraction,
                "actual_retained_fraction": method.actual_retained_fraction,
                "full_cache_metric": cell.full_cache_metric, "solvable": cell.solvable,
                "method": method.method, "method_key": method.method_key,
                "metric_mean": method.metric.mean, "metric_stdev": method.metric.stdev,
                "metric_samples": method.metric.samples,
                "metric_ci_low": method.metric.ci_low, "metric_ci_high": method.metric.ci_high,
                "scoring_advantage": None if advantage is None else advantage.mean,
                "advantage_ci_low": None if advantage is None else advantage.ci_low,
                "advantage_ci_high": None if advantage is None else advantage.ci_high,
                "prompt_retained": method.prompt_retained,
                "generated_retained": method.generated_retained,
                "total_retained": method.total_retained,
                "generated_position_mean": method.generated_position_mean,
                "kv_bytes": method.kv_bytes, "flags": " | ".join(cell.flags),
            })
    return records


def write_analysis(results: Sequence[CellResult], path_stem: str, config: dict) -> None:
    records = flat_rows(results)
    with open(f"{path_stem}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["workload"])
        writer.writeheader()
        writer.writerows(records)
    with open(f"{path_stem}.json", "w") as handle:
        json.dump({"config": {**config, "resampling_unit": RESAMPLING_UNIT,
                              "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                              "confidence": CONFIDENCE},
                   "flags": sorted({flag for cell in results for flag in cell.flags}),
                   "cells": records}, handle, indent=2)


def format_advantage(results: Sequence[CellResult]) -> str:
    header = (f"{'workload':<10}{'redun':<7}{'regime':<17}{'method':<30}{'prm':>5}{'gen':>5}"
              f"{'p_frac':>7}{'retain':>7}{'metric':>8}{'random':>8}{'advantage':>11}"
              f"{'95% CI':>18}{'n':>4}  flags")
    lines = [header, "-" * len(header)]
    for cell in sorted(results, key=lambda c: (c.workload, c.redundancy, c.regime,
                                               c.prompt_length, c.generation_length,
                                               c.requested_retained_fraction)):
        baseline = next((m for m in cell.methods if m.method_key == BASELINE_KEY), None)
        for method in cell.methods:
            if method.method_key == BASELINE_KEY:
                continue
            advantage = method.scoring_advantage
            interval = ("        n/a       " if advantage is None
                        else f"[{advantage.ci_low:+.2f}, {advantage.ci_high:+.2f}]".rjust(18))
            lines.append(
                f"{cell.workload:<10}{cell.redundancy:<7}{cell.regime:<17}{method.method:<30}"
                f"{cell.prompt_length:>5.0f}{cell.generation_length:>5}"
                f"{cell.prompt_fraction:>7.2f}{cell.requested_retained_fraction:>7.2f}"
                f"{method.metric.mean:>8.2f}"
                f"{'   n/a' if baseline is None else format(baseline.metric.mean, '>8.2f')}"
                f"{'        n/a' if advantage is None else format(advantage.mean, '>+11.2f')}"
                f"{interval}{method.metric.samples:>4}  "
                f"{'; '.join(cell.flags)[:60]}")
    return "\n".join(lines)
