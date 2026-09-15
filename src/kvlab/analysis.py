"""Offline analysis of a boundary sweep. Reads saved rows; never runs inference.

The central quantity is deliberately a difference and not a verdict:

    scoring_advantage = scored_method_metric - random_metric

reported as a continuous number with an interval, per cell. No threshold is applied
and no cell is declared "the boundary": where a difference stops mattering is a
judgement about the difference, and printing a cut-off would bury that judgement
inside the tool. The transition is something to look at in the plots, not something
this module decides.

Resampling unit: **one (task_seed, example) pair**, which is one task instance. The
sweep varies two independent sources of randomness and this module keeps them apart:
a task seed chooses the facts, the operations, the filler and the target, while an
eviction seed chooses only what a random method throws away. They are nested, not
crossed with equal standing -- several eviction draws run against the *same* task --
so eviction draws are averaged within their task instance first and the bootstrap
resamples task instances. Pooling the two would let a method with five draws per task
look five times more precisely measured than a deterministic method that ran once,
which is an artefact of the sweep and not a property of the method.

The dispersion each source contributes is reported separately: `metric.stdev` is the
spread across task instances, `eviction_stdev` is the mean spread across draws within
one instance. A random baseline whose eviction_stdev is near zero is not varying in
practice, however many seeds it was given.

An advantage is bootstrapped *paired* -- the same task instance contributes to the
scored method and to the baseline in the same replicate -- because the two ran on the
same task and treating them as independent would overstate the interval. No p-values:
an interval that overlaps zero says what needs saying.

Four sanity checks run before any of it is worth reading, and each one flags rather
than silently drops:

- **solvable** -- the full cache's own score on this cell. Where the model cannot do
  the task uncompressed, nothing below it is evidence about compression.
- **equal memory** -- every compressed method in a cell should hold the same retained
  KV, since that is what the sweep controls. Checked to slot granularity.
- **random varies** -- once eviction is active, different eviction seeds against the
  *same* task must retain visibly different parts of the trace, or the random baseline
  is not random in practice.
- **compression** -- requested retained fraction against the fraction actually
  achieved, both reported.
- **ablation separability** -- a trace-ablation arm that ran out of the class it
  targets spills into the other class, at which point both arms are removing mostly
  the same positions and a null result between them means nothing. This is the
  failure that would most easily be read as "the trace does not matter".
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

RESAMPLING_UNIT = "one (task_seed, example) pair; eviction draws are averaged inside it"
BOOTSTRAP_REPLICATES = 2000
CONFIDENCE = 0.95
#: Retained KV is counted in whole positions per layer per head, so two methods that
#: are meant to hold the same cache may legitimately differ by rounding within one
#: position. More than that is a real mismatch.
MEMORY_TOLERANCE_TOKENS = 1.0
BASELINE_KEY = "random"
CONTROL_KEY = "full"
NO_EVICTION_SEED = -1
ABLATION_PREFIX = "ablate-"

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
    #: Mean spread across eviction draws within one task instance, and how many draws
    #: each instance had. Zero draws means the method is deterministic given the task,
    #: which is a different statement from a draw that never moved.
    eviction_stdev: float
    eviction_draws: int
    #: None when the baseline had no feasible configuration in this cell, which is a
    #: different statement from an advantage of zero and must not be read as one.
    scoring_advantage: Estimate | None
    prompt_retained: float
    generated_retained: float
    total_retained: float
    generated_position_mean: float
    #: For a trace ablation, how much of its removal came from the class it targets
    #: and how much spilled into the other. Spill is what makes two arms converge.
    ablated_targeted: float
    ablated_other: float
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

        def units_of(method_rows):
            """Metrics keyed by task instance, with eviction draws kept together."""
            units: dict[tuple[int, int], list[float]] = {}
            for row in method_rows:
                units.setdefault((row.task_seed, row.example), []).append(row.metric_value)
            return units

        baseline_units = {unit: mean(values)
                          for unit, values in units_of(by_method.get(BASELINE_KEY, [])).items()}
        control = by_method.get(CONTROL_KEY, [])
        full_metric = mean(row.metric_value for row in control) if control else None
        solvable = full_metric is not None and full_metric >= solvable_at

        methods = []
        for method_key, method_rows in sorted(by_method.items()):
            if method_key == CONTROL_KEY:
                continue
            units = units_of(method_rows)
            values = [mean(draws) for draws in units.values()]
            draws_per_unit = max(len(draws) for draws in units.values())
            spread = mean(pstdev(draws) for draws in units.values())
            paired = [(mean(draws), baseline_units[unit])
                      for unit, draws in units.items() if unit in baseline_units]
            advantage = None
            if method_key != BASELINE_KEY and paired:
                low, high = _paired_bootstrap(paired, rng)
                differences = [scored - base for scored, base in paired]
                advantage = Estimate(mean(differences), pstdev(differences), len(differences),
                                     low, high)
            methods.append(MethodResult(
                method=method_rows[0].method, method_key=method_key,
                metric=_estimate(values, rng), scoring_advantage=advantage,
                eviction_stdev=spread,
                eviction_draws=0 if method_rows[0].eviction_seed == NO_EVICTION_SEED
                else draws_per_unit,
                prompt_retained=mean(row.prompt_retained for row in method_rows),
                generated_retained=mean(row.generated_retained for row in method_rows),
                total_retained=mean(row.total_retained for row in method_rows),
                generated_position_mean=mean(row.generated_position_mean
                                             for row in method_rows),
                ablated_targeted=mean(row.ablated_targeted for row in method_rows),
                ablated_other=mean(row.ablated_other for row in method_rows),
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

    for method in methods:
        if method.method_key.startswith(ABLATION_PREFIX) and method.ablated_other > 0:
            flags.append(
                f"{method.method_key} exhausted the class it targets and removed "
                f"{method.ablated_other:.0f} positions from the other one: at this budget the "
                "two ablation arms remove mostly the same positions, so a null between them "
                "is not evidence that the trace does not matter")

    baseline = [row for row in group if row.method_key == BASELINE_KEY]
    if baseline:
        evicting = any(row.generated_retained < row.cached_generation_length
                       for row in baseline)
        # Compared within a task instance, so a spread that is really task variety
        # cannot be mistaken for the draw varying.
        by_task: dict[tuple[int, int], list[float]] = {}
        for row in baseline:
            by_task.setdefault((row.task_seed, row.example), []).append(
                row.generated_position_mean)
        replicated = [positions for positions in by_task.values() if len(positions) > 1]
        if evicting and replicated and all(pstdev(p) == 0 for p in replicated):
            flags.append("random evicted but every eviction seed retained the same positions "
                         "for a given task: the baseline is not varying and cannot be read "
                         "as a random draw")
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
                "eviction_stdev": method.eviction_stdev,
                "eviction_draws": method.eviction_draws,
                "metric_ci_low": method.metric.ci_low, "metric_ci_high": method.metric.ci_high,
                "scoring_advantage": None if advantage is None else advantage.mean,
                "advantage_ci_low": None if advantage is None else advantage.ci_low,
                "advantage_ci_high": None if advantage is None else advantage.ci_high,
                "prompt_retained": method.prompt_retained,
                "generated_retained": method.generated_retained,
                "total_retained": method.total_retained,
                "generated_position_mean": method.generated_position_mean,
                "ablated_targeted": method.ablated_targeted,
                "ablated_other": method.ablated_other,
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
