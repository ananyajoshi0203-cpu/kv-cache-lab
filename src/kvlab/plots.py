"""Figures for a boundary sweep, drawn from saved analysis and never from inference.

Four views, because the hypothesis has two halves and a table cannot show a
transition:

A. task metric against retained fraction, per method, faceted by shape. Where does
   each method break, and does the score-free baseline break at the same place.
B. scoring advantage against the prompt's share of the context. If selection matters
   because unique information is concentrated in the prompt, this is where it shows.
C. the same against the generated share, needle and multistep drawn separately
   because they probe opposite halves of the cache.
D. advantage as a heatmap over prompt and generation length, for the shape of the
   region rather than its slices.

Everything reads the flat records from kvlab.analysis. Cells flagged unsolvable are
drawn hollow rather than dropped: a method scoring zero because the model could not
do the task is not the same finding as a method scoring zero because its cache was
compressed, and hiding those rows would make the second look more common than it is.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

logger = logging.getLogger(__name__)

BASELINE_KEY = "random"
FIGURE_DPI = 140


def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _group(records, *keys):
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(tuple(record[key] for key in keys), []).append(record)
    return grouped


def _save(fig, path: str) -> str:
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    _pyplot().close(fig)
    logger.info("wrote %s", path)
    return path


def plot_metric_against_retention(records: Sequence[dict], out_dir: str) -> list[str]:
    """Plot A."""
    plt = _pyplot()
    written = []
    for (workload, regime, redundancy), group in _group(
            records, "workload", "regime", "redundancy").items():
        shapes = sorted({(r["generation_length"], round(r["prompt_length"])) for r in group})
        if not shapes:
            continue
        fig, axes = plt.subplots(1, len(shapes), figsize=(4 * len(shapes), 3.6), squeeze=False,
                                 sharey=True)
        for axis, shape in zip(axes[0], shapes):
            generation, prompt = shape
            cells = [r for r in group
                     if r["generation_length"] == generation
                     and round(r["prompt_length"]) == prompt]
            for (method,), points in sorted(_group(cells, "method").items()):
                baseline = points[0]["method_key"] == BASELINE_KEY
                curve = sorted(points, key=lambda r: r["requested_retained_fraction"])
                axis.plot([p["requested_retained_fraction"] for p in curve],
                          [p["metric_mean"] for p in curve],
                          marker="s" if baseline else "o",
                          linewidth=2.4 if baseline else 1.3, label=method)
            control = next((r["full_cache_metric"] for r in cells
                            if r["full_cache_metric"] is not None), None)
            if control is not None:
                axis.axhline(control, color="0.4", linestyle=":", linewidth=1,
                             label="full cache")
            axis.set_title(f"prompt {prompt} / gen {generation}", fontsize=9)
            axis.set_xlabel("retained fraction")
            axis.set_ylim(-0.05, 1.05)
        axes[0][0].set_ylabel("task metric")
        axes[0][-1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"A. {workload} / {regime} / redundancy {redundancy}", fontsize=10)
        written.append(_save(fig, f"{out_dir}/A_{workload}_{regime}_{redundancy}.png"))
    return written


def _advantage_scatter(records, out_dir, x_key, title, stem):
    plt = _pyplot()
    written = []
    for (workload,), group in _group(records, "workload").items():
        scored = [r for r in group
                  if r["method_key"] != BASELINE_KEY and r["scoring_advantage"] is not None]
        if not scored:
            continue
        fig, axis = plt.subplots(figsize=(6.4, 4.2))
        for (method,), points in sorted(_group(scored, "method").items()):
            solved = [p for p in points if p["solvable"]]
            unsolved = [p for p in points if not p["solvable"]]
            line = axis.scatter([p[x_key] for p in solved], [p["scoring_advantage"] for p in solved],
                                s=26, label=method)
            axis.scatter([p[x_key] for p in unsolved],
                         [p["scoring_advantage"] for p in unsolved], s=26,
                         facecolors="none", edgecolors=line.get_facecolor(), linewidths=0.8)
        axis.axhline(0, color="0.3", linewidth=1)
        axis.set_xlabel(x_key.replace("_", " "))
        axis.set_ylabel("scoring advantage over random")
        axis.set_title(f"{title} -- {workload} (hollow: full cache unsolved)", fontsize=10)
        axis.legend(fontsize=7)
        written.append(_save(fig, f"{out_dir}/{stem}_{workload}.png"))
    return written


def plot_advantage_against_prompt_fraction(records: Sequence[dict], out_dir: str) -> list[str]:
    """Plot B, the one the prompt-fragility hypothesis lives or dies on."""
    return _advantage_scatter(records, out_dir, "prompt_fraction",
                              "B. scoring advantage vs prompt share of context", "B")


def plot_advantage_against_generated_fraction(records: Sequence[dict], out_dir: str) -> list[str]:
    """Plot C."""
    return _advantage_scatter(records, out_dir, "generated_fraction",
                              "C. scoring advantage vs generated share of context", "C")


def plot_advantage_heatmap(records: Sequence[dict], out_dir: str) -> list[str]:
    """Plot D. Averaged over retained fraction and redundancy, which is stated in the
    title because averaging over a transition is exactly how one gets hidden."""
    plt = _pyplot()
    written = []
    for (workload,), group in _group(records, "workload").items():
        scored = [r for r in group
                  if r["method_key"] != BASELINE_KEY and r["scoring_advantage"] is not None]
        methods = sorted({r["method"] for r in scored})
        if not methods:
            continue
        prompts = sorted({round(r["prompt_length"]) for r in scored})
        generations = sorted({r["generation_length"] for r in scored})
        fig, axes = plt.subplots(1, len(methods), figsize=(3.4 * len(methods), 3.2),
                                 squeeze=False)
        for axis, method in zip(axes[0], methods):
            cells = _group([r for r in scored if r["method"] == method],
                           "prompt_length", "generation_length")
            grid = [[float("nan")] * len(prompts) for _ in generations]
            for (prompt, generation), points in cells.items():
                row, column = generations.index(generation), prompts.index(round(prompt))
                grid[row][column] = sum(p["scoring_advantage"] for p in points) / len(points)
            image = axis.imshow(grid, cmap="RdBu_r", vmin=-1, vmax=1, origin="lower",
                                aspect="auto")
            axis.set_xticks(range(len(prompts)), prompts, fontsize=7)
            axis.set_yticks(range(len(generations)), generations, fontsize=7)
            axis.set_xlabel("prompt length", fontsize=8)
            axis.set_title(method, fontsize=8)
        axes[0][0].set_ylabel("generation length", fontsize=8)
        fig.colorbar(image, ax=axes[0].tolist(), shrink=0.85, label="advantage over random")
        fig.suptitle(f"D. {workload}: mean advantage over retained fraction and redundancy",
                     fontsize=10)
        written.append(_save(fig, f"{out_dir}/D_{workload}.png"))
    return written


PLOTS = (plot_metric_against_retention, plot_advantage_against_prompt_fraction,
         plot_advantage_against_generated_fraction, plot_advantage_heatmap)


def draw_all(records: Sequence[dict], out_dir: str) -> list[str]:
    import os
    os.makedirs(out_dir, exist_ok=True)
    return [path for plot in PLOTS for path in plot(records, out_dir)]
