"""Check a grid against the tokenizer that will run it, before any inference.

Prompt lengths are token counts, and token counts are a property of a tokenizer, not
of a workload. The floors quoted for the full study were measured on GPT-2 while the
study is configured for Qwen2.5, which is not evidence about the same quantity: the
two tokenizers segment the same evidence differently, so a cell that fits on one can
overflow on the other. Overflowing is not a harmless miss either -- it makes the
prompt run long, which moves redundancy and prompt length together and destroys the
one comparison the redundancy hypothesis rests on.

So the floors are computed here, per workload and per redundancy level, by building
each example with no filler at all and measuring it with the selected model's own
tokenizer. Only the tokenizer and the model's config are loaded; the weights are not
touched, so this costs seconds rather than GPU-hours.

The check refuses the run rather than warning about it when the study's controlled
comparisons are not available: a grid where no shape can hold every redundancy level
cannot answer the redundancy question, and discovering that after the compute is
spent is the expensive way to find out.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .boundary import WORKLOADS, Shape, prompt_band

logger = logging.getLogger(__name__)

CONTEXT_OVERFLOW = "prompt plus generation exceeds the model's context window"
BELOW_FLOOR = "the workload cannot build a prompt this short at this redundancy level"


@dataclass(frozen=True)
class Floor:
    """The shortest prompt a workload can build at a redundancy level, in this
    tokenizer's tokens, with no filler at all."""

    workload: str
    redundancy: str
    minimum_prompt_tokens: int


@dataclass(frozen=True)
class CellCheck:
    workload: str
    redundancy: str
    prompt_length: int
    generation_length: int
    feasible: bool
    reason: str


@dataclass(frozen=True)
class Preflight:
    model: str
    context_limit: int
    floors: tuple[Floor, ...]
    cells: tuple[CellCheck, ...]
    #: Per workload, the shapes where every requested redundancy level fits. These are
    #: the only shapes at which a redundancy comparison is possible at all.
    comparable_shapes: dict[str, tuple[Shape, ...]]
    problems: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.problems

    @property
    def feasible_cells(self) -> tuple[CellCheck, ...]:
        return tuple(cell for cell in self.cells if cell.feasible)


def floor_for(tokenizer, workload_key: str, redundancy: str) -> int:
    """Built at a target of one token, which drives the filler to nothing and leaves
    exactly the evidence and the instructions. Generic across workloads: no workload
    has to publish a floor of its own for this to work."""
    example = WORKLOADS[workload_key].build(tokenizer, 1, 0, 1, 0, redundancy)
    return len(tokenizer(example.context + example.question).input_ids)


def check(tokenizer, *, model: str, context_limit: int, workload_keys, shapes,
          redundancies=None, require_comparable_shapes: int = 1) -> Preflight:
    floors, cells, comparable, problems = [], [], {}, []

    for workload_key in workload_keys:
        workload = WORKLOADS[workload_key]
        levels = (tuple(workload.redundancies) if redundancies is None else
                  tuple(level for level in redundancies if level in workload.redundancies))
        if not levels:
            problems.append(f"{workload_key} varies {', '.join(workload.redundancies)}, none of "
                            "which was requested, so it would contribute no rows")
            continue

        level_floors = {level: floor_for(tokenizer, workload_key, level) for level in levels}
        floors.extend(Floor(workload_key, level, tokens) for level, tokens in level_floors.items())

        fits: dict[Shape, list[str]] = {}
        for prompt, generation in shapes:
            for level in levels:
                reason = ""
                if prompt + generation > context_limit:
                    reason = (f"{CONTEXT_OVERFLOW}: {prompt} + {generation} > {context_limit} "
                              f"for {model}")
                elif level_floors[level] > prompt + prompt_band(prompt):
                    reason = (f"{BELOW_FLOOR}: {level} needs {level_floors[level]} tokens before "
                              f"any filler, target is {prompt} +/- {prompt_band(prompt)}")
                cells.append(CellCheck(workload_key, level, prompt, generation,
                                       not reason, reason))
                if not reason:
                    fits.setdefault((prompt, generation), []).append(level)

        comparable[workload_key] = tuple(shape for shape, got in fits.items()
                                         if len(got) == len(levels))
        if not fits:
            problems.append(f"{workload_key} has no feasible cell in this grid at all")
        elif len(levels) > 1 and len(comparable[workload_key]) < require_comparable_shapes:
            problems.append(
                f"{workload_key} varies {len(levels)} redundancy levels but only "
                f"{len(comparable[workload_key])} shape(s) hold all of them, and "
                f"{require_comparable_shapes} were required: the redundancy comparison this "
                f"study is for cannot be made. Shortest prompt that would work: "
                f"{max(level_floors.values())} tokens plus its band")

    return Preflight(model=model, context_limit=context_limit, floors=tuple(floors),
                     cells=tuple(cells), comparable_shapes=comparable,
                     problems=tuple(problems))


def run(model: str, *, workload_keys, shapes, redundancies=None,
        require_comparable_shapes: int = 1) -> Preflight:
    """Load the selected model's tokenizer and config -- not its weights -- and check
    the grid against them."""
    from transformers import AutoConfig, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    config = AutoConfig.from_pretrained(model)
    limit = getattr(config, "max_position_embeddings", None) or getattr(config, "n_positions", 0)
    logger.info("preflight: %s, context window %d", model, limit)
    return check(tokenizer, model=model, context_limit=limit, workload_keys=workload_keys,
                 shapes=shapes, redundancies=redundancies,
                 require_comparable_shapes=require_comparable_shapes)


def format_report(preflight: Preflight) -> str:
    lines = [f"preflight for {preflight.model} (context window {preflight.context_limit})", ""]
    lines.append(f"{'workload':<12}{'redundancy':<12}{'floor (tokens)':>15}")
    lines.append("-" * 39)
    for floor in preflight.floors:
        lines.append(f"{floor.workload:<12}{floor.redundancy:<12}"
                     f"{floor.minimum_prompt_tokens:>15}")

    feasible = preflight.feasible_cells
    lines += ["", f"grid cells: {len(feasible)} of {len(preflight.cells)} feasible"]
    for cell in preflight.cells:
        if not cell.feasible:
            lines.append(f"  no  {cell.workload}/{cell.redundancy} "
                         f"{cell.prompt_length}x{cell.generation_length}: {cell.reason}")
    for workload_key, shapes in preflight.comparable_shapes.items():
        rendered = ", ".join(f"{p}x{g}" for p, g in shapes) or "none"
        lines.append(f"  {workload_key}: every redundancy level fits at {rendered}")

    lines += ["", "OK" if preflight.ok else "REFUSED:"]
    lines += [f"  {problem}" for problem in preflight.problems]
    return "\n".join(lines)
