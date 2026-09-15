"""Sweep for the boundary where a score starts to beat a coin flip.

Workload x redundancy x (prompt, generation) shape x retained fraction, against the
score-free prompt-protected random baseline, in two regimes: methods as published,
and methods with the prompt held out of reach. Read kvlab.boundary's module
docstring first. Every method in a row is held to the same final cache size, and the
nominal budget each needed to get there is different and is recorded.

    pip install torch transformers
    python examples/run_boundary_sweep.py --out results/smoke
    python examples/run_boundary_sweep.py --config experiments/smoke.json --out results/smoke

Defaults are sized for a CPU run on distilgpt2, which is a plumbing check and not
evidence about how a modern model uses its cache. The checked-in configurations under
experiments/ describe the runs that are.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import boundary                                          # noqa: E402
from kvlab.log import configure, get_logger                         # noqa: E402

log = get_logger("boundary.cli")


def ints(text):
    return tuple(int(part) for part in text.split(","))


def floats(text):
    return tuple(float(part) for part in text.split(","))


def words(text):
    return tuple(part.strip() for part in text.split(",") if part.strip())


def shapes(text):
    """--shapes 128x64,256x512 -- explicit pairs, because the interesting matrix is
    not the full cross product of prompt and generation lengths."""
    pairs = []
    for part in text.split(","):
        prompt, _, generation = part.strip().lower().partition("x")
        pairs.append((int(prompt), int(generation)))
    return tuple(pairs)


def build_parser(explicit_only=False):
    """With explicit_only, unsupplied flags are absent from the parse rather than
    filled with defaults, which is how a configuration file can be overridden by the
    command line without a default silently counting as an override."""
    ap = argparse.ArgumentParser(
        argument_default=argparse.SUPPRESS if explicit_only else None)
    ap.add_argument("--config", default="", help="JSON experiment configuration; any flag "
                                                 "given on the command line overrides it")
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--workloads", type=words, default=tuple(boundary.WORKLOADS))
    ap.add_argument("--shapes", type=shapes, default=boundary.DEFAULT_SHAPES,
                    help="prompt x generation pairs, e.g. 128x64,256x512")
    ap.add_argument("--retain", type=floats, default=boundary.DEFAULT_FRACTIONS,
                    help="fractions of the full cache every method is held to; the per-method "
                         "budget that reaches each is derived and recorded, never shared")
    ap.add_argument("--redundancy", type=words, default=None,
                    help="prompt-redundancy levels; omit for every level each workload varies")
    ap.add_argument("--seeds", type=ints, default=(0, 1, 2, 3, 4))
    ap.add_argument("--examples", type=int, default=1, help="task examples per seed")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    return ap


CONFIG_KEYS = ("model", "workloads", "shapes", "retain", "redundancy", "seeds", "examples")


def resolve(argv=None):
    """Command line over configuration file over defaults."""
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    if not args.config:
        return args

    given = vars(build_parser(explicit_only=True).parse_args(argv))
    with open(args.config) as handle:
        config = json.load(handle)
    for key in CONFIG_KEYS:
        if key not in config or key in given:
            continue
        value = config[key]
        if key == "shapes":
            value = tuple(tuple(pair) for pair in value)
        elif isinstance(value, list):
            value = tuple(value)
        setattr(args, key, value)
    return args


def main():
    args = resolve()
    configure(verbose=args.verbose)

    result = boundary.sweep(
        args.model, workload_keys=args.workloads, shapes=args.shapes,
        retained_fractions=args.retain, redundancies=args.redundancy,
        seeds=args.seeds, examples=args.examples)
    summaries = boundary.summarize(result.rows)
    log.info("\n%s", boundary.format_summary(summaries, "metric"))
    for skip in result.skipped:
        log.info("skipped %s/%s on %s/%s at prompt=%d gen=%d retain=%.2f: %s",
                 skip.regime, skip.method_key, skip.workload, skip.redundancy,
                 skip.prompt_length, skip.generation_length, skip.retained_fraction, skip.reason)

    if args.out:
        import torch
        import transformers
        config = {
            "model": result.cfg.name, "layers": result.cfg.layers,
            "kv_heads": result.cfg.n_kv_heads, "head_dim": result.cfg.head_dim,
            "workloads": list(args.workloads),
            "workload_notes": {key: boundary.WORKLOADS[key].note for key in args.workloads},
            "shapes": [list(shape) for shape in args.shapes],
            "retained_fractions": list(args.retain),
            "redundancy": list(args.redundancy) if args.redundancy else "every level per workload",
            "seeds": list(args.seeds), "examples_per_seed": args.examples,
            "budget_semantics": "every method in a cell is held to the same total_budget; "
                                "generation_budget bounds the generated cache only, so a "
                                "prompt-protected run holds prompt_length + generation_budget "
                                "and the two numbers are never interchanged",
            "kv_bytes_kind": boundary.KV_BYTES_KIND,
            "decode_wall_seconds_note": "Python-level wall clock, for bookkeeping only; "
                                        "not a throughput measurement",
            "skipped": [vars(skip) for skip in result.skipped],
            "torch": torch.__version__, "transformers": transformers.__version__,
        }
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        boundary.write_rows(result.rows, summaries, args.out, config)
        log.info("wrote %s.csv, %s.jsonl and %s.json", args.out, args.out, args.out)


if __name__ == "__main__":
    main()
