"""Sweep for the boundary where a score starts to beat a coin flip.

Prompt length x generation length x retained fraction of the cache, against the
score-free prompt-protected random baseline, in two regimes: methods as published,
and methods with the prompt held out of reach. Read kvlab.boundary's module
docstring first. Every method in a row is held to the same final cache size, and
the nominal budget each needed to get there is different and is recorded.

    pip install torch transformers
    python examples/run_boundary_sweep.py --out results/boundary

Defaults are sized for a CPU run on distilgpt2. Every axis is a flag, so widen it
where the machine allows.
"""

import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--workload", default="needle", choices=sorted(boundary.WORKLOADS))
    ap.add_argument("--prompts", type=ints, default=(128, 384),
                    help="target prompt lengths in tokens")
    ap.add_argument("--gen", type=ints, default=(48,), help="generation lengths in tokens")
    ap.add_argument("--retain", type=floats, default=(0.25, 0.5, 0.75),
                    help="fraction of the full cache every method is held to; the per-method "
                         "budget that reaches it is derived and recorded, never shared")
    ap.add_argument("--seeds", type=ints, default=(0, 1, 2, 3, 4))
    ap.add_argument("--examples", type=int, default=1, help="examples per seed")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    configure(verbose=args.verbose)

    result = boundary.sweep(
        args.model, workload_key=args.workload, prompt_lengths=args.prompts,
        generation_lengths=args.gen, retained_fractions=args.retain,
        seeds=args.seeds, examples=args.examples)
    workload = boundary.WORKLOADS[args.workload]
    summaries = boundary.summarize(result.rows)
    log.info("\n%s", boundary.format_summary(summaries, workload.task_metric))
    for skip in result.skipped:
        log.info("skipped %s/%s at prompt=%d gen=%d retain=%.2f: %s", skip.regime, skip.method_key,
                 skip.prompt_length, skip.generation_length, skip.retained_fraction, skip.reason)

    if args.out:
        import torch
        import transformers
        config = {
            "model": result.cfg.name, "layers": result.cfg.layers,
            "kv_heads": result.cfg.n_kv_heads, "head_dim": result.cfg.head_dim,
            "workload": workload.key, "workload_note": workload.note,
            "task_metric": workload.task_metric,
            "prompt_lengths": list(args.prompts), "generation_lengths": list(args.gen),
            "retained_fractions": list(args.retain), "seeds": list(args.seeds),
            "examples_per_seed": args.examples,
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
