"""Reference-method benchmark on a small CPU model: memory saved vs perplexity cost.
For standardized long-context accuracy, use the KVPress backend and evals (see README).

    pip install torch transformers
    python examples/run_benchmark.py [--model gpt2] [--out results/run]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import benchmark
from kvlab.log import configure, get_logger

log = get_logger("benchmark.cli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--prefill", type=int, default=384)
    ap.add_argument("--cont", type=int, default=64)
    ap.add_argument("--ratio", type=float, default=0.75,
                    help="target fraction of cache bytes removed, applied to every method")
    ap.add_argument("--out", default="")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    configure(verbose=args.verbose)

    methods = ["full", "h2o", "snapkv", "cake", "kivi"]
    rows, cfg = benchmark.run(methods, model_name=args.model,
                              prefill=args.prefill, cont=args.cont, ratio=args.ratio)
    log.info("\n%s", benchmark.format_table(rows, cfg))
    if args.out:
        import torch
        import transformers
        config = {"model": cfg.name, "layers": cfg.layers, "kv_heads": cfg.n_kv_heads,
                  "head_dim": cfg.head_dim, "prefill": args.prefill, "cont": args.cont,
                  "ratio": args.ratio, "methods": methods,
                  "torch": torch.__version__, "transformers": transformers.__version__}
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        benchmark.write_results(rows, args.out, config)
        log.info("wrote %s.csv and %s.json", args.out, args.out)


if __name__ == "__main__":
    main()
