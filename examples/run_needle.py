"""Passkey retrieval under compression on a small CPU model. The ratio 0.0 row
is the uncompressed control; read every other row as a delta from it.

    pip install torch transformers
    python examples/run_needle.py [--model gpt2] [--method snapkv] [--ratios 0,0.5,0.75]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import needle
from kvlab.backends import ReferenceBackend
from kvlab.log import configure, get_logger

log = get_logger("needle.cli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilgpt2")
    ap.add_argument("--method", default="snapkv", choices=["snapkv", "h2o", "kivi", "full"])
    ap.add_argument("--ratios", default="0,0.5,0.75",
                    help="comma-separated fractions of cache bytes to remove")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--sentences", type=int, default=30)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    configure(verbose=args.verbose)

    backend = ReferenceBackend(model=args.model, method_key=args.method)
    ratios = [float(r) for r in args.ratios.split(",")]
    rows = needle.run_needle(backend, ratios, n_examples=args.n, filler_sentences=args.sentences)
    log.info("model=%s method=%s\n%s", args.model, args.method, needle.format_table(rows))


if __name__ == "__main__":
    main()
