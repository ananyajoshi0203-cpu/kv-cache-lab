"""Turn saved sweep rows into cell-level results, flags and figures. No inference.

    python examples/run_boundary_analysis.py --rows results/smoke.jsonl \
        --out results/smoke_analysis --plots results/plots

Reports the scoring advantage over the score-free baseline as a continuous number
with a bootstrap interval, per cell, and applies no threshold: where a difference
stops mattering is a judgement about the difference, not something this script gets
to make. Sanity flags are printed first, because a cell whose full-cache control
cannot do the task says nothing about compression.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import analysis                                          # noqa: E402
from kvlab.log import configure, get_logger                         # noqa: E402

log = get_logger("analysis.cli")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True, help="a sweep's .jsonl or .csv")
    ap.add_argument("--out", default="", help="path stem for the analysis csv and json")
    ap.add_argument("--plots", default="", help="directory for the figures")
    ap.add_argument("--solvable-at", type=float, default=0.5,
                    help="full-cache score below which a cell is flagged as telling us "
                         "nothing about compression")
    ap.add_argument("--bootstrap-seed", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    configure(verbose=args.verbose)

    rows = analysis.load_rows(args.rows)
    results = analysis.analyse(rows, solvable_at=args.solvable_at, seed=args.bootstrap_seed)
    log.info("%d rows -> %d cells", len(rows), len(results))

    flags = sorted({flag for cell in results for flag in cell.flags})
    if flags:
        log.info("sanity flags:")
        for flag in flags:
            log.info("  %s", flag)
    else:
        log.info("sanity flags: none")
    unsolved = sum(not cell.solvable for cell in results)
    log.info("cells whose full cache is below the solvable threshold: %d of %d",
             unsolved, len(results))
    log.info("resampling unit for every interval: %s", analysis.RESAMPLING_UNIT)
    log.info("\n%s", analysis.format_advantage(results))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        analysis.write_analysis(results, args.out, {"rows": args.rows,
                                                    "solvable_at": args.solvable_at,
                                                    "bootstrap_seed": args.bootstrap_seed})
        log.info("wrote %s.csv and %s.json", args.out, args.out)
    if args.plots:
        from kvlab import plots
        written = plots.draw_all(analysis.flat_rows(results), args.plots)
        log.info("wrote %d figures to %s", len(written), args.plots)


if __name__ == "__main__":
    main()
