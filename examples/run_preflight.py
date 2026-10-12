"""Check a grid against the tokenizer that will run it, before spending anything.

    python examples/run_preflight.py --config experiments/full_study.json

Loads the selected model's tokenizer and config, never its weights, and reports the
prompt floor for every workload and redundancy level, which grid cells are feasible,
and which shapes can hold every redundancy level. Exits non-zero when the study's
controlled comparisons are not available in the grid as configured.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import preflight                                         # noqa: E402
from kvlab.log import configure, get_logger                         # noqa: E402
from run_boundary_sweep import resolve                              # noqa: E402

log = get_logger("preflight.cli")


def main():
    args = resolve()
    configure(verbose=args.verbose)
    report = preflight.run(args.model, workload_keys=args.workloads, shapes=args.shapes,
                           redundancies=args.redundancy,
                           require_comparable_shapes=args.require_comparable_shapes)
    log.info("\n%s", preflight.format_report(report))
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
