# Smoke run

`experiments/smoke.json` on distilgpt2, CPU. **This is evidence about the plumbing and
about nothing else.** It exists to show that the workloads build, the runner sweeps,
results serialize, the analysis reads them back and the figures draw.

1296 rows, 36 cells, 6 task instances per cell (3 task seeds x 2 examples), 3 eviction
draws for each stochastic method, 14 figures. The preflight ran first and is recorded
in `run_config.json`.

## What it confirms, and what it does not

distilgpt2 cannot do `multistep`. Every multistep cell scores zero and every one is
flagged by the full-cache solvability check: **24 of 36 cells unsolvable**. That is
the check working. Reading a compression conclusion out of those cells is exactly the
mistake the check exists to prevent.

`needle` it can do, so those cells carry the only interpretable numbers here, and even
those are a two-shape CPU run at three retained fractions. They are not a result.

The trace ablation is flagged unseparable in five cells: a distilgpt2 trace holds very
few numeric positions, so the numeric arm exhausts its class and spills into the other
one, at which point both arms remove mostly the same positions. That is the guard
working. It is also the reason the intervention belongs at mild compression, and the
reason a null between the arms here would mean nothing.

## Files

<dl>
<dt>analysis.csv</dt><dd>one record per (cell, method): metric with bootstrap interval,
scoring advantage over the random baseline, retained KV split by prompt and generated,
achieved compression, analytical KV bytes, and any sanity flags.</dd>
<dt>run_config.json</dt><dd>what produced it, including how many cells each skip reason
accounted for.</dd>
<dt>A_needle_published_none.png, B_needle.png</dt><dd>two of the fourteen figures. The
rest regenerate from the raw rows in one command; they are not checked in because they
are derived and large.</dd>
</dl>

Raw rows (`results/smoke.{csv,jsonl,json}`, ~600KB) are deliberately not committed.

## Reproducing

```bash
python examples/run_boundary_sweep.py --config experiments/smoke.json --out results/smoke
python examples/run_boundary_analysis.py --rows results/smoke.jsonl \
    --out results/smoke_analysis --plots results/plots
```
