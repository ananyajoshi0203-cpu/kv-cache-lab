# Experiment configurations

Three tiers, smallest first. Each is a `--config` for `examples/run_boundary_sweep.py`;
any flag given on the command line overrides the file.

```bash
python examples/run_boundary_sweep.py --config experiments/smoke.json --out results/smoke
python examples/run_boundary_analysis.py --rows results/smoke.jsonl \
    --out results/smoke_analysis --plots results/plots
```

| Config | What it is for | Decodes | Cost |
|---|---|---|---|
| `smoke.json` | plumbing only, distilgpt2, CPU | 504 measured | ~7 min on a laptop, measured |
| `pilot.json` | the smallest run that could answer the question | ~5,200 | ~24 GPU-hours, estimated |
| `full_study.json` | the intended grid | ~37,000 | ~205 GPU-hours, estimated |

The smoke figure is measured: 504 decodes at 0.79 s each. The other two are arithmetic
over the grid, not observations, and they assume a rate this repository has not yet
measured on a GPU. Run the pilot before believing the full-study number.

## Read the smoke run for what it is

distilgpt2 cannot do multistep. Every multistep cell in the smoke run is expected to
score zero and to be flagged unsolvable by the full-cache check, and that is the
check working. Nothing in a smoke run is evidence about how a modern model uses its
cache.

## Where the compute goes

Cost is dominated by generation length and by `output_attentions=True`, which the
harness needs because the scorers read real attention weights and which forces eager
attention. Per group of (workload, redundancy, shape) the runner performs
`units x (1 + methods x retained_fractions)` decodes, where a unit is one
(seed, example) pair: the full cache ignores the budget so it decodes once per unit,
and everything else decodes once per unit per retained fraction.

The estimates above assume roughly 0.08 s per generated token for a 1.5B model with
eager attention and per-step Python decoding, which is the harness's actual shape and
not a throughput claim. They also discount the grid by 20% for cells no method can
reach, which is roughly what the smoke run skipped; the real fraction depends on the
shapes and is reported per run. Two reductions are available and neither is implemented here:
the score-free methods do not need attention weights at all, and prefill attention
could be freed after the prompt scorers have run.

## Sizing a configuration

Run the preflight. It loads the selected model's tokenizer and config -- never its
weights -- and reports the prompt floor for each workload and redundancy level, which
grid cells are feasible, and which shapes can hold every redundancy level:

```bash
python examples/run_preflight.py --config experiments/full_study.json
```

The sweep runs the same check before any inference and **refuses to start** when no
shape can hold every redundancy level, because such a grid cannot answer the
redundancy question however long it runs.

**Floors are a property of the tokenizer, not of the workload.** The same evidence
segments differently, and the difference is not small: high redundancy needs 265
tokens under GPT-2 and **310** under Qwen2.5. A floor measured on one model is not
evidence about another, and using GPT-2's would have put prompt 256 in the Qwen grid
as if it were feasible. `full_study_preflight.json` is the report for the intended
study, regenerated from the command above rather than hand-written.
