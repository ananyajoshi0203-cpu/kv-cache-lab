# Experiment configurations

Three tiers, smallest first. Each is a `--config` for `examples/run_boundary_sweep.py`;
any flag given on the command line overrides the file.

```bash
python examples/run_boundary_sweep.py --config experiments/smoke.json --out results/smoke
python examples/run_boundary_analysis.py --rows results/smoke.jsonl \
    --out results/smoke_analysis --plots results/plots
```

| Config | What it is for | Decodes | Rough cost |
|---|---|---|---|
| `smoke.json` | plumbing only, distilgpt2, CPU | ~800 | ~20 min on a laptop |
| `pilot.json` | the smallest run that could answer the question | ~11k | ~60 GPU-hours |
| `full_study.json` | the intended grid | ~40k | ~220 GPU-hours |

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
not a throughput claim. Two reductions are available and neither is implemented here:
the score-free methods do not need attention weights at all, and prefill attention
could be freed after the prompt scorers have run.

## Sizing a configuration

`kvlab.multistep.minimum_prompt_tokens` reports the shortest prompt each redundancy
level can build before any filler. A shape whose prompt sits below that floor is
refused rather than run, because a high-redundancy prompt that overflows its target
moves redundancy and prompt length together and neither can then be read. Check the
floors against your tokenizer before adding short-prompt shapes.
