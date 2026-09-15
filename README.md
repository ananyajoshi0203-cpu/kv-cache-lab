# kv-cache-lab

A reproducible harness for comparing KV-cache optimization methods on equal terms.

The KV cache is the memory an autoregressive transformer keeps so it never
recomputes past tokens. It is also the main reason long-context inference is
expensive: the cache grows linearly with sequence length, and past a point it, not
the model weights, is what fills the GPU. A large body of work now attacks this from
several angles, and the recurring finding is that no single method is best in every
setting. This repository takes that finding seriously and builds the tooling to test
it: an analytical cost model, a taxonomy of the methods, and a comparison harness
that stands on established libraries and benchmarks rather than a bespoke one.

The design goal is credibility. Method implementations come from NVIDIA
[KVPress](https://github.com/NVIDIA/kvpress); accuracy is meant to come from the
same long-context benchmarks the literature reports (RULER, LongBench, SCBench,
IFEval), invoked through KVPress's evaluation CLI — this repository catalogs and
wires those suites but has not yet produced published result artifacts. Every
method is compared at a matched compression ratio so the numbers mean something.
The repository also ships small reference implementations of a few methods so the
mechanism is legible, but these are for understanding, not for headline numbers:
where a reference simplifies its paper, the class docstring and registry entry
say exactly what is omitted.

## The cost model

Everything follows from one equation. For a token, the cache stores a key and a
value vector in every attention head of every layer:

```
bytes_per_token = 2 · layers · kv_heads · head_dim · dtype_bytes
total_cache     = bytes_per_token · context_length · batch
```

For an MHA 7B configuration like Llama-2-7B (32 layers, 32 KV heads, head_dim 128,
fp16) this is 0.5 MB per token, so a 128K context needs 64 GB of cache, which alone
exceeds an 80 GB A100 once weights and activations are resident. On a 24 GB card
the cache fills near 48K tokens. GQA models divide this by their grouping factor:
Llama-3.1-8B keeps 8 KV heads instead of 32, so 0.125 MB per token and 16 GB at
128K. `kvlab.memory` computes these figures with no dependencies, and
`examples/run_memory_demo.py` reproduces them.

Every optimization reduces one term of that equation. This gives a clean way to
organize the field:

| Lever | Term | Family | Idea |
|-------|------|--------|------|
| context | `context_length` | Eviction | drop tokens that will not be needed |
| dtype | `dtype_bytes` | Compression | store each number in fewer bits |
| location | where it lives | Hybrid memory | move the cache to CPU or disk (lossless) |
| formula | the whole term | Alternative attention | redesign attention so there is no per-token cache |
| mixed | several | Hybrid approaches | combine the above |

The registry in `kvlab.registry` catalogs the methods from the survey against these
levers, and records which KVPress press implements each one where a mapping exists.

## Why fair comparison is not obvious

Two rules keep the comparison honest, and most casual benchmarks miss at least one.

First, a fixed control. The uncompressed cache defines maximum quality and maximum
memory; every method is reported as a delta from it.

Second, iso-ratio evaluation. Methods are compared at the same compression ratio,
not the same hyperparameters. If one method simply keeps more tokens it will look
better for a trivial reason. Fixing the ratio isolates the quality difference that
actually matters.

Quality itself is plural. Perplexity is cheap but misleading: a method can hold
perplexity while silently ignoring instructions. That failure mode is documented in
*The Pitfalls of KV Cache Compression* and is the reason IFEval is part of the
benchmark set rather than perplexity alone.

## The boundary experiment

Every evictor here is a scoring function, which means the harness cannot answer
whether the scores are what is doing the work. *Random Attention*
([arXiv:2609.03430](https://arxiv.org/abs/2609.03430)) reports that keeping the
prompt and then evicting uniformly at random within each head matches the strongest
scored evictor. That makes a score-free method the control the others have to beat,
and it makes one question worth answering before any new method is worth proposing:
**where does score-free eviction stop being enough?**

`examples/run_boundary_sweep.py` sweeps prompt length, generation length and the
retained fraction of the cache to find that crossing. It is not a new method and it
does not try to be.

### The budget is the part that is easy to get wrong

`PromptProtectedRandom` takes a budget over the **generated** cache; H2O takes one
over the **whole** cache; SnapKV, CAKE and OBCache take one over the **prompt** and
then let the cache grow with every generated token. Hand all of them the number 128
and they hold three different amounts of memory, and the one that quietly holds the
most looks like it won.

So the sweep controls the one thing that is comparable: the **final cache size**.
Each method is then given whatever budget reaches it, derived separately and
recorded per row.

<dl>
<dt>H2O</dt><dd>budget = target (it bounds the whole cache at every step)</dd>
<dt>SnapKV, CAKE, OBCache</dt><dd>budget = target − generated tokens (it only ever
compresses the prompt, then accumulates)</dd>
<dt>Prompt-protected random</dt><dd>budget = target − prompt length (it refuses to
touch the prompt)</dd>
</dl>

Where a method has no budget that reaches the target, the cell is **skipped with a
reason** rather than fudged into the table, and two of those skips are results in
their own right:

- Below a retained fraction that holds the whole prompt, a prompt-protected method
  has no feasible configuration at all. On a 192-token prompt with 32 generated
  tokens that floor is 0.86, so the method cannot be run over most of the
  compression range. This is the honest form of "the protected prompt makes total
  memory differ", and it is why an iso-total-memory claim is never made on its
  behalf.
- An after-prefill press cannot reach a target smaller than the generated tokens it
  accumulates. With a 128-token prompt and 384 generated tokens, SnapKV, CAKE and
  OBCache have no configuration at any interesting compression ratio, because they
  never touch the generated cache at all — exactly the regime long reasoning traces
  live in.

### Two regimes

**Published** runs every method as its paper defines it. The prompt/generated split
in the accounting columns is what to read: at equal memory, how much of each cache
went to the prompt.

**Prompt-protected** wraps methods so the prompt is untouchable and the target is
spent on the generated cache alone, isolating what the score bought from whether
the score happened to keep the prompt. A wrapped method is reported under its own
key (`pp-h2o`) and its own name, never under the published one.

SnapKV, CAKE and OBCache have **no prompt-protected form** here, and that is a
finding rather than an omission: they vote with an observation window of prompt
queries, and once the prompt is protected only the single query of one decode step
remains. A scorer reading one query is a different scorer wearing the same name.
Giving them a faithful decode-time form is method design, which this phase
deliberately does not do.

### Two workloads, because the cache has two halves

`needle` probes **prompt** retention: a passkey buried in filler. Its answer comes out
within the first few generated tokens, so a budget over the generated cache has
nothing to bite on. That is a property of the task, not a bug, and it is why it
cannot be the only one.

`multistep` probes **generated-trace** retention. Quantities are stated in the prompt
and a chain of operations combines them, each step consuming the running total the
model just wrote, so by the last step the operand it needs exists only in its own
generated KV. Scored on the last integer written and on nothing about the shape of
the reasoning; chains whose answer equals an operand or an intermediate are rejected
and redrawn, so a model that copies a number or truncates its trace cannot score by
accident.

### Redundancy is the second independent variable

`multistep` can state each fact once, twice or four times in genuinely different
sentences, dealt round by round so a fact's restatements land at different depths
rather than clustering, while filler shrinks to hold the prompt at its target length.
Same facts, same operations, same answer: the only thing that moves is the share of
prompt KV carrying information the model has already seen elsewhere.

That is what separates the two hypotheses. If scoring matters because unique
information is concentrated in the prompt, advantage should track the prompt's share
of the context. If it matters because of information density, redundancy should
predict the crossover better than context fraction does.

Evidence has a floor, and the harness enforces it: 115 prompt tokens at low
redundancy, 165 at medium, 265 at high, before any filler. A prompt target below its
level's floor is **refused with a reason**, never quietly run long — otherwise
redundancy and prompt length would move together and neither could be read.

### What is measured

Task accuracy, not perplexity — the repository already knows perplexity hides the
failure eviction actually causes. Alongside it, every row carries the workload and
redundancy level, prompt KV, generated KV and total KV retained, the mean original
position of the retained generated KV, the analytical KV bytes, the compression
ratio actually achieved, and the nominal budget that method was handed.

Two things the numbers do **not** claim. The byte figures are **analytical** — slot
counts multiplied through the cost model, never allocator readings — and the per-row
wall clock is Python-level bookkeeping, not a throughput measurement.

### Analysis, offline

The sweep writes rows; a separate script turns them into cells and figures without
touching a model:

```bash
python examples/run_boundary_sweep.py --config experiments/smoke.json --out results/smoke
python examples/run_boundary_analysis.py --rows results/smoke.jsonl \
    --out results/analysis --plots results/plots
```

The quantity is a continuous difference and not a verdict:

```
scoring_advantage = scored_method_metric - random_metric
```

reported per cell with a bootstrap interval and **no threshold** — where a difference
stops mattering is a judgement about the difference, and a cut-off buried in a tool
is a judgement nobody sees being made. The resampling unit is one `(seed, example)`
pair, resampled *paired* for an advantage because both methods ran on the same task
instance. No p-values.

Four checks run before any of it is worth reading, and each flags the cell rather
than dropping it: whether the **full cache** can do the task at all (if not, nothing
below it is evidence about compression), whether every method in a cell really held
**equal memory**, whether the **requested** retained fraction matches the **achieved**
one, and whether the random baseline's seeds actually retained different positions.

Figures: task metric against retained fraction per method and shape; advantage
against the prompt's share of context; advantage against the generated share, needle
and multistep drawn apart; and advantage as a heatmap over prompt and generation
length. Cells whose control failed are drawn hollow, not hidden.

`experiments/` holds three checked-in configurations — a CPU smoke run that proves
only the plumbing, a pilot, and the full grid — with their compute estimates. **Do
not read distilgpt2 results as evidence about modern KV-cache behaviour**: it cannot
do `multistep` at all, and every one of its multistep cells is expected to be flagged
unsolvable.

## Architecture

```
kv-cache-lab/
  src/kvlab/
    memory.py       cost model and GPU fit, dependency-free
    registry.py     methods by family and lever, mapped to KVPress presses
    scenarios.py    deployment scenarios and the methods that fit them
    backends.py     ReferenceBackend (in-repo) and KVPressBackend (NVIDIA KVPress)
    methods.py      reference implementations of prompt-protected random, H2O,
                    SnapKV, OBCache, CAKE, KIVI
    decode.py       step-wise decode loop with a per-step method hook
    accounting.py   prompt KV vs generated KV, and the analytical bytes behind them
    boundary.py     the boundary sweep: one target cache, a budget per method
    multistep.py    a task whose answer lives in the model's own generated trace
    analysis.py     offline: scoring advantage, bootstrap intervals, sanity flags
    plots.py        the four figures, drawn from saved analysis only
    evals.py        standard benchmarks (RULER, LongBench, SCBench, IFEval)
    needle.py       passkey retrieval eval, runnable on CPU
    benchmark.py    iso-ratio runner with CSV/JSON output
    log.py          logging setup
  examples/         runnable entry points
  tests/            unit tests for the cost model and registry
  docs/TAXONOMY.md  every method mapped to its lever
```

The split matters: `memory`, `registry`, and `scenarios` are pure Python and always
run. The backends and evals require torch, transformers, and optionally KVPress, and
are only imported when used.

## Quickstart

The cost model and taxonomy run anywhere, no downloads:

```bash
python examples/run_memory_demo.py
```

The reference benchmark runs a small model on CPU and reports memory saved against
perplexity cost, every method at the same target compression ratio:

```bash
pip install torch transformers
python examples/run_benchmark.py --ratio 0.75 --out results/reference
```

Perplexity understates eviction damage, so a passkey-retrieval eval closes the
loop from method to an accuracy number (the ratio 0.0 row is the uncompressed
control):

```bash
python examples/run_needle.py --method snapkv --ratios 0,0.5,0.75
```

The boundary sweep asks where a score starts to beat a coin flip, holding every
method to the same final cache size, then the analysis turns its rows into cells and
figures without rerunning anything:

```bash
python examples/run_boundary_sweep.py --config experiments/smoke.json --out results/smoke
python examples/run_boundary_analysis.py --rows results/smoke.jsonl \
    --out results/analysis --plots results/plots
```

For real long-context methods, use KVPress through the same interface:

```python
from kvlab.backends import KVPressBackend

backend = KVPressBackend(model="meta-llama/Llama-3.1-8B-Instruct", press="expected_attention")
answer = backend.generate(context=long_document, question="...", ratio=0.5)
```

Standardized accuracy numbers come from KVPress's evaluation CLI over the benchmarks
in `kvlab.evals`; `evals.kvpress_eval_hint()` prints the invocation.

## Standard benchmarks

| Benchmark | Measures | Source |
|-----------|----------|--------|
| RULER | synthetic retrieval and tracking at controlled length | [NVIDIA/RULER](https://github.com/NVIDIA/RULER) |
| LongBench | real-world multi-doc QA, summarization, code | [THUDM/LongBench](https://github.com/THUDM/LongBench) |
| SCBench | full cache lifecycle including reuse and multi-turn | [arXiv:2412.10319](https://arxiv.org/abs/2412.10319) |
| IFEval | instruction following under compression | [arXiv:2311.07911](https://arxiv.org/abs/2311.07911) |
| InfiniteBench | tasks beyond 100K tokens | [OpenBMB/InfiniteBench](https://github.com/OpenBMB/InfiniteBench) |

## Status

Runnable now: the cost model, the registry and scenario map, and five reference
methods on a small CPU model — H2O (one-shot, plus online heavy-hitter decoding
via the step hook; not parity-tested against the authors' code), SnapKV (with the
paper's pooling step), OBCache in first-order form (value-aware scoring), a
CAKE-style layer allocator (supported by ragged-cache mask fitting, whose
correctness is pinned by logit-equivalence tests on a RoPE model), and KIVI as a
fake-quant reference (simulated error, analytical sizes) — plus the iso-ratio
perplexity benchmark and the passkey retrieval eval. The harness runs on
transformers 5. Wired and ready for a GPU host: the KVPress backend (including
KVzip, a strong published multi-turn baseline) and the benchmark catalog.

`accounting.py` splits retained KV into prompt and generated, which the iso-ratio
accounting cannot: methods now publish the indices they gathered with, and a
`CacheLedger` composes those across decode steps to recover each surviving
position's index in the original sequence. It refuses to guess when a method
reshapes the cache without publishing what it did.

Both accountings cover retained KV storage. Scorer side-state (H2O's accumulated
statistics, quantization scales) is not yet counted; it is negligible for the
current methods but must be counted for any method that keeps per-chunk metadata.

Trained eviction methods (LookaheadKV, ForesightKV) are catalogued with their
published numbers rather than reimplemented, and hybrid-memory and
alternative-attention methods are documented rather than reimplemented, since
they belong to serving engines (vLLM) or require retraining. `docs/TAXONOMY.md`
records the full map and the intended next steps; note that refinements within
a scoring family only separate at real context lengths, so toy-scale numbers
compare families, not papers.

## References

See [REFERENCES.md](REFERENCES.md) for the full list. Core sources: the survey
(arXiv:2603.20397), the pitfalls paper (arXiv:2510.00231), KVPress
(arXiv:2510.00636), and Random Attention (arXiv:2609.03430), which the boundary
experiment exists to test rather than to extend.

## License

MIT, see [LICENSE](LICENSE). KVPress is Apache-2.0 and is used as an optional
dependency, not vendored.
