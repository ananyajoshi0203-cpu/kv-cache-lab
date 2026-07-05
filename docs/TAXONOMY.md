# Taxonomy: every method mapped to its lever

This is the deep-dive companion to the README. It maps all ~28 methods from the
survey onto the formula they attack, so you always know *why* a method exists.

```
bytes_per_token = 2 × layers × kv_heads × head_dim × dtype_bytes
total_cache     = bytes_per_token × context_length × batch
```

---

## 1. Eviction — lever: `context_length`

Drop tokens you probably won't need again. Cheapest to run; the risk is throwing
away something that mattered.

| Method | Idea in one line | Phase | Paper |
|--------|------------------|-------|-------|
| **H2O** ✅ | Keep recent tokens + "heavy hitters" (highest accumulated attention). | Decoding | 2306.14048 |
| **SnapKV** ✅ | An observation window votes for important prompt tokens; pool + keep. | After-prefill | 2404.14469 |
| Ada-KV | Give attention-dispersed heads more budget, sparse heads less. Plugs into SnapKV. | After-prefill | 2407.11550 |
| NACL | Single-shot eviction: proxy-token scoring + random eviction, done once. | Prefill | 2408.03675 |
| InfiniPot | On-device: distill essentials whenever the fixed cache nears overflow. | Prefill | 2410.01518 |
| HashEvict | Locality-sensitive hashing approximates importance *before* attention. | Decoding | 2412.16187 |
| MorphKV | Constant-size cache; scores old tokens from recent patterns, no early bias. | Decoding | 2503.00979 |
| RocketKV | Coarse SnapKV eviction, then fine-grained sparse attention over pages. | Decoding | 2502.14051 |
| KVzip | Query-agnostic: keep tokens that best let the model *reconstruct* the context. | After-prefill | 2505.23416 |

**Watch out (pitfalls paper):** plain eviction permanently deletes tokens, so
it's a poor fit for multi-turn chat — a later turn may need what you dropped.

## 2. Compression — lever: `dtype_bytes`

Keep every token, but store each number in fewer bits (or a smaller subspace).

| Method | Idea in one line | Paper |
|--------|------------------|-------|
| **KIVI** ✅ | Tuning-free 2-bit: per-channel keys (outlier channels), per-token values. | 2402.02750 |
| KVQuant | Ultra-low-bit: pre-RoPE + non-uniform + outlier preservation, up to 10M ctx. | 2401.18079 |
| PALU | Low-rank: SVD-decompose the KV projections, cache the small latent. | 2407.21118 |
| MiniCache | Merge near-identical adjacent deep layers via spherical interpolation. | 2405.14366 |

## 3. Hybrid memory — lever: *where the cache lives*

Don't shrink the cache — **move** it. Size on the accelerator drops; accuracy is
untouched (lossless). Cost shifts to bandwidth and system complexity.

| Method | Idea in one line | Paper |
|--------|------------------|-------|
| PagedAttention (vLLM) | OS-style paging: non-contiguous KV blocks + block table + sharing. | 2309.06180 |
| InfiniGen | Cache in CPU RAM; predict + prefetch only the critical KV to GPU. | 2406.19707 |
| LayerKV | Offload layers during prefill so offload ≤ prefill time; big TTFT wins. | 2410.00428 |
| INF2 | Attention-near-storage: compute attention on accelerators by the storage. | 2502.09989 |
| KVPR | Overlap partial KV recompute on GPU with CPU→GPU transfer. | 2411.17089 |
| Oneiros | Multi-tenant: page idle model weights off-GPU to grow cache room. | survey |
| CLO | CPU-light offload; reuse loaded KV across similar decode steps. | survey |

## 4. Alternative attention — lever: *the formula itself*

Redesign attention so there is no linear-growing per-token cache. Best asymptotic
scaling, but needs a retrained model and tends to lag on hard reasoning.

| Method | Complexity | Paper |
|--------|-----------|-------|
| Linear Attention | O(N) time, O(1) decode memory | 2006.16236 |
| Log-Linear Attention | O(N log N) time, O(log N) memory | 2506.04761 |
| Local Linear Attention | interpolates linear ↔ softmax | survey |
| Kimi Linear | delta attention + gating, 3:1 hybrid with softmax; ~75% KV cut | 2510.xxxxx |

## 5. Hybrid approaches — lever: `mixed`

Combine eviction + compression + offloading.

| Method | Idea in one line | Paper |
|--------|------------------|-------|
| FlexGen | Compression + GPU/CPU/disk offload as a cost LP; high-throughput batch. | 2303.06865 |
| Q-Hitter | Sparse + quantized: rank by attention importance AND quant robustness. | MLSys'24 |
| ShadowKV | Low-rank keys on GPU, values on CPU, sparse landmark retrieval. | 2410.21465 |
| TailorKV | Quantize shallow layers on GPU; offload deep layers to CPU. | 2505.xxxxx |

---

## Where this goes next (roadmap)

The natural progression of this lab, roughly in order of effort:

1. **More eviction/compression methods** — Ada-KV, MorphKV, KVQuant, PALU. All
   fit the CPU harness today. Great first contributions.
2. **A better quality metric than perplexity** — a needle-in-a-haystack retrieval
   task and a **multi-instruction eval (IFEval-style)**. This is where the
   *pitfalls* paper lives: show that a method with "fine" perplexity still drops
   specific instructions. This is the experiment worth writing up.
3. **Real long-context on GPU** — swap distilgpt2 for Llama-3-8B / Qwen2.5-14B,
   measure at 32K–128K, and reproduce headline memory/throughput claims.
4. **A hybrid-memory demo** — even a toy PagedAttention (paged block table over
   the cache) makes the "lossless, moves not shrinks" idea concrete.

The through-line for a workshop paper: *benchmarks say compression is free;
build the eval that shows when it isn't, across this whole taxonomy.*
