# References

## Surveys and analysis
- KV Cache Optimization Strategies for Scalable and Efficient LLM Inference. arXiv:2603.20397.
- The Pitfalls of KV Cache Compression. Chen et al. ACL 2026. arXiv:2510.00231.
- SCBench: A KV Cache-Centric Analysis of Long-Context Methods. ICLR 2025. arXiv:2412.10319.
- Towards Efficient LLM Serving: A Survey on System-Aware KV Cache Optimization. ACL 2026.
  https://github.com/jjiantong/Awesome-KV-Cache-Optimization
- Rethinking KV Cache Eviction via a Unified Information-Theoretic Objective. arXiv:2604.25975.

## Libraries and tooling
- NVIDIA KVPress. https://github.com/NVIDIA/kvpress
- KVPress leaderboard. https://huggingface.co/spaces/nvidia/kvpress-leaderboard
- Expected Attention (KVPress paper). arXiv:2510.00636.
- vLLM (PagedAttention). https://github.com/vllm-project/vllm

## Benchmarks
- RULER. https://github.com/NVIDIA/RULER
- LongBench. https://github.com/THUDM/LongBench
- IFEval. arXiv:2311.07911.
- InfiniteBench. https://github.com/OpenBMB/InfiniteBench

## Methods, by family

### Eviction
- H2O: Heavy-Hitter Oracle. arXiv:2306.14048.
- SnapKV. arXiv:2404.14469.
- Ada-KV. arXiv:2407.11550.
- NACL. arXiv:2408.03675.
- InfiniPot. arXiv:2410.01518.
- HashEvict. arXiv:2412.16187.
- MorphKV. arXiv:2503.00979.
- RocketKV. arXiv:2502.14051.
- KVzip. arXiv:2505.23416.
- CAKE: Cascading and Adaptive KV Cache Eviction with Layer Preferences. arXiv:2503.12491.
- OBCache: Optimal Brain KV Cache Pruning. arXiv:2510.07651.
- SABlock: Semantic-Aware Eviction with Adaptive Block Size. arXiv:2510.22556.
- ForesightKV: Learning Long-Term Contribution for Reasoning Models. arXiv:2602.03203.
- LookaheadKV: Eviction by Glimpsing into the Future without Generation. arXiv:2603.10899.
- ReST-KV: Layer-wise Output Reconstruction with Spatial-Temporal Smoothing. arXiv:2605.08840.
- MomentKV: Closing the Directional Gap in KV Cache Eviction. arXiv:2606.01563.
- InfoKV: Information-Aware Compression for Long Reasoning. arXiv:2606.26875.
- AnchorKV: Safety-Aware Compression via Soft Penalty with a Refusal Anchor. arXiv:2606.17872.

### Compression
- KIVI. arXiv:2402.02750.
- KVQuant. arXiv:2401.18079.
- PALU. arXiv:2407.21118.
- MiniCache. arXiv:2405.14366.

### Hybrid memory
- PagedAttention. arXiv:2309.06180.
- InfiniGen. arXiv:2406.19707.
- LayerKV. arXiv:2410.00428.
- INF2. arXiv:2502.09989.
- KVPR. arXiv:2411.17089.

### Alternative attention
- Transformers are RNNs (linear attention). arXiv:2006.16236.
- Log-Linear Attention. arXiv:2506.04761.

### Hybrid approaches
- FlexGen. arXiv:2303.06865.
- ShadowKV. arXiv:2410.21465.
- Q-Hitter. MLSys 2024.
