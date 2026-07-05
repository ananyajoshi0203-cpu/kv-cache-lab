"""Catalog of KV-cache optimization methods, grouped by family and by the term
of the cost model each one attacks. The `kvpress` field names the NVIDIA KVPress
press that implements a method where one exists."""

from __future__ import annotations

from dataclasses import dataclass

FAMILIES = ["Eviction", "Compression", "HybridMemory", "AltAttention", "Hybrid"]

LEVER_EXPLAINED = {
    "context":  "drop tokens -> smaller context_length",
    "dtype":    "fewer bits per number -> smaller dtype_bytes",
    "location": "move the cache off the GPU (size unchanged, lossless)",
    "formula":  "replace attention so there is no per-token cache (needs retraining)",
    "mixed":    "combine several levers",
}


@dataclass(frozen=True)
class Method:
    key: str
    name: str
    family: str
    lever: str
    status: str          # implemented | planned | conceptual
    summary: str
    paper: str = ""
    phase: str = ""
    kvpress: str = ""    # KVPress press class, if available


REGISTRY: list[Method] = [
    # Eviction
    Method("h2o", "H2O", "Eviction", "context", "implemented",
           "Retain recent tokens plus the highest accumulated-attention tokens; evict the rest.",
           "arXiv:2306.14048", "Decoding", "ObservedAttentionPress"),
    Method("snapkv", "SnapKV", "Eviction", "context", "implemented",
           "Vote for important prompt tokens using a recent observation window, then compress.",
           "arXiv:2404.14469", "After-prefill", "SnapKVPress"),
    Method("ada-kv", "Ada-KV", "Eviction", "context", "planned",
           "Head-wise budget: reallocate cache from sparse heads to dispersed heads. Wraps a scorer.",
           "arXiv:2407.11550", "After-prefill", "AdaKVPress"),
    Method("nacl", "NACL", "Eviction", "context", "planned",
           "Single-shot eviction combining proxy-token scoring with random eviction.",
           "arXiv:2408.03675", "Prefill"),
    Method("infinipot", "InfiniPot", "Eviction", "context", "planned",
           "On-device continual distillation of the context when the fixed cache nears overflow.",
           "arXiv:2410.01518", "Prefill"),
    Method("hashevict", "HashEvict", "Eviction", "context", "planned",
           "Pre-attention eviction using locality-sensitive hashing to approximate importance.",
           "arXiv:2412.16187", "Decoding"),
    Method("morphkv", "MorphKV", "Eviction", "context", "planned",
           "Constant-size cache scored from recent attention patterns, removing early-token bias.",
           "arXiv:2503.00979", "Decoding"),
    Method("rocketkv", "RocketKV", "Eviction", "context", "planned",
           "Coarse eviction followed by fine-grained sparse attention over paged tokens.",
           "arXiv:2502.14051", "Decoding"),
    Method("kvzip", "KVzip", "Eviction", "context", "planned",
           "Query-agnostic eviction that keeps tokens best able to reconstruct the context.",
           "arXiv:2505.23416", "After-prefill", "KVzipPress"),

    # Compression
    Method("kivi", "KIVI", "Compression", "dtype", "implemented",
           "2-bit KV quantization: per-channel for keys, per-token for values.",
           "arXiv:2402.02750", "Decoding"),
    Method("kvquant", "KVQuant", "Compression", "dtype", "planned",
           "Ultra-low-bit with pre-RoPE, non-uniform levels, and outlier preservation.",
           "arXiv:2401.18079", "Decoding"),
    Method("palu", "PALU", "Compression", "dtype", "planned",
           "Low-rank: cache a small SVD latent instead of the full key and value.",
           "arXiv:2407.21118", "Decoding"),
    Method("minicache", "MiniCache", "Compression", "dtype", "planned",
           "Merge near-identical adjacent deep layers via spherical interpolation.",
           "arXiv:2405.14366", "Decoding"),

    # Hybrid memory
    Method("pagedattention", "PagedAttention", "HybridMemory", "location", "conceptual",
           "Paged KV blocks and a block table, enabling sharing. Lossless. Native in vLLM.",
           "arXiv:2309.06180", "Serving"),
    Method("infinigen", "InfiniGen", "HybridMemory", "location", "conceptual",
           "Cache in CPU memory; predict and prefetch only the critical KV pairs to the GPU.",
           "arXiv:2406.19707", "Decoding"),
    Method("layerkv", "LayerKV", "HybridMemory", "location", "conceptual",
           "Layer-wise offload during prefill to cut time-to-first-token.",
           "arXiv:2410.00428", "Prefill"),
    Method("inf2", "INF2", "HybridMemory", "location", "conceptual",
           "Attention-near-storage: run attention on accelerators attached to storage.",
           "arXiv:2502.09989", "Serving"),
    Method("kvpr", "KVPR", "HybridMemory", "location", "conceptual",
           "Overlap partial KV recomputation on the GPU with CPU-to-GPU transfer.",
           "arXiv:2411.17089", "Decoding"),
    Method("oneiros", "Oneiros", "HybridMemory", "location", "conceptual",
           "Multi-tenant serving that pages idle model weights off-GPU to grow cache room.",
           "survey:2603.20397", "Serving"),
    Method("clo", "CLO", "HybridMemory", "location", "conceptual",
           "CPU-light offload that reuses loaded KV across similar decode steps.",
           "survey:2603.20397", "Decoding"),

    # Alternative attention
    Method("linear-attn", "Linear Attention", "AltAttention", "formula", "conceptual",
           "Kernel feature map gives O(N) time and O(1) decode memory; needs training.",
           "arXiv:2006.16236", "Architecture"),
    Method("loglinear-attn", "Log-Linear Attention", "AltAttention", "formula", "conceptual",
           "Fenwick-tree summaries give O(N log N) time and O(log N) memory.",
           "arXiv:2506.04761", "Architecture"),
    Method("local-linear-attn", "Local Linear Attention", "AltAttention", "formula", "conceptual",
           "Attention as local linear regression, interpolating linear and softmax.",
           "survey:2603.20397", "Architecture"),
    Method("kimi-linear", "Kimi Linear", "AltAttention", "formula", "conceptual",
           "Gated delta attention hybridized 3:1 with softmax; large KV reduction at long context.",
           "arXiv:2510.xxxxx", "Architecture"),

    # Hybrid
    Method("flexgen", "FlexGen", "Hybrid", "mixed", "conceptual",
           "Compression plus GPU/CPU/disk offload as a cost LP, for high-throughput batch.",
           "arXiv:2303.06865", "Serving"),
    Method("q-hitter", "Q-Hitter", "Hybrid", "mixed", "planned",
           "Rank tokens by attention importance and quantization robustness; keep top-K quantized.",
           "MLSys 2024", "Decoding"),
    Method("shadowkv", "ShadowKV", "Hybrid", "mixed", "planned",
           "Low-rank keys on GPU, values offloaded to CPU, sparse landmark retrieval at decode.",
           "arXiv:2410.21465", "Decoding"),
    Method("tailorkv", "TailorKV", "Hybrid", "mixed", "planned",
           "Quantize shallow layers on GPU; offload deep layers to CPU with dynamic top-k fetch.",
           "arXiv:2505.xxxxx", "Decoding"),
]


def by_family(family: str) -> list[Method]:
    return [m for m in REGISTRY if m.family == family]


def by_status(status: str) -> list[Method]:
    return [m for m in REGISTRY if m.status == status]


def get(key: str) -> Method:
    for m in REGISTRY:
        if m.key == key:
            return m
    raise KeyError(f"unknown method '{key}'")


def summary_table() -> str:
    tag = {"implemented": "[x]", "planned": "[ ]", "conceptual": "[~]"}
    lines = []
    for fam in FAMILIES:
        methods = by_family(fam)
        runnable = sum(1 for m in methods if m.status == "implemented")
        lines.append(f"\n{fam}  ({len(methods)} methods, {runnable} runnable)")
        lines.append("-" * 64)
        for m in methods:
            kp = f"  kvpress:{m.kvpress}" if m.kvpress else ""
            lines.append(f"  {tag[m.status]} {m.name:<24} lever={m.lever:<9} {m.paper}{kp}")
    lines.append("\n[x] runnable here   [ ] catalogued/stub   [~] needs GPU, hardware, or retraining")
    return "\n".join(lines)


if __name__ == "__main__":
    from .log import configure
    log = configure()
    log.info("registry: %d methods", len(REGISTRY))
    log.info(summary_table())
