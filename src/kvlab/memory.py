"""Analytical KV-cache cost model. No torch, no downloads."""

from __future__ import annotations

from dataclasses import dataclass

GB = 1024 ** 3
MB = 1024 ** 2

DTYPE_BYTES = {"fp32": 4.0, "fp16": 2.0, "bf16": 2.0, "int8": 1.0, "int4": 0.5, "int2": 0.25}


@dataclass(frozen=True)
class ModelConfig:
    name: str
    layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int

    @property
    def hidden_size(self) -> int:
        return self.n_heads * self.head_dim

    @property
    def gqa_ratio(self) -> float:
        return self.n_heads / self.n_kv_heads


MODELS = {
    "distilgpt2":  ModelConfig("distilgpt2",  6,  12, 12, 64),
    "gpt2":        ModelConfig("gpt2",         12, 12, 12, 64),
    "gpt2-medium": ModelConfig("gpt2-medium",  24, 16, 16, 64),
    "llama2-7b":   ModelConfig("llama2-7b",    32, 32, 32, 128),
    "llama2-13b":  ModelConfig("llama2-13b",   40, 40, 40, 128),
    "mistral-7b":  ModelConfig("mistral-7b",   32, 32, 8,  128),
    "llama3-8b":   ModelConfig("llama3-8b",    32, 32, 8,  128),
    "qwen2.5-14b": ModelConfig("qwen2.5-14b",  48, 40, 8,  128),
    "llama2-70b":  ModelConfig("llama2-70b",   80, 64, 8,  128),
}

# Raw VRAM in GB. Usable room is lower once weights and activations are resident.
GPUS = {"T4": 16, "L4": 24, "RTX 3090": 24, "RTX 4090": 24,
        "A100 40GB": 40, "A100 80GB": 80, "H100 80GB": 80}


def bytes_per_token(cfg: ModelConfig, dtype: str = "fp16", batch: int = 1) -> float:
    return 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim * DTYPE_BYTES[dtype] * batch


def cache_bytes(cfg: ModelConfig, context_len: int, dtype: str = "fp16", batch: int = 1) -> float:
    return bytes_per_token(cfg, dtype, batch) * context_len


def max_context_on_gpu(cfg: ModelConfig, gpu_gb: float, dtype: str = "fp16", batch: int = 1) -> int:
    return int((gpu_gb * GB) // bytes_per_token(cfg, dtype, batch))


def fits_on_gpu(cfg: ModelConfig, context_len: int, gpu_gb: float,
                dtype: str = "fp16", batch: int = 1) -> bool:
    return cache_bytes(cfg, context_len, dtype, batch) <= gpu_gb * GB


def human(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024 or unit == "TB":
            return f"{int(num_bytes)} B" if unit == "B" else f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024


def savings(baseline_bytes: float, optimized_bytes: float) -> str:
    return "inf" if optimized_bytes <= 0 else f"{baseline_bytes / optimized_bytes:.1f}x"
