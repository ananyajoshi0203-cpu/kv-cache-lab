"""Tests for the torch-free memory core. Run: `pytest tests/` (or just python this file)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import memory   # noqa: E402


def test_per_token_half_mb_for_7b_fp16():
    cfg = memory.MODELS["llama2-7b"]
    per = memory.bytes_per_token(cfg, "fp16")
    assert abs(per - 0.5 * memory.MB) < 1  # exactly 524288 bytes


def test_128k_is_64gb():
    cfg = memory.MODELS["llama2-7b"]
    total = memory.cache_bytes(cfg, 128 * 1024, "fp16")
    assert abs(total - 64 * memory.GB) < 1


def test_4090_fills_near_48k():
    cfg = memory.MODELS["llama2-7b"]
    mx = memory.max_context_on_gpu(cfg, 24, "fp16")
    assert 48 * 1024 <= mx <= 49 * 1024   # ~49,152 tokens


def test_int4_is_4x_smaller_than_fp16():
    cfg = memory.MODELS["llama2-7b"]
    fp16 = memory.cache_bytes(cfg, 1000, "fp16")
    int4 = memory.cache_bytes(cfg, 1000, "int4")
    assert abs(fp16 / int4 - 4.0) < 1e-6


def test_gqa_70b_cheaper_per_token_than_7b():
    per_7b = memory.bytes_per_token(memory.MODELS["llama2-7b"], "fp16")
    per_70b = memory.bytes_per_token(memory.MODELS["llama2-70b"], "fp16")
    assert per_70b < per_7b   # the GQA surprise


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all memory tests passed")
