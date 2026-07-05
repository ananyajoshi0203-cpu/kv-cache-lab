"""Dependency-free tour of the cost model. Reproduces the survey's memory figures,
shows how each lever changes them, and prints the method registry and scenario map.

    python examples/run_memory_demo.py [--verbose]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import memory, registry, scenarios
from kvlab.log import configure, get_logger

log = get_logger("demo")


def section(title):
    log.info("\n%s\n%s\n%s", "=" * 70, title, "=" * 70)


def headline_numbers():
    section("1. Cost model vs the survey: 7B at 128K")
    cfg = memory.MODELS["llama2-7b"]
    log.info("llama2-7b: %s per token (fp16)", memory.human(memory.bytes_per_token(cfg, "fp16")))
    log.info("128K tokens: %s", memory.human(memory.cache_bytes(cfg, 128 * 1024, "fp16")))
    for gpu, gb in memory.GPUS.items():
        log.info("  %-11s %3d GB fills at ~%dK tokens", gpu, gb, memory.max_context_on_gpu(cfg, gb) // 1024)


def levers():
    section("2. Each lever shrinks one term")
    cfg = memory.MODELS["llama2-7b"]
    base = memory.cache_bytes(cfg, 128 * 1024, "fp16")
    log.info("baseline 128K fp16: %s", memory.human(base))
    q4 = memory.cache_bytes(cfg, 128 * 1024, "int4")
    log.info("dtype  int4        : %-11s %s", memory.human(q4), memory.savings(base, q4))
    ev = memory.cache_bytes(cfg, 8 * 1024, "fp16")
    log.info("context evict to 8K: %-11s %s", memory.human(ev), memory.savings(base, ev))
    g = memory.cache_bytes(memory.MODELS["llama3-8b"], 128 * 1024, "fp16")
    log.info("heads  GQA (8 kv)  : %-11s %s", memory.human(g), memory.savings(base, g))


def gqa_surprise():
    section("3. GQA: 70B cheaper per token than 7B")
    for name in ("llama2-7b", "llama2-70b"):
        cfg = memory.MODELS[name]
        log.info("%-11s kv_heads=%-3d -> %s/token", name, cfg.n_kv_heads, memory.human(memory.bytes_per_token(cfg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true")
    configure(verbose=ap.parse_args().verbose)
    headline_numbers()
    levers()
    gqa_surprise()
    section("4. Method registry")
    log.info(registry.summary_table())
    section("5. Scenario map")
    for s in scenarios.SCENARIOS:
        log.info("%-42s prefer: %s", s.title, ", ".join(s.prefer))


if __name__ == "__main__":
    main()
