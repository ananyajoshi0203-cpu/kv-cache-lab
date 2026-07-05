"""Standard long-context benchmarks used to compare KV-cache methods. The point of
this module is to keep the harness on established, citable evaluations rather than a
homemade test. Canonical accuracy numbers come from the KVPress evaluation CLI, which
runs these datasets at fixed compression ratios; this module catalogs them, loads the
ones with public HuggingFace datasets, and records what each one measures."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Benchmark:
    key: str
    name: str
    measures: str
    hf_dataset: str      # HuggingFace dataset id, or "" if it needs a generator
    source: str


BENCHMARKS = [
    Benchmark("ruler", "RULER",
              "Synthetic long-context: needle retrieval, variable tracking, aggregation, at controlled lengths.",
              "", "https://github.com/NVIDIA/RULER"),
    Benchmark("longbench", "LongBench",
              "Real-world long-context: multi-doc QA, summarization, few-shot, code.",
              "THUDM/LongBench", "https://github.com/THUDM/LongBench"),
    Benchmark("longbench_v2", "LongBench v2",
              "Harder, longer real-world tasks with multiple-choice scoring.",
              "THUDM/LongBench-v2", "https://github.com/THUDM/LongBench"),
    Benchmark("scbench", "SCBench",
              "KV-cache-centric: full cache lifecycle including reuse and multi-turn shared context.",
              "microsoft/SCBench", "https://arxiv.org/abs/2412.10319"),
    Benchmark("ifeval", "IFEval",
              "Instruction following; exposes instructions silently dropped under compression.",
              "google/IFEval", "https://arxiv.org/abs/2311.07911"),
    Benchmark("infinitebench", "InfiniteBench",
              "Tasks beyond 100K tokens.",
              "xinrongzhang2022/InfiniteBench", "https://github.com/OpenBMB/InfiniteBench"),
]


def get(key: str) -> Benchmark:
    for b in BENCHMARKS:
        if b.key == key:
            return b
    raise KeyError(f"unknown benchmark '{key}'")


def load(key: str, split: str = "test", limit: int | None = None):
    """Load a benchmark's examples via HuggingFace datasets. Raises for benchmarks
    that ship a generator rather than a static dataset (RULER)."""
    b = get(key)
    if not b.hf_dataset:
        raise NotImplementedError(
            f"{b.name} is generated, not a static dataset. Build it with the tools at {b.source}.")
    from datasets import load_dataset
    ds = load_dataset(b.hf_dataset, split=split)
    return ds.select(range(min(limit, len(ds)))) if limit else ds


def kvpress_eval_hint(press: str, dataset: str = "ruler", model: str = "meta-llama/Llama-3.1-8B-Instruct",
                      ratio: float = 0.5) -> str:
    """Pointer to the canonical, standardized evaluation. KVPress ships a CLI under
    its evaluation/ directory; flags evolve, so check evaluation/README before running."""
    return (f"# canonical numbers: run KVPress's evaluation CLI\n"
            f"#   git clone https://github.com/NVIDIA/kvpress && cd kvpress\n"
            f"#   see evaluation/README.md for exact flags, then run roughly:\n"
            f"#   python evaluation/evaluate.py --model {model} --dataset {dataset} "
            f"--press_name {press} --compression_ratio {ratio}")
