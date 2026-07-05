"""Deployment scenarios and the method families that fit each one. The survey's
premise is that no single method wins everywhere; this encodes that as data."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    prefer: list[str]
    avoid: list[str]
    why: str


SCENARIOS = [
    Scenario("long_context", "Long context, single request (>1M tokens)",
             ["rocketkv", "kvzip", "kivi", "kvquant"], ["pagedattention"],
             "Footprint dominates; eviction and compression cut it with little accuracy loss."),
    Scenario("high_throughput", "High-throughput datacenter serving",
             ["pagedattention", "oneiros", "shadowkv"], ["flexgen"],
             "Concurrency and sharing win, and paging is lossless. FlexGen suits only large batches."),
    Scenario("edge", "Edge or memory-limited device",
             ["infinipot", "tailorkv", "kivi"], ["oneiros", "pagedattention"],
             "Eviction and compression fit tiny VRAM; hybrid memory needs bandwidth the edge lacks."),
    Scenario("multi_turn", "Multi-turn conversation",
             ["kvzip", "shadowkv", "rocketkv"], ["h2o"],
             "Plain eviction permanently drops tokens a later turn may need."),
    Scenario("prefill_heavy", "Prefill-heavy, low time-to-first-token",
             ["nacl", "hashevict", "layerkv", "minicache"], [],
             "Single-shot eviction and layer-wise offload cut TTFT on long prompts."),
    Scenario("reasoning", "Accuracy-critical reasoning",
             ["pagedattention", "infinigen", "layerkv"], ["linear-attn", "loglinear-attn", "h2o"],
             "Lossless hybrid memory preserves accuracy; aggressive compression hurts reasoning."),
    Scenario("minimal_change", "Minimal model modification",
             ["snapkv", "ada-kv", "kivi"], ["linear-attn", "kimi-linear"],
             "Fine-tuning-free drop-ins only; new attention mechanisms require retraining."),
]


def recommend(scenario_key: str) -> Scenario:
    for s in SCENARIOS:
        if s.key == scenario_key:
            return s
    raise KeyError(f"unknown scenario '{scenario_key}'")


if __name__ == "__main__":
    from .log import configure
    log = configure()
    for s in SCENARIOS:
        log.info("\n%s", s.title)
        log.info("  prefer: %s", ", ".join(s.prefer))
        log.info("  avoid : %s", ", ".join(s.avoid) or "-")
        log.info("  %s", s.why)
