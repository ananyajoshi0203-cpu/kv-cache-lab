"""Passkey retrieval: the minimal closed loop from a compression method to an
accuracy number, runnable on CPU. A passkey sentence is buried at a controlled
depth in filler text and the model must reproduce the passkey after the cache is
compressed. This complements perplexity, which barely moves when eviction drops
a fact the model then cannot retrieve.

Every ratio sees the identical examples (same seed), so rows are paired and the
ratio 0.0 row is the uncompressed control: small models bound the achievable
ceiling, and each method is read as a delta from that control, not in absolute
terms. Standardized long-context numbers still come from the KVPress evaluation
CLI (see kvlab.evals); this is the in-repo smoke test of the same design.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)

FILLER_SENTENCES = [
    "The morning train to the coast was delayed by fog again.",
    "A good sourdough loaf needs at least twelve hours of slow fermentation.",
    "The observatory on the ridge photographs variable stars every clear night.",
    "Local farmers rotate barley and clover to keep the soil in good heart.",
    "The city library extended its opening hours for the exam season.",
    "Migrating cranes rest on the flooded meadows south of the river.",
    "The workshop repairs violins that are older than the town hall.",
    "New tram tickets can be validated with a tap at either door.",
    "The lighthouse keeper's log records every storm since the twenties.",
    "A patch of wild thyme grows between the stones of the old wall.",
    "The chess club meets above the bakery on the first Tuesday of the month.",
    "Volunteers repainted the ferry landing before the summer crowds arrived.",
]

NEEDLE = "The secret passkey is {passkey}."
QUESTION = " The secret passkey is"


@dataclass(frozen=True)
class NeedleRow:
    ratio: float
    correct: int
    total: int

    @property
    def accuracy(self) -> float:
        return self.correct / self.total


def make_example(passkey: int, filler_sentences: int, depth: float,
                 rng: random.Random) -> tuple[str, str]:
    """Context with the passkey sentence inserted at `depth` (0 start, 1 end),
    and the question that asks for it back."""
    filler = [rng.choice(FILLER_SENTENCES) for _ in range(filler_sentences)]
    filler.insert(round(depth * len(filler)), NEEDLE.format(passkey=passkey))
    return " ".join(filler), QUESTION


def run_needle(backend, ratios: Sequence[float], n_examples: int = 8,
               filler_sentences: int = 30, seed: int = 0) -> list[NeedleRow]:
    """Retrieval accuracy per compression ratio. Depths sweep 0..1 across the
    examples so shallow and deep placements are both represented."""
    rows = []
    for ratio in ratios:
        rng = random.Random(seed)
        correct = 0
        for i in range(n_examples):
            passkey = rng.randint(10000, 99999)
            depth = i / max(1, n_examples - 1)
            context, question = make_example(passkey, filler_sentences, depth, rng)
            answer = backend.generate(context, question, ratio=ratio, max_new_tokens=8)
            hit = str(passkey) in answer
            correct += hit
            logger.debug("ratio=%.2f depth=%.2f passkey=%s hit=%s", ratio, depth, passkey, hit)
        rows.append(NeedleRow(ratio, correct, n_examples))
        logger.info("ratio=%.2f accuracy=%d/%d", ratio, correct, n_examples)
    return rows


def format_table(rows: Iterable[NeedleRow]) -> str:
    header = f"{'ratio':>7}{'correct':>10}{'accuracy':>10}"
    lines = [header, "-" * len(header)]
    for r in rows:
        lines.append(f"{r.ratio:>7.2f}{r.correct:>7}/{r.total:<2}{r.accuracy:>10.2f}")
    return "\n".join(lines)
