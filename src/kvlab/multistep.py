"""A synthetic task whose answer lives in the model's own generated trace.

Passkey retrieval (kvlab.needle) probes one failure mode: can the model still read
something out of the *prompt* after the prompt cache has been compressed. It cannot
probe the other one, because the answer is emitted within the first few generated
tokens, before a budget over the generated cache has anything to bite on.

This task inverts that. Several quantities are stated in the prompt, and a chain of
sequential operations combines them. Each step consumes the running total the model
just produced, so by the last step the operand the model needs exists *only* in the
tokens it generated itself. Evicting the wrong generated KV therefore breaks the
answer in a way evicting prompt KV does not, which is the whole reason the task is
here.

Deterministic from a seed, scored automatically, no external service. Nothing about
hidden or private reasoning is used or required: the trace is ordinary visible
tokens, which is exactly the KV under study, and no chain-of-thought labels are
manufactured or graded. Only the final answer is scored.

**Redundancy is the experimental variable.** Each fact can be stated once or several
times in different words, and the filler shrinks to keep the prompt at its target
length. So for a fixed prompt length and a fixed answer, raising redundancy raises
the share of prompt KV carrying information the model has already seen elsewhere,
without changing the facts, the operations, or the answer. That is the controlled
form of "information density", and it is what separates a redundancy explanation of
the scoring boundary from a context-fraction explanation.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

PLACES = ["Ardun", "Belmar", "Ceyla", "Dorrin", "Eskvale", "Falter", "Ghent Row",
          "Harlow Bend", "Ilmar", "Jessup", "Kelvedon", "Lorne"]

UNIT = "crates"
FACILITY = "depot"

#: Restatements of one fact. Different sentences carrying the same value, so raising
#: redundancy adds evidence rather than repeating a string. Order is fixed, and a
#: level takes a prefix of it, so a fact stated twice is stated the same two ways in
#: every example at that level.
STATEMENTS = [
    "The {facility} at {place} holds {value} {unit}.",
    "A stock count at {place} came back at {value} {unit}.",
    "{place}'s {facility} is entered in the register as {value} {unit}.",
    "The quarterly audit confirmed {value} {unit} standing at {place}.",
    "Logistics lists {place} at {value} {unit} on hand.",
    "Receiving at {place} reconciled to {value} {unit}.",
]

REDUNDANCY_LEVELS = {"low": 1, "medium": 2, "high": 4}

#: Filler in the same register as the evidence, and deliberately free of numerals:
#: prose from another domain would make the facts findable by topic alone, while
#: numeric filler would make the scoring rule ambiguous.
FILLER = [
    "The loading bay reopens once the inspection paperwork clears.",
    "Drivers rotate through the northern route on alternating weeks.",
    "A new conveyor was fitted over the winter shutdown.",
    "The night shift signs for anything arriving after the office closes.",
    "Pallet wrap is ordered through the regional purchasing office.",
    "Forklift certification is refreshed at the start of every season.",
    "The yard floods when the culvert behind the fence backs up.",
    "Returns are staged separately from outbound stock.",
    "The weighbridge calibration certificate hangs in the gatehouse.",
    "Seasonal staff are briefed in the canteen before their first shift.",
    "A spare generator sits behind the maintenance shed.",
    "The dispatch board is updated by hand at the end of each run.",
    "Cold storage runs on its own circuit for exactly this reason.",
    "Barcode scanners are docked overnight in the supervisor's office.",
    "The access road is single track past the old weighbridge.",
    "Waste cardboard is baled on Thursdays if the press is working.",
    "The site manager walks the perimeter before locking up.",
    "Deliveries are refused without a booking reference.",
    "The canteen roof was patched after the storm last autumn.",
    "Shift handover notes are kept in the logbook by the door.",
]

OPERATIONS = ["add", "subtract"]
ANSWER_PATTERN = re.compile(r"-?\d+")


@dataclass(frozen=True)
class Chain:
    """The facts, the operations over them, and what they come to."""

    places: tuple[str, ...]
    values: tuple[int, ...]
    operations: tuple[str, ...]      # one per step after the first
    running: tuple[int, ...]         # the running total after every step
    answer: int


def build_chain(steps: int, rng: random.Random) -> Chain:
    """A chain whose answer is distinct from every value the model will have written
    down on the way there.

    Without that, a model that copies an operand or stops one step early scores by
    accident, and the task would be measuring truncation rather than arithmetic.
    Rejection is seeded, so the chain is still a deterministic function of the seed.
    """
    for _ in range(200):
        places = tuple(rng.sample(PLACES, steps))
        values = tuple(rng.randint(120, 980) for _ in range(steps))
        operations = tuple(rng.choice(OPERATIONS) for _ in range(steps - 1))
        running = [values[0]]
        for operation, value in zip(operations, values[1:]):
            running.append(running[-1] + value if operation == "add" else running[-1] - value)
        answer = running[-1]
        if answer > 0 and answer not in set(values) | set(running[:-1]):
            return Chain(places, values, operations, tuple(running), answer)
    raise RuntimeError(f"no {steps}-step chain with a distinct answer after 200 draws")


def evidence_sentences(chain: Chain, redundancy: str) -> list[str]:
    """Every restatement of every fact, ordered round by round rather than fact by
    fact, so a fact's restatements end up spread across the prompt instead of
    clustered. Otherwise redundancy would be confounded with locality: a fact stated
    four times in one paragraph survives or dies as a block."""
    repeats = REDUNDANCY_LEVELS[redundancy]
    return [STATEMENTS[index].format(facility=FACILITY, place=place, value=value, unit=UNIT)
            for index in range(repeats)
            for place, value in zip(chain.places, chain.values)]


def instructions(chain: Chain) -> str:
    lines = [f"Step 1: start with the {UNIT} at {chain.places[0]}."]
    for index, (operation, place) in enumerate(zip(chain.operations, chain.places[1:]), start=2):
        lines.append(f"Step {index}: {operation} the {UNIT} at {place}.")
    return ("\n\nWork through these steps in order, writing the running total after each one, "
            "then state the final number.\n" + "\n".join(lines))


PROMPT_TAIL = "\n\nWorking:\nStep 1:"


def filler_sentences(count: int, rng: random.Random) -> list[str]:
    """Dealt from a reshuffled deck rather than drawn with replacement, so the filler
    does not quietly become redundant evidence of its own and confound the variable
    the task exists to control."""
    dealt: list[str] = []
    while len(dealt) < count:
        dealt.extend(rng.sample(FILLER, len(FILLER)))
    return dealt[:count]


def compose(chain: Chain, redundancy: str, filler_count: int, rng: random.Random) -> str:
    """Evidence spread through filler, so no restatement of a fact sits next to
    another and depth is not confounded with redundancy."""
    evidence = evidence_sentences(chain, redundancy)
    body = filler_sentences(filler_count, rng)
    if evidence:
        stride = (len(body) + 1) / len(evidence)
        for offset, sentence in enumerate(evidence):
            body.insert(min(len(body), round(offset * stride) + offset), sentence)
    return " ".join(body) + instructions(chain)


def score_answer(generated: str, answer: int) -> float:
    """The last integer the model wrote, compared with the truth.

    The last one and not any one: searching the whole trace would pay a model for
    emitting numbers until one lands, which is the verbosity reward this task has to
    avoid. Nothing about the shape of the reasoning is scored, and a trace with no
    number in it scores zero rather than being given the benefit of the doubt.
    """
    numbers = ANSWER_PATTERN.findall(generated)
    return float(bool(numbers) and int(numbers[-1]) == answer)


def minimum_prompt_tokens(tokenizer, redundancy: str, steps: int, seed: int = 0) -> int:
    """The shortest prompt this workload can build at a redundancy level: all of the
    evidence, all of the instructions, and no filler at all.

    A cell whose target sits below this cannot be run without the prompt overflowing,
    which would make redundancy and prompt length move together. Configurations are
    sized against this rather than against a guess.
    """
    import random as _random

    chain = build_chain(steps, _random.Random(seed))
    context = compose(chain, redundancy, 0, _random.Random(seed))
    return len(tokenizer(context + PROMPT_TAIL).input_ids)
