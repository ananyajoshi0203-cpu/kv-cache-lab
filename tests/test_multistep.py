"""The multistep workload: is it actually measuring what it claims to.

A synthetic task is only worth running if its answer cannot be reached by accident,
its redundancy setting changes redundancy and nothing else, and the same seed gives
the same question every time. Those three are what these tests pin, because a weak
task produces a clean-looking result that means nothing. Run: `pytest tests/`.
"""

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import multistep                                          # noqa: E402
from kvlab.boundary import WORKLOADS, Example, prompt_band           # noqa: E402

STEPS = 4
LEVELS = tuple(multistep.REDUNDANCY_LEVELS)


class WordTokenizer:
    """Whitespace tokenisation. The workload only ever asks a tokenizer for a
    length, so word counts stand in for the real thing."""

    def __call__(self, text, return_tensors=None):
        return type("Encoded", (), {"input_ids": text.split()})()


@pytest.mark.parametrize("seed", range(6))
def test_the_answer_cannot_be_reached_by_copying_or_stopping_early(seed):
    """A chain whose answer equals an operand, or a total the model passes through on
    the way, would score a model that copies a number or truncates its own trace. The
    task would then be measuring truncation, not arithmetic."""
    chain = multistep.build_chain(STEPS, random.Random(seed))

    assert len(chain.values) == STEPS
    assert len(chain.operations) == STEPS - 1
    assert chain.answer > 0
    assert chain.answer not in chain.values
    assert chain.answer not in chain.running[:-1]
    assert len(set(chain.places)) == STEPS

    replayed = [chain.values[0]]
    for operation, value in zip(chain.operations, chain.values[1:]):
        replayed.append(replayed[-1] + value if operation == "add" else replayed[-1] - value)
    assert tuple(replayed) == chain.running


class ScriptedRandom:
    """Hands build_chain the chains it is told to, in order, so the rejection path is
    exercised rather than waited for: a colliding chain is rare enough that sampling
    seeds tests nothing."""

    def __init__(self, chains):
        self.chains = list(chains)
        self.drawn = -1
        self.values: list[int] = []
        self.operations: list[str] = []

    def sample(self, population, k):
        self.drawn = min(self.drawn + 1, len(self.chains) - 1)
        places, values, operations = self.chains[self.drawn]
        self.values, self.operations = list(values), list(operations)
        return list(places)

    def randint(self, low, high):
        return self.values.pop(0)

    def choice(self, sequence):
        return self.operations.pop(0)


PLACES_4 = tuple(multistep.PLACES[:STEPS])
# answer 100, which is also an operand: a model that copies an operand would score.
COLLIDING = (PLACES_4, (200, 100, 100, 100), ("add", "subtract", "subtract"))
# answer 550, distinct from every operand and every running total before it.
CLEAN = (PLACES_4, (200, 100, 300, 50), ("add", "add", "subtract"))


def test_a_colliding_chain_is_rejected_and_redrawn():
    assert multistep.build_chain(STEPS, ScriptedRandom([COLLIDING, CLEAN])).answer == 550
    with pytest.raises(RuntimeError, match="distinct answer"):
        multistep.build_chain(STEPS, ScriptedRandom([COLLIDING]))


@pytest.mark.parametrize("seed", range(4))
def test_redundancy_restates_the_evidence_and_changes_nothing_else(seed):
    """The control the whole redundancy hypothesis rests on: same facts, same
    operations, same answer, only how often each fact is stated."""
    chain = multistep.build_chain(STEPS, random.Random(seed))
    by_level = {level: multistep.evidence_sentences(chain, level) for level in LEVELS}

    counts = [len(by_level[level]) for level in LEVELS]
    assert counts == sorted(counts) and len(set(counts)) == len(LEVELS)
    for level in LEVELS:
        sentences = by_level[level]
        assert len(sentences) == multistep.REDUNDANCY_LEVELS[level] * STEPS
        assert len(set(sentences)) == len(sentences), "restatements must not be one string repeated"
        for place, value in zip(chain.places, chain.values):
            mentions = [s for s in sentences if place in s]
            assert len(mentions) == multistep.REDUNDANCY_LEVELS[level]
            assert all(str(value) in s for s in mentions)
    # Every fact is restated once before any fact is restated twice, so a fact's
    # copies land at different depths instead of clustering.
    high = by_level[max(LEVELS, key=lambda level: multistep.REDUNDANCY_LEVELS[level])]
    assert {place in sentence for place, sentence in zip(chain.places, high[:STEPS])} == {True}


@pytest.mark.parametrize("level", LEVELS)
def test_the_same_seed_asks_the_same_question(level):
    def built(seed):
        return WORKLOADS["multistep"].build(WordTokenizer(), 200, 0, 1, seed, level)

    assert built(3) == built(3)
    assert built(3).answer != built(4).answer


@pytest.mark.parametrize("level", LEVELS)
def test_prompt_length_is_held_while_redundancy_moves(level):
    """Filler shrinks as evidence grows, so redundancy changes the share of the prompt
    carrying information the model has already seen, not the size of the prompt."""
    tokenizer = WordTokenizer()
    target = 400
    example = WORKLOADS["multistep"].build(tokenizer, target, 0, 1, 0, level)

    assert isinstance(example, Example)
    assert abs(len(tokenizer(example.context + example.question).input_ids) - target) \
        <= prompt_band(target)
    assert example.facts == STEPS
    assert example.statements == multistep.REDUNDANCY_LEVELS[level] * STEPS
    assert multistep.minimum_prompt_tokens(tokenizer, level, STEPS) < target


@pytest.mark.parametrize("trace, hits", [
    ("Step 1: 408. Step 2: 145. Step 3: 1038. Step 4: {answer}", True),
    ("Step 1: 408. The final number is {answer}, I think.", True),
    # The answer appears, but the model kept writing: only the last number counts, so
    # emitting candidates until one lands buys nothing.
    ("{answer} or maybe 17", False),
    ("Step 1: 408. Step 2: 145.", False),
    ("no numbers at all here", False),
    ("", False),
])
def test_only_the_final_number_is_scored(trace, hits):
    chain = multistep.build_chain(STEPS, random.Random(0))
    example = Example("", "", str(chain.answer))
    assert WORKLOADS["multistep"].score(
        example, trace.format(answer=chain.answer)) == float(hits)
