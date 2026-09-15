"""The trace-ablation diagnostic: does it remove what it says it removes.

This is the intervention that decides whether the multistep workload's generated axis
means anything, so its own correctness has to be pinned rather than assumed. The
prompt must survive untouched, the two arms must reach for opposite token classes,
both must end holding the same amount of KV so the comparison is at equal memory, and
the harness must notice when an arm ran out of the class it targets and started
removing the same positions as the other arm. Run: `pytest tests/`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab.ablation import NUMERIC, OTHER, TraceAblation, numeric_token_ids   # noqa: E402
from kvlab.decode import generate_stepwise                                    # noqa: E402
from kvlab.methods import FullCache                                           # noqa: E402

BATCH, HEADS, DIM = 1, 2, 2
NUMERIC_TOKENS = frozenset({7, 8, 9})
PROMPT = 5
# Alternating classes, so a class runs out only when the budget says it should.
TRACE = [7, 1, 8, 2, 9, 3, 7, 4, 8, 5, 9, 6]


class WordVocabulary:
    """A tokenizer stub exposing only get_vocab, which is all the scan reads."""

    def __init__(self, vocab):
        self._vocab = vocab

    def get_vocab(self):
        return self._vocab


def coded(positions, layers=1):
    values = torch.tensor(list(positions), dtype=torch.float32)
    key = values.view(1, 1, -1, 1).expand(BATCH, HEADS, len(values), DIM).clone()
    return tuple((key.clone(), key.clone()) for _ in range(layers))


def drive(method, trace=TRACE, prompt_length=PROMPT, layers=1):
    """Prefill, then one decode step per trace token, exactly as the real loop does."""
    cache = coded(range(prompt_length), layers)
    method.apply(cache, None)
    for offset, token in enumerate(trace):
        new = coded([prompt_length + offset], layers)
        cache = tuple((torch.cat([k, nk], dim=2), torch.cat([v, nv], dim=2))
                      for (k, v), (nk, nv) in zip(cache, new))
        method.observe(token)
        cache = method.step(cache, None)
    return cache


def kept(cache, layer=0, head=0):
    return [int(p) for p in cache[layer][0][0, head, :, 0].tolist()]


def arm(target, budget, seed=0):
    return TraceAblation(generation_budget=budget, target=target,
                         numeric_tokens=NUMERIC_TOKENS, seed=seed)


def test_the_vocabulary_scan_finds_the_tokens_with_digits():
    vocab = {"hello": 0, "12": 1, "Ġ3rd": 2, "world": 3, "x9": 4, "!": 5}
    assert numeric_token_ids(WordVocabulary(vocab)) == {1, 2, 4}


@pytest.mark.parametrize("target, budget", [
    (NUMERIC, 8), (OTHER, 8), (NUMERIC, 4), (OTHER, 4), (NUMERIC, 11), (OTHER, 11),
])
def test_an_arm_keeps_the_whole_prompt_and_holds_its_budget(target, budget):
    method = arm(target, budget)
    survivors = kept(drive(method))

    assert survivors[:PROMPT] == list(range(PROMPT))
    assert len(survivors) == PROMPT + budget
    assert survivors == sorted(survivors)
    assert method.kept_len(PROMPT + len(TRACE)) == PROMPT + budget


@pytest.mark.parametrize("budget", [8, 9, 10])
def test_the_two_arms_reach_for_opposite_classes_at_the_same_memory(budget):
    """Same budget, same retained KV, opposite halves of the trace: that is what makes
    a difference between them attributable to the class and not to the memory."""
    numeric_positions = {PROMPT + i for i, token in enumerate(TRACE) if token in NUMERIC_TOKENS}
    numeric_arm, other_arm = arm(NUMERIC, budget), arm(OTHER, budget)
    numeric_kept = set(kept(drive(numeric_arm))) - set(range(PROMPT))
    other_kept = set(kept(drive(other_arm))) - set(range(PROMPT))

    assert len(numeric_kept) == len(other_kept) == budget
    survived_numeric = len(numeric_kept & numeric_positions)
    other_survived_numeric = len(other_kept & numeric_positions)
    assert survived_numeric < other_survived_numeric, \
        "the numeric arm must end holding fewer numeric positions than the other arm"
    assert numeric_arm.removed_other == other_arm.removed_other == 0, \
        "neither class should have run out at this budget"


def test_an_arm_that_exhausts_its_class_records_the_spill():
    """The failure that would read as 'the trace does not matter': once the targeted
    class is gone, both arms remove the same positions."""
    numeric_count = sum(token in NUMERIC_TOKENS for token in TRACE)
    method = arm(NUMERIC, budget=1)
    drive(method)

    assert method.removed_targeted == numeric_count
    assert method.removed_other == len(TRACE) - 1 - numeric_count > 0


@pytest.mark.parametrize("seed, other_seed, same", [(3, 3, True), (3, 4, False)])
def test_the_eviction_seed_alone_decides_which_of_a_class_goes(seed, other_seed, same):
    assert (kept(drive(arm(OTHER, 8, seed))) == kept(drive(arm(OTHER, 8, other_seed)))) is same


def test_a_trace_out_of_step_with_the_cache_is_refused():
    """observe() and the cache have to stay aligned, or the classes describe positions
    that are no longer there and the arms silently target the wrong tokens."""
    method = arm(NUMERIC, 2)
    cache = coded(range(PROMPT))
    method.apply(cache, None)
    grown = coded(range(PROMPT + 4))
    with pytest.raises(RuntimeError, match="observe"):
        method.step(grown, None)


def test_the_decode_loop_reports_every_token_it_appends():
    """The hook has to fire once per cached generated token, or a diagnostic that
    depends on it is reading a stale trace."""
    seen = []

    class Watching(FullCache):
        def observe(self, token_id):
            seen.append(token_id)

    model = pytest.importorskip("kvlab.model")
    try:
        loaded, tokenizer, _ = model.load_model("hf-internal-testing/tiny-random-LlamaForCausalLM")
    except OSError as error:
        pytest.skip(f"cannot download the tiny test model: {error}")

    torch.manual_seed(0)
    ids = torch.randint(1, loaded.config.vocab_size, (1, 12))
    steps = 6
    generated = generate_stepwise(loaded, ids, Watching(), max_new_tokens=steps,
                                  eos_token_id=tokenizer.eos_token_id)
    # The final token's KV never enters the cache, so it is never observed.
    assert seen == generated[0, :len(seen)].tolist()
    assert len(seen) == steps - 1
