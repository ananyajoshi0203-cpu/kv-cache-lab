"""Provenance accounting: does the ledger really know where surviving KV came from.

The ledger is driven by a scripted method that keeps exactly the cache indices it
is told to and publishes them, so every expected split is fixed by construction
before the ledger runs. Index composition across steps is the part worth pinning:
after the first eviction a method's indices address the *compacted* cache, so a
ledger that forgot to compose them would still produce plausible-looking totals
and the wrong prompt/generated split. Run: `pytest tests/`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab.accounting import CacheLedger, KVAccount                   # noqa: E402
from kvlab.memory import MODELS, bytes_per_token                      # noqa: E402
from kvlab.methods import KVMethod, _gather_tokens                    # noqa: E402

CFG = MODELS["distilgpt2"]
BATCH, DIM = 1, 4


class Scripted(KVMethod):
    """Keeps exactly the cache indices the script names, and publishes them."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def apply(self, past_key_values, attentions):
        self.last_indices = None
        return past_key_values

    def step(self, past_key_values, attentions):
        keep = self.script[self.calls]
        self.calls += 1
        if keep is None:
            self.last_indices = None
            return past_key_values
        out, indices = [], []
        for key, value in past_key_values:
            idx = torch.tensor(keep).view(1, 1, -1).expand(key.shape[0], key.shape[1], len(keep))
            indices.append(idx)
            out.append((_gather_tokens(key, idx), _gather_tokens(value, idx)))
        self.last_indices = tuple(indices)
        return tuple(out)


class Silent(Scripted):
    """Gathers without publishing: the mistake the ledger has to catch."""

    def step(self, past_key_values, attentions):
        out = super().step(past_key_values, attentions)
        self.last_indices = None
        return out


def cache_of(length, layers=CFG.layers, heads=CFG.n_kv_heads):
    key = torch.zeros(BATCH, heads, length, DIM)
    return tuple((key.clone(), key.clone()) for _ in range(layers))


def grow(cache):
    new = torch.zeros(BATCH, cache[0][0].shape[1], 1, DIM)
    return tuple((torch.cat([k, new], dim=2), torch.cat([v, new], dim=2)) for k, v in cache)


def drive(method, prompt_length, steps, layers=CFG.layers, heads=CFG.n_kv_heads):
    cache = cache_of(prompt_length, layers, heads)
    ledger = CacheLedger()
    ledger.start(cache)
    cache = method.apply(cache, None)
    ledger.compact(method, cache)
    for _ in range(steps):
        cache = grow(cache)
        ledger.append()
        cache = method.step(cache, None)
        ledger.compact(method, cache)
    return ledger


@pytest.mark.parametrize("prompt_length, script, survivors, prompt_tokens, generated_tokens", [
    # One step. Cache is prompt 0..4 plus generated 5; keeping [0,1,2,5] keeps
    # three prompt positions and one generated one.
    (5, [[0, 1, 2, 5]], [0, 1, 2, 5], 3, 1),
    # Two steps, and the second script addresses the *compacted* cache: after the
    # first step the cache holds original positions [0,1,2,5], the second appends
    # 6, so keeping indices [0,3,4] keeps original positions [0,5,6].
    (5, [[0, 1, 2, 5], [0, 3, 4]], [0, 5, 6], 1, 2),
    # A method that drops nothing keeps the whole sequence.
    (5, [None, None, None], [0, 1, 2, 3, 4, 5, 6, 7], 5, 3),
    # Everything generated is evicted every step.
    (4, [[0, 1, 2, 3], [0, 1, 2, 3]], [0, 1, 2, 3], 4, 0),
])
def test_ledger_splits_prompt_from_generated(prompt_length, script, survivors,
                                             prompt_tokens, generated_tokens):
    ledger = drive(Scripted(script), prompt_length, len(script))
    account = ledger.account(CFG)
    assert ledger.retained_positions() == survivors
    slots = CFG.layers * CFG.n_kv_heads
    assert account.prompt_slots == prompt_tokens * slots
    assert account.generated_slots == generated_tokens * slots
    assert account.prompt_retained == prompt_tokens
    assert account.generated_retained == generated_tokens
    assert account.total_retained == prompt_tokens + generated_tokens
    assert account.prompt_length == prompt_length
    assert account.generated_length == len(script)


@pytest.mark.parametrize("dtype", ["fp16", "fp32", "int2"])
def test_bytes_and_ratio_follow_the_cost_model(dtype):
    """Cross-checked against kvlab.memory, which derives per-token bytes from the
    model config by a different expression than accounting derives slot bytes."""
    prompt_length, script = 6, [[0, 1, 2, 3, 6], [0, 1, 4]]
    account = drive(Scripted(script), prompt_length, len(script)).account(CFG, dtype=dtype)
    per_token = bytes_per_token(CFG, dtype)

    assert account.total_kv_bytes == pytest.approx(per_token * account.total_retained)
    assert account.prompt_kv_bytes == pytest.approx(per_token * account.prompt_retained)
    assert account.generated_kv_bytes == pytest.approx(per_token * account.generated_retained)
    assert account.prompt_kv_bytes + account.generated_kv_bytes == pytest.approx(account.total_kv_bytes)
    assert account.uncompressed_kv_bytes == pytest.approx(
        per_token * (prompt_length + len(script)))
    assert account.compression_ratio == pytest.approx(
        1 - account.total_kv_bytes / account.uncompressed_kv_bytes)


def test_ledger_refuses_a_method_that_hides_its_gather():
    """Silent eviction is the one failure that would corrupt every split quietly,
    so it has to raise rather than be inferred."""
    with pytest.raises(RuntimeError, match="must set last_indices"):
        drive(Silent([[0, 1, 2]]), prompt_length=5, steps=1)


def test_account_refuses_a_config_for_a_different_model():
    ledger = drive(Scripted([None]), prompt_length=4, steps=1, layers=2, heads=3)
    with pytest.raises(ValueError, match="different model"):
        ledger.account(CFG)


@pytest.mark.parametrize("field, value", [
    ("prompt_slots", -1),
    ("prompt_slots", 2 * 3 * 4 + 1),     # layers * heads * prompt_length, plus one
    ("generated_slots", 2 * 3 * 5 + 1),  # layers * heads * generated_length, plus one
])
def test_an_impossible_account_is_rejected(field, value):
    base = dict(model="stub", layers=2, kv_heads=3, head_dim=8, dtype="fp16",
                prompt_length=4, generated_length=5, prompt_slots=10, generated_slots=10)
    with pytest.raises(ValueError, match="impossible"):
        KVAccount(**{**base, field: value})
