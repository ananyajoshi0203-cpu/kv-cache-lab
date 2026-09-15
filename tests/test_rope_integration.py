"""Logit-level correctness of the harness on a RoPE model (tiny random Llama).

These tests answer a question the unit tests cannot: after non-contiguous
eviction, is compressed inference still positionally exact? HuggingFace caches
keys post-RoPE, so a gathered key keeps its original rotary phase, and the
decode/scoring paths pass explicit absolute position_ids. Each test pins one
link of that argument against an oracle computed a different way:

- harness scoring with an uncompressed cache == one direct full forward;
- the step-wise decode loop == model.generate, token for token;
- an evicted cache == the full cache with the same tokens hidden by a 4D mask
  (exact equality is only possible if gathered keys keep their positions);
- ragged per-layer budgets run end-to-end through mask fitting;
- a prompt-protected random cache decodes at prompt + generation budget,
  which is the size its budget actually names;
- every method's provenance accounting balances against the cache it really held.

Requires downloading a few-MB test model; skipped when that fails.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab.accounting import CacheLedger                            # noqa: E402
from kvlab.benchmark import _score                                  # noqa: E402
from kvlab.decode import generate_stepwise                          # noqa: E402
from kvlab.methods import (                                         # noqa: E402
    CAKE, FullCache, PromptProtectedRandom, _gather_tokens, build,
)
from kvlab.model import cache_to_tuples, load_model, per_layer_mask_fit, tuples_to_cache  # noqa: E402

TINY_LLAMA = "hf-internal-testing/tiny-random-LlamaForCausalLM"
SEQ, CONT = 40, 12


@pytest.fixture(scope="module")
def llama():
    try:
        model, tok, cfg = load_model(TINY_LLAMA)
    except OSError as err:
        pytest.skip(f"cannot download {TINY_LLAMA}: {err}")
    torch.manual_seed(0)
    vocab = model.config.vocab_size
    ids = torch.randint(1, vocab, (1, SEQ))
    return model, tok, cfg, ids


def _prefill(model, ids):
    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=True)
    return cache_to_tuples(out.past_key_values), out.attentions, out.logits


def test_harness_scoring_matches_one_direct_forward(llama):
    model, _, _, ids = llama
    cache_len = SEQ - CONT - 1
    pkv, _, _ = _prefill(model, ids[:, :cache_len])
    harness_ppl = _score(model, ids[:, cache_len: SEQ - 1], ids[:, cache_len + 1: SEQ],
                         pkv, start_pos=cache_len)

    with torch.no_grad():
        direct = model(ids[:, : SEQ - 1])
    logits = direct.logits[:, cache_len:]
    targets = ids[:, cache_len + 1: SEQ]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    assert harness_ppl == pytest.approx(float(torch.exp(nll)), rel=1e-4)


def test_stepwise_decode_matches_hf_generate(llama):
    model, tok, _, ids = llama
    steps = 8
    ours = generate_stepwise(model, ids, FullCache(), max_new_tokens=steps)
    with torch.no_grad():
        theirs = model.generate(ids, max_new_tokens=steps, do_sample=False,
                                pad_token_id=tok.eos_token_id)
    assert ours[0].tolist() == theirs[0, SEQ:].tolist()


def test_evicted_cache_equals_masked_full_attention(llama):
    model, _, _, ids = llama
    pkv, _, prefill_logits = _prefill(model, ids)
    kept = sorted(set(range(0, SEQ, 3)) | set(range(SEQ - 8, SEQ)))
    dropped = [p for p in range(SEQ) if p not in kept]

    B, H = pkv[0][0].shape[:2]
    idx = torch.tensor(kept).view(1, 1, -1).expand(B, H, len(kept))
    evicted = tuple((_gather_tokens(k, idx), _gather_tokens(v, idx)) for k, v in pkv)

    next_id = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)
    true_position = torch.tensor([[SEQ]])
    with torch.no_grad(), per_layer_mask_fit(model, [len(kept)] * len(evicted)):
        compressed = model(input_ids=next_id, past_key_values=tuples_to_cache(evicted),
                           position_ids=true_position, use_cache=False)

    mask = torch.zeros(1, 1, 1, SEQ + 1)
    mask[:, :, :, dropped] = torch.finfo(torch.float32).min
    with torch.no_grad():
        oracle = model(input_ids=next_id, past_key_values=tuples_to_cache(pkv),
                       position_ids=true_position, attention_mask=mask, use_cache=False)

    torch.testing.assert_close(compressed.logits, oracle.logits, rtol=1e-4, atol=1e-4)


def test_wrong_query_position_changes_logits(llama):
    """The counterfactual behind decode.py's explicit position tracking: using
    the compressed cache length as the query position (what a naive loop does)
    must NOT reproduce the true-position logits."""
    model, _, _, ids = llama
    pkv, _, prefill_logits = _prefill(model, ids)
    kept = list(range(SEQ // 2, SEQ))
    idx = torch.tensor(kept).view(1, 1, -1).expand(pkv[0][0].shape[0], pkv[0][0].shape[1], len(kept))
    evicted = tuple((_gather_tokens(k, idx), _gather_tokens(v, idx)) for k, v in pkv)
    next_id = prefill_logits[:, -1].argmax(dim=-1, keepdim=True)

    def logits_at(position):
        with torch.no_grad(), per_layer_mask_fit(model, [len(kept)] * len(evicted)):
            return model(input_ids=next_id, past_key_values=tuples_to_cache(evicted),
                         position_ids=torch.tensor([[position]]), use_cache=False).logits

    assert not torch.allclose(logits_at(SEQ), logits_at(len(kept)), rtol=1e-4, atol=1e-4)


def test_ragged_budgets_decode_on_rope(llama):
    model, _, _, ids = llama
    method = CAKE(budget=16, window=4)
    tokens = generate_stepwise(model, ids, method, max_new_tokens=4)
    layer_budgets = method._layer_budgets
    assert len(set(layer_budgets)) >= 1 and sum(layer_budgets) == 16 * len(layer_budgets)
    assert tokens.shape[1] == 4
    assert ((tokens >= 0) & (tokens < model.config.vocab_size)).all()


def test_prompt_protected_random_decodes_at_prompt_plus_generation_budget(llama):
    """End-to-end on a real cache: the budget bounds the generated half only, so
    the cache settles at SEQ + budget positions and not at budget."""
    model, _, _, ids = llama
    budget, steps = 6, 10
    observed = []

    class Recording(PromptProtectedRandom):
        def step(self, past_key_values, attentions):
            out = super().step(past_key_values, attentions)
            observed.append([k.shape[2] for k, _ in out])
            return out

    tokens = generate_stepwise(model, ids, Recording(generation_budget=budget, seed=0),
                               max_new_tokens=steps)
    assert tokens.shape[1] == steps
    assert [lengths[0] for lengths in observed] == [
        SEQ + min(budget, generated + 1) for generated in range(steps - 1)]
    assert all(len(set(lengths)) == 1 for lengths in observed), "layers must stay uniform"


@pytest.mark.parametrize("key, kwargs, protects_the_prompt", [
    ("full", {}, True),
    ("random", {"generation_budget": 4, "seed": 0}, True),
    ("h2o", {"budget": SEQ // 2, "recent": 4}, False),
    ("snapkv", {"budget": SEQ // 2, "window": 4}, False),
    ("cake", {"budget": SEQ // 2, "window": 4}, False),
    ("obcache", {"budget": SEQ // 2, "recent": 4}, False),
])
def test_provenance_balances_for_every_method(llama, key, kwargs, protects_the_prompt):
    """The ledger runs against real caches here, not scripted ones: it has to
    balance for a per-head random draw and a ragged per-layer budget alike, and
    only the prompt-protected methods may come back holding the whole prompt."""
    model, _, cfg, ids = llama
    steps = 8
    ledger = CacheLedger()
    generate_stepwise(model, ids, build(key, **kwargs), max_new_tokens=steps, ledger=ledger)
    account = ledger.account(cfg)

    assert account.prompt_length == SEQ
    # The final generated token's KV never enters the cache: nothing attends to it.
    assert account.generated_length == steps - 1
    assert account.total_retained == account.prompt_retained + account.generated_retained
    assert (account.prompt_retained == SEQ) is protects_the_prompt
    assert account.total_kv_bytes <= account.uncompressed_kv_bytes
