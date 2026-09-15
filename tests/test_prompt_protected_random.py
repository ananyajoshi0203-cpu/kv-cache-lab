"""Invariants of the prompt-protected random baseline (arXiv:2609.03430).

The method computes no score, so there is no scoring path to check. What there is
instead is a set of structural promises -- the prompt survives, the generated half
is held to its own budget, keys and values agree, and the draw is both reproducible
from a seed and genuinely different at every step and in every head. Those are what
these tests pin.

Cache positions are coded into the tensors (position s stores the value s in every
element), so the surviving positions are read back off the tensors rather than
re-derived with the production drawing code. Run: `pytest tests/`.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab.methods import PromptProtectedRandom, build                # noqa: E402

BATCH, HEADS, DIM = 1, 4, 2


def coded(positions, layers=1):
    """A one-layer-per-entry cache whose every element holds its own position."""
    values = torch.tensor(list(positions), dtype=torch.float32)
    key = values.view(1, 1, -1, 1).expand(BATCH, HEADS, len(values), DIM).clone()
    return tuple((key.clone(), key.clone()) for _ in range(layers))


def decode(method, prompt_length, steps, layers=1):
    """Prefill, then `steps` decode steps that each append one new position.
    Returns the cache after every step."""
    cache = coded(range(prompt_length), layers)
    method.apply(cache, None)
    history = []
    for step in range(steps):
        new = coded([prompt_length + step], layers)
        cache = tuple((torch.cat([k, nk], dim=2), torch.cat([v, nv], dim=2))
                      for (k, v), (nk, nv) in zip(cache, new))
        cache = method.step(cache, None)
        history.append(cache)
    return history


def positions(key, head=0, batch=0):
    return [int(p) for p in key[batch, head, :, 0].tolist()]


@pytest.mark.parametrize("prompt_length, budget, steps", [
    (6, 3, 10),     # budget bites early and stays bitten
    (1, 2, 8),      # a one-token prompt is still protected
    (9, 0, 6),      # nothing generated is ever kept
    (4, 20, 5),     # budget never bites: no eviction at all
])
def test_prompt_kv_is_never_evicted(prompt_length, budget, steps):
    method = PromptProtectedRandom(generation_budget=budget, seed=0)
    for cache in decode(method, prompt_length, steps):
        for key, _ in cache:
            for head in range(HEADS):
                kept = positions(key, head)
                assert kept[:prompt_length] == list(range(prompt_length))


@pytest.mark.parametrize("prompt_length, budget, steps", [
    (6, 3, 10),
    (1, 2, 8),
    (9, 0, 6),
    (4, 20, 5),
])
def test_generated_half_is_held_to_its_own_budget_in_order(prompt_length, budget, steps):
    method = PromptProtectedRandom(generation_budget=budget, seed=1)
    for step, cache in enumerate(decode(method, prompt_length, steps)):
        # `step + 1` generated positions have entered the cache by now; eviction is
        # irreversible, so the pool can never exceed the budget once it has bitten.
        expected_generated = min(budget, step + 1)
        for key, _ in cache:
            for head in range(HEADS):
                kept = positions(key, head)
                generated = [p for p in kept if p >= prompt_length]
                assert len(generated) == expected_generated
                assert len(kept) == prompt_length + expected_generated
                assert kept == sorted(kept)
                assert len(set(kept)) == len(kept)


def test_keys_and_values_keep_identical_positions():
    method = PromptProtectedRandom(generation_budget=3, seed=2)
    for cache in decode(method, prompt_length=5, steps=8, layers=2):
        for key, value in cache:
            assert torch.equal(key, value)


@pytest.mark.parametrize("seed, other_seed, same_draw", [(7, 7, True), (7, 8, False)])
def test_the_seed_alone_determines_the_draw(seed, other_seed, same_draw):
    def draws(s):
        method = PromptProtectedRandom(generation_budget=4, seed=s)
        return [positions(cache[0][0], head)
                for cache in decode(method, prompt_length=5, steps=12)
                for head in range(HEADS)]

    assert (draws(seed) == draws(other_seed)) is same_draw


def test_a_reused_instance_reproduces_its_own_draws():
    """apply() reseeds, so running the same object twice is the same experiment."""
    method = PromptProtectedRandom(generation_budget=4, seed=3)
    first = [positions(cache[0][0]) for cache in decode(method, 5, 12)]
    second = [positions(cache[0][0]) for cache in decode(method, 5, 12)]
    assert first == second


def test_the_draw_evolves_across_steps_heads_and_layers():
    """The failure this guards against is an RNG reset per step, which would make
    every step, head and layer pick the same positions."""
    budget, prompt_length, steps, layers = 5, 4, 14, 3
    method = PromptProtectedRandom(generation_budget=budget, seed=4)
    history = decode(method, prompt_length, steps, layers)

    def generated(cache, layer, head):
        return tuple(p for p in positions(cache[layer][0], head) if p >= prompt_length)

    saturated = [c for c in history if len(generated(c, 0, 0)) == budget]
    assert len(saturated) > 2, "the budget never bit, so there is nothing to test"

    across_steps = {generated(c, 0, 0) for c in saturated}
    across_heads = {generated(saturated[-1], 0, head) for head in range(HEADS)}
    across_layers = {generated(saturated[-1], layer, 0) for layer in range(layers)}
    assert len(across_steps) > 1
    assert len(across_heads) > 1
    assert len(across_layers) > 1


@pytest.mark.parametrize("prompt_length, budget, total_length", [
    (100, 32, 400),     # budget bites: prompt is protected on top of it
    (100, 32, 110),     # only 10 generated so far, all of them fit
    (100, 0, 400),      # prompt-only cache
])
def test_kept_len_is_prompt_plus_generation_budget(prompt_length, budget, total_length):
    method = build("random", generation_budget=budget, seed=0)
    method.apply(coded(range(prompt_length)), None)
    generated = total_length - prompt_length
    assert method.kept_len(total_length) == prompt_length + min(budget, generated)


@pytest.mark.parametrize("build_kwargs, action, error", [
    ({"generation_budget": -1}, None, ValueError),
    ({"generation_budget": 4}, "step", RuntimeError),
])
def test_misuse_is_refused_not_guessed(build_kwargs, action, error):
    with pytest.raises(error):
        method = PromptProtectedRandom(**build_kwargs)
        if action == "step":
            method.step(coded(range(6)), None)
