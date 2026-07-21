"""Step-wise greedy decode loop with a per-step method hook.

transformers' generate() gives a method no way to reshape the cache mid-decode,
so decoding-phase eviction (online H2O, MorphKV, and any method that reacts to
what the model actually attends to while generating) had nowhere to run. This
loop calls method.apply() once after prefill and method.step() after every
decode step, and tracks true token positions explicitly so position embeddings
stay correct after eviction shrinks the cache.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def generate_stepwise(model, ids, method, max_new_tokens: int = 64, eos_token_id: int | None = None):
    """Greedy-decode continuation ids of shape [batch, <=max_new_tokens]."""
    import torch

    from .model import cache_to_tuples, tuples_to_cache

    with torch.no_grad():
        out = model(ids, use_cache=True, output_attentions=True)
    pkv = cache_to_tuples(out.past_key_values)
    if len(out.attentions) != len(pkv):
        raise RuntimeError(
            f"got {len(out.attentions)} attention tensors for {len(pkv)} cache layers; "
            "the model must run with attn_implementation='eager' to expose attentions")
    pkv = method.apply(pkv, out.attentions)

    position = ids.shape[1]
    next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated = [next_id]
    for _ in range(max_new_tokens - 1):
        if eos_token_id is not None and (next_id == eos_token_id).all():
            break
        position_ids = torch.full((ids.shape[0], 1), position, dtype=torch.long, device=ids.device)
        with torch.no_grad():
            out = model(input_ids=next_id, past_key_values=tuples_to_cache(pkv),
                        position_ids=position_ids, use_cache=True, output_attentions=True)
        pkv = method.step(cache_to_tuples(out.past_key_values), out.attentions)
        position += 1
        next_id = out.logits[:, -1].argmax(dim=-1, keepdim=True)
        generated.append(next_id)
    return torch.cat(generated, dim=1)
