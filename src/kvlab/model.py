"""Load a causal LM and read its cache-relevant config, plus conversion between
transformers Cache objects and the plain (key, value) tuples the reference methods
operate on. Requires torch and transformers; kept separate so the cost model stays
dependency-free."""

from __future__ import annotations

import logging
from contextlib import contextmanager

from .memory import ModelConfig

logger = logging.getLogger(__name__)


def load_model(name: str = "distilgpt2", device: str = "cpu"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("loading %s on %s", name, device)
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Eager attention is required for output_attentions: SDPA (the default since
    # transformers 5) returns an empty attentions tuple, which would silently
    # starve the attention-based scorers.
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.float32, attn_implementation="eager").to(device).eval()

    hf = model.config
    n_heads = getattr(hf, "num_attention_heads", None) or hf.n_head
    n_layers = getattr(hf, "num_hidden_layers", None) or hf.n_layer
    n_kv = getattr(hf, "num_key_value_heads", None) or n_heads
    hidden = getattr(hf, "hidden_size", None) or hf.n_embd
    head_dim = getattr(hf, "head_dim", None) or (hidden // n_heads)

    return model, tok, ModelConfig(name, n_layers, n_heads, n_kv, head_dim)


@contextmanager
def per_layer_mask_fit(model, key_lens):
    """Let a ragged cache -- different kept lengths per layer, as produced by
    layer-adaptive budgets (CAKE, Ada-KV) -- run through a stock forward pass.

    The model builds one additive attention mask sized to a single cache length,
    which raises a shape error as soon as layers disagree. Cache positions are
    uniformly visible (zero) in that mask and only the causal tail over the new
    queries differs, so trimming the mask's key axis to each layer's true length
    (or left-padding it with visible zeros) is exact, not an approximation.
    Registers a forward-pre-hook per attention module; removed on exit."""
    import torch.nn.functional as F

    blocks = model.transformer.h if hasattr(model, "transformer") else model.model.layers

    def fit(key_len):
        def hook(module, args, kwargs):
            mask = kwargs.get("attention_mask")
            if mask is None:
                return None
            target = key_len + mask.shape[-2]
            if mask.shape[-1] > target:
                kwargs["attention_mask"] = mask[..., -target:]
            elif mask.shape[-1] < target:
                kwargs["attention_mask"] = F.pad(mask, (target - mask.shape[-1], 0))
            return args, kwargs
        return hook

    handles = []
    try:
        for block, key_len in zip(blocks, key_lens):
            attn = block.attn if hasattr(block, "attn") else block.self_attn
            handles.append(attn.register_forward_pre_hook(fit(key_len), with_kwargs=True))
        yield
    finally:
        for handle in handles:
            handle.remove()


def cache_to_tuples(cache):
    """A model's past_key_values as a tuple over layers of (key, value) tensors.
    transformers < 5 returns a Cache with to_legacy_cache(); transformers 5
    removed the legacy API, leaving per-layer .keys/.values attributes."""
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    if hasattr(cache, "layers"):
        return tuple((layer.keys, layer.values) for layer in cache.layers)
    return tuple(cache)


def tuples_to_cache(past_key_values):
    """The inverse: wrap (key, value) tuples back into the Cache object the
    model's forward pass requires."""
    from transformers import DynamicCache

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(past_key_values)
    return DynamicCache(past_key_values)
