"""Load a causal LM and read its cache-relevant config, plus conversion between
transformers Cache objects and the plain (key, value) tuples the reference methods
operate on. Requires torch and transformers; kept separate so the cost model stays
dependency-free."""

from __future__ import annotations

import logging

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
