"""Load a causal LM and read its cache-relevant config. Requires torch and
transformers; kept separate so the cost model stays dependency-free."""

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

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32).to(device).eval()

    hf = model.config
    n_heads = getattr(hf, "num_attention_heads", None) or hf.n_head
    n_layers = getattr(hf, "num_hidden_layers", None) or hf.n_layer
    n_kv = getattr(hf, "num_key_value_heads", None) or n_heads
    hidden = getattr(hf, "hidden_size", None) or hf.n_embd
    head_dim = getattr(hf, "head_dim", None) or (hidden // n_heads)

    return model, tok, ModelConfig(name, n_layers, n_heads, n_kv, head_dim)
