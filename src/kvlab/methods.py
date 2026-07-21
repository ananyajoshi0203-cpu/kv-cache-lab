"""Reference implementations of a few KV-cache methods, written for clarity so the
mechanism is legible. Production runs use the KVPress backend; these exist to make
eviction and quantization concrete and to validate against KVPress. Each method
transforms a HuggingFace past_key_values, a tuple over layers of (key, value) with
shape [batch, kv_heads, seq, head_dim]."""

from __future__ import annotations

import torch

from .memory import ModelConfig


class KVMethod:
    key = "base"
    name = "Base"
    family = "-"
    lever = "-"
    bits = 16

    def apply(self, past_key_values, attentions):
        raise NotImplementedError

    def step(self, past_key_values, attentions):
        """Called after each decode step with the grown cache and that step's
        attentions. Prefill-only methods leave the cache alone; decoding-phase
        methods (online H2O, MorphKV) evict here."""
        return past_key_values

    def kept_len(self, orig_len: int) -> int:
        return orig_len

    def kv_bytes(self, orig_len: int, cfg: ModelConfig) -> float:
        """Cache size after this method, in bytes. The default assumes every kept
        token is stored uniformly at `self.bits`; methods with mixed precision
        (e.g. a full-precision residual window) must override."""
        per_elem = 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim
        return per_elem * (self.bits / 8.0) * self.kept_len(orig_len)


def _key_scores(attn, n_kv_heads, window=None):
    if window is not None:
        attn = attn[:, :, -window:, :]
    scores = attn.sum(dim=2)
    b, qh, k = scores.shape
    if qh != n_kv_heads:
        scores = scores.view(b, n_kv_heads, qh // n_kv_heads, k).sum(dim=2)
    return scores


def _evict_indices(scores, budget, recent):
    """Kept-token indices [batch, heads, budget] in original order, or None when
    the sequence already fits the budget."""
    B, H, S = scores.shape
    if S <= budget:
        return None
    recent = min(recent, budget)
    heavy = min(budget - recent, S - recent)
    idx_recent = torch.arange(S - recent, S, device=scores.device).view(1, 1, recent).expand(B, H, recent)
    _, idx_heavy = scores[..., : S - recent].topk(heavy, dim=-1)
    return torch.cat([idx_heavy, idx_recent], dim=-1).sort(dim=-1).values


def _gather_tokens(tensor, idx):
    return tensor.gather(2, idx.unsqueeze(-1).expand(*idx.shape, tensor.shape[-1]))


def _evict(key, value, scores, budget, recent):
    idx = _evict_indices(scores, budget, recent)
    if idx is None:
        return key, value
    return _gather_tokens(key, idx), _gather_tokens(value, idx)


def _fake_quant(x, bits, reduce_dim):
    if bits >= 16:
        return x
    qmax = (2 ** bits) - 1
    xmin = x.amin(dim=reduce_dim, keepdim=True)
    xmax = x.amax(dim=reduce_dim, keepdim=True)
    scale = (xmax - xmin).clamp(min=1e-8) / qmax
    q = ((x - xmin) / scale).round().clamp(0, qmax)
    return q * scale + xmin


class FullCache(KVMethod):
    key, name, family, lever, bits = "full", "Full cache", "Baseline", "none", 16

    def apply(self, past_key_values, attentions):
        return past_key_values


class H2O(KVMethod):
    """One-shot eviction after prefill, then true online H2O during decoding:
    attention scores accumulate across steps (surviving the evictions that
    reshape the cache) and each step evicts back down to the budget."""

    key, name, family, lever, bits = "h2o", "H2O", "Eviction", "context", 16

    def __init__(self, budget=128, recent=32):
        self.budget, self.recent = budget, recent
        self._acc: list | None = None

    def _evict_layer(self, k, v, scores):
        idx = _evict_indices(scores, self.budget, self.recent)
        if idx is None:
            return (k, v), scores
        return (_gather_tokens(k, idx), _gather_tokens(v, idx)), scores.gather(-1, idx)

    def apply(self, past_key_values, attentions):
        out, self._acc = [], []
        for (k, v), attn in zip(past_key_values, attentions):
            kv, acc = self._evict_layer(k, v, _key_scores(attn, k.shape[1]))
            out.append(kv)
            self._acc.append(acc)
        return tuple(out)

    def step(self, past_key_values, attentions):
        out, acc_next = [], []
        for i, ((k, v), attn) in enumerate(zip(past_key_values, attentions)):
            scores = _key_scores(attn, k.shape[1])
            if self._acc is not None:
                prev = self._acc[i]
                scores[..., : prev.shape[-1]] += prev
            kv, acc = self._evict_layer(k, v, scores)
            out.append(kv)
            acc_next.append(acc)
        self._acc = acc_next
        return tuple(out)

    def kept_len(self, orig_len):
        return min(self.budget, orig_len)


class SnapKV(KVMethod):
    key, name, family, lever, bits = "snapkv", "SnapKV", "Eviction", "context", 16

    def __init__(self, budget=128, window=32):
        self.budget, self.window = budget, window

    def apply(self, past_key_values, attentions):
        return tuple(_evict(k, v, _key_scores(attn, k.shape[1], self.window), self.budget, self.window)
                     for (k, v), attn in zip(past_key_values, attentions))

    def kept_len(self, orig_len):
        return min(self.budget, orig_len)


class KIVIQuant(KVMethod):
    key, name, family, lever = "kivi", "KIVI", "Compression", "dtype"

    def __init__(self, bits=2, residual=16):
        self.bits = bits
        self.residual = residual

    def apply(self, past_key_values, attentions):
        out = []
        for k, v in past_key_values:
            S = k.shape[2]
            r = min(self.residual, S)
            kq = torch.cat([_fake_quant(k[:, :, : S - r], self.bits, 2), k[:, :, S - r:]], dim=2)
            vq = torch.cat([_fake_quant(v[:, :, : S - r], self.bits, 3), v[:, :, S - r:]], dim=2)
            out.append((kq, vq))
        return tuple(out)

    def kv_bytes(self, orig_len: int, cfg: ModelConfig) -> float:
        per_elem = 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim
        r = min(self.residual, orig_len)
        return per_elem * ((orig_len - r) * self.bits / 8.0 + r * 2.0)


_IMPLEMENTED = {"full": FullCache, "h2o": H2O, "snapkv": SnapKV, "kivi": KIVIQuant}


def build(key: str, **kwargs) -> KVMethod:
    if key in _IMPLEMENTED:
        return _IMPLEMENTED[key](**kwargs)
    from .registry import get
    m = get(key)
    raise NotImplementedError(
        f"{m.name} is catalogued (status={m.status}, lever={m.lever}) but not implemented here. "
        f"Use the KVPress backend if a press exists ({m.kvpress or 'none'}), "
        f"or add a KVMethod subclass. Paper: {m.paper}")


def implemented_keys() -> list[str]:
    return list(_IMPLEMENTED)
