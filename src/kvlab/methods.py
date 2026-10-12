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
    phase = "after-prefill"

    #: Indices this method last gathered with, per layer, [batch, kv_heads, kept],
    #: relative to the cache it was handed. None means nothing was dropped. A
    #: method publishes these so an outside observer (kvlab.accounting.CacheLedger)
    #: can follow where each surviving position came from without the method
    #: keeping any provenance state of its own.
    last_indices: tuple | None = None

    def apply(self, past_key_values, attentions):
        raise NotImplementedError

    def step(self, past_key_values, attentions):
        """Called after each decode step with the grown cache and that step's
        attentions. Prefill-only methods leave the cache alone; decoding-phase
        methods (online H2O, MorphKV) evict here."""
        self.last_indices = None
        return past_key_values

    def observe(self, token_id: int) -> None:
        """Called with each generated token as its KV enters the cache, before
        step(). Cache methods ignore it: attention weights are what they score on,
        and a method that needed to know which token it was looking at would be
        reading the output it is meant to be agnostic to. It exists for diagnostics
        that deliberately target the cache by token identity."""

    def kept_len(self, orig_len: int) -> int:
        return orig_len

    def kv_bytes(self, orig_len: int, cfg: ModelConfig) -> float:
        """Cache size after this method, in bytes. The default assumes every kept
        token is stored uniformly at `self.bits`; methods with mixed precision
        (e.g. a full-precision residual window) must override."""
        per_elem = 2 * cfg.layers * cfg.n_kv_heads * cfg.head_dim
        return per_elem * (self.bits / 8.0) * self.kept_len(orig_len)


def _key_scores(attn, n_kv_heads, window=None):
    """Per-token importance from attention weights, [batch, kv_heads, seq].
    Under GQA, the query heads sharing a KV head are summed; sum vs mean does
    not change any within-layer top-k because the group size is uniform."""
    if window is not None:
        attn = attn[:, :, -window:, :]
    scores = attn.sum(dim=2)
    b, qh, k = scores.shape
    if qh != n_kv_heads:
        scores = scores.view(b, n_kv_heads, qh // n_kv_heads, k).sum(dim=2)
    return scores


def _pool_scores(scores, kernel_size):
    """SnapKV's clustering step: max-pool scores along the key axis so a
    high-scoring token lifts its neighbours and contiguous spans survive
    together, instead of isolated tokens surrounded by evicted context."""
    import torch.nn.functional as F
    return F.max_pool1d(scores, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)


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


def _evict_tracked(key, value, scores, budget, recent):
    """`_evict`, plus the indices it kept, so a caller can record provenance."""
    idx = _evict_indices(scores, budget, recent)
    if idx is None:
        return key, value, None
    return _gather_tokens(key, idx), _gather_tokens(value, idx), idx


def _evict(key, value, scores, budget, recent):
    return _evict_tracked(key, value, scores, budget, recent)[:2]


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
    phase = "none"

    def apply(self, past_key_values, attentions):
        self.last_indices = None
        return past_key_values


class PromptProtectedRandom(KVMethod):
    """Random Attention (arXiv:2609.03430): keep the prompt, evict the generated
    cache uniformly at random, compute no score anywhere. It is the control the
    rest of this repository is measured against -- if a scorer cannot beat a
    coin flip that was told only "do not touch the prompt", its selection signal
    is not what is doing the work.

    Budget semantics, which are the whole point of the class. `generation_budget`
    bounds the *generated* half of the cache only. The cache this method holds is

        prompt_length + generation_budget

    positions, so a generated-token budget and a total-token budget are different
    quantities and this harness never converts between them implicitly. See
    kvlab.boundary.Budget, which carries both and is explicit about which one a
    given comparison holds fixed.

    Granularity. The paper draws "uniformly at random within each attention head".
    A HuggingFace cache stores one entry per *KV* head, which is the finest
    granularity any draw over it can have: on an MHA model (kv_heads == n_heads,
    which includes distilgpt2 and GPT-2) this is exactly the paper's per-head
    draw, and under GQA the query heads sharing a KV head necessarily share its
    draw. Layers draw independently, as do decode steps.

    Eviction is irreversible, so each step draws over the generated positions
    still in the cache rather than over every position ever generated -- evicting
    uniformly at random from the live pool, which is what the paper describes.
    The paper specifies no recency window and none is added here.
    """

    key, name, family, lever, bits = "random", "Prompt-protected random", "Eviction", "context", 16
    phase = "decoding"

    def __init__(self, generation_budget: int = 128, seed: int = 0):
        if generation_budget < 0:
            raise ValueError(f"generation_budget must be non-negative, got {generation_budget}")
        self.generation_budget = generation_budget
        self.seed = seed
        self.prompt_length: int | None = None
        self._generator = torch.Generator()

    def apply(self, past_key_values, attentions):
        """Prefill leaves the cache alone: it is all prompt, and the prompt is
        protected. Recording its length here is what makes the protection
        possible, and reseeding here makes a reused instance reproducible."""
        self.prompt_length = past_key_values[0][0].shape[2]
        self._generator.manual_seed(self.seed)
        self.last_indices = None
        return past_key_values

    def step(self, past_key_values, attentions):
        if self.prompt_length is None:
            raise RuntimeError("apply() must run on the prefilled cache before step(); "
                               "without it the method does not know where the prompt ends")
        out, indices = [], []
        for k, v in past_key_values:
            idx = self._draw(k)
            indices.append(idx)
            out.append((k, v) if idx is None else (_gather_tokens(k, idx), _gather_tokens(v, idx)))
        self.last_indices = tuple(indices)
        return tuple(out)

    def _draw(self, key):
        """Kept indices [batch, kv_heads, prompt_length + generation_budget] in
        chronological order, or None while the generated cache is within budget."""
        batch, heads, seq = key.shape[:3]
        protected = min(self.prompt_length, seq)
        pool = seq - protected
        if pool <= self.generation_budget:
            return None
        draws = [torch.randperm(pool, generator=self._generator)[: self.generation_budget].sort().values
                 for _ in range(batch * heads)]
        generated = protected + torch.stack(draws).view(batch, heads, self.generation_budget)
        prompt = torch.arange(protected).view(1, 1, protected).expand(batch, heads, protected)
        return torch.cat([prompt, generated], dim=-1).to(key.device)

    def kept_len(self, orig_len: int) -> int:
        """`orig_len` is read as a total sequence length: prompt plus generated.
        Before apply() has seen a prefill there is no prompt boundary to protect,
        so nothing is dropped."""
        if self.prompt_length is None:
            return orig_len
        generated = max(0, orig_len - self.prompt_length)
        return self.prompt_length + min(self.generation_budget, generated)


class H2O(KVMethod):
    """One-shot eviction after prefill, then true online H2O during decoding:
    attention scores accumulate across steps (surviving the evictions that
    reshape the cache) and each step evicts back down to the budget."""

    key, name, family, lever, bits = "h2o", "H2O", "Eviction", "context", 16
    phase = "decoding"

    def __init__(self, budget=128, recent=32):
        self.budget, self.recent = budget, recent
        self._acc: list | None = None

    def _evict_layer(self, k, v, scores):
        idx = _evict_indices(scores, self.budget, self.recent)
        if idx is None:
            return (k, v), scores, None
        return (_gather_tokens(k, idx), _gather_tokens(v, idx)), scores.gather(-1, idx), idx

    def apply(self, past_key_values, attentions):
        out, self._acc, indices = [], [], []
        for (k, v), attn in zip(past_key_values, attentions):
            kv, acc, idx = self._evict_layer(k, v, _key_scores(attn, k.shape[1]))
            out.append(kv)
            self._acc.append(acc)
            indices.append(idx)
        self.last_indices = tuple(indices)
        return tuple(out)

    def step(self, past_key_values, attentions):
        out, acc_next, indices = [], [], []
        for i, ((k, v), attn) in enumerate(zip(past_key_values, attentions)):
            scores = _key_scores(attn, k.shape[1])
            if self._acc is not None:
                prev = self._acc[i]
                scores[..., : prev.shape[-1]] += prev
            kv, acc, idx = self._evict_layer(k, v, scores)
            out.append(kv)
            acc_next.append(acc)
            indices.append(idx)
        self._acc = acc_next
        self.last_indices = tuple(indices)
        return tuple(out)

    def kept_len(self, orig_len):
        return min(self.budget, orig_len)


class SnapKV(KVMethod):
    """Observation-window voting with max-pooled clustering (arXiv:2404.14469):
    the last `window` queries score every prompt token, pooling keeps clusters
    of context together, and the window itself is always retained."""

    key, name, family, lever, bits = "snapkv", "SnapKV", "Eviction", "context", 16

    def __init__(self, budget=128, window=32, pool=7):
        self.budget, self.window, self.pool = budget, window, pool

    def apply(self, past_key_values, attentions):
        out, indices = [], []
        for (k, v), attn in zip(past_key_values, attentions):
            scores = _key_scores(attn, k.shape[1], self.window)
            if self.pool > 1:
                scores = _pool_scores(scores, self.pool)
            ek, ev, idx = _evict_tracked(k, v, scores, self.budget, self.window)
            out.append((ek, ev))
            indices.append(idx)
        self.last_indices = tuple(indices)
        return tuple(out)

    def kept_len(self, orig_len):
        return min(self.budget, orig_len)


class OBCache(KVMethod):
    """Value-aware eviction (OBCache, arXiv:2510.07651) in first-order reference
    form: saliency is accumulated attention mass weighted by the L2 norm of each
    token's value vector, the diagonal term of the paper's output-perturbation
    objective. Two tokens with equal attention differ in saliency when their
    values differ in magnitude -- the signal attention-only scorers cannot see.
    The Hessian-based correction is not reproduced."""

    key, name, family, lever, bits = "obcache", "OBCache 1st-order", "Eviction", "context", 16

    def __init__(self, budget=128, recent=32):
        self.budget, self.recent = budget, recent

    def apply(self, past_key_values, attentions):
        out, indices = [], []
        for (k, v), attn in zip(past_key_values, attentions):
            scores = _key_scores(attn, k.shape[1]) * v.norm(dim=-1)
            ek, ev, idx = _evict_tracked(k, v, scores, self.budget, self.recent)
            out.append((ek, ev))
            indices.append(idx)
        self.last_indices = tuple(indices)
        return tuple(out)

    def kept_len(self, orig_len):
        return min(self.budget, orig_len)


class CAKE(KVMethod):
    """Layer-adaptive budget allocation (CAKE, arXiv:2503.12491) in reference
    form. Each layer's preference combines spatial dispersion (entropy of its
    mean attention distribution over the observation window) with temporal shift
    (variance of that distribution across window queries); the global token
    budget -- `budget` per layer on average -- is split proportionally, then each
    layer evicts with SnapKV-style window scores. One-shot after prefill; the
    paper's cascading prefill management is not reproduced. The preference
    statistics round-trip through Python floats, which forces device sync --
    fine for this CPU reference, not for a production implementation."""

    key, name, family, lever, bits = "cake", "CAKE-style", "Eviction", "context", 16

    def __init__(self, budget=128, window=32, pool=7, tau1=1.0, tau2=1.0):
        self.budget, self.window, self.pool = budget, window, pool
        self.tau1, self.tau2 = tau1, tau2
        self._layer_budgets: list[int] | None = None

    def _preference(self, attn) -> float:
        window_attn = attn[:, :, -self.window:, :].mean(dim=(0, 1))
        dist = window_attn.mean(dim=0)
        dist = dist / dist.sum().clamp(min=1e-9)
        dispersion = -(dist * dist.clamp(min=1e-9).log()).sum()
        shift = window_attn.var(dim=0).mean()
        return float((dispersion + 1e-6) ** (1 / self.tau1) * (shift + 1e-6) ** (1 / self.tau2))

    def _allocate(self, preferences: list[float], lens: list[int]) -> list[int]:
        n = len(lens)
        total = min(self.budget * n, sum(lens))
        floors = [min(self.window, length) for length in lens]
        if sum(floors) >= total:
            return [min(self.budget, length) for length in lens]
        alloc = [float(f) for f in floors]
        for _ in range(n):
            remaining = total - sum(alloc)
            if remaining <= 1e-9:
                break
            open_layers = [i for i in range(n) if alloc[i] < lens[i]]
            if not open_layers:
                break
            weight_sum = sum(preferences[i] for i in open_layers)
            for i in open_layers:
                share = preferences[i] / weight_sum if weight_sum > 0 else 1 / len(open_layers)
                alloc[i] = min(float(lens[i]), alloc[i] + remaining * share)
        budgets = [int(a) for a in alloc]
        by_remainder = sorted(range(n), key=lambda i: alloc[i] - budgets[i], reverse=True)
        for i in by_remainder:
            if sum(budgets) >= total:
                break
            if budgets[i] < lens[i]:
                budgets[i] += 1
        return budgets

    def apply(self, past_key_values, attentions):
        preferences = [self._preference(attn) for attn in attentions]
        lens = [k.shape[2] for k, _ in past_key_values]
        self._layer_budgets = self._allocate(preferences, lens)
        out, indices = [], []
        for (k, v), attn, layer_budget in zip(past_key_values, attentions, self._layer_budgets):
            scores = _key_scores(attn, k.shape[1], self.window)
            if self.pool > 1:
                scores = _pool_scores(scores, self.pool)
            ek, ev, idx = _evict_tracked(k, v, scores, layer_budget, self.window)
            out.append((ek, ev))
            indices.append(idx)
        self.last_indices = tuple(indices)
        return tuple(out)

    def kept_len(self, orig_len):
        return min(self.budget, orig_len)

    def kv_bytes(self, orig_len: int, cfg: ModelConfig) -> float:
        if self._layer_budgets is None:
            return super().kv_bytes(orig_len, cfg)
        per_layer_elem = 2 * cfg.n_kv_heads * cfg.head_dim
        return per_layer_elem * (self.bits / 8.0) * sum(min(b, orig_len) for b in self._layer_budgets)


class KIVIQuant(KVMethod):
    """Fake-quantization reference of KIVI (arXiv:2402.02750): tensors are
    quantized and immediately dequantized, so the *error* of 2-bit storage is
    simulated while the tensors stay full-precision floats. kv_bytes reports the
    analytical low-bit size, not allocated memory; real memory and latency
    numbers require a genuine quantized-cache backend (KVPressQuantBackend)."""

    key, name, family, lever = "kivi", "KIVI fake-quant", "Compression", "dtype"
    phase = "none"

    def __init__(self, bits=2, residual=16):
        self.bits = bits
        self.residual = residual

    def apply(self, past_key_values, attentions):
        self.last_indices = None
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


_IMPLEMENTED = {"full": FullCache, "random": PromptProtectedRandom, "h2o": H2O,
                "snapkv": SnapKV, "cake": CAKE, "obcache": OBCache, "kivi": KIVIQuant}


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
