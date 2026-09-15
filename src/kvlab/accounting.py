"""Where the surviving KV actually came from.

Every number in this module is **analytical**. A slot count is counted off the
cache tensors; bytes are that count multiplied through the cost model in
kvlab.memory. Nothing here reads an allocator. That is not a shortcut: the
reference methods keep full-precision tensors and simulate low-bit storage, and a
method that evicts per head leaves the tensor shape unchanged for the heads that
kept everything, so measured GPU bytes would answer a different question from the
one asked. Results files record these as analytical KV bytes and never as memory.

The distinction the module exists for is prompt KV against generated KV. A
compressed cache is usually reported as one number, which cannot separate a method
that spent its budget on the prompt from one that spent it on the model's own
output -- and that separation is the whole claim under test in arXiv:2609.03430.
CacheLedger follows every position's original index through eviction so the split
is measured rather than assumed, for scored and score-free methods alike.

A *slot* is one retained KV position, in one layer, in one KV head. Slots are what
bytes are proportional to, so they are what the ledger counts; the per-layer token
counts a reader expects (`prompt_retained` and friends) are derived from them.
"""

from __future__ import annotations

from dataclasses import dataclass

from .memory import DTYPE_BYTES, ModelConfig

KV_TENSORS_PER_TOKEN = 2  # one key and one value


@dataclass(frozen=True)
class KVAccount:
    """A compressed cache broken down by the provenance of what it kept."""

    model: str
    layers: int
    kv_heads: int
    head_dim: int
    dtype: str
    prompt_length: int          # prompt tokens, before any eviction
    generated_length: int       # generated tokens that entered the cache
    prompt_slots: int           # retained prompt slots, over all layers and heads
    generated_slots: int
    #: Mean original sequence index of the retained generated slots, 0.0 when there
    #: are none. Two methods can retain the same count and still keep entirely
    #: different parts of the trace, and this is the cheapest summary that separates
    #: them: it moves when a method is recency-biased, and it moves between seeds of
    #: a random method even though the count cannot.
    generated_position_mean: float = 0.0

    def __post_init__(self) -> None:
        for label, slots, length in (("prompt", self.prompt_slots, self.prompt_length),
                                     ("generated", self.generated_slots, self.generated_length)):
            ceiling = self.layers * self.kv_heads * length
            if not 0 <= slots <= ceiling:
                raise ValueError(
                    f"{slots} retained {label} slots is impossible for {length} {label} tokens "
                    f"across {self.layers} layers and {self.kv_heads} KV heads (ceiling {ceiling})")

    @property
    def total_slots(self) -> int:
        return self.prompt_slots + self.generated_slots

    @property
    def uncompressed_slots(self) -> int:
        return self.layers * self.kv_heads * (self.prompt_length + self.generated_length)

    def _bytes(self, slots: int) -> float:
        return KV_TENSORS_PER_TOKEN * self.head_dim * DTYPE_BYTES[self.dtype] * slots

    @property
    def prompt_kv_bytes(self) -> float:
        return self._bytes(self.prompt_slots)

    @property
    def generated_kv_bytes(self) -> float:
        return self._bytes(self.generated_slots)

    @property
    def total_kv_bytes(self) -> float:
        return self._bytes(self.total_slots)

    @property
    def uncompressed_kv_bytes(self) -> float:
        return self._bytes(self.uncompressed_slots)

    @property
    def compression_ratio(self) -> float:
        """Fraction of cache bytes removed, the same sense as benchmark.run's
        `ratio`, so the two harnesses can be read side by side."""
        if self.uncompressed_slots == 0:
            return 0.0
        return 1.0 - self.total_slots / self.uncompressed_slots

    def _per_head_layer(self, slots: int) -> float:
        """Slots read back as tokens: what one layer of one head holds. A mean
        because heads need not agree once a method evicts them independently."""
        return slots / (self.layers * self.kv_heads)

    @property
    def prompt_retained(self) -> float:
        return self._per_head_layer(self.prompt_slots)

    @property
    def generated_retained(self) -> float:
        return self._per_head_layer(self.generated_slots)

    @property
    def total_retained(self) -> float:
        return self._per_head_layer(self.total_slots)


class CacheLedger:
    """Follows the original sequence index of every position a method keeps.

    Methods publish the indices they gathered with (`KVMethod.last_indices`,
    relative to the cache they were handed). Composing those across decode steps
    recovers each surviving position's index in the original sequence, which is
    what separates prompt KV from generated KV. Methods hold no provenance state
    of their own, so this works for a scored evictor and a random one alike.
    """

    def __init__(self) -> None:
        self.prompt_length = 0
        self.generated_length = 0
        self._positions: list = []

    def start(self, past_key_values) -> None:
        """Called on the prefilled cache, before the method touches it."""
        import torch

        self.prompt_length = past_key_values[0][0].shape[2]
        self.generated_length = 0
        self._positions = [
            torch.arange(key.shape[2], device=key.device)
            .view(1, 1, -1).expand(key.shape[0], key.shape[1], key.shape[2]).clone()
            for key, _ in past_key_values
        ]

    def append(self) -> None:
        """One decode step appended one position to every layer."""
        import torch

        position = self.prompt_length + self.generated_length
        self.generated_length += 1
        self._positions = [
            torch.cat([tracked, torch.full((*tracked.shape[:-1], 1), position,
                                           dtype=tracked.dtype, device=tracked.device)], dim=-1)
            for tracked in self._positions
        ]

    def compact(self, method, past_key_values) -> None:
        """Apply whatever the method just did, and refuse to guess if it did
        something it did not publish."""
        indices = method.last_indices
        if indices is not None:
            self._positions = [tracked if idx is None else tracked.gather(-1, idx.to(tracked.device))
                               for tracked, idx in zip(self._positions, indices)]
        for layer, ((key, _), tracked) in enumerate(zip(past_key_values, self._positions)):
            if key.shape[2] != tracked.shape[-1]:
                raise RuntimeError(
                    f"{type(method).__name__} layer {layer} holds {key.shape[2]} cache positions "
                    f"but accounts for {tracked.shape[-1]}: a method that reshapes the cache must "
                    "set last_indices, otherwise its provenance cannot be followed")

    def retained_positions(self, layer: int = 0, head: int = 0, batch: int = 0) -> list[int]:
        """Original sequence indices still in the cache, in cache order. The
        prompt/generated split is a summary of this; which *part* of the prompt a
        method kept is the question it is here to let someone ask next."""
        return [int(p) for p in self._positions[layer][batch, head].tolist()]

    def account(self, cfg: ModelConfig, dtype: str = "fp16") -> KVAccount:
        heads = self._positions[0].shape[1]
        if (len(self._positions), heads) != (cfg.layers, cfg.n_kv_heads):
            raise ValueError(
                f"tracked a {len(self._positions)}-layer, {heads}-head cache but was handed the "
                f"config for {cfg.name} ({cfg.layers} layers, {cfg.n_kv_heads} KV heads); the "
                "byte figures would be for a different model than the one that ran")
        prompt_slots = sum(int((tracked < self.prompt_length).sum()) for tracked in self._positions)
        total_slots = sum(int(tracked.numel()) for tracked in self._positions)
        generated_slots = total_slots - prompt_slots
        generated_sum = sum(float(tracked[tracked >= self.prompt_length].sum())
                            for tracked in self._positions)
        batch = self._positions[0].shape[0]
        return KVAccount(
            model=cfg.name, layers=cfg.layers, kv_heads=cfg.n_kv_heads, head_dim=cfg.head_dim,
            dtype=dtype, prompt_length=self.prompt_length, generated_length=self.generated_length,
            prompt_slots=prompt_slots // batch,
            generated_slots=generated_slots // batch,
            generated_position_mean=generated_sum / generated_slots if generated_slots else 0.0)
