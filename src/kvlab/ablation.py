"""A diagnostic that asks whether the generated trace is load-bearing at all.

The multistep workload was described as one whose later operands "exist only in the
generated KV". That claim was wrong and is withdrawn. Every fact and every operation
is stated in the prompt, so a model with its prompt cache intact can in principle
recompute the answer from scratch at any point. The trace is a *shortcut*, not a
private store, and whether a model actually leans on it is an empirical question that
no amount of task design can settle by construction: any task deterministic enough to
score automatically is recomputable from its own prompt.

So it is measured instead. TraceAblation keeps the whole prompt and removes generated
KV, choosing what to remove by what the token was:

- **ablate-numeric** removes the positions holding digits first. In a multistep trace
  those are the running totals the next step would consume.
- **ablate-other** removes the positions holding everything else first.

Both arms keep the prompt, hold the same generation budget, and therefore end at the
same retained KV. They differ only in which part of the trace goes. If the numeric arm
degrades and the other does not, the model was reading its own intermediate results
out of the cache, and results about generated-cache compression on this workload mean
something. If they degrade alike, the trace was not carrying the computation and the
workload needs redesigning before its generated axis is worth reporting.

**The arms only separate while the targeted class lasts.** If a budget removes more
positions than the target class contains, the arm spills into the other class and both
arms end up removing mostly the same positions, at which case a null between them
means nothing. A distilgpt2 trace of 199 tokens held 8 numeric positions, so a removal
of 117 was not a test of anything. kvlab.analysis flags this per cell from the two
counts each arm records; run the intervention at mild compression, where the removal
is smaller than the numeric class.

This is not a cache-compression method and must never be reported as one. It is also
not a perfectly matched control: numeric and non-numeric positions differ in more than
their class -- numbers cluster at step boundaries, so the two arms remove from
different parts of the trace -- which is why every row records the mean original
position of what survived, and why the two counts each arm actually removed are
reported rather than assumed. Nothing here reads or scores hidden reasoning: it
operates on the model's own visible output tokens, which are the KV under study.
"""

from __future__ import annotations

import logging

from .methods import KVMethod, _gather_tokens

logger = logging.getLogger(__name__)

NUMERIC, OTHER = "numeric", "other"
TARGETS = (NUMERIC, OTHER)


def numeric_token_ids(tokenizer) -> frozenset[int]:
    """Vocabulary ids whose token contains a digit. Read off the vocabulary directly
    rather than by decoding every id, which is the same answer far faster."""
    vocabulary = tokenizer.get_vocab()
    return frozenset(index for token, index in vocabulary.items()
                     if any(character.isdigit() for character in token))


class TraceAblation(KVMethod):
    """Keeps the prompt, removes generated KV by what the token was. See the module
    docstring: a diagnostic, not a method."""

    family, lever, phase, bits = "Diagnostic", "context", "decoding", 16

    def __init__(self, *, generation_budget: int, target: str, numeric_tokens: frozenset[int],
                 seed: int = 0):
        if target not in TARGETS:
            raise ValueError(f"target must be one of {TARGETS}, got {target!r}")
        self.key = f"ablate-{target}"
        self.name = f"Trace ablation ({target} first)"
        self.target = target
        self.generation_budget = generation_budget
        self.numeric_tokens = numeric_tokens
        self.seed = seed
        self.prompt_length: int | None = None
        self.removed_targeted = 0
        self.removed_other = 0
        self._numeric: list[bool] = []
        self._generator = None

    def apply(self, past_key_values, attentions):
        import torch

        self.prompt_length = past_key_values[0][0].shape[2]
        self._generator = torch.Generator().manual_seed(self.seed)
        self._numeric = []
        self.removed_targeted = self.removed_other = 0
        self.last_indices = None
        return past_key_values

    def observe(self, token_id: int) -> None:
        """One generated token has just entered the cache. Its class is what decides
        whether this method will reach for it."""
        self._numeric.append(int(token_id) in self.numeric_tokens)

    def step(self, past_key_values, attentions):
        import torch

        if self.prompt_length is None:
            raise RuntimeError("apply() must run on the prefilled cache before step()")
        prompt = self.prompt_length
        generated = past_key_values[0][0].shape[2] - prompt
        if generated != len(self._numeric):
            raise RuntimeError(
                f"{generated} generated positions in the cache but {len(self._numeric)} "
                "observed tokens; the decode loop must call observe() for every token it "
                "appends or the classes no longer line up with the cache")
        if generated <= self.generation_budget:
            self.last_indices = None
            return past_key_values

        wanted = self.target == NUMERIC
        targeted = [i for i, numeric in enumerate(self._numeric) if numeric == wanted]
        other = [i for i, numeric in enumerate(self._numeric) if numeric != wanted]
        removals = generated - self.generation_budget
        # Uniform inside a class, so the two arms differ in which class they reach for
        # and not in how they pick within it.
        from_targeted = min(removals, len(targeted))
        dropped = set(self._sample(targeted, from_targeted, torch))
        dropped |= set(self._sample(other, removals - from_targeted, torch))
        self.removed_targeted += from_targeted
        self.removed_other += removals - from_targeted

        keep = [i for i in range(generated) if i not in dropped]
        self._numeric = [self._numeric[i] for i in keep]
        positions = list(range(prompt)) + [prompt + i for i in keep]

        index = torch.tensor(positions)
        out, indices = [], []
        for key, value in past_key_values:
            batch, heads = key.shape[:2]
            expanded = index.view(1, 1, -1).expand(batch, heads, len(positions)).to(key.device)
            out.append((_gather_tokens(key, expanded), _gather_tokens(value, expanded)))
            indices.append(expanded)
        self.last_indices = tuple(indices)
        return tuple(out)

    def _sample(self, pool: list[int], count: int, torch) -> list[int]:
        if count <= 0:
            return []
        chosen = torch.randperm(len(pool), generator=self._generator)[:count]
        return [pool[int(i)] for i in chosen]

    def kept_len(self, orig_len: int) -> int:
        if self.prompt_length is None:
            return orig_len
        generated = max(0, orig_len - self.prompt_length)
        return self.prompt_length + min(self.generation_budget, generated)
