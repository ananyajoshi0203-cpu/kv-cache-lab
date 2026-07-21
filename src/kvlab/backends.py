"""Backends that actually compress a model's KV cache.

Two are provided. ReferenceBackend runs the legible implementations in
kvlab.methods and is the default for teaching and CPU experiments. KVPressBackend
wraps NVIDIA KVPress (https://github.com/NVIDIA/kvpress), which supplies 20+
peer-reviewed presses behind a uniform compression_ratio interface and is the
path for real long-context runs. Both expose the same generate() signature so a
benchmark can swap them without changing the eval."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


# KVPress presses keyed by the registry method they implement. Extend as needed.
KVPRESS_PRESSES = {
    "h2o": "ObservedAttentionPress",
    "snapkv": "SnapKVPress",
    "ada-kv": "AdaKVPress",
    "kvzip": "KVzipPress",
    "streaming": "StreamingLLMPress",
    "tova": "TOVAPress",
    "knorm": "KnormPress",
    "expected_attention": "ExpectedAttentionPress",
}


class KVPressBackend:
    """Compress and generate with a KVPress press. Requires `pip install kvpress`
    and a GPU-class model such as meta-llama/Llama-3.1-8B-Instruct."""

    def __init__(self, model="meta-llama/Llama-3.1-8B-Instruct", press="expected_attention", device_map="auto"):
        import kvpress
        from transformers import pipeline

        press_name = KVPRESS_PRESSES.get(press, press)
        if not hasattr(kvpress, press_name):
            raise ValueError(f"KVPress has no press '{press_name}'")
        self.press_cls = getattr(kvpress, press_name)
        self.press_name = press_name
        self.pipe = pipeline("kv-press-text-generation", model=model, device_map=device_map, dtype="auto")
        logger.info("KVPress backend: model=%s press=%s", model, press_name)

    def generate(self, context: str, question: str = "", ratio: float = 0.5) -> str:
        press = self.press_cls(compression_ratio=ratio)
        return self.pipe(context, question=question, press=press)["answer"]


class KVPressQuantBackend:
    """Quantized KV cache via transformers QuantizedCache, driven through the
    KVPress pipeline. Requires `pip install optimum-quanto`."""

    def __init__(self, model="meta-llama/Llama-3.1-8B-Instruct", nbits=4, device_map="auto"):
        from transformers import pipeline
        self.nbits = nbits
        self.pipe = pipeline("kv-press-text-generation", model=model, device_map=device_map, dtype="auto")

    def generate(self, context: str, question: str = "", ratio: float = 0.0) -> str:
        from transformers import QuantizedCache
        cache = QuantizedCache(backend="quanto", nbits=self.nbits)
        return self.pipe(context, question=question, cache=cache)["answer"]


class ReferenceBackend:
    """The in-repo reference methods on a small model. Runs on CPU; for learning
    and for checking that a press behaves as expected before scaling up."""

    def __init__(self, model="distilgpt2", device="cpu", method_key="snapkv"):
        from .model import load_model
        self.method_key = method_key
        self.model, self.tok, self.cfg = load_model(model, device)

    def generate(self, context: str, question: str = "", ratio: float = 0.5, max_new_tokens=64) -> str:
        from .benchmark import derive_kwargs
        from .decode import generate_stepwise
        from .methods import build

        ids = self.tok(context + question, return_tensors="pt").input_ids
        method = build(self.method_key,
                       **derive_kwargs(self.method_key, ids.shape[1], ratio, recent=32, cfg=self.cfg))
        gen = generate_stepwise(self.model, ids, method, max_new_tokens=max_new_tokens,
                                eos_token_id=self.tok.eos_token_id)
        return self.tok.decode(gen[0], skip_special_tokens=True)
