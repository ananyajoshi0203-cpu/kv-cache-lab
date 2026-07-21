"""Invariant tests for the reference eviction and quantization methods. Keys and
values are position-coded (token s stores the value s in every element) so the
surviving positions can be read back from the tensors themselves rather than
re-deriving them with the production scoring code. Run: `pytest tests/`."""

import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kvlab import memory                                            # noqa: E402
from kvlab.methods import (                                         # noqa: E402
    CAKE, H2O, FullCache, KIVIQuant, OBCache, SnapKV,
    _evict, _fake_quant, _key_scores, build,
)

B, H, S, D = 1, 2, 24, 4
BUDGET, RECENT = 8, 3


def position_coded_kv():
    positions = torch.arange(S, dtype=torch.float32)
    k = positions.view(1, 1, S, 1).expand(B, H, S, D).clone()
    return k, k.clone()


def kept_positions(key):
    return {int(p) for p in key[0, 0, :, 0].tolist()}


def test_evict_always_keeps_recent_window():
    k, v = position_coded_kv()
    scores = torch.arange(S, 0, -1, dtype=torch.float32).view(1, 1, S).expand(B, H, S)
    ek, _ = _evict(k, v, scores, BUDGET, RECENT)
    assert kept_positions(ek) >= set(range(S - RECENT, S))


def test_evict_respects_budget():
    k, v = position_coded_kv()
    scores = torch.rand(B, H, S)
    ek, ev = _evict(k, v, scores, BUDGET, RECENT)
    assert ek.shape == (B, H, BUDGET, D)
    assert ev.shape == (B, H, BUDGET, D)


def test_evict_noop_within_budget():
    k, v = position_coded_kv()
    ek, ev = _evict(k, v, torch.rand(B, H, S), S, RECENT)
    assert torch.equal(ek, k) and torch.equal(ev, v)


def test_evict_keeps_top_scored_old_tokens():
    k, v = position_coded_kv()
    chosen = set(range(0, (BUDGET - RECENT) * 2, 2))
    scores = torch.zeros(B, H, S)
    scores[:, :, sorted(chosen)] = 1.0
    ek, _ = _evict(k, v, scores, BUDGET, RECENT)
    assert kept_positions(ek) == chosen | set(range(S - RECENT, S))


def test_evict_pairs_values_with_keys():
    k, v = position_coded_kv()
    ek, ev = _evict(k, v, torch.rand(B, H, S), BUDGET, RECENT)
    assert torch.equal(ek, ev)


def test_key_scores_groups_query_heads_for_gqa():
    n_kv, group = 2, 3
    queries = 5
    attn = torch.ones(B, n_kv * group, queries, S)
    scores = _key_scores(attn, n_kv)
    assert scores.shape == (B, n_kv, S)
    assert torch.allclose(scores, torch.full_like(scores, float(queries * group)))


def test_key_scores_window_ignores_earlier_queries():
    outside, inside = 2, 9
    window = 4
    attn = torch.zeros(B, H, S, S)
    attn[:, :, : S - window, outside] = 1.0
    attn[:, :, S - window:, inside] = 1.0
    scores = _key_scores(attn, H, window=window)
    assert scores.argmax(dim=-1).unique().item() == inside


def test_snapkv_pooling_keeps_clusters_together():
    k, v = position_coded_kv()
    spike = 8
    pool = 3
    attn = torch.zeros(B, H, S, S)
    attn[:, :, -RECENT:, spike] = 1.0
    budget = RECENT + pool
    (ok_pooled, _), = SnapKV(budget=budget, window=RECENT, pool=pool).apply(((k.clone(), v.clone()),), (attn,))
    assert kept_positions(ok_pooled) >= set(range(spike - pool // 2, spike + pool // 2 + 1))
    (ok_unpooled, _), = SnapKV(budget=budget, window=RECENT, pool=1).apply(((k.clone(), v.clone()),), (attn,))
    assert spike in kept_positions(ok_unpooled)


def test_methods_report_the_len_they_produce():
    k, v = position_coded_kv()
    attns = tuple(torch.rand(B, H, S, S) for _ in range(2))
    pkv = tuple((k.clone(), v.clone()) for _ in range(2))
    for method in (H2O(budget=BUDGET, recent=RECENT), SnapKV(budget=BUDGET, window=RECENT), FullCache()):
        out = method.apply(tuple((k.clone(), v.clone()) for k, v in pkv), attns)
        assert all(ok.shape[2] == method.kept_len(S) for ok, _ in out)


def _uniform_attention():
    return torch.full((B, H, S, S), 1.0 / S)


def _one_hot_attention(target):
    attn = torch.zeros(B, H, S, S)
    attn[:, :, :, target] = 1.0
    return attn


def test_cake_layer_budgets_sum_to_global_budget():
    torch.manual_seed(3)
    n_layers = 3
    k, v = position_coded_kv()
    pkv = tuple((k.clone(), v.clone()) for _ in range(n_layers))
    attns = tuple(torch.rand(B, H, S, S).softmax(dim=-1) for _ in range(n_layers))
    method = CAKE(budget=BUDGET, window=RECENT)
    method.apply(pkv, attns)
    assert sum(method._layer_budgets) == BUDGET * n_layers


def test_cake_gives_dispersed_layers_more_budget():
    k, v = position_coded_kv()
    pkv = tuple((k.clone(), v.clone()) for _ in range(2))
    dispersed = _uniform_attention()
    dispersed[:, :, ::2, 0] += 0.01
    focused = _one_hot_attention(target=3)
    method = CAKE(budget=BUDGET, window=RECENT)
    method.apply(pkv, (dispersed, focused))
    dispersed_budget, focused_budget = method._layer_budgets
    assert dispersed_budget > focused_budget


def test_cake_every_layer_keeps_its_window():
    k, v = position_coded_kv()
    pkv = tuple((k.clone(), v.clone()) for _ in range(2))
    out = CAKE(budget=BUDGET, window=RECENT).apply(pkv, (_uniform_attention(), _one_hot_attention(3)))
    for ok, _ in out:
        assert kept_positions(ok) >= set(range(S - RECENT, S))


def test_cake_bytes_equal_flat_budget_when_layers_identical():
    cfg = memory.MODELS["distilgpt2"]
    k, v = position_coded_kv()
    pkv = tuple((k.clone(), v.clone()) for _ in range(cfg.layers))
    attns = tuple(_uniform_attention() for _ in range(cfg.layers))
    method = CAKE(budget=BUDGET, window=RECENT)
    method.apply(pkv, attns)
    assert method.kv_bytes(S, cfg) == memory.cache_bytes(cfg, BUDGET, "fp16")


def test_obcache_value_norm_breaks_attention_ties():
    k, v = position_coded_kv()
    big_value, small_value = 4, 5
    v = torch.ones(B, H, S, D)
    v[:, :, big_value] = 10.0
    out = OBCache(budget=RECENT + 1, recent=RECENT).apply(((k, v),), (_uniform_attention(),))
    (ok, _), = out
    assert kept_positions(ok) >= {big_value} | set(range(S - RECENT, S))
    assert small_value not in kept_positions(ok)


def test_obcache_matches_h2o_when_value_norms_equal():
    k, v = position_coded_kv()
    v = torch.ones(B, H, S, D)
    attn = torch.rand(B, H, S, S).softmax(dim=-1)
    (ok_ob, _), = OBCache(budget=BUDGET, recent=RECENT).apply(((k.clone(), v.clone()),), (attn,))
    (ok_h2o, _), = H2O(budget=BUDGET, recent=RECENT).apply(((k.clone(), v.clone()),), (attn,))
    assert kept_positions(ok_ob) == kept_positions(ok_h2o)


def test_kivi_residual_untouched():
    torch.manual_seed(0)
    k, v = torch.randn(B, H, S, D), torch.randn(B, H, S, D)
    residual = 5
    (qk, qv), = KIVIQuant(bits=2, residual=residual).apply(((k, v),), attentions=None)
    assert torch.equal(qk[:, :, S - residual:], k[:, :, S - residual:])
    assert torch.equal(qv[:, :, S - residual:], v[:, :, S - residual:])


def test_fake_quant_16bit_identity():
    torch.manual_seed(1)
    x = torch.randn(B, H, S, D)
    assert torch.equal(_fake_quant(x, 16, 2), x)


def test_fake_quant_error_bounded_by_step_size():
    torch.manual_seed(2)
    bits, reduce_dim = 2, 2
    x = torch.randn(B, H, S, D)
    q = _fake_quant(x, bits, reduce_dim)
    step = (x.amax(dim=reduce_dim, keepdim=True) - x.amin(dim=reduce_dim, keepdim=True)) / (2 ** bits - 1)
    assert ((q - x).abs() <= step / 2 + 1e-6).all()


def test_full_cache_bytes_match_cost_model():
    cfg = memory.MODELS["distilgpt2"]
    assert FullCache().kv_bytes(S, cfg) == memory.cache_bytes(cfg, S, "fp16")


def test_kivi_bytes_between_pure_2bit_and_fp16():
    cfg = memory.MODELS["distilgpt2"]
    kivi = KIVIQuant(bits=2, residual=5).kv_bytes(S, cfg)
    assert memory.cache_bytes(cfg, S, "int2") < kivi < memory.cache_bytes(cfg, S, "fp16")


def test_build_raises_for_catalogued_but_unimplemented():
    try:
        build("ada-kv")
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected NotImplementedError for ada-kv")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all method tests passed")
