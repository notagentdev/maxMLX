"""Decode-path kernels against the computations they replace.

router_gate (bf16 matvec in fp32), moe_decode (a routed SwitchGLU row in two
launches, 3- and 4-bit), decode_attention (split-K over a main buffer plus a
tail, with sinks and padded GQA groups) and TailKVCache (decode writes in a
tail, folded for everything else) must agree with the stock MLX paths; tiny
Flint and Qwen3-Next models must decode the same logits with the tail path
forced on.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="the kernels need MLX")
pytest.importorskip("mlx_lm")
if not mx.metal.is_available():
    pytest.skip("Metal kernels", allow_module_level=True)

import mlx.nn as nn  # noqa: E402
from mlx_lm.models.cache import KVCache, make_prompt_cache  # noqa: E402
from mlx_lm.models.switch_layers import SwitchGLU  # noqa: E402

from maxmlx.mlx_backend import (  # noqa: E402
    archs,
    decode_attention,
    decode_fusion,
    moe_decode,
    router_gate,
)

archs.register_all()


def test_router_gate_matches_fp32_matvec():
    mx.random.seed(0)
    gate = nn.Linear(2048, 256, bias=False)
    gate.weight = gate.weight.astype(mx.bfloat16)
    x = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)
    ref = gate(x.astype(mx.float32))
    got = router_gate.gate_logits(x, gate.weight)
    assert got.shape == ref.shape and got.dtype == mx.float32
    assert mx.abs(got - ref).max().item() < 1e-5


@pytest.mark.parametrize("bits", [3, 4])
def test_moe_decode_matches_switch_glu(bits):
    mx.random.seed(1)
    switch = SwitchGLU(2048, 512, 8)
    nn.quantize(switch, group_size=64, bits=bits)
    switch.update(nn.utils.tree_map(
        lambda a: a.astype(mx.bfloat16) if a.dtype == mx.float32 else a, switch.parameters()
    ))
    assert moe_decode.fits(switch)
    x = mx.random.normal((1, 1, 2048)).astype(mx.bfloat16)
    inds = mx.array([[[5, 0, 7, 2, 3, 6]]], dtype=mx.uint32)
    ref = switch(x, inds).astype(mx.float32)
    got = moe_decode.experts(switch, x, inds).astype(mx.float32)
    assert got.shape == ref.shape
    assert mx.abs(got - ref).max().item() <= 2e-2 * mx.abs(ref).max().item()


@pytest.mark.parametrize("hq,hk,d", [(32, 4, 128), (12, 2, 64), (8, 8, 128), (16, 2, 256)])
@pytest.mark.parametrize("tail", [0, 1, 37])
def test_decode_attention_matches_sdpa(hq, hk, d, tail):
    mx.random.seed(2)
    T = 3000
    q = mx.random.normal((1, hq, 1, d)).astype(mx.bfloat16)
    k = mx.random.normal((1, hk, T + tail, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, hk, T + tail, d)).astype(mx.bfloat16)
    sinks = mx.random.normal((hq,)).astype(mx.bfloat16)
    ref = mx.fast.scaled_dot_product_attention(
        q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32),
        scale=d**-0.5, sinks=sinks.astype(mx.float32),
    )
    assert decode_attention.supported(q, k, None)
    got = decode_attention.decode_attention(
        q, k[..., :T, :], v[..., :T, :], scale=d**-0.5, sinks=sinks,
        tail=(k[..., T:, :], v[..., T:, :]) if tail else None,
    )
    assert mx.abs(got.astype(mx.float32) - ref).max().item() < 2e-2


def test_tail_cache_is_a_kv_cache_outside_decode():
    mx.random.seed(3)
    tail, ref = decode_attention.TailKVCache(), KVCache()
    tail.TAIL = 4  # fold often
    prompt = mx.random.normal((1, 2, 10, 16))
    tail.update_and_fetch(prompt, prompt * 2)
    ref.update_and_fetch(prompt, prompt * 2)
    for i in range(11):
        kv = mx.random.normal((1, 2, 1, 16))
        mk, mv, tk, tv = tail.append_decode(kv, kv * 2)
        rk, rv = ref.update_and_fetch(kv, kv * 2)
        assert mx.array_equal(mx.concatenate([mk, tk], axis=2), rk)
        assert mx.array_equal(mx.concatenate([mv, tv], axis=2), rv)
        assert tail.offset == ref.offset
    assert tail.trim(3) == ref.trim(3)
    window = mx.random.normal((1, 2, 3, 16))
    got_k, _ = tail.update_and_fetch(window, window)
    ref_k, _ = ref.update_and_fetch(window, window)
    assert mx.array_equal(got_k, ref_k)
    tail.append_decode(window[..., :1, :], window[..., :1, :])
    ref.update_and_fetch(window[..., :1, :], window[..., :1, :])
    for a, b in zip(tail.state, ref.state):
        assert mx.array_equal(a, b)


def _tiny_flint():
    from mlx_lm.models.flint import Model, ModelArgs

    args = ModelArgs(
        model_type="flint", hidden_size=128, num_hidden_layers=4, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32, num_experts=8,
        num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
        mlp_only_layers=[0, 1],
        layer_types=["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        sliding_window=9, rms_norm_eps=1e-6, vocab_size=256,
    )
    mx.random.seed(4)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    nn.quantize(model, group_size=64, bits=4)  # the attention fusion needs quantized projections
    mx.eval(model.parameters())
    return model


def _reference_attention(q, k, v, *, scale, sinks=None, tail=None):
    if tail is not None:
        k = mx.concatenate([k, tail[0]], axis=2)
        v = mx.concatenate([v, tail[1]], axis=2)
    return mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, sinks=sinks)


def test_tiny_flint_tail_path_is_the_stock_attention(monkeypatch):
    # The integration (tail cache, folds, path choice) is checked exactly: the
    # kernel is swapped for MLX's sdpa over the same main + tail contents, so
    # the logits must equal the stock path's bit for bit. The kernel's own
    # numerics are the tests above (a random model amplifies bf16 noise too
    # much to compare them through it).
    model = _tiny_flint()
    prompt = mx.array([[(7 * i) % 251 for i in range(40)]])
    tokens = [(11 * i + 3) % 251 for i in range(24)]

    def logits_along():
        cache = make_prompt_cache(model)
        model(prompt, cache=cache)
        rows = [model(mx.array([[t]]), cache=cache)[0, -1] for t in tokens]
        mx.eval(rows)
        return rows, cache

    try:
        decode_fusion.install(model)
        ref, _ = logits_along()
        monkeypatch.setattr(decode_attention, "MIN_KEYS", 16)
        monkeypatch.setattr(decode_attention.TailKVCache, "TAIL", 8)  # fold mid-run
        monkeypatch.setattr(decode_attention, "decode_attention", _reference_attention)
        got, cache = logits_along()
        assert any(isinstance(c, decode_attention.TailKVCache) and c._tail_n for c in cache)
    finally:
        decode_fusion.uninstall()
    for a, b in zip(ref, got):
        assert mx.array_equal(a, b)


def test_tiny_qwen3_next_tail_path_is_the_stock_attention(monkeypatch):
    # Ornith's attention (Qwen3-Next: gated output, partial RoPE), checked the
    # same way as Flint's above.
    from mlx_lm.models.qwen3_next import Model, ModelArgs

    args = ModelArgs(
        model_type="qwen3_next", hidden_size=128, num_hidden_layers=4, intermediate_size=128,
        num_attention_heads=4, linear_num_value_heads=2, linear_num_key_heads=2,
        linear_key_head_dim=32, linear_value_head_dim=32, linear_conv_kernel_dim=4,
        num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
        shared_expert_intermediate_size=64, mlp_only_layers=[], moe_intermediate_size=64,
        rms_norm_eps=1e-6, vocab_size=256, num_key_value_heads=2, rope_theta=10000.0,
        partial_rotary_factor=0.25, max_position_embeddings=512, head_dim=64,
    )
    mx.random.seed(5)
    model = Model(args)
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    prompt = mx.array([[(5 * i) % 251 for i in range(40)]])
    tokens = [(13 * i + 1) % 251 for i in range(24)]

    def logits_along():
        cache = make_prompt_cache(model)
        model(prompt, cache=cache)
        rows = [model(mx.array([[t]]), cache=cache)[0, -1] for t in tokens]
        mx.eval(rows)
        return rows, cache

    try:
        assert decode_fusion.install(model)["attn_tail"] == 1
        monkeypatch.setattr(decode_attention, "MIN_KEYS", 1 << 30)  # stock path first
        ref, _ = logits_along()
        monkeypatch.setattr(decode_attention, "MIN_KEYS", 16)
        monkeypatch.setattr(decode_attention.TailKVCache, "TAIL", 8)
        monkeypatch.setattr(decode_attention, "decode_attention", _reference_attention)
        got, cache = logits_along()
        assert any(isinstance(c, decode_attention.TailKVCache) and c._tail_n for c in cache)
    finally:
        decode_fusion.uninstall()
    for a, b in zip(ref, got):
        assert mx.array_equal(a, b)
