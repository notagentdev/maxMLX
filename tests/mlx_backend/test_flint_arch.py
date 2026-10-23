"""The vendored Flint architecture and what MaxToken layers on top of it.

A tiny random Flint (two dense layers, two MoE layers, full + sliding-window
attention with a 9-token window, attention sinks) stands in for the 40B: it
must be what mlx-lm resolves for model_type "flint", its fused router must pick
the reference experts and weights, prefill pacing must not change the logits
(the window is narrower than the prompt, so the sliding mask matters), and the
compiled MoE decode must agree with the stock forward.
"""

import pytest

mx = pytest.importorskip("mlx.core", reason="Flint needs MLX")
pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

from maxtoken.mlx_backend import archs, decode_fusion, prefill_pacing  # noqa: E402

archs.register_all()


def _tiny_model():
    from mlx_lm.models.flint import Model, ModelArgs

    layer_types = ["full_attention", "sliding_attention", "sliding_attention", "full_attention"]
    args = ModelArgs(
        model_type="flint",
        hidden_size=64,
        num_hidden_layers=4,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        mlp_only_layers=[0, 1],
        layer_types=layer_types,
        sliding_window=9,
        rms_norm_eps=1e-6,
        vocab_size=256,
    )
    mx.random.seed(0)
    model = Model(args)
    for layer in model.model.layers:
        attn = layer.self_attn
        attn.softmax_offset = mx.random.normal(attn.softmax_offset.shape)
    mx.eval(model.parameters())
    return model


def test_mlx_lm_resolves_flint_to_the_vendored_model():
    from mlx_lm.utils import _get_classes

    from maxtoken.mlx_backend.archs import flint

    model_cls, args_cls = _get_classes({"model_type": "flint"})
    assert model_cls is flint.Model and args_cls is flint.ModelArgs


def test_fused_router_matches_the_reference_router():
    from maxtoken.mlx_backend.archs.kernels import expert_routing

    mx.random.seed(1)
    logits = mx.random.normal((16, 256)) * 3
    inds, weights = expert_routing(
        logits, mx.zeros((256,)), top_k=6, multiplier=1.0, score_eps=0.0, route_scale=1.0
    )
    scores = mx.sigmoid(logits)
    ref_inds = mx.argpartition(-scores, kth=5, axis=-1)[..., :6]
    ref_w = mx.take_along_axis(scores, ref_inds, axis=-1)
    ref_w = ref_w / ref_w.sum(axis=-1, keepdims=True)
    for row in range(16):
        got = dict(zip(inds[row].tolist(), weights[row].tolist()))
        want = dict(zip(ref_inds[row].tolist(), ref_w[row].tolist()))
        assert got.keys() == want.keys()
        for expert, w in want.items():
            assert abs(got[expert] - w) < 1e-6


def test_paced_prefill_matches_unpaced():
    model = _tiny_model()
    tokens = mx.array([[i % 251 for i in range(3, 3 + 48)]])
    ref = model(tokens, cache=make_prompt_cache(model))
    mx.eval(ref)
    try:
        assert prefill_pacing.install(model, pace=1) == 1
        cache = make_prompt_cache(model)
        out = model(tokens, cache=cache)
        mx.eval(out)
        assert mx.array_equal(ref, out)
        # The paced prefill left a cache a decode step can continue from.
        nxt = model(mx.array([[7]]), cache=cache)
        ref_cache = make_prompt_cache(model)
        prefill_pacing.uninstall()
        model(tokens, cache=ref_cache)
        ref_nxt = model(mx.array([[7]]), cache=ref_cache)
        assert mx.allclose(nxt, ref_nxt, atol=1e-5)
    finally:
        prefill_pacing.uninstall()


def test_compiled_moe_decode_matches_stock():
    model = _tiny_model()
    prompt = mx.array([[5, 9, 13, 17, 21, 25, 29, 33, 37, 41, 45, 49]])
    cache = make_prompt_cache(model)
    model(prompt, cache=cache)
    ref = model(mx.array([[3]]), cache=cache)
    mx.eval(ref)
    try:
        report = decode_fusion.install(model)
        assert report["moe_compiled"] == 2
        cache = make_prompt_cache(model)
        model(prompt, cache=cache)
        out = model(mx.array([[3]]), cache=cache)
        assert mx.allclose(ref, out, atol=1e-4)
    finally:
        decode_fusion.uninstall()
