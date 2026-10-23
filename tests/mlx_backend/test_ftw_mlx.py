"""FTW-MLX zero-copy expert store: repack roundtrip for both disk layouts.

Needs mlx (the store imports mapped tensors via DLPack); the repack itself is
exercised through a crafted checkpoint in tmp_path, with MAXMLX_MLX_FTW_DIR
pointing the cache there too.
"""

import json
import os

import numpy as np
import pytest

mlx = pytest.importorskip("mlx.core", reason="mapped store needs mlx")

from maxmlx.mlx_backend.ftw_mlx import (  # noqa: E402 -- after importorskip
    MappedExpertStore,
    _PAGE,
    repack_experts,
)

# tests/ is not a package; load the sibling module's helpers by path.
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "_offload_helpers", os.path.join(os.path.dirname(__file__), "test_offload.py")
)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
_fake_glu_tensors = _helpers._fake_glu_tensors
write_safetensors = _helpers.write_safetensors


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "classic"])
@pytest.mark.parametrize("stacked", [True, False], ids=["stacked", "per-expert"])
def test_repack_and_mapped_views_roundtrip(tmp_path, monkeypatch, stacked, packed):
    monkeypatch.setenv("MAXMLX_MLX_FTW_DIR", str(tmp_path / "ftw-cache"))
    monkeypatch.setenv("MAXMLX_MLX_FTW_PACK", "1" if packed else "0")
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    rng = np.random.default_rng(1)
    tensors = _fake_glu_tensors(rng, num_experts=5, stacked=stacked, prefix="model.mlp")
    write_safetensors(model_dir / "m.safetensors", tensors)

    out_dir = repack_experts(str(model_dir), ["model.mlp.switch_mlp"])
    manifest = json.load(open(os.path.join(out_dir, "manifest.json")))
    for meta in manifest["tensors"].values():
        assert meta["offset"] % _PAGE == 0  # the whole point of the repack

    store = MappedExpertStore(out_dir)
    params = store.glu_params("model.mlp.switch_mlp")

    def stacked_source(proj, part):
        if stacked:
            return tensors[f"model.mlp.switch_mlp.{proj}.{part}"]
        return np.stack(
            [tensors[f"model.mlp.experts.{e}.{proj}.{part}"] for e in range(5)]
        )

    if packed:
        # gate and up interleaved PER EXPERT into one tensor; the classic
        # per-proj entries for gate/up must be gone.
        assert "gate_proj" not in params and "up_proj" not in params
        got = np.array(params["gate_up_proj"]["weight"])
        g = stacked_source("gate_proj", "weight")
        u = stacked_source("up_proj", "weight")
        want = np.stack(
            [np.concatenate([g[e], u[e]], axis=0) for e in range(5)]
        )
        assert got.shape == want.shape
        assert got.tolist() == want.tolist()
    else:
        got = np.array(params["gate_proj"]["weight"])
        want = stacked_source("gate_proj", "weight")
        assert got.shape == want.shape
        assert got.tolist() == want.tolist()
    down = np.array(params["down_proj"]["weight"])
    assert down.tolist() == stacked_source("down_proj", "weight").tolist()

    # idempotent: a second call must reuse the manifest, not rewrite the store
    mtime = os.path.getmtime(os.path.join(out_dir, "experts.ftwm"))
    assert repack_experts(str(model_dir), ["model.mlp.switch_mlp"]) == out_dir
    assert os.path.getmtime(os.path.join(out_dir, "experts.ftwm")) == mtime


def _store_dir(tmp_path, shape, dtype="U32"):
    """A one-tensor store written by hand: ``t`` at offset 0."""
    data = np.arange(int(np.prod(shape)), dtype=np.uint32).reshape(shape)
    (tmp_path / "experts.ftwm").write_bytes(data.tobytes())
    (tmp_path / "manifest.json").write_text(json.dumps(
        {"version": 2, "source": {}, "tensors": {"t": {"offset": 0, "shape": list(shape), "dtype": dtype}}}
    ))
    return data


def test_expert_views_are_one_page_aligned_array_per_expert(tmp_path):
    data = _store_dir(tmp_path, (4, 4096))  # 16 KiB per expert
    store = MappedExpertStore(str(tmp_path), per_expert=True)
    assert store.tensors == {}  # no stacked view may reach a command buffer
    views = store.expert_views("t")
    assert len(views) == 4
    for e, v in enumerate(views):
        assert np.array(v).tolist() == data[e].tolist()


def test_expert_views_refuse_experts_that_straddle_pages(tmp_path):
    _store_dir(tmp_path, (4, 1000))
    with pytest.raises(ValueError, match="page-aligned"):
        MappedExpertStore(str(tmp_path), per_expert=True).expert_views("t")


def test_per_expert_switch_glu_matches_the_stacked_gather():
    """Routing through a stack of only the selected experts gives the same
    numbers as gathering over the whole stacked layer, on the unsorted (decode)
    and the sorted (prefill) path alike."""
    import mlx.core as mx
    from mlx_lm.models.switch_layers import SwiGLU

    from maxmlx.mlx_backend.ftw_mlx import _ExpertViews, _per_expert_switch_glu
    from maxmlx.mlx_backend.moe_pack import classes

    E, D, H, K = 16, 128, 64, 4
    mx.random.seed(0)
    q = {"group_size": 64, "bits": 4, "mode": "affine"}
    gu = mx.quantize(mx.random.normal((E, 2 * H, D)) * 0.05, **{k: q[k] for k in ("group_size", "bits")})
    dn = mx.quantize(mx.random.normal((E, D, H)) * 0.05, **{k: q[k] for k in ("group_size", "bits")})

    Proj, Packed, _ = classes()
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear

    down = QuantizedSwitchLinear(H, D, E, bias=False, group_size=64, bits=4)
    down.weight, down.scales, down.biases = dn
    stacked = Packed(Proj(*gu, group_size=64, bits=4, mode="affine"), down, SwiGLU(), H)
    per = _per_expert_switch_glu()(
        _ExpertViews([[t[e] for e in range(E)] for t in gu], [[t[e] for e in range(E)] for t in dn]),
        SwiGLU(), H, q, q,
    )
    assert per.per_expert
    for tokens in (1, 3, 40):  # 40 x top-4 = 160 pairs takes the sorted path
        x = mx.random.normal((1, tokens, D)).astype(mx.bfloat16)
        idx = mx.random.randint(0, E, (1, tokens, K)).astype(mx.uint32)
        want, got = stacked(x, idx), per(x, idx)
        assert got.shape == want.shape
        assert mx.allclose(got, want, atol=1e-2, rtol=1e-2).item(), tokens
