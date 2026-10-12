"""Qwen3.8-Flash-Next's hyper-connection mix (Qwen4ExpGatedResidual) for one
decode row in two Metal kernels instead of ~12 MLX ops.

The stock module, for h = 4 streams x 2560 (bf16):

    normed = rms_norm per stream (fp32 math, weight centred at zero) -> bf16
    mix    = sigmoid(up(silu(down(normed) / 4)))          down 10240->320, up 320->10240
    mixed  = mean over the 4 streams of mix * normed       -> 2560
    inject = 2 * sigmoid(inject(normed) / 4)               10240->4  (not on the final mixer)

all three projections 4-bit affine, group 32, bf16 scales and biases. A decode
token runs it 97 times (twice per layer, once for the final mixer), each time
as a chain of norms, three quantized matmuls and the elementwise glue.

    kernel 1  every threadgroup reduces the 4 stream norms itself (10240 bf16
              reads from cache), then one simdgroup per output row computes the
              down rows (silu applied) and the inject rows (gate applied)
    kernel 2  one thread per column: the 4 up rows of that column, sigmoid,
              times the normed input, mean over the streams

Intermediate values are rounded to bf16 where the stock path rounds them.
"""
from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx

HC, D = 4, 2560
H = HC * D            # 10240
R = 320               # low rank
WORDS = H // 8        # uint32 per down/inject row
GROUPS = H // 32
UP_WORDS = R // 8     # 40
UP_GROUPS = R // 32   # 10
SG_PER_TG = 8

_K1_SRC = f"""
    const uint tid = thread_position_in_threadgroup.x;
    const uint tg = threadgroup_position_in_grid.x;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    threadgroup float part[{SG_PER_TG}][{HC}];
    threadgroup float rs[{HC}];
    threadgroup bfloat16_t nrm[{H}];

    float ss[{HC}] = {{0.0f, 0.0f, 0.0f, 0.0f}};
    for (uint i = tid; i < {H}; i += {SG_PER_TG * 32}) {{
        float v = float(x[i]);
        ss[i / {D}] += v * v;
    }}
    for (uint s = 0; s < {HC}; s++) {{
        float t = simd_sum(ss[s]);
        if (lane == 0) part[sg][s] = t;
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < {HC}) {{
        float t = 0.0f;
        for (uint k = 0; k < {SG_PER_TG}; k++) t += part[k][tid];
        rs[tid] = metal::rsqrt(t / {D}.0f + eps[0]);
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // the normed input, once per threadgroup, rounded to bf16 as the stock norm's output is
    for (uint i = tid; i < {H}; i += {SG_PER_TG * 32}) {{
        nrm[i] = bfloat16_t(float(x[i]) * rs[i / {D}] * (1.0f + float(nw[i])));
    }}
    if (tg == 0 && tid < {HC}) rs_out[tid] = rs[tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint row = tg * {SG_PER_TG} + sg;
    if (row >= n_rows[0]) return;
    const bool is_down = row < {R};
    const uint r = is_down ? row : row - {R};
    device const uint32_t* W = is_down ? wd + r * {WORDS} : wi + r * {WORDS};
    device const bfloat16_t* S = is_down ? sd + r * {GROUPS} : si + r * {GROUPS};
    device const bfloat16_t* B = is_down ? bd + r * {GROUPS} : bi + r * {GROUPS};

    // lane handles whole 32-element groups: group g = 4 words
    float acc = 0.0f;
    for (uint g = lane; g < {GROUPS}; g += 32) {{
        float qx = 0.0f, sx = 0.0f;
        for (uint w = 0; w < 4; w++) {{
            const uint32_t packed = W[g * 4 + w];
            const uint base = g * 32 + w * 8;
            for (uint k = 0; k < 8; k++) {{
                const float nv = float(nrm[base + k]);
                qx += float((packed >> (4 * k)) & 15u) * nv;
                sx += nv;
            }}
        }}
        acc += float(S[g]) * qx + float(B[g]) * sx;
    }}
    acc = simd_sum(acc);
    if (lane == 0) {{
        float d = float(bfloat16_t(float(bfloat16_t(acc)) * 0.25f));
        if (is_down) {{
            down_out[r] = float(bfloat16_t(d / (1.0f + metal::exp(-d))));
        }} else {{
            inj_out[r] = bfloat16_t(2.0f * float(bfloat16_t(1.0f / (1.0f + metal::exp(-d)))));
        }}
    }}
"""

COLS_PER_TG = 8

_K2_SRC = f"""
    // one simdgroup per up row; a threadgroup = {COLS_PER_TG} columns x {HC} streams
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const uint tg = threadgroup_position_in_grid.x;
    threadgroup float val[{HC}][{COLS_PER_TG}];
    const uint s = sg / {COLS_PER_TG};
    const uint c = tg * {COLS_PER_TG} + (sg % {COLS_PER_TG});
    const uint row = s * {D} + c;
    device const uint32_t* W = wu + row * {UP_WORDS};
    device const bfloat16_t* S = su + row * {UP_GROUPS};
    device const bfloat16_t* B = bu + row * {UP_GROUPS};
    float acc = 0.0f;
    for (uint u = lane; u < {UP_WORDS}; u += 32) {{
        const uint32_t packed = W[u];
        float qx = 0.0f, sx = 0.0f;
        for (uint k = 0; k < 8; k++) {{
            const float v = down_in[u * 8 + k];
            qx += float((packed >> (4 * k)) & 15u) * v;
            sx += v;
        }}
        acc += float(S[u / 4]) * qx + float(B[u / 4]) * sx;
    }}
    acc = simd_sum(acc);
    if (lane == 0) {{
        const float up = float(bfloat16_t(acc));
        const float m = float(bfloat16_t(1.0f / (1.0f + metal::exp(-up))));
        const float nv = float(bfloat16_t(float(x[row]) * rs_in[s] * (1.0f + float(nw[row]))));
        val[s][sg % {COLS_PER_TG}] = float(bfloat16_t(m * nv));
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < {COLS_PER_TG}) {{
        float total = 0.0f;
        for (uint t = 0; t < {HC}; t++) total += val[t][lane];
        mixed[tg * {COLS_PER_TG} + lane] = bfloat16_t(total * 0.25f);
    }}
"""

_K1 = mx.fast.metal_kernel(
    name="niwaki_hc_down_inject",
    input_names=["x", "nw", "wd", "sd", "bd", "wi", "si", "bi", "eps", "n_rows"],
    output_names=["down_out", "inj_out", "rs_out"],
    source=_K1_SRC,
)
_K2 = mx.fast.metal_kernel(
    name="niwaki_hc_up_mix",
    input_names=["x", "nw", "wu", "su", "bu", "down_in", "rs_in"],
    output_names=["mixed"],
    source=_K2_SRC,
)

_PARAMS: Dict[int, Any] = {}


def _params(mod):
    p = _PARAMS.get(id(mod))
    if p is None:
        down, up = mod.input_mix_weight_down, mod.input_mix_weight_up
        has_inj = "block_inject_weight" in mod
        if has_inj:
            inj = mod.block_inject_weight
            wi, si, bi = inj.weight, inj.scales, inj.biases
        else:  # the final mixer: a dummy row set the kernel never reads
            wi, si, bi = down.weight[:1], down.scales[:1], down.biases[:1]
        p = (mod.hc_norm.weight, down.weight, down.scales, down.biases, wi, si, bi,
             up.weight, up.scales, up.biases, mx.array([float(mod.hc_norm.eps)], dtype=mx.float32),
             mx.array([R + (HC if has_inj else 0)], dtype=mx.uint32), has_inj)
        _PARAMS[id(mod)] = p
    return p


def supported(mod) -> bool:
    try:
        lins = [mod.input_mix_weight_down, mod.input_mix_weight_up]
        if "block_inject_weight" in mod:
            lins.append(mod.block_inject_weight)
        return (
            mod.hc_count == HC and mod.hidden_size == D
            and mod.hc_norm.group_size == D
            and all(int(getattr(l, "bits", 0)) == 4 and int(getattr(l, "group_size", 0)) == 32
                    and str(getattr(l, "mode", "affine")) == "affine" for l in lins)
            and tuple(mod.input_mix_weight_down.weight.shape) == (R, WORDS)
            and tuple(mod.input_mix_weight_up.weight.shape) == (H, UP_WORDS)
        )
    except AttributeError:
        return False


def hc_fused(mod, h):
    """The module's output for one decode row ``h`` (1, 1, 10240) bf16."""
    nw, wd, sd, bd, wi, si, bi, wu, su, bu, eps, n_rows, has_inj = _params(mod)
    x = h.reshape(H)
    n = int(R + (HC if has_inj else 0))
    tgs = (n + SG_PER_TG - 1) // SG_PER_TG
    down, inj, rs = _K1(
        inputs=[x, nw, wd, sd, bd, wi, si, bi, eps, n_rows],
        grid=(tgs * SG_PER_TG * 32, 1, 1), threadgroup=(SG_PER_TG * 32, 1, 1),
        output_shapes=[(R,), (HC,), (HC,)], output_dtypes=[mx.float32, mx.bfloat16, mx.float32],
    )
    (mixed,) = _K2(
        inputs=[x, nw, wu, su, bu, down, rs],
        grid=(D // COLS_PER_TG * HC * COLS_PER_TG * 32, 1, 1), threadgroup=(HC * COLS_PER_TG * 32, 1, 1),
        output_shapes=[(D,)], output_dtypes=[mx.bfloat16],
    )
    mixed = mixed.reshape(1, 1, D)
    if not has_inj:
        return mixed
    return mixed, h, inj.reshape(1, 1, HC)


_ORIG: Dict[str, Any] = {}


def install(model) -> int:
    """Route every supported hyper-connection's decode rows through the kernels."""
    lm = model.language_model
    mods = [m for layer in lm.model.layers for m in (layer.attn_hyper_connection, layer.mlp_hyper_connection)]
    mods.append(lm.model.hyper_connection_mixer)
    ok = {id(m) for m in mods if supported(m)}
    if not ok:
        return 0
    cls = type(mods[0])
    if "call" not in _ORIG:
        original = cls.__call__
        _ORIG["call"], _ORIG["cls"] = original, cls

        def call(self, hyper_input):
            if (id(self) in _ENABLED and hyper_input.ndim == 3 and hyper_input.shape[:2] == (1, 1)
                    and hyper_input.dtype == mx.bfloat16):
                return hc_fused(self, hyper_input)
            return original(self, hyper_input)

        cls.__call__ = call
    _ENABLED.update(ok)
    return len(ok)


_ENABLED: set = set()


def uninstall() -> None:
    cls = _ORIG.pop("cls", None)
    if cls is not None:
        cls.__call__ = _ORIG.pop("call")
    _ENABLED.clear()
