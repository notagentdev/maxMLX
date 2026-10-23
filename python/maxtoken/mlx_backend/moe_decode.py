"""The routed experts of one decode row in two launches (gate+up+SwiGLU, down).

A decode step's routed experts are a handful of tiny matvecs per layer: on
Flint, 6 of 256 experts, gate/up 512 x 2048 and down 2048 x 512, 3-bit. MLX
runs them as three gather_qmm launches plus the activation, each with little
work in flight, and the block is latency-bound, not bandwidth-bound: 3.9 ms
of an 11.2 ms step in the model (47 layers, 388 MB -> ~100 GB/s), and the
same time at 4 bits (30% more bytes) as at 3 (measured).

Kernel A computes h = silu(x W_gate^T) * (x W_up^T) for every selected expert
in one launch: a simdgroup owns (expert, block of intermediate rows), each
lane one 64-value quantization group of x held in registers, the gate and up
rows of the block streamed through it. Kernel B computes every expert's
h W_down^T: a simdgroup owns (expert, block of output rows), each lane one
64-value group of h, I/64 lanes per row reduced by lane shuffles. Weights are
read in MLX's packed affine layout (little-endian bit stream, per-group bf16
scale and bias: w = scale * q + bias), accumulated in fp32.

``SwitchGLU`` keeps its contract: one row in, the (..., K, H) per-expert
outputs back; the block's own weighting and sum follow unchanged. Rows of
more than one position (prefill, verify windows) keep the stock path.
"""

from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx

_DOT = """
    // Dot of one packed 64-value group of B-bit codes with the floats v[].
    #define GROUP_DOT(acc, p, v, B)                                           \\
        {                                                                     \\
            uint wv[2 * B];                                                   \\
            _Pragma("clang loop unroll(full)")                               \\
            for (int j = 0; j < 2 * B; j++) {                                 \\
                wv[j] = (p)[j];                                               \\
            }                                                                 \\
            _Pragma("clang loop unroll(full)")                               \\
            for (int i = 0; i < 64; i++) {                                    \\
                const int bit = i * B;                                        \\
                const int wi = bit >> 5, off = bit & 31;                      \\
                uint qv = wv[wi] >> off;                                      \\
                if (off + B > 32) {                                           \\
                    qv |= wv[wi + 1] << (32 - off);                           \\
                }                                                             \\
                acc += float(qv & ((1u << B) - 1u)) * v[i];                  \\
            }                                                                 \\
        }
"""

_GATE_UP = _DOT + """
    constexpr int NG = H / 64;          // groups per gate/up row (== 32 lanes)
    uint lane = thread_index_in_simdgroup;
    uint rb = threadgroup_position_in_grid.y;
    uint k = threadgroup_position_in_grid.z;

    // This lane's quantization group of x: 64 bf16 values, 16-byte loads.
    float xs[64];
    float xsum = 0.0f;
    auto xv = (const device uint4*)(x) + lane * 8;
    #pragma clang loop unroll(full)
    for (int j = 0; j < 8; j++) {
        uint4 w = xv[j];
        uint ww[4] = {w.x, w.y, w.z, w.w};
        #pragma clang loop unroll(full)
        for (int c = 0; c < 4; c++) {
            xs[8 * j + 2 * c] = as_type<float>(ww[c] << 16);
            xs[8 * j + 2 * c + 1] = as_type<float>(ww[c] & 0xFFFF0000u);
        }
    }
    #pragma clang loop unroll(full)
    for (int i = 0; i < 64; i++) {
        xsum += xs[i];
    }
    uint e = inds[k];
    #pragma clang loop unroll(full)
    for (int rr = 0; rr < RB; rr++) {
        int r = int(rb) * RB + rr;
        size_t row = size_t(e) * I + r;
        float dg = 0.0f, du = 0.0f;
        GROUP_DOT(dg, gw + row * (NG * 2 * BITS) + lane * 2 * BITS, xs, BITS)
        GROUP_DOT(du, uw + row * (NG * 2 * BITS) + lane * 2 * BITS, xs, BITS)
        float g = float(gs[row * NG + lane]) * dg + float(gb[row * NG + lane]) * xsum;
        float u = float(us[row * NG + lane]) * du + float(ub[row * NG + lane]) * xsum;
        g = simd_sum(g);
        u = simd_sum(u);
        if (lane == 0) {
            h[k * I + r] = g / (1.0f + metal::fast::exp(-g)) * u;
        }
    }
"""

_DOWN = _DOT + """
    uint lane = thread_index_in_simdgroup;
    uint rb = threadgroup_position_in_grid.y;
    uint k = threadgroup_position_in_grid.z;
    constexpr int NG = I / 64;          // groups per down row
    constexpr int RP = 32 / NG;         // rows per pass of the simdgroup
    uint e = inds[k];
    int g = lane % NG, rl = lane / NG;
    float hs[64];
    float hsum = 0.0f;
    auto hv = (const device float4*)(h + k * I + g * 64);
    #pragma clang loop unroll(full)
    for (int j = 0; j < 16; j++) {
        float4 f = hv[j];
        hs[4 * j] = f.x;
        hs[4 * j + 1] = f.y;
        hs[4 * j + 2] = f.z;
        hs[4 * j + 3] = f.w;
        hsum += f.x + f.y + f.z + f.w;
    }
    #pragma clang loop unroll(full)
    for (int p = 0; p < RB / RP; p++) {
        int r = int(rb) * RB + p * RP + rl;
        size_t row = size_t(e) * H + r;
        float d = 0.0f;
        GROUP_DOT(d, dw + row * (NG * 2 * BITS) + g * 2 * BITS, hs, BITS)
        float v = float(ds[row * NG + g]) * d + float(db[row * NG + g]) * hsum;
        #pragma clang loop unroll(full)
        for (int m = 1; m < NG; m <<= 1) {
            v += simd_shuffle_xor(v, m);
        }
        if (g == 0) {
            out[k * H + r] = OutT(v);
        }
    }
"""

_gate_up = mx.fast.metal_kernel(
    name="maxtoken_moe_decode_gate_up",
    input_names=["x", "gw", "gs", "gb", "uw", "us", "ub", "inds"],
    output_names=["h"],
    source=_GATE_UP,
)
_down = mx.fast.metal_kernel(
    name="maxtoken_moe_decode_down",
    input_names=["h", "dw", "ds", "db", "inds"],
    output_names=["out"],
    source=_DOWN,
)

# Rows per simdgroup (measured on an M1 Max, Flint shapes).
GATE_UP_ROWS = 1
DOWN_ROWS = 4

_PATCHED: Dict[str, Any] = {}


def _proj_ok(p, rows: int, cols: int) -> bool:
    return (
        type(p).__name__ == "QuantizedSwitchLinear"
        and getattr(p, "group_size", None) == 64
        and getattr(p, "bits", None) in (3, 4)
        and getattr(p, "mode", "affine") == "affine"
        and "biases" in p
        and tuple(p.weight.shape[1:]) == (rows, cols * p.bits // 32)
        and p.scales.dtype == mx.bfloat16
    )


def fits(switch) -> bool:
    """A SwitchGLU these kernels serve: affine q3/q4 g64 experts, hidden 2048,
    intermediate a multiple of 64 dividing the lanes, SwiGLU."""
    g, u, d = (getattr(switch, n, None) for n in ("gate_proj", "up_proj", "down_proj"))
    if g is None or u is None or d is None:
        return False
    H = int(g.weight.shape[-1] * 32 // max(1, getattr(g, "bits", 32)))
    I = int(g.weight.shape[1])
    return (
        H == 2048
        and I % 64 == 0 and 32 % (I // 64) == 0
        and I % GATE_UP_ROWS == 0 and H % DOWN_ROWS == 0
        and DOWN_ROWS % (32 // (I // 64)) == 0
        and _proj_ok(g, I, H) and _proj_ok(u, I, H) and _proj_ok(d, H, I)
        and g.bits == u.bits == d.bits
        and type(getattr(switch, "activation", None)).__name__ == "SwiGLU"
    )


def experts(switch, x: mx.array, indices: mx.array) -> mx.array:
    """Per-expert outputs (..., K, H) of one row ``x`` (..., H)."""
    g, u, d = switch.gate_proj, switch.up_proj, switch.down_proj
    bits = g.bits
    I = int(g.weight.shape[1])
    H = int(d.weight.shape[1])
    K = int(indices.size)
    inds = indices.reshape(K).astype(mx.uint32)
    common = [("H", H), ("I", I), ("BITS", bits)]
    (h,) = _gate_up(
        inputs=[x.reshape(H), g.weight, g.scales, g.biases, u.weight, u.scales, u.biases, inds],
        template=common + [("RB", GATE_UP_ROWS)],
        grid=(32, I // GATE_UP_ROWS, K),
        threadgroup=(32, 1, 1),
        output_shapes=[(K * I,)],
        output_dtypes=[mx.float32],
    )
    (out,) = _down(
        inputs=[h, d.weight, d.scales, d.biases, inds],
        template=common + [("RB", DOWN_ROWS), ("OutT", x.dtype)],
        grid=(32, H // DOWN_ROWS, K),
        threadgroup=(32, 1, 1),
        output_shapes=[(K, H)],
        output_dtypes=[x.dtype],
    )
    return out.reshape(*indices.shape, H)


def install(model_switches) -> int:
    """Serve the single-row calls of these SwitchGLUs with the kernels.
    Returns how many were bound."""
    from mlx_lm.models import switch_layers as sl

    switches = [s for s in model_switches if fits(s)]
    if not switches:
        return 0
    if "original" not in _PATCHED:
        original = sl.SwitchGLU.__call__
        _PATCHED["original"] = original
        bound = _PATCHED.setdefault("ids", set())

        def call(self, x, indices, _orig=original):
            if id(self) in bound and x.size == x.shape[-1] and x.dtype == mx.bfloat16:
                return experts(self, x, indices)
            return _orig(self, x, indices)

        sl.SwitchGLU.__call__ = call
    _PATCHED["ids"].update(id(s) for s in switches)
    return len(switches)


def uninstall() -> None:
    from mlx_lm.models import switch_layers as sl

    original = _PATCHED.pop("original", None)
    if original is not None:
        sl.SwitchGLU.__call__ = original
    _PATCHED.clear()
