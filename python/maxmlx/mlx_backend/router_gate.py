"""The MoE router's logits for a decode row: a bf16 matvec accumulated in fp32.

Flint routes in fp32 (``self.gate(x.astype(mx.float32))``) with a bf16
weight, so every decode step and every MoE layer MLX first casts the whole
256 x 2048 weight to fp32 and then runs an fp32 matvec over 256 output rows,
which spreads over too few threadgroups to stream: 1.8 ms per token over 47
layers on an M1 Max (28 GB/s), more than the lm_head. Pre-casting the weight
once still left 1.4 ms (the fp32 matvec alone).

This kernel reads the bf16 weight as it is stored, one simdgroup per expert
row with 16-byte loads, the input converted to fp32 in registers and the
products summed in fp32: the reference computation (bf16 values are exact in
fp32), up to fp32 summation order. Rows of more than one position (prefill,
verify windows) keep the stock path.
"""

from __future__ import annotations

from typing import Any, Dict

import mlx.core as mx

_SOURCE = """
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.y;
    // K / 8 uint4 words per row (8 bf16 each); lanes stride over them.
    auto wr = (const device uint4*)(w) + size_t(row) * (K / 8);
    auto xr = (const device uint4*)(x);
    float acc = 0.0f;
    for (int i = lane; i < K / 8; i += 32) {
        uint4 a = wr[i];
        uint4 b = xr[i];
        uint wa[4] = {a.x, a.y, a.z, a.w};
        uint xb[4] = {b.x, b.y, b.z, b.w};
        for (int e = 0; e < 4; e++) {
            acc += as_type<float>(wa[e] << 16) * as_type<float>(xb[e] << 16);
            acc += as_type<float>(wa[e] & 0xFFFF0000u) * as_type<float>(xb[e] & 0xFFFF0000u);
        }
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        out[row] = acc;
    }
"""

_kernel = mx.fast.metal_kernel(
    name="maxmlx_router_gate_bf16",
    input_names=["x", "w"],
    output_names=["out"],
    source=_SOURCE,
)

_PATCHED: Dict[str, Any] = {}


def gate_logits(x: mx.array, weight: mx.array) -> mx.array:
    """fp32 logits of one bf16 row ``x`` (..., K) against ``weight`` (E, K)."""
    E, K = weight.shape
    (out,) = _kernel(
        inputs=[x.reshape(K), weight],
        template=[("K", K)],
        grid=(32, E, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(E,)],
        output_dtypes=[mx.float32],
    )
    return out.reshape(*x.shape[:-1], E)


def _fits(gate) -> bool:
    w = getattr(gate, "weight", None)
    return (
        type(gate).__name__ == "Linear"
        and "bias" not in gate
        and w is not None
        and w.ndim == 2
        and w.dtype == mx.bfloat16
        and w.shape[1] % 256 == 0
    )


def install(blocks) -> int:
    """Route the decode rows of these MoE blocks' bf16 gates through the
    kernel. Returns how many gates were bound."""
    import mlx.nn as nn

    gates = [b.gate for b in blocks if _fits(getattr(b, "gate", None))]
    if not gates:
        return 0
    if "original" not in _PATCHED:
        original = nn.Linear.__call__
        _PATCHED["original"] = original
        bound = _PATCHED.setdefault("ids", set())

        def call(self, x, _orig=original):
            # One position whose input is the bf16 row cast up: the cast is
            # exact, so the kernel reads the bf16 values directly.
            if id(self) in bound and x.size == x.shape[-1]:
                return gate_logits(x.astype(mx.bfloat16), self.weight)
            return _orig(self, x)

        nn.Linear.__call__ = call
    _PATCHED["ids"].update(id(g) for g in gates)
    return len(gates)


def uninstall() -> None:
    import mlx.nn as nn

    original = _PATCHED.pop("original", None)
    if original is not None:
        nn.Linear.__call__ = original
    _PATCHED.clear()
