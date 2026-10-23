"""Decode attention over long KV caches: a tail cache and a split-K kernel.

Measured on Flint (14 full-attention layers, 32 query / 4 KV heads of 128) at
16k positions, MLX's decode attention cost 0.52 ms per layer, 7.3 of a
19.6 ms step, and most of it was not attention:

* The cache write. KVCache.update_and_fetch slice-updates one position into
  the whole buffer. Under the engine's pipelined decode the previous step's
  attention still holds that buffer when the next step is encoded, so MLX
  cannot update it in place and copies all of it: ~220 us per layer at 16k,
  every token. TailKVCache sends decode positions to a 256-position tail
  instead (copying that is cheap) and folds the tail into the main buffer
  once it is full, one copy per 256 tokens.
* The attention itself. MLX's vector kernel streams the cache at ~120 GB/s;
  the split-K kernel below reaches 180-240 GB/s from 16k positions up (a
  plain read of the same arrays tops out at ~250 GB/s on an M1 Max).

Pass 1 gives every (KV head, chunk of positions) one simdgroup; chunks are
sized so 128 simdgroups run (more were slower: per-chunk setup and the
memory system, measured). The chunks of the main buffer come first, then
those of the tail. A chunk is walked in blocks of 32 positions, with the GQA
group's (up to 8) query heads as one 8-row tile on the simdgroup matrix
units:

* S^T = K Q^T as four 8x8 tiles, the keys loaded straight into the tiles'
  register layout. The head dimension is permuted (column c of tile j is
  index (c / 2) * D/4 + 2 j + c % 2) so a lane's elements of a row are one
  contiguous quarter, read with 16-byte loads; Q, V and the output share
  the permutation, which a dot product does not see;
* online softmax per head (lane shuffles across the tile rows);
* O += P V with P staged through threadgroup memory to transpose it.

Pass 2 merges the chunks of each head, with the attention sink (a learned
logit whose value is zero) in the denominator. The result equals
mx.fast.scaled_dot_product_attention up to fp32 summation order.

K/V are read in place through their strides (row stride a multiple of 8
elements, as the caches allocate them). Short contexts keep MLX's kernel and
the plain cache write, which are as fast there.
"""

from typing import Optional, Tuple

import mlx.core as mx
from mlx_lm.models.cache import KVCache

_PASS1 = """
    constexpr int DT = D / 8;  // 8-wide tiles along the head dimension
    uint lane = thread_index_in_simdgroup;
    uint kvh = threadgroup_position_in_grid.y;
    uint c = threadgroup_position_in_grid.z;
    uint C = threadgroups_per_grid.z;
    // Chunks [0, C1) walk the main cache, the rest walk the tail.
    int C1 = n_main_chunks[0];
    bool in_tail = int(c) >= C1;
    int chunk = in_tail ? chunk_size[1] : chunk_size[0];
    int T = in_tail ? k2_shape[2] : k_shape[2];
    int start = (in_tail ? int(c) - C1 : int(c)) * chunk;
    int end = min(T, start + chunk);
    auto kp = in_tail ? k2 : k;
    auto vp = in_tail ? v2 : v;
    int64_t ks1 = in_tail ? k2_strides[1] : k_strides[1];
    int64_t ks2 = in_tail ? k2_strides[2] : k_strides[2];
    int64_t vs1 = in_tail ? v2_strides[1] : v_strides[1];
    int64_t vs2 = in_tail ? v2_strides[2] : v_strides[2];

    // The GQA group's queries, scaled, as one 8-row tile (rows past G zero).
    threadgroup float qs[8 * D];
    threadgroup float pt[32 * 8];  // a block's probabilities, keys x heads
    threadgroup float hc[8];       // a block's per-head rescale factors
    float sc = scale[0];
    // Tile j, column cc covers head-dim index (cc / 2) * (D / 4) + 2 j + cc % 2:
    // a lane's two columns over all tiles are then one contiguous quarter of
    // a row, read with 16-byte loads. Q, K, V and O share the permutation.
    for (int i = lane; i < 8 * D; i += 32) {
        int h = i / D;
        int j = (i % D) / 8, cc = i % 8;
        int d = (cc / 2) * (D / 4) + 2 * j + cc % 2;
        qs[i] = h < G ? float(q[(kvh * G + h) * D + d]) * sc : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Q^T tiles (d x heads) stay in registers; O (heads x d) accumulates.
    simdgroup_matrix<float, 8, 8> O[DT];
    #pragma clang loop unroll(full)
    for (int j = 0; j < DT; j++) {
        O[j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    }
    // This lane's two elements of every 8x8 tile: row fm, columns fn, fn + 1.
    short qid = lane / 4;
    short fm = (qid & 4) + ((lane / 2) % 4);
    short fn = (qid & 2) * 2 + (lane % 2) * 2;

    // Online softmax state of heads fn, fn + 1 (score tiles are keys x heads).
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    auto kbase = kp + kvh * ks1;
    auto vbase = vp + kvh * vs1;

    for (int b = start; b < end; b += 32) {
        // S^T = K Q^T, four tiles of 8 keys (four independent MMA chains).
        simdgroup_matrix<float, 8, 8> S[4];
        int key_n[4];
        const device uint4* kr[4];
        #pragma clang loop unroll(full)
        for (int n = 0; n < 4; n++) {
            S[n] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            key_n[n] = b + n * 8 + fm;
            kr[n] = (const device uint4*)(kbase + (key_n[n] < end ? key_n[n] : start) * ks2
                                          + (fn / 2) * (D / 4));
        }
        // Four head-dim tiles per step: one 16-byte load per key row.
        #pragma clang loop unroll(full)
        for (int u = 0; u < DT / 4; u++) {
            uint4 w[4];
            #pragma clang loop unroll(full)
            for (int n = 0; n < 4; n++) {
                w[n] = kr[n][u];
            }
            #pragma clang loop unroll(full)
            for (int jj = 0; jj < 4; jj++) {
                simdgroup_matrix<float, 8, 8> Qj;
                simdgroup_load(Qj, qs + (4 * u + jj) * 8, D, ulong2(0, 0), true);
                #pragma clang loop unroll(full)
                for (int n = 0; n < 4; n++) {
                    simdgroup_matrix<float, 8, 8> Kt;
                    uint x = w[n][jj];
                    Kt.thread_elements()[0] = as_type<float>(x << 16);
                    Kt.thread_elements()[1] = as_type<float>(x & 0xFFFF0000u);
                    simdgroup_multiply_accumulate(S[n], Kt, Qj, S[n]);
                }
            }
        }
        float s[4][2];
        float bm0 = -INFINITY, bm1 = -INFINITY;
        #pragma clang loop unroll(full)
        for (int n = 0; n < 4; n++) {
            bool ok = key_n[n] < end;
            s[n][0] = ok ? S[n].thread_elements()[0] : -INFINITY;
            s[n][1] = ok ? S[n].thread_elements()[1] : -INFINITY;
            bm0 = max(bm0, s[n][0]);
            bm1 = max(bm1, s[n][1]);
        }
        // Lanes that share fn differ in lane bits 1, 2 and 4.
        bm0 = max(bm0, simd_shuffle_xor(bm0, 2));
        bm1 = max(bm1, simd_shuffle_xor(bm1, 2));
        bm0 = max(bm0, simd_shuffle_xor(bm0, 4));
        bm1 = max(bm1, simd_shuffle_xor(bm1, 4));
        bm0 = max(bm0, simd_shuffle_xor(bm0, 16));
        bm1 = max(bm1, simd_shuffle_xor(bm1, 16));
        float mn0 = max(m0, bm0), mn1 = max(m1, bm1);
        float c0 = metal::fast::exp(m0 - mn0), c1 = metal::fast::exp(m1 - mn1);
        float ps0 = 0.0f, ps1 = 0.0f;
        #pragma clang loop unroll(full)
        for (int n = 0; n < 4; n++) {
            float p0 = metal::fast::exp(s[n][0] - mn0);
            float p1 = metal::fast::exp(s[n][1] - mn1);
            pt[(n * 8 + fm) * 8 + fn] = p0;
            pt[(n * 8 + fm) * 8 + fn + 1] = p1;
            ps0 += p0;
            ps1 += p1;
        }
        ps0 += simd_shuffle_xor(ps0, 2);
        ps1 += simd_shuffle_xor(ps1, 2);
        ps0 += simd_shuffle_xor(ps0, 4);
        ps1 += simd_shuffle_xor(ps1, 4);
        ps0 += simd_shuffle_xor(ps0, 16);
        ps1 += simd_shuffle_xor(ps1, 16);
        l0 = l0 * c0 + ps0;
        l1 = l1 * c1 + ps1;
        m0 = mn0;
        m1 = mn1;
        if ((lane & 22) == 0) {
            hc[fn] = c0;
            hc[fn + 1] = c1;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // O = O * corr + P V, P loaded transposed (heads x keys).
        float cr = hc[fm];
        #pragma clang loop unroll(full)
        for (int j = 0; j < DT; j++) {
            O[j].thread_elements()[0] *= cr;
            O[j].thread_elements()[1] *= cr;
        }
        #pragma clang loop unroll(full)
        for (int n = 0; n < 4; n++) {
            simdgroup_matrix<float, 8, 8> Pm;
            simdgroup_load(Pm, pt + n * 64, 8, ulong2(0, 0), true);
            int key = b + n * 8 + fm;
            auto vr = (const device uint4*)(vbase + (key < end ? key : start) * vs2
                                            + (fn / 2) * (D / 4));
            uint vw[DT];
            #pragma clang loop unroll(full)
            for (int u = 0; u < DT / 4; u++) {
                uint4 w = vr[u];
                vw[4 * u] = w.x;
                vw[4 * u + 1] = w.y;
                vw[4 * u + 2] = w.z;
                vw[4 * u + 3] = w.w;
            }
            #pragma clang loop unroll(full)
            for (int j = 0; j < DT; j++) {
                simdgroup_matrix<float, 8, 8> Vt;
                Vt.thread_elements()[0] = as_type<float>(vw[j] << 16);
                Vt.thread_elements()[1] = as_type<float>(vw[j] & 0xFFFF0000u);
                simdgroup_multiply_accumulate(O[j], Pm, Vt, O[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }

    if ((lane & 22) == 0) {
        if (fn < G) {
            part_m[(kvh * G + fn) * C + c] = m0;
            part_l[(kvh * G + fn) * C + c] = l0;
        }
        if (fn + 1 < G) {
            part_m[(kvh * G + fn + 1) * C + c] = m1;
            part_l[(kvh * G + fn + 1) * C + c] = l1;
        }
    }
    if (fm < G) {
        size_t row = size_t(kvh * G + fm) * C + c;
        #pragma clang loop unroll(full)
        for (int j = 0; j < DT; j++) {
            int d = (fn / 2) * (D / 4) + 2 * j;
            part_acc[row * D + d] = O[j].thread_elements()[0];
            part_acc[row * D + d + 1] = O[j].thread_elements()[1];
        }
    }
"""

_PASS2 = """
    constexpr int P = D / 32;
    uint lane = thread_index_in_simdgroup;
    uint qh = threadgroup_position_in_grid.y;
    int C = part_m_shape[1];
    float M = HAS_SINKS ? float(sinks[qh]) : -INFINITY;
    for (int c = 0; c < C; c++) {
        M = max(M, part_m[qh * C + c]);
    }
    float total = HAS_SINKS ? metal::fast::exp(float(sinks[qh]) - M) : 0.0f;
    float acc[P];
    for (int i = 0; i < P; i++) {
        acc[i] = 0.0f;
    }
    for (int c = 0; c < C; c++) {
        size_t row = size_t(qh) * C + c;
        float w = metal::fast::exp(part_m[row] - M);
        total += part_l[row] * w;
        for (int i = 0; i < P; i++) {
            acc[i] += part_acc[row * D + lane * P + i] * w;
        }
    }
    for (int i = 0; i < P; i++) {
        out[qh * D + lane * P + i] = OutT(acc[i] / total);
    }
"""

_pass1 = mx.fast.metal_kernel(
    name="maxtoken_decode_attn_splitk",
    input_names=["q", "k", "v", "k2", "v2", "scale", "chunk_size", "n_main_chunks"],
    output_names=["part_acc", "part_m", "part_l"],
    source=_PASS1,
    ensure_row_contiguous=False,
)
_pass2 = mx.fast.metal_kernel(
    name="maxtoken_decode_attn_merge",
    input_names=["part_acc", "part_m", "part_l", "sinks"],
    output_names=["out"],
    source=_PASS2,
)

# Below this many positions MLX's kernel and the plain cache write are as fast
# (measured; at 1k MLX's kernel alone is faster).
MIN_KEYS = 2048
# Heads of 256 (Qwen3-Next attention): the kernel holds twice the accumulators
# and streams at ~130 GB/s, so the tail path pays from ~3k positions on
# (measured on Ornith's shape, 16 query / 2 KV heads, 10 layers pipelined).
MIN_KEYS_WIDE = 3072
# Simdgroups in flight: chunks are sized so (KV heads x chunks) reaches this.
TARGET_GROUPS = 128
MIN_CHUNK = 32
# Keys per simdgroup in the (short) tail segment.
TAIL_CHUNK = 64


def min_keys(head_dim: int) -> int:
    """Context length from which the tail path is worth it at this head size."""
    return MIN_KEYS if head_dim <= 128 else MIN_KEYS_WIDE


def supported(queries: mx.array, keys: mx.array, mask) -> bool:
    """One query token, one sequence, no mask, GQA-shaped heads (at most 8
    query heads per KV head). Whether the context is long enough to be worth
    it (MIN_KEYS) is the caller's call: ``keys`` may be the new position only."""
    if mask is not None or queries.ndim != 4 or keys.ndim != 4:
        return False
    B, Hq, L, D = queries.shape
    Hk = keys.shape[1]
    return (
        B == 1 and L == 1
        and D % 32 == 0 and D <= 256
        and Hk > 0 and Hq % Hk == 0 and Hq // Hk <= 8
        and queries.dtype == keys.dtype
    )


def decode_attention(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    scale: float,
    sinks: Optional[mx.array] = None,
    tail: Optional[Tuple[mx.array, mx.array]] = None,
) -> mx.array:
    """softmax(q k^T * scale [+ sink]) v for queries (1, Hq, 1, D) against
    keys/values (1, Hk, T, D), followed by the positions in ``tail`` (keys,
    values of the same layout, at least one position) if given; see
    supported()."""
    _, Hq, _, D = queries.shape
    Hk, T = keys.shape[1], keys.shape[2]
    G = Hq // Hk
    chunk = max(MIN_CHUNK, -(-T * Hk // TARGET_GROUPS))
    chunk = -(-chunk // 32) * 32
    C1 = -(-T // chunk)
    if tail is None:
        k2, v2, C2 = keys, values, 0
    else:
        k2, v2 = tail
        C2 = -(-k2.shape[2] // TAIL_CHUNK)
    C = C1 + C2
    part_acc, part_m, part_l = _pass1(
        inputs=[
            mx.contiguous(queries).reshape(Hq, D),
            keys,
            values,
            k2,
            v2,
            mx.array([scale], dtype=mx.float32),
            mx.array([chunk, TAIL_CHUNK], dtype=mx.int32),
            mx.array([C1], dtype=mx.int32),
        ],
        template=[("D", D), ("G", G)],
        grid=(32, Hk, C),
        threadgroup=(32, 1, 1),
        output_shapes=[(Hq * C, D), (Hq, C), (Hq, C)],
        output_dtypes=[mx.float32, mx.float32, mx.float32],
    )
    has_sinks = sinks is not None
    (out,) = _pass2(
        inputs=[part_acc, part_m, part_l, sinks if has_sinks else part_m],
        template=[("D", D), ("OutT", queries.dtype), ("HAS_SINKS", has_sinks)],
        grid=(32, Hq, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(Hq, D)],
        output_dtypes=[queries.dtype],
    )
    return out.reshape(1, Hq, 1, D)


class TailKVCache(KVCache):
    """A KVCache whose one-position decode writes go to a small tail buffer
    (see the module docstring for why). ``append_decode`` is the decode
    path; everything else -- prefill, verify windows, state, trim -- folds
    the tail into the main buffer first and is then the stock KVCache."""

    TAIL = 256

    def __init__(self):
        super().__init__()
        self._tail_k = None
        self._tail_v = None
        self._tail_n = 0

    def _fold(self):
        n = self._tail_n
        if not n:
            return
        self._tail_n = 0
        main = self.offset - n
        tk, tv = self._tail_k[..., :n, :], self._tail_v[..., :n, :]
        if main == 0 or self.keys is None:
            self.keys, self.values = tk, tv
        else:
            self.keys = mx.concatenate([self.keys[..., :main, :], tk], axis=2)
            self.values = mx.concatenate([self.values[..., :main, :], tv], axis=2)

    def append_decode(self, keys, values):
        """Append one position; returns the main buffer's keys and values
        and the tail's, for decode_attention(..., tail=...)."""
        if self._tail_n == self.TAIL:
            self._fold()
        if self._tail_k is None or self._tail_k.shape[1:2] != keys.shape[1:2]:
            B, H, _, D = keys.shape
            self._tail_k = mx.zeros((B, H, self.TAIL, D), keys.dtype)
            self._tail_v = mx.zeros((B, H, self.TAIL, values.shape[3]), values.dtype)
        n = self._tail_n
        self._tail_k[..., n : n + 1, :] = keys
        self._tail_v[..., n : n + 1, :] = values
        self._tail_n = n + 1
        self.offset += 1
        main = self.offset - self._tail_n
        return (
            self.keys[..., :main, :],
            self.values[..., :main, :],
            self._tail_k[..., : self._tail_n, :],
            self._tail_v[..., : self._tail_n, :],
        )

    def update_and_fetch(self, keys, values):
        self._fold()
        return super().update_and_fetch(keys, values)

    @property
    def state(self):
        self._fold()
        return KVCache.state.fget(self)

    @state.setter
    def state(self, v):
        KVCache.state.fset(self, v)
        self._tail_n = 0

    def trim(self, n):
        self._fold()
        return super().trim(n)

    def to_quantized(self, group_size: int = 64, bits: int = 4):
        self._fold()
        return super().to_quantized(group_size=group_size, bits=bits)

    def empty(self):
        return self.keys is None and not self._tail_n

    @property
    def nbytes(self):
        tail = 0 if self._tail_k is None else self._tail_k.nbytes + self._tail_v.nbytes
        return super().nbytes + tail

    @classmethod
    def merge(cls, caches):
        for c in caches:
            if isinstance(c, TailKVCache):
                c._fold()
        return KVCache.merge(caches)
