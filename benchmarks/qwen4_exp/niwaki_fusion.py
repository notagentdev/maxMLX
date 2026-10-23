"""Fewer kernel launches per decode token on Qwen3.8-Flash-Next (qwen4_exp).

A decode token of the Niwaki checkpoint takes ~46 ms of GPU time for ~3.6 GB of
weights, which the memory system streams in ~10 ms: like Ornith before
decode_fusion, the token is paid in dispatch latency, not bytes. Each of the
48 layers is a chain of small kernels: two hyper-connection mixes (an RMS norm
over 4 x 2560, a low-rank down/up pair, a sigmoid, a mean over the streams, an
injection gate), the mixer, the MoE block, and the elementwise injections
between them.

Everything in a layer except the mixer is a pure function of its inputs, so
per layer and for the decode shape only (one row, batch 1):

  pre   attn_hyper_connection(h)                         -> mixed, hyper, w
  post  hyper + inject(branch, w) -> mlp_hyper_connection
        -> mlp(mixed) -> hyper2 + inject(branch2, w2)    -> h'

are each ``mx.compile``d with the layer's weights as trace constants (the
same pattern as maxmlx's decode_fusion for Ornith's MoE blocks). The
stateful mixers (gated delta net, QSA attention with its caches) and the PLE
layer (a host gather) keep the eager path, as does every prefill shape.
The final 4-stream mixer and the LM head are compiled together.
"""
from __future__ import annotations

from typing import Any, Dict

_COMPILED: Dict[int, Any] = {}
_GDN_FULL: Dict[int, Any] = {}
_ORIGINAL: Dict[str, Any] = {}


class _TraceCache(list):
    """[conv_state, recurrent_state] for a gated-delta forward inside a trace:
    the stock forward reads and writes cache[0]/cache[1] and, having no
    ``advance``, leaves the bookkeeping to the caller."""

    lengths = None


def _decode_shape(x) -> bool:
    return x.ndim == 3 and x.shape[0] == 1 and x.shape[1] == 1


def install(model) -> Dict[str, int]:
    import mlx.core as mx

    lm = model.language_model
    layers = list(lm.model.layers)
    layer_cls = type(layers[0])

    for layer in layers:
        def pre(h, _l=layer):
            return _l.attn_hyper_connection(h)

        def post(hyper, branch, w, _l=layer):
            inj = branch[..., None, :] * w[..., None]
            h = hyper + inj.reshape(*hyper.shape)
            mixed, hyper2, w2 = _l.mlp_hyper_connection(h)
            b2 = _l.mlp(mixed)
            inj2 = b2[..., None, :] * w2[..., None]
            return hyper2 + inj2.reshape(*hyper2.shape)

        _COMPILED[id(layer)] = (mx.compile(pre), mx.compile(post))

    if "layer_call" not in _ORIGINAL:
        original = layer_cls.__call__
        _ORIGINAL["layer_call"] = original
        _ORIGINAL["layer_cls"] = layer_cls

        def call(self, hidden_states, input_ids, mask, cache, position_ids):
            fns = _COMPILED.get(id(self))
            if fns is None or not _decode_shape(hidden_states):
                return original(self, hidden_states, input_ids, mask, cache, position_ids)
            pre, post = fns
            full = _GDN_FULL.get(id(self))
            if (
                full is not None
                and mask is None
                and cache is not None
                and cache[0] is not None
                and cache[1] is not None
            ):
                out, cache[0], cache[1] = full(hidden_states, cache[0], cache[1])
                if hasattr(cache, "advance"):
                    cache.advance(1)
                return out
            if "ple" in self:
                hidden_states = hidden_states + self.ple(hidden_states, input_ids, cache, mask)
            mixed, hyper, w = pre(hidden_states)
            if self.is_linear:
                branch = self.linear_attn(mixed, mask=mask, cache=cache)
            else:
                branch = self.self_attn(mixed, mask=mask, cache=cache, position_ids=position_ids)
            return post(hyper, branch, w)

        layer_cls.__call__ = call

    # The tail: final hyper-connection mixer + LM head, one compiled graph.
    head = getattr(lm, "lm_head", None)
    mixer = lm.model.hyper_connection_mixer
    n_tail = 0
    if head is not None and "model_call" not in _ORIGINAL:
        model_cls = type(lm.model)
        mixer_cls = type(mixer)
        orig_mixer = mixer_cls.__call__
        _ORIGINAL["mixer_call"] = orig_mixer
        # Through the ORIGINAL call: the patched one below would route the
        # trace back into this compiled function.
        _COMPILED[id(mixer)] = mx.compile(lambda h: orig_mixer(mixer, h))

        def mixer_call(self, hyper_input, _orig=orig_mixer):
            fn = _COMPILED.get(id(self))
            if fn is not None and hyper_input.ndim == 3 and hyper_input.shape[:2] == (1, 1):
                return fn(hyper_input)
            return _orig(self, hyper_input)

        mixer_cls.__call__ = mixer_call
        _ORIGINAL["mixer_cls"] = mixer_cls
        _ORIGINAL["model_call"] = model_cls
        n_tail = 1
    return {"layers": len(layers), "tail": n_tail}


def uninstall() -> None:
    cls = _ORIGINAL.pop("layer_cls", None)
    if cls is not None:
        cls.__call__ = _ORIGINAL.pop("layer_call")
    mcls = _ORIGINAL.pop("mixer_cls", None)
    if mcls is not None:
        mcls.__call__ = _ORIGINAL.pop("mixer_call")
    _ORIGINAL.clear()
    _COMPILED.clear()


# -- the post-cast healing maps in 8 bits ------------------------------------------


def quantize_maps(model, bits: int = 8, group_size: int = 64) -> Dict[str, float]:
    """Niwaki applies a trained bf16 2560 x 2560 map ``T`` to every MoE block's
    output (``niwaki_flash_load._mapped``: ``y.astype(bf16) @ T``). That is
    600 MiB of bf16 read per decode token, the largest dense tensor class of
    the checkpoint. Stored as ``T.T`` in 8-bit affine groups of 64 it is half
    the bytes. T is close to the identity, so what is quantized is T - I:
    ``y + quantized_matmul(y, (T - I)q, transpose=True)`` = ``y @ T``.
    The bf16 maps are dropped afterwards (600 MiB less memory too).
    Returns the worst relative error measured on a random probe."""
    import mlx.core as mx

    layers = list(model.language_model.model.layers)
    worst = 0.0
    n = 0
    for layer in layers:
        blk = layer.mlp
        T = blk.__dict__.get("T") if hasattr(blk, "__dict__") else None
        if T is None and "T" in blk:
            T = blk["T"]
        if T is None:
            continue
        # T is close to the identity: its large diagonal would set every
        # group's scale and drown the small corrections. Quantize T - I and
        # add y back exactly.
        eye = mx.eye(T.shape[0], dtype=mx.float32)
        delta = (T.astype(mx.float32) - eye).T
        w, s, b = mx.quantize(mx.contiguous(delta).astype(mx.bfloat16),
                              group_size=group_size, bits=bits)
        probe = mx.random.normal((4, T.shape[0])).astype(mx.bfloat16)
        ref = (probe @ T).astype(mx.float32)
        got = (probe.astype(mx.float32) + mx.quantized_matmul(
            probe, w, s, b, transpose=True, group_size=group_size, bits=bits
        ).astype(mx.float32))
        err = float(mx.linalg.norm(got - ref) / mx.maximum(mx.linalg.norm(ref), 1e-9))
        worst = max(worst, err)
        mx.eval(w, s, b)
        blk._Tq = (w, s, b)
        cls = type(blk)
        base = cls.__mro__[1]

        def call(self, x, _base=base):
            y = _base.__call__(self, x)
            qw, qs, qb = self._Tq
            yb = y.astype(mx.bfloat16)
            out = yb + mx.quantized_matmul(yb, qw, qs, qb, transpose=True,
                                           group_size=group_size, bits=bits)
            return out.astype(y.dtype)

        cls.__call__ = call
        del blk["T"]
        n += 1
    return {"maps": n, "worst_rel_err": worst}


# -- the gated-delta input projections as one matmul --------------------------------

_GDN_PROJ = ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")


def fuse_gdn_projections(model) -> int:
    """The gated-delta block's four input projections read the same input and
    are all 4-bit g32 here: their rows are concatenated into ONE quantized
    linear, and four small slice modules hand its pieces back to the stock
    forward (the first one computes the fused matmul, the other three reuse
    it for the same input). Three launches fewer per recurrent layer; the
    originals are dropped, so memory stays the same."""
    import mlx.core as mx
    import mlx.nn as nn

    from maxmlx.mlx_backend.decode_fusion import _fusable, _fuse_linears

    class _Holder:
        def __init__(self, fused, splits):
            self.fused, self.splits, self.x, self.out = fused, splits, None, None

    class _Slice(nn.Module):
        def __init__(self, holder, i):
            super().__init__()
            self._holder, self._i = holder, i

        def __call__(self, x):
            h = self._holder
            if h.x is not x:
                h.out = mx.split(h.fused(x), h.splits, axis=-1)
                h.x = x
            return h.out[self._i]

    n = 0
    for layer in model.language_model.model.layers:
        g = getattr(layer, "linear_attn", None) if layer.is_linear else None
        if g is None:
            continue
        lins = [g[name] for name in _GDN_PROJ]
        if not _fusable(lins):
            continue
        sizes = [int(lin.weight.shape[0]) for lin in lins]
        splits = [sizes[0], sizes[0] + sizes[1], sizes[0] + sizes[1] + sizes[2]]
        holder = _Holder(_fuse_linears(lins), splits)
        for i, name in enumerate(_GDN_PROJ):
            g[name] = _Slice(holder, i)
        n += 1
    return n



def compile_gdn_layers(model) -> int:
    """Whole recurrent layers (no PLE) as ONE compiled function each for the
    decode shape: attention hyper-connection, the gated delta net (its conv
    and recurrent states passed in and returned), the injection, the MLP
    hyper-connection, the MoE block and its injection. Call after install()."""
    import mlx.core as mx

    n = 0
    for layer in model.language_model.model.layers:
        if not layer.is_linear or "ple" in layer or id(layer) not in _COMPILED:
            continue
        g = layer.linear_attn
        # The decode conv weight is cached on first use with an mx.eval, which
        # cannot run inside a trace: build it here.
        weight = g.conv1d.weight[:, :, 0].T.astype(mx.float32)
        mx.eval(weight)
        g._qwen3_5_decode_conv_weight = (id(g.conv1d.weight), weight)
        _pre, post = _COMPILED[id(layer)]

        def full(h, conv, rec, _l=layer, _post=post):
            tc = _TraceCache([conv, rec])
            mixed, hyper, w = _l.attn_hyper_connection(h)
            branch = _l.linear_attn(mixed, mask=None, cache=tc)
            return _post(hyper, branch, w), tc[0], tc[1]

        _GDN_FULL[id(layer)] = mx.compile(full)
        n += 1
    return n


def patch_speculative_verifier() -> None:
    """mlx-vlm's exact speculative verifier re-implements every layer, the MoE
    block included: it reads ``switch_mlp.gate_proj``/``up_proj`` (which the
    mapped store packs into one ``gate_up_proj``) and it never applies
    Niwaki's healing map ``T`` -- a verify window would compute a different
    model than plain decode. Top-level MoE blocks therefore go through their
    own module call (packed experts, ``T``); the shared expert inside keeps the
    verifier's path."""
    from mlx_vlm.models.qwen4_exp.language import Qwen4ExpExactSpeculativeVerifier as V

    if getattr(V, "_niwaki_patched", False):
        return
    original = V._feed_forward

    def feed_forward(self, module, x):
        if hasattr(module, "switch_mlp"):
            return module(x)
        return original(self, module, x)

    V._feed_forward = feed_forward
    V._niwaki_patched = True
    import os

    if os.environ.get("BATCHED_VERIFY", "1") == "1":
        # The exact verifier computes quantized linears token by token
        # ("singleton-equivalent", _target_verify_timewise) wherever its own
        # small-M kernel does not cover the shape, so a window of T rows costs
        # ~T single decode steps and speculation cannot pay. Plain module
        # calls run the window as one batched matmul. The GDN per-position
        # state capture for rollback is untouched; only the rounding of the
        # verify rows may differ from single-row decode near ties.
        V._linear = lambda self, linear, x: linear(x)
        V._linears = lambda self, linears, x: tuple(lin(x) for lin in linears)
