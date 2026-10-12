"""FTW-MLX: zero-copy mmapped expert store for the MLX backend.

The CUDA engine's FTW format repacks checkpoints so expert banks can be served
without per-request copies. This is its Apple-silicon analogue: the switch-GLU
expert tensors are repacked ONCE into a page-aligned flat file, and at serve
time each stacked tensor is memory-mapped and imported into MLX **zero-copy**
(``mx.from_dlpack`` on a ``np.memmap`` — unified memory makes the same pages
GPU-readable, the same trick llama.cpp's Metal backend uses for its weights).

The payoff, compared to the slot-cache offload path: expert serving becomes a
plain ``gather_qmm`` over the mapped store — no slot cache, no per-layer CPU
syncs, no speculation, no misses. Residency is managed by the OS page cache:
hot experts stay in RAM, cold ones fault in from SSD at first touch, and under
memory pressure clean file-backed pages are simply evicted (never swapped).
``mx.get_active_memory`` therefore only shows the dense weights; the experts'
real cost is reclaimable page cache.

Why a repack is needed at all: safetensors gives no alignment guarantees (the
JSON header makes every offset arbitrary), while Metal needs dtype-aligned
buffer offsets and ``np.memmap`` needs page-aligned mapping offsets. The repack
also stacks per-expert layouts (OLMoE-era conversions) into ``[E, ...]`` form.

Layout: ``experts.ftwm`` (page-aligned raw tensors) + ``manifest.json``:
{"version": 1, "source": {size, mtime}, "tensors": {name: {offset, shape, dtype}}}
"""

from __future__ import annotations

import json
import os
import threading
from typing import Any, Dict, List, Tuple

import numpy as np

from maxtoken.utils import init_logger

from .offload import _NP_DTYPES, SafetensorsIndex, _iter_modules, _is_switch_glu

logger = init_logger(__name__)

_PAGE = 16384  # Apple silicon VM page size; also satisfies mmap allocation granularity
_PROJS = ("gate_proj", "up_proj", "down_proj")
_PARTS = ("weight", "scales", "biases")


def _cache_dir(model_dir: str) -> str:
    src = os.stat(model_dir)
    tag = f"{os.path.basename(os.path.normpath(model_dir))}-{src.st_ino}"
    root = os.environ.get(
        "MAXTOKEN_MLX_FTW_DIR",
        os.path.join(os.path.expanduser("~"), ".cache", "maxtoken", "mlx-ftw"),
    )
    return os.path.join(root, tag)


def _source_stamp(index: SafetensorsIndex) -> Dict[str, int]:
    shards = sorted({e.shard for e in index.entries.values()})
    total = sum(os.path.getsize(s) for s in shards)
    mtime = max(int(os.path.getmtime(s)) for s in shards)
    return {"size": total, "mtime": mtime}


def _pack_enabled() -> bool:
    """Whether the store lays gate and up out interleaved per expert
    (``gate_up_proj``), so the two projections run as ONE gather (moe_pack.py).
    On by default; MAXTOKEN_MLX_FTW_PACK=0 keeps the classic per-proj layout."""
    return os.environ.get("MAXTOKEN_MLX_FTW_PACK", "1") != "0"


def expert_residency() -> str:
    """How much of the store a forward pulls into memory (MAXTOKEN_MLX_EXPERT_RESIDENCY).

    ``layer`` (default): one Metal buffer per stacked tensor. The GPU cannot
    fault pages in, so Metal makes a buffer wholly resident before a command
    buffer that uses it runs — a gather over 8 of 256 experts makes the whole
    layer resident (measured: 256 of 256 MiB after one 8-expert gather). Every
    token passes all MoE layers, so the whole store must sit in memory; under
    pressure its clean file pages are the first to be evicted and every token
    reads them back (Ornith at 3 tok/s beside ~12 GB of apps).

    ``expert``: one buffer per expert. A MoE block reads its routing to the
    host (one sync per layer), stacks only the experts it selected and gathers
    over that stack, so only experts actually used are made resident (8 of 256
    MiB measured). Costs the per-layer sync and the small stack copy, and
    skips the fused/compiled decode paths that bind the stacked tensors."""
    value = os.environ.get("MAXTOKEN_MLX_EXPERT_RESIDENCY", "layer").strip().lower()
    return "expert" if value == "expert" else "layer"


def _expert_slice(src, e: int, E: int, stacked: bool) -> Tuple[str, int, int]:
    """Byte range of expert ``e`` inside one source tensor."""
    if stacked:
        per = (src.end - src.start) // E
        return (src.shard, src.start + e * per, src.start + (e + 1) * per)
    s = src[e]
    return (s.shard, s.start, s.end)


def _switch_glu_tensor_names(index: SafetensorsIndex, glu_paths: List[str]):
    """Yield (store_name, byte_slices, shape, dtype) for every store tensor.

    Byte slices are (shard, start, end) ranges written in order. Stacked
    sources contribute one slice, per-expert sources one per expert (their
    concatenation IS the stacked layout, row-major). With packing enabled,
    each GLU's gate and up tensors are interleaved PER EXPERT into a single
    ``gate_up_proj`` tensor — affine groups run along the input axis, so the
    concatenation along output features is exact by construction."""

    def sources(path: str, proj: str, part: str):
        stacked = f"{path}.{proj}.{part}"
        if stacked in index:
            e = index[stacked]
            return e, int(e.shape[0]), True
        alt = path.rsplit(".", 1)[0] + ".experts"
        per = []
        i = 0
        while f"{alt}.{i}.{proj}.{part}" in index:
            per.append(index[f"{alt}.{i}.{proj}.{part}"])
            i += 1
        if per:
            return per, len(per), False
        return None, 0, False

    for path in glu_paths:
        pack = _pack_enabled()
        if pack:
            for part in _PARTS:
                g, Eg, _ = sources(path, "gate_proj", part)
                u, Eu, _ = sources(path, "up_proj", part)
                if g is None or u is None or Eg != Eu or Eg == 0:
                    pack = False
                    break
        if pack:
            for part in _PARTS:
                g, E, g_stacked = sources(path, "gate_proj", part)
                u, _, u_stacked = sources(path, "up_proj", part)
                slices: List[Tuple[str, int, int]] = []
                for e in range(E):
                    slices.append(_expert_slice(g, e, E, g_stacked))
                    slices.append(_expert_slice(u, e, E, u_stacked))
                first = g if g_stacked else g[0]
                if g_stacked:
                    shape = [E, 2 * int(first.shape[1]), *first.shape[2:]]
                else:
                    shape = [E, 2 * int(first.shape[0]), *first.shape[1:]]
                yield f"{path}.gate_up_proj.{part}", slices, shape, first.dtype
        for proj in (("down_proj",) if pack else _PROJS):
            for part in _PARTS:
                src, E, stacked = sources(path, proj, part)
                if src is None:
                    continue
                if stacked:
                    yield (
                        f"{path}.{proj}.{part}",
                        [(src.shard, src.start, src.end)],
                        list(src.shape),
                        src.dtype,
                    )
                else:
                    yield (
                        f"{path}.{proj}.{part}",
                        [(s.shard, s.start, s.end) for s in src],
                        [E, *src[0].shape],
                        src[0].dtype,
                    )


def repack_experts(model_dir: str, glu_paths: List[str]) -> str:
    """Write (or reuse) the page-aligned expert store for a checkpoint.

    Returns the cache directory containing ``experts.ftwm`` + ``manifest.json``.
    Idempotent: a manifest matching the source's size/mtime short-circuits.
    """
    index = SafetensorsIndex(model_dir)
    out_dir = _cache_dir(model_dir)
    manifest_path = os.path.join(out_dir, "manifest.json")
    store_path = os.path.join(out_dir, "experts.ftwm")
    stamp = {**_source_stamp(index), "packed": _pack_enabled()}
    if os.path.exists(manifest_path) and os.path.exists(store_path):
        try:
            manifest = json.load(open(manifest_path))
            if manifest.get("version") == 2 and manifest.get("source") == stamp:
                return out_dir
        except Exception:  # noqa: BLE001 -- corrupt cache: rebuild below
            pass

    os.makedirs(out_dir, exist_ok=True)
    tensors: Dict[str, Any] = {}
    tmp = store_path + ".tmp"
    total = 0
    handles: Dict[str, Any] = {}
    try:
        with open(tmp, "wb") as out:
            pos = 0
            for name, slices, shape, dtype in _switch_glu_tensor_names(
                index, glu_paths
            ):
                pad = (-pos) % _PAGE
                out.write(b"\0" * pad)
                pos += pad
                tensors[name] = {"offset": pos, "shape": shape, "dtype": dtype}
                for shard, start, end in slices:
                    src = handles.get(shard)
                    if src is None:
                        src = handles[shard] = open(shard, "rb")
                    src.seek(start)
                    remaining = end - start
                    while remaining:
                        chunk = src.read(min(remaining, 64 << 20))
                        out.write(chunk)
                        remaining -= len(chunk)
                        pos += len(chunk)
                total += 1
    finally:
        for fh in handles.values():
            fh.close()
    os.replace(tmp, store_path)
    with open(manifest_path, "w") as f:
        json.dump({"version": 2, "source": stamp, "tensors": tensors}, f)
    logger.info(
        f"FTW-MLX expert store: repacked {total} tensors "
        f"({os.path.getsize(store_path) / 2**30:.2f} GiB) into {out_dir}"
    )
    return out_dir


class MappedExpertStore:
    """Zero-copy mx views over the repacked expert store, one per tensor name.

    Holds the numpy memmap bases alive for the lifetime of the store — the mx
    arrays alias their pages (DLPack import), so dropping the bases would leave
    dangling device pointers.
    """

    def __init__(self, store_dir: str, per_expert: bool = False):
        import mlx.core as mx

        self._mx = mx
        self._bases: List[Any] = []
        self._base_by_name: Dict[str, Any] = {}
        self._dtype: Dict[str, str] = {}
        manifest = json.load(open(os.path.join(store_dir, "manifest.json")))
        self.path = os.path.join(store_dir, "experts.ftwm")
        self.per_expert = per_expert
        # Stacked views are built on first use: in per-expert mode none may
        # ever reach a command buffer (it would make its whole layer resident).
        self.tensors: Dict[str, Any] = {}
        for name, meta in manifest["tensors"].items():
            np_dt = _NP_DTYPES[meta["dtype"]]
            base = np.memmap(
                self.path,
                dtype=np_dt,
                mode="r",  # read-only shared mapping: stays file-backed page cache under GPU access (mode "c" turned touched pages into anonymous GPU copies, measured 2026-09-26)
                offset=meta["offset"],
                shape=tuple(meta["shape"]),
            )
            self._bases.append(base)
            self._base_by_name[name] = base
            self._dtype[name] = meta["dtype"]
            if not per_expert:
                self.tensors[name] = self._wrap(name, base)
        self._prefetch_lock = threading.Lock()
        self._prefetching = False
        self._advise_map: Any = None

    def _wrap(self, name: str, view) -> Any:
        arr = self._mx.from_dlpack(view)
        return arr.view(self._mx.bfloat16) if self._dtype[name] == "BF16" else arr

    def expert_views(self, name: str) -> List[Any]:
        """One zero-copy mx array per expert of a stacked tensor, each its own
        Metal buffer, so a forward makes only the experts it uses resident.
        Raises ValueError when an expert's bytes are not page-aligned (a buffer
        that is not would be imported as a copy)."""
        base = self._base_by_name[name]
        stride = base.nbytes // base.shape[0]
        if stride % _PAGE or (base.ctypes.data % _PAGE):
            raise ValueError(f"{name}: experts are not page-aligned ({stride} B each)")
        return [self._wrap(name, base[e]) for e in range(base.shape[0])]

    # -------------------------------------------------------------- prefetch

    def _advise_handle(self):
        """A parallel plain mmap of the store used only for madvise/page-in;
        it shares the page cache with the DLPack-imported memmaps."""
        import mmap as mmap_mod

        if self._advise_map is None:
            f = open(self.path, "rb")
            self._advise_map = mmap_mod.mmap(f.fileno(), 0, prot=mmap_mod.PROT_READ)
            self._advise_file = f  # keep the fd alive with the map
        return self._advise_map

    def start_prefetch(self) -> None:
        """Kick a background sweep that advises the store's pages into memory
        IN FILE (= layer) ORDER, so page-in runs ahead of the forward that
        consumes the layers in the same order. This is the unified-memory
        analogue of streaming expert uploads on a second stream: on a cold
        cache the prefill otherwise pays every page fault synchronously inside
        its evals. Idempotent while a sweep is running; near-free when warm."""
        import mmap as mmap_mod

        if self.per_expert:
            # Reading the whole store back in is what per-expert residency
            # exists to avoid; hot experts come back on demand.
            return
        with self._prefetch_lock:
            if self._prefetching:
                return
            self._prefetching = True

        def sweep():
            import time as _time

            try:
                m = self._advise_handle()
                size = len(m)
                # Small chunks with an explicit yield: ``mmap.madvise`` does
                # NOT release the GIL, so a background sweep in 256 MB chunks
                # strangled the scheduler thread in 96 ms bites — measured as
                # 1.4 s prompt rounds and 0.7 s gaps during a concurrent
                # bench (the "B>=2 server tax" in large part). 32 MB keeps
                # each GIL hold near 10 ms and the sleep lets the scheduler
                # run between chunks; the sweep still finishes a 17 GiB
                # store in a few seconds.
                chunk = 32 << 20
                for off in range(0, size, chunk):
                    m.madvise(
                        mmap_mod.MADV_WILLNEED, off, min(chunk, size - off)
                    )
                    _time.sleep(0.002)
            except Exception as exc:  # noqa: BLE001 -- advisory only
                logger.warning(f"expert-store prefetch sweep failed: {exc!r}")
            finally:
                with self._prefetch_lock:
                    self._prefetching = False

        threading.Thread(target=sweep, name="ftw-mlx-prefetch", daemon=True).start()

    def mlock_all(self) -> bool:
        """Pin the whole store into memory (MAXTOKEN_MLX_MLOCK=1): the pressure
        counterpart of llama.cpp's host-register trick. Only sensible when the
        store fits in RAM with headroom — pinned pages cannot be reclaimed, so
        this trades system elasticity for immunity against expert-page eviction."""
        import ctypes

        try:
            libc = ctypes.CDLL(None, use_errno=True)
            total = 0
            for base in self._bases:
                addr = base.ctypes.data
                length = base.nbytes
                if libc.mlock(ctypes.c_void_p(addr), ctypes.c_size_t(length)) != 0:
                    err = ctypes.get_errno()
                    logger.warning(
                        f"mlock failed after {total / 2**30:.2f} GiB (errno {err}); "
                        "raise the memlock rlimit or drop MAXTOKEN_MLX_MLOCK"
                    )
                    return False
                total += length
            logger.info(f"expert store pinned: {total / 2**30:.2f} GiB mlocked")
            return True
        except Exception as exc:  # noqa: BLE001 -- opt-in nicety
            logger.warning(f"mlock unavailable: {exc!r}")
            return False

    def glu_params(self, glu_path: str) -> Dict[str, Dict[str, Any]]:
        """{proj: {part: array}} for one switch-GLU; ``gate_up_proj`` when the
        store holds the packed interleaved layout."""
        out: Dict[str, Dict[str, Any]] = {}
        for proj in (*_PROJS, "gate_up_proj"):
            parts = {
                part: self.tensors[f"{glu_path}.{proj}.{part}"]
                for part in _PARTS
                if f"{glu_path}.{proj}.{part}" in self.tensors
            }
            if parts:
                out[proj] = parts
        return out


class _ExpertViews:
    """Per-expert arrays of one switch-GLU. A plain holder, not an nn.Module
    attribute value, so 256 x 6 arrays per layer never join the model's
    parameter tree (parameters() walks, evals and quantize predicates)."""

    def __init__(self, gate_up: List[List[Any]], down: List[List[Any]]):
        self.gate_up = gate_up  # [weight, scales, biases?], each a list per expert
        self.down = down


_PER_EXPERT_CLS: Any = None
# MAXTOKEN_MLX_EXPERT_TRACE=<file>: one line per per-expert MoE call,
# "<layer> <rows> <expert ids...>", for studying routing predictability.
_TRACE: Any = (
    open(os.environ["MAXTOKEN_MLX_EXPERT_TRACE"], "a", buffering=1)
    if os.environ.get("MAXTOKEN_MLX_EXPERT_TRACE")
    else None
)


def _per_expert_switch_glu():
    """SwitchGLU over per-expert buffers (defined lazily: importing this module
    must not pull mlx). Same math as moe_pack.PackedSwitchGLU — gate+up as one
    gathered projection, the token sort for wide inputs — over a stack of
    only the experts the routing selected."""
    global _PER_EXPERT_CLS
    if _PER_EXPERT_CLS is not None:
        return _PER_EXPERT_CLS
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.switch_layers import _gather_sort, _scatter_unsort

    class PerExpertSwitchGLU(nn.Module):
        # Read by decode_fusion and whole_moe: they bind stacked tensors and
        # compile the block, and this forward syncs on its routing.
        per_expert = True

        def __init__(self, views, activation, split_at, q_gate_up, q_down):
            super().__init__()
            self._views = views
            self.activation = activation
            self._split_at = int(split_at)
            self._q_gate_up = q_gate_up
            self._q_down = q_down
            self._layer = -1  # set at attach; names the block in the routing trace

        def _gather(self, x, parts, q, used, idx, do_sort):
            stacked = [mx.stack([p[e] for e in used]) for p in parts]
            return mx.gather_qmm(
                x,
                stacked[0],
                stacked[1],
                stacked[2] if len(stacked) > 2 else None,
                rhs_indices=idx,
                transpose=True,
                sorted_indices=do_sort,
                **q,
            )

        def __call__(self, x, indices):
            # The one host sync per layer: which experts this forward needs.
            flat = np.asarray(indices).reshape(-1)
            used, local = np.unique(flat, return_inverse=True)
            used = used.tolist()
            if _TRACE is not None:
                _TRACE.write(f"{self._layer} {flat.size // indices.shape[-1]} {' '.join(map(str, used))}\n")
            idx = mx.array(local.reshape(indices.shape).astype(np.uint32))
            x = mx.expand_dims(x, (-2, -3))
            do_sort = indices.size >= 64
            inv_order = None
            if do_sort:
                x, idx, inv_order = _gather_sort(x, idx)
            packed = self._gather(x, self._views.gate_up, self._q_gate_up, used, idx, do_sort)
            x_gate, x_up = mx.split(packed, [self._split_at], axis=-1)
            x = self._gather(
                self.activation(x_up, x_gate), self._views.down, self._q_down, used, idx, do_sort
            )
            if do_sort:
                x = _scatter_unsort(x, inv_order, indices.shape)
            return x.squeeze(-2)

    _PER_EXPERT_CLS = PerExpertSwitchGLU
    return PerExpertSwitchGLU


def _quant(lin) -> Dict[str, Any]:
    return {
        "group_size": int(getattr(lin, "group_size", 64)),
        "bits": int(getattr(lin, "bits", 4)),
        "mode": str(getattr(lin, "mode", "affine")),
    }


def _attach_per_expert(model, store: "MappedExpertStore", glu_paths, glu_mods) -> None:
    """Replace every switch-GLU by a PerExpertSwitchGLU over ``store``'s
    per-expert views. All views are built before any module is touched, so a
    store this mode cannot serve raises with the model unchanged."""
    built = []
    for path, mod in zip(glu_paths, glu_mods, strict=True):
        names = {
            proj: [f"{path}.{proj}.{part}" for part in _PARTS
                   if f"{path}.{proj}.{part}" in store._base_by_name]
            for proj in ("gate_up_proj", "down_proj")
        }
        if not names["gate_up_proj"] or not names["down_proj"]:
            raise ValueError("per-expert residency needs the packed store (MAXTOKEN_MLX_FTW_PACK=1)")
        views = _ExpertViews(
            [store.expert_views(n) for n in names["gate_up_proj"]],
            [store.expert_views(n) for n in names["down_proj"]],
        )
        split_at = int(store._base_by_name[names["gate_up_proj"][0]].shape[1]) // 2
        built.append((path, mod, views, split_at))
    cls = _per_expert_switch_glu()
    for layer, (path, mod, views, split_at) in enumerate(built):
        parent_path, attr = path.rsplit(".", 1)
        # The whole SwitchGLU goes, so its lazy checkpoint arrays are never
        # materialized (the same reason the packed path replaces it).
        glu = cls(views, mod.activation, split_at, _quant(mod.gate_proj), _quant(mod.down_proj))
        glu._layer = layer
        setattr(_resolve_module(model, parent_path), attr, glu)


def _resolve_module(model, path: str):
    """Walk a dotted parameter path (list indices included) to its module."""
    obj = model
    for name in path.split("."):
        obj = obj[int(name)] if name.isdigit() else getattr(obj, name)
    return obj


def attach_mapped_experts(model, model_dir: str) -> int:
    """Serve every switch-GLU's experts from the zero-copy mapped store.

    Must run on a ``lazy=True``-loaded model. Unlike the slot-cache offload this
    keeps mlx-lm's own SwitchGLU modules and forward (including its sorted-gather
    prefill path) — only their weight/scales/biases parameters are replaced by
    the mapped views. Returns the number of GLUs rewired.
    """
    index = SafetensorsIndex(model_dir)
    glu_paths = []
    glu_mods = []
    for path, mod in _iter_modules(model):
        if not (path and _is_switch_glu(mod)):
            continue
        if f"{path}.{_PROJS[0]}.{_PARTS[0]}" in index or (
            f"{path.rsplit('.', 1)[0]}.experts.0.{_PROJS[0]}.{_PARTS[0]}" in index
        ):
            glu_paths.append(path)
            glu_mods.append(mod)
    if not glu_paths:
        raise ValueError("no switch-GLU expert tensors found for the mapped store")

    store_dir = repack_experts(model_dir, glu_paths)
    if expert_residency() == "expert":
        store = MappedExpertStore(store_dir, per_expert=True)
        try:
            _attach_per_expert(model, store, glu_paths, glu_mods)
        except ValueError as exc:
            logger.warning(f"per-expert residency unavailable ({exc}); stacked layers instead")
        else:
            model._maxtoken_mapped_store = store
            logger.info(
                f"expert serving: zero-copy mmap store, per-expert residency "
                f"({len(glu_paths)} MoE layers, {os.path.getsize(store.path) / 2**30:.2f} GiB "
                "file-backed; only the experts a forward routes to are made resident)"
            )
            return len(glu_paths)
    store = MappedExpertStore(store_dir)
    packed_count = 0
    for path, mod in zip(glu_paths, glu_mods, strict=True):
        params = store.glu_params(path)
        packed = params.get("gate_up_proj")
        if packed is not None:
            # Replace the whole SwitchGLU with the packed form BEFORE the
            # model evaluates its parameters: the dropped gate/up modules
            # still hold their lazy checkpoint arrays, and keeping them
            # reachable would materialize ~10 GB the mapped store exists to
            # avoid.
            from .moe_pack import classes

            PackedQuantizedProjection, PackedSwitchGLU, _ = classes()
            down = mod.down_proj
            dp = params["down_proj"]
            down.weight = dp["weight"]
            down.scales = dp["scales"]
            down.biases = dp["biases"]
            proj = PackedQuantizedProjection(
                packed["weight"],
                packed["scales"],
                packed.get("biases"),
                group_size=int(getattr(mod.gate_proj, "group_size", 64)),
                bits=int(getattr(mod.gate_proj, "bits", 4)),
                mode=str(getattr(mod.gate_proj, "mode", "affine")),
            )
            split_at = int(packed["weight"].shape[1]) // 2
            parent_path, attr = path.rsplit(".", 1)
            setattr(
                _resolve_module(model, parent_path),
                attr,
                PackedSwitchGLU(proj, down, mod.activation, split_at),
            )
            packed_count += 1
            continue
        for proj, parts in params.items():
            lin = getattr(mod, proj)
            lin.weight = parts["weight"]
            lin.scales = parts["scales"]
            lin.biases = parts["biases"]
    if packed_count:
        logger.info(
            f"MoE pack: {packed_count} switch-GLUs serve gate+up as one gather "
            "(interleaved mapped store)"
        )
    # The store must outlive the model; hang it off the model object.
    model._maxtoken_mapped_store = store
    logger.info(
        f"expert serving: zero-copy mmap store ({len(glu_paths)} MoE layers, "
        f"{os.path.getsize(store.path) / 2**30:.2f} GiB file-backed, OS-managed residency)"
    )
    if os.environ.get("MAXTOKEN_MLX_MLOCK") == "1":
        store.mlock_all()
    return len(glu_paths)
