"""Build an mlx-vlm ``qwen4_exp_mtp`` drafter from the native MTP tensors that
Strata's ``tools/mtp_fetch.py`` range-read out of Qwen/Qwen3.8-Flash-Next
(31 BF16 tensors, 5.2 GB, one raw file each under ``<src>/tensors``).

mlx-vlm's own splitter (``Qwen4ExpMTPSplitter``) expects a checkpoint
directory with safetensors shards; this feeds it the raw tensors instead and
reuses its transform (fused gate_up split into gate/up, prefix stripping) and
its affine quantization (the router gate stays full precision). The drafter's
text config is the target's (Niwaki): the MTP layer has the base model's
geometry, which Niwaki keeps.

    ~/.venvs/qwen4-spike/bin/python build_mtp_drafter.py <mtp_src> <target_dir> <out_dir> [bits] [expert_bits]
"""
import json
import os
import shutil
import sys

import mlx.core as mx
import numpy as np
from mlx_vlm.speculative.drafters.qwen4_exp_mtp.split import Qwen4ExpMTPSplitter

src, target, out = (os.path.abspath(a) for a in sys.argv[1:4])
bits = int(sys.argv[4]) if len(sys.argv) > 4 else 4
# The routed experts are ~85% of the layer's bytes; with 32 GB they have to be
# narrower than the rest for the drafter to fit beside the target (Strata keeps
# its MTP experts in 2 bits too).
expert_bits = int(sys.argv[5]) if len(sys.argv) > 5 else bits

inv = json.load(open(os.path.join(src, "mtp-inventory.json")))
verified = os.path.join(src, "tensors", "verified.json")
if not os.path.exists(verified):
    sys.exit("tensors not verified yet: run `mtp_fetch.py verify --out <src>` first")

np_dtype = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32}
tensors = {}
for t in inv["tensors"]:
    raw = np.fromfile(os.path.join(src, "tensors", t["name"] + ".bin"), dtype=np_dtype[t["dtype"]])
    a = mx.array(raw.reshape(t["shape"]))
    if t["dtype"] == "BF16":
        a = a.view(mx.bfloat16)
    tensors[t["name"]] = a
print(f"read {len(tensors)} tensors, {sum(t['bytes'] for t in inv['tensors']) / 1e9:.2f} GB", flush=True)

source_config = json.load(open(os.path.join(target, "config.json")))
splitter = Qwen4ExpMTPSplitter()
text_config = splitter.read_text_config(source_config)
# Niwaki narrowed its routed experts (moe_intermediate_size 448); the MTP layer
# keeps the base model's 640. The tensors decide.
text_config["moe_intermediate_size"] = int(tensors["mtp.layers.0.mlp.experts.down_proj"].shape[-1])
text_config["shared_expert_intermediate_size"] = int(
    tensors["mtp.layers.0.mlp.shared_expert.gate_proj.weight"].shape[0])
weights = splitter.transform(tensors, text_config, source_is_mlx=False)
quantization = None
for key in list(weights):
    w = weights[key]
    if not splitter.should_quantize_key(key) or w.ndim < 2 or w.shape[-1] % 64:
        continue
    b = expert_bits if ".switch_mlp." in key else bits
    wq, scales, biases = mx.quantize(w, group_size=64, bits=b)
    stem = key[: -len(".weight")]
    weights[key], weights[stem + ".scales"], weights[stem + ".biases"] = wq, scales, biases
    quantization = {"group_size": 64, "bits": bits, "mode": "affine"}
if quantization is not None and expert_bits != bits:
    for proj in ("gate_proj", "up_proj", "down_proj"):
        quantization[f"layers.0.mlp.switch_mlp.{proj}"] = {
            "group_size": 64, "bits": expert_bits, "mode": "affine"}
mx.eval(list(weights.values()))

os.makedirs(out, exist_ok=True)
mx.save_safetensors(os.path.join(out, "model.safetensors"), weights, metadata={"format": "mlx"})
config = {
    "model_type": splitter.output_model_type,
    "text_config": text_config,
    "block_size": splitter.depth(text_config) + splitter.block_size_extra,
    "tie_word_embeddings": bool(text_config.get("tie_word_embeddings", False)),
}
if quantization is not None:
    config["quantization"] = quantization
    config["quantization_config"] = quantization
json.dump(dict(sorted(config.items())), open(os.path.join(out, "config.json"), "w"), indent=2)
for name in splitter.tokenizer_files:
    if os.path.exists(os.path.join(target, name)):
        shutil.copy(os.path.join(target, name), os.path.join(out, name))
size = os.path.getsize(os.path.join(out, "model.safetensors"))
print(f"drafter written to {out}: {size / 2**20:.0f} MiB, quantization {quantization}", flush=True)
for k in sorted(weights)[:6]:
    print("  ", k, weights[k].shape, weights[k].dtype)
