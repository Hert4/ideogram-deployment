"""Run Ideogram 4's fp8 checkpoints on Apple Silicon GPU (MPS) via int8.

MPS has no float8 dtype, so the model's weight-only fp8 Linears cannot dequantize
on-device (`float8_e4m3fn -> bfloat16` is unimplemented on MPS). We re-quantize
each fp8 weight to int8 (per-output-row symmetric) — a cast MPS *does* support —
and swap in Int8Linear layers. Everything else in the model runs unchanged in
bf16 on MPS.

int8 weights are the same size as fp8 (1 byte), so this does not reduce memory;
its only purpose is to make on-GPU dequant possible. Per-row int8 generally
matches or beats e4m3 fp8 precision for bounded weights, so quality is preserved.
"""

import json
import posixpath

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors import safe_open

FP8 = torch.float8_e4m3fn
SCALE_SUFFIX = ".weight_scale"


class Int8Linear(nn.Module):
    """Linear holding an int8 weight + per-row fp32 scale; dequantizes per call.

    Mirrors ideogram4.quantized_loading.Fp8Linear but with int8 so the dequant
    cast is supported on MPS.
    """

    weight: torch.Tensor
    weight_scale: torch.Tensor
    bias: torch.Tensor | None

    def __init__(self, in_features, out_features, bias, compute_dtype):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = compute_dtype
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=torch.int8))
        self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
        if bias:
            self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
        else:
            self.bias = None

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
        bias = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, bias)


def swap_linears_to_int8(module, keys, compute_dtype, *, prefix=""):
    """Replace each nn.Linear that has a saved weight_scale with an Int8Linear.

    `keys` is any container supporting `in` (a set of checkpoint keys, or the fp8
    state_dict itself), so this is drop-in for swap_linears_to_fp8.
    """
    for name, child in list(module.named_children()):
        cp = f"{prefix}{name}"
        if isinstance(child, nn.Linear) and f"{cp}{SCALE_SUFFIX}" in keys:
            setattr(
                module, name,
                Int8Linear(child.in_features, child.out_features,
                           child.bias is not None, compute_dtype),
            )
        else:
            swap_linears_to_int8(child, keys, compute_dtype, prefix=f"{cp}.")


def requant_fp8_to_int8(w_fp8_cpu, scale_cpu):
    """fp8 weight + per-row scale -> int8 weight + new per-row scale (CPU only).

    The fp8->float32 cast is unsupported on MPS, so this must run on CPU tensors.
    """
    real = w_fp8_cpu.to(torch.float32) * scale_cpu.to(torch.float32).unsqueeze(1)
    amax = real.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    new_scale = amax / 127.0
    q = torch.clamp(torch.round(real / new_scale), -127, 127).to(torch.int8)
    return q, new_scale.squeeze(1).to(torch.float32)


def load_int8_state_dict(model, state_dict, device, dtype, *, assign=False, strict=True):
    """Drop-in for load_fp8_state_dict: re-quantizes fp8 -> int8 then loads.

    `state_dict` must be on CPU. Used for the text encoder (loaded first, so the
    transient cost of materializing `prepared` on-device is acceptable).
    """
    prepared = {}
    bases = {k[: -len(SCALE_SUFFIX)] for k in state_dict if k.endswith(SCALE_SUFFIX)}
    for base in bases:
        q, s = requant_fp8_to_int8(state_dict[base + ".weight"], state_dict[base + SCALE_SUFFIX])
        prepared[base + ".weight"] = q.to(device)
        prepared[base + SCALE_SUFFIX] = s.to(device)
    for k, v in state_dict.items():
        if k in prepared or k.endswith(SCALE_SUFFIX):
            continue
        if k.endswith(".weight") and k[: -len(".weight")] in bases:
            continue
        prepared[k] = v.to(device, dtype) if v.is_floating_point() else v.to(device)
    missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
    if unexpected:
        raise RuntimeError(f"unexpected int8 keys: {unexpected[:8]}")
    if missing and strict:
        raise RuntimeError(f"missing int8 keys: {missing[:8]}")
    if not assign:
        model.to(device)


# ---- streaming transformer loader (meta init -> low peak memory) ----

def _resolve_shards(repo, index_filename):
    try:
        idx = hf_hub_download(repo_id=repo, filename=index_filename)
        with open(idx) as f:
            wmap = json.load(f)["weight_map"]
        d = posixpath.dirname(index_filename)
        return [hf_hub_download(repo_id=repo, filename=posixpath.join(d, s) if d else s)
                for s in sorted(set(wmap.values()))]
    except EntryNotFoundError:
        single = index_filename.removesuffix(".index.json")
        return [hf_hub_download(repo_id=repo, filename=single)]


def _assign(model, key, tensor):
    obj = model
    parts = key.split(".")
    for p in parts[:-1]:
        obj = getattr(obj, p)
    leaf = parts[-1]
    if leaf in obj._parameters:
        obj._parameters[leaf] = nn.Parameter(tensor, requires_grad=False)
    else:
        obj._buffers[leaf] = tensor


def stream_build_int8_transformer(repo, index_filename, tcfg, transformer_cls, device, dtype):
    """Build an Ideogram4 transformer with int8 weights, streaming straight to `device`.

    Meta init means the 9.3B params are never materialized as dense fp32 on CPU.
    Peak stays ~= the int8 model (~9.3 GB) plus a one-tensor transient.
    """
    shards = _resolve_shards(repo, index_filename)

    # Collect all keys + the (small) per-row scales up front.
    scales = {}
    for sp in shards:
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(SCALE_SUFFIX):
                    scales[k] = f.get_tensor(k)
    scale_key_set = set(scales.keys())

    with torch.device("meta"):
        model = transformer_cls(tcfg)
        model = model.to(dtype)
        swap_linears_to_int8(model, scale_key_set, dtype)

    for sp in shards:
        with safe_open(sp, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith(SCALE_SUFFIX):
                    continue
                base = k[: -len(".weight")] if k.endswith(".weight") else None
                if base is not None and base + SCALE_SUFFIX in scale_key_set:
                    q, s = requant_fp8_to_int8(f.get_tensor(k), scales[base + SCALE_SUFFIX])
                    _assign(model, k, q.to(device))
                    _assign(model, base + SCALE_SUFFIX, s.to(device))
                else:
                    v = f.get_tensor(k)
                    v = v.to(device, dtype) if v.is_floating_point() else v.to(device)
                    _assign(model, k, v)

    # Recompute the one computed buffer (rotary inv_freq), absent from the checkpoint.
    rmod = model.rotary_emb
    hd = rmod.head_dim
    inv_freq = 1.0 / (tcfg.rope_theta ** (torch.arange(0, hd, 2, dtype=torch.float32, device=device) / hd))
    rmod._buffers["inv_freq"] = inv_freq

    leftover = [n for n, p in list(model.named_parameters()) + list(model.named_buffers()) if p.is_meta]
    if leftover:
        raise RuntimeError(f"meta tensors left after load: {leftover[:8]}")
    model.eval()
    return model
