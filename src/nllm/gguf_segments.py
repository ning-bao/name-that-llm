"""Read GGUF layer segments of the open-weight candidate models without ever loading a model whole.

GLM-5.3-Flash and GLM-5.3 are two of the 18 candidates and their weights are public. Kaggle hosts them cut into
GGUF files of a few layers each (`jinghengwu/ntl-flash-iq3xxs-segs`, `jinghengwu/ntl-glm53-iq3xxs-segs-a` / `-b`),
117 GB and 278 GB at about 3 bits per weight. This module reads single tensors out of such files:

- `read_header` parses a file's metadata and tensor table (it works on a file that is still downloading);
- `Segments` finds a tensor in a directory of segments and unpacks it; the routed experts (97% of the weights)
  are unpacked on the GPU a few experts at a time, by decoders checked against gguf-py's (tests/);
- `load_layer` fills one transformers decoder layer with everything but its routed experts.

Tensor names are llama.cpp's (its `conversion/glm.py` wrote the files). LAYER_NAMES and `Segments.layer_tensor`
undo that converter's renames and reshapes; every shape is checked against the transformers module. Needs the
`gguf` package (the `candidates` extra) for the small tensors' decoders.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import torch

SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
# The library's _keep_in_fp32_modules_strict for these architectures.
FP32_NAMES = ("e_score_correction_bias", "conv1d", "dt_bias", "A_log")
# transformers layer parameter -> GGUF tensor name inside `blk.{i}.`, where the converter only renames.
LAYER_NAMES = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
    "mlp.gate.weight": "ffn_gate_inp.weight",
    "mlp.gate.e_score_correction_bias": "exp_probs_b.bias",
    "mlp.shared_experts.gate_proj.weight": "ffn_gate_shexp.weight",
    "mlp.shared_experts.up_proj.weight": "ffn_up_shexp.weight",
    "mlp.shared_experts.down_proj.weight": "ffn_down_shexp.weight",
    # sparse (MLA) attention and its indexer
    "self_attn.q_a_proj.weight": "attn_q_a.weight",
    "self_attn.q_a_layernorm.weight": "attn_q_a_norm.weight",
    "self_attn.q_b_proj.weight": "attn_q_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn_kv_a_mqa.weight",
    "self_attn.kv_a_layernorm.weight": "attn_kv_a_norm.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.indexer.wq_b.weight": "indexer.attn_q_b.weight",
    "self_attn.indexer.wk.weight": "indexer.attn_k.weight",
    "self_attn.indexer.k_norm.weight": "indexer.k_norm.weight",
    "self_attn.indexer.k_norm.bias": "indexer.k_norm.bias",
    "self_attn.indexer.weights_proj.weight": "indexer.proj.weight",
    "self_attn.indexer.index_kpool_compress_ape": "indexer_compressor_ape.weight",
    "self_attn.indexer.index_kpool_compress_gate": "indexer_compressor_gate.weight",
    # linear (KDA) attention, GLM-5.3-Flash only
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.b_proj.weight": "ssm_beta.weight",
    "self_attn.g_a_proj.weight": "ssm_g_a.weight",
    "self_attn.g_b_proj.weight": "ssm_g_b.weight",
    "self_attn.o_norm.weight": "ssm_norm.weight",
    "self_attn.forget_gate.dt_bias": "ssm_dt.bias",
    "self_attn.forget_gate.f_a_proj.weight": "ssm_f_a.weight",
    "self_attn.forget_gate.f_b_proj.weight": "ssm_f_b.weight",
    # hyper-connections, GLM-5.3-Flash only
    **{f"{part}_hc.{x}": f"hc_{part}_{x}.weight" for part in ("attn", "ffn") for x in ("fn", "base", "scale")},
}


def _read(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]


def _string(f) -> str:
    return f.read(_read(f, "<Q")).decode("utf-8", errors="replace")


def _value(f, kind: int):
    if kind in SCALAR:
        return _read(f, SCALAR[kind])
    if kind == 8:
        return _string(f)
    if kind == 9:
        item, n = _read(f, "<I"), _read(f, "<Q")
        return [_value(f, item) for _ in range(n)]
    raise ValueError(f"unknown GGUF value type {kind}")


def read_header(path, long_array: int = 4096) -> tuple[dict, dict]:
    """(metadata, {tensor name: (shape, GGML type id, absolute byte offset, number of elements)}) of a GGUF file.

    Only the header is read. Shapes are in numpy order (GGUF stores dimensions reversed). Metadata arrays longer
    than `long_array` (the vocabulary) are replaced by a note."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        _version, n_tensors, n_kv = _read(f, "<I"), _read(f, "<Q"), _read(f, "<Q")
        meta = {}
        for _ in range(n_kv):
            key, kind = _string(f), _read(f, "<I")
            value = _value(f, kind)
            meta[key] = f"<array of {len(value)}>" if isinstance(value, list) and len(value) > long_array else value
        infos = []
        for _ in range(n_tensors):
            name, n_dims = _string(f), _read(f, "<I")
            dims = [_read(f, "<Q") for _ in range(n_dims)]
            infos.append((name, tuple(reversed(dims)), _read(f, "<I"), _read(f, "<Q")))
        align = meta.get("general.alignment", 32)
        base = (f.tell() + align - 1) // align * align
    return meta, {name: (shape, kind, base + off, int(np.prod(shape, dtype=np.int64))) for name, shape, kind, off in infos}


def _bits(x: torch.Tensor, shifts, mask: int) -> torch.Tensor:
    """x[..., None] >> shifts & mask, as in gguf-py's numpy decoders."""
    return (x.unsqueeze(-1) >> torch.tensor(shifts, dtype=torch.uint8, device=x.device)) & mask


def dequant_iq2_s(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """gguf.quants.IQ2_S.dequantize_blocks in torch: uint8 [n, 82] -> float32 [n, 256]; `grid` is [1024, 8]."""
    n = blocks.shape[0]
    d = blocks[:, :2].contiguous().view(torch.float16).float()
    qs, signs, qh, scales = blocks[:, 2:34], blocks[:, 34:66], blocks[:, 66:74], blocks[:, 74:82]
    db = (d * (0.5 + _bits(scales, [0, 4], 0x0F).reshape(n, -1).float()) * 0.25).reshape(n, -1, 1, 1)
    sign = torch.where(_bits(signs, list(range(8)), 1) == 0, 1.0, -1.0).reshape(n, -1, 2, 8)
    index = qs.to(torch.int64) | (_bits(qh, [0, 2, 4, 6], 3).reshape(n, -1).to(torch.int64) << 8)
    return (db * grid[index].reshape(n, -1, 2, 8) * sign).reshape(n, -1)


def dequant_iq3_s(blocks: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """gguf.quants.IQ3_S.dequantize_blocks in torch: uint8 [n, 110] -> float32 [n, 256]; `grid` is [512, 4]."""
    n = blocks.shape[0]
    d = blocks[:, :2].contiguous().view(torch.float16).float()
    qs, qh, signs, scales = blocks[:, 2:66], blocks[:, 66:74], blocks[:, 74:106], blocks[:, 106:110]
    db = (d * (1 + 2 * _bits(scales, [0, 4], 0x0F).reshape(n, -1).float())).reshape(n, -1, 1, 1)
    sign = torch.where(_bits(signs, list(range(8)), 1) == 0, 1.0, -1.0).reshape(n, -1, 4, 8)
    index = qs.to(torch.int64) | (_bits(qh, list(range(8)), 1).reshape(n, -1).to(torch.int64) << 8)
    return (db * grid[index].reshape(n, -1, 4, 8) * sign).reshape(n, -1)


def torch_decoders() -> dict:
    """{GGML type: (torch decoder, gguf-py reference class)} for the formats the experts are stored in."""
    from gguf import quants
    from gguf.constants import GGMLQuantizationType as T

    return {T.IQ2_S: (dequant_iq2_s, quants.IQ2_S), T.IQ3_S: (dequant_iq3_s, quants.IQ3_S)}


class Segments:
    """The tensors of every *.gguf file in a directory, by GGUF name. `dtype` is what weights are unpacked to."""

    def __init__(self, directory, dtype: torch.dtype = torch.bfloat16):
        self.dtype, self.where, self.maps, self.grids = dtype, {}, {}, {}
        for path in sorted(Path(directory).glob("*.gguf")):
            try:
                tensors = read_header(path)[1]
            except (struct.error, ValueError, OSError):  # a download that has not written its header yet
                continue
            for name, info in tensors.items():
                self.where.setdefault(name, (path, *info))

    def raw(self, name: str):
        """(raw bytes as uint8 [n_blocks, block bytes], shape, GGML type)."""
        from gguf.constants import GGML_QUANT_SIZES, GGMLQuantizationType

        path, shape, kind, offset, n = self.where[name]
        kind = GGMLQuantizationType(kind)
        block, size = GGML_QUANT_SIZES[kind]
        if path not in self.maps:
            self.maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return self.maps[path][offset: offset + n // block * size].reshape(-1, size), shape, kind

    def numpy(self, name: str) -> np.ndarray:
        """A whole tensor as float32, by gguf-py's decoders (for everything but the routed experts)."""
        from gguf import quants

        data, shape, kind = self.raw(name)
        if kind.name == "F32":
            return np.frombuffer(data.tobytes(), dtype=np.float32).reshape(shape).copy()
        byte_shape = quants.quant_shape_to_byte_shape(shape, kind)
        return quants.dequantize(np.ascontiguousarray(data).reshape(byte_shape), kind)

    def stacked(self, name: str, ids: list[int], device) -> torch.Tensor:
        """Slices `ids` along the first axis of a large tensor (the experts), unpacked on `device`."""
        from gguf import quants

        data, shape, kind = self.raw(name)
        per = data.shape[0] // shape[0]
        part = np.concatenate([data[e * per: (e + 1) * per] for e in ids])
        decoders = torch_decoders()
        if kind in decoders:
            fn, ref = decoders[kind]
            if (kind, device) not in self.grids:
                ref.init_grid()
                self.grids[kind, device] = torch.from_numpy(ref.grid.reshape(ref.grid_shape).copy()).to(device)
            out = fn(torch.from_numpy(part).to(device), self.grids[kind, device])
            return out.reshape(len(ids), *shape[1:]).to(self.dtype)
        byte_shape = quants.quant_shape_to_byte_shape((len(ids), *shape[1:]), kind)
        return torch.from_numpy(quants.dequantize(part.reshape(byte_shape), kind)).to(device, self.dtype)

    def experts(self, i: int, device):
        """A function ids -> (gate_up [n, 2I, D], down [n, D, I]) for those routed experts of layer i."""
        def read(ids: list[int]):
            gate = self.stacked(f"blk.{i}.ffn_gate_exps.weight", ids, device)
            up = self.stacked(f"blk.{i}.ffn_up_exps.weight", ids, device)
            return torch.cat([gate, up], dim=1), self.stacked(f"blk.{i}.ffn_down_exps.weight", ids, device)
        return read

    def layer_tensor(self, i: int, name: str) -> np.ndarray:
        """Parameter `name` of decoder layer i as transformers holds it."""
        b = f"blk.{i}."
        if name == "self_attn.conv1d.weight":  # one grouped convolution over q, k, v; stored as three
            return np.concatenate([self.numpy(f"{b}ssm_conv1d_{x}.weight").reshape(-1, 1, 4) for x in "qkv"], axis=0)
        if name == "self_attn.forget_gate.A_log":  # stored as -exp(A_log)
            return np.log(-self.numpy(b + "ssm_a"))
        if name == "self_attn.kv_b_proj.weight":  # stored per head as k_b (transposed) and v_b
            k_b, v_b = self.numpy(b + "attn_k_b.weight"), self.numpy(b + "attn_v_b.weight")
            return np.concatenate([k_b.transpose(0, 2, 1), v_b], axis=1).reshape(-1, k_b.shape[1])
        return self.numpy(b + LAYER_NAMES[name])


def set_tensor(module: torch.nn.Module, name: str, value: torch.Tensor) -> None:
    """Replace a parameter or buffer of `module`, addressed by dotted name, with `value`."""
    *path, leaf = name.split(".")
    for part in path:
        module = getattr(module, part)
    if leaf in module._parameters:
        module._parameters[leaf] = torch.nn.Parameter(value, requires_grad=False)
    else:
        module._buffers[leaf] = value


def load_layer(layer_class, cfg, segments: Segments, i: int, device):
    """Decoder layer i (a transformers `layer_class(cfg, i)`) with every tensor on `device` except the routed
    experts, which stay unallocated: candidate_lm reads those a few at a time."""
    with torch.device("meta"):
        layer = layer_class(cfg, i)
    for name, meta in {**dict(layer.named_parameters()), **dict(layer.named_buffers())}.items():
        if name.startswith("mlp.experts."):
            continue
        value = torch.from_numpy(np.ascontiguousarray(segments.layer_tensor(i, name)))
        if tuple(value.shape) != tuple(meta.shape):
            raise ValueError(f"layer {i} {name}: GGUF shape {tuple(value.shape)}, expected {tuple(meta.shape)}")
        keep32 = any(k in name for k in FP32_NAMES)
        set_tensor(layer, name, value.to(device, torch.float32 if keep32 else segments.dtype))
    return layer.eval()
