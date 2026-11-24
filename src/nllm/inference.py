"""Scoring with a saved fine-tuned encoder or LoRA decoder: the Kaggle submission path (report section H3).

Needs only torch, tokenizers, safetensors and transformers' model classes, so a Kaggle notebook can import it
from a dataset without peft or the training code. Token ids come from the saved tokenizer.json through the
`tokenizers` library, independent of the installed transformers version; they match the training tokenizer
(encoder.encode) exactly (tests/test_inference.py). Logit columns follow LABELS order.
"""

import contextlib
import json
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from nllm.data import LABELS


def tokenize(model_dir, texts, max_length: int = 512) -> list[list[int]]:
    """Token ids with the model's special tokens, truncated like training (encoder.encode)."""
    tok = Tokenizer.from_file(str(Path(model_dir) / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length)
    return [e.ids for e in tok.encode_batch(list(texts))]


def load(model_dir, device):
    """The fine-tuned classifier in eval mode, from local files only."""
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_dir), attn_implementation="sdpa", local_files_only=True
    )
    if model.config.num_labels != len(LABELS):
        raise ValueError(f"{model_dir} has {model.config.num_labels} labels, expected {len(LABELS)}")
    return model.to(device).eval()


def tokenize_decoder(model_dir, texts, max_length: int = 512) -> list[list[int]]:
    """Decoder token ids like encoder.encode: special-token strings in the text are read as text, truncated to
    max_length - 1, then EOS."""
    model_dir = Path(model_dir)
    tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(max_length - 1)
    tok.encode_special_tokens = True
    eos = json.loads((model_dir / "tokenizer_config.json").read_text(encoding="utf-8"))["eos_token"]
    eos_id = tok.token_to_id(eos)
    return [e.ids + [eos_id] for e in tok.encode_batch(list(texts))]


def load_decoder(base_dir, adapter_dir, pad_id: int, dtype=torch.float16, device_map=None):
    """A LoRA-trained decoder classifier in eval mode: base weights from base_dir, the adapters (lora_A/lora_B)
    merged into them by hand as W += B @ A * alpha / r, and the trained head copied in. No peft needed.

    `device_map` is passed to from_pretrained ("auto" splits an 8B model across two T4s); None keeps it on the CPU.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForSequenceClassification

    adapter_dir = Path(adapter_dir)
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("use_dora") or config.get("use_rslora") or config.get("fan_in_fan_out"):
        raise ValueError(f"{adapter_dir}: only plain LoRA adapters are supported")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(base_dir), num_labels=len(LABELS), attn_implementation="sdpa", dtype=dtype, pad_token_id=pad_id,
        local_files_only=True, device_map=device_map,
    )
    if model.config.pad_token_id != pad_id:
        raise RuntimeError("pad_token_id did not reach the model config; pooling would read the wrong token")
    modules = dict(model.named_modules())
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    scale = config["lora_alpha"] / config["r"]
    merged, copied = 0, 0
    with torch.no_grad():
        for key, value in state.items():
            name = key.removeprefix("base_model.model.")
            if ".lora_A." in name:
                weight = modules[name.split(".lora_A.")[0]].weight
                b = state[key.replace(".lora_A.", ".lora_B.")]
                weight += (b.float() @ value.float() * scale).to(weight.device, weight.dtype)
                merged += 1
            elif ".lora_B." not in name:  # the fully trained head (modules_to_save)
                module, _, param = name.rpartition(".")
                getattr(modules[module], param).copy_(value)
                copied += 1
    if merged != len(config["target_modules"]) * model.config.num_hidden_layers or not copied:
        raise RuntimeError(f"{adapter_dir}: merged {merged} LoRA pairs and copied {copied} head tensors")
    return model.eval()


POOLED_HEADS, POOLED_CONFIG = "pooled_heads.safetensors", "pooled.json"


def pooled_features(states, attention_mask, layers, norms) -> torch.Tensor:
    """The pooled decoder's features (training and inference share this): for each depth in `layers`, the mean of
    states[depth] over the passage's tokens with the first token masked, through that depth's LayerNorm, in fp32,
    concatenated. States on another GPU (a model split by device_map) are moved to the mask's device."""
    mask = attention_mask.clone()
    mask[:, 0] = 0
    weights = (mask / mask.sum(1, keepdim=True).clamp(min=1)).unsqueeze(-1).float()
    return torch.cat([norm((states[i].to(weights.device).float() * weights).sum(1))
                      for i, norm in zip(layers, norms)], dim=-1)


class PooledClassifier(torch.nn.Module):
    """Inference form of encoder.PooledDecoder: a truncated base (adapters merged) plus the per-depth LayerNorms
    and the plain head. Returns an object with .logits, like a transformers classifier."""

    def __init__(self, base, layers, hidden_size: int):
        super().__init__()
        self.base = base
        self.layers = tuple(int(i) for i in layers)
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(hidden_size) for _ in self.layers)
        self.head = torch.nn.Linear(hidden_size * len(self.layers), len(LABELS))

    def forward(self, input_ids, attention_mask):
        states = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                           use_cache=False).hidden_states
        out = self.head(pooled_features(states, attention_mask, self.layers, self.norms))
        return type("Output", (), {"logits": out})()


def load_pooled_decoder(base_dir, adapter_dir, dtype=torch.float16, device_map=None):
    """A pooled (LoRA24-style) decoder in eval mode without peft: the base from base_dir truncated to the saved
    keep_layers (weights above are never loaded), final norm off, the adapters merged by hand as
    W += B @ A * alpha / r, and the LayerNorms and plain head (fp32) loaded from pooled_heads.safetensors.

    `device_map` goes to from_pretrained ("auto" may split the base across two T4s); None keeps it on the CPU.
    The heads sit on the embedding's device, where predict_logits sends the inputs.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    adapter_dir = Path(adapter_dir)
    pooled = json.loads((adapter_dir / POOLED_CONFIG).read_text(encoding="utf-8"))
    config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if config.get("use_dora") or config.get("use_rslora") or config.get("fan_in_fan_out"):
        raise ValueError(f"{adapter_dir}: only plain LoRA adapters are supported")
    keep = int(pooled["keep_layers"])
    base_config = AutoConfig.from_pretrained(str(base_dir), local_files_only=True)
    base_config.num_hidden_layers = keep
    if getattr(base_config, "layer_types", None):
        base_config.layer_types = base_config.layer_types[:keep]
    base = AutoModel.from_pretrained(str(base_dir), config=base_config, dtype=dtype, attn_implementation="sdpa",
                                     local_files_only=True, device_map=device_map)
    if len(base.layers) != keep:
        raise RuntimeError(f"expected {keep} layers, loaded {len(base.layers)}")
    base.norm = torch.nn.Identity()
    modules = dict(base.named_modules())
    state = load_file(str(adapter_dir / "adapter_model.safetensors"))
    scale = config["lora_alpha"] / config["r"]
    merged = 0
    with torch.no_grad():
        for key, value in state.items():
            name = key.removeprefix("base_model.model.")
            if ".lora_A." in name:
                weight = modules[name.split(".lora_A.")[0]].weight
                b = state[key.replace(".lora_A.", ".lora_B.")]
                weight += (b.float() @ value.float() * scale).to(weight.device, weight.dtype)
                merged += 1
            elif ".lora_B." not in name:
                raise RuntimeError(f"{adapter_dir}: unexpected adapter tensor {key}")
    if merged != len(config["target_modules"]) * keep:
        raise RuntimeError(f"{adapter_dir}: merged {merged} LoRA pairs, expected {len(config['target_modules']) * keep}")
    model = PooledClassifier(base, pooled["layers"], int(pooled["hidden_size"]))
    heads = load_file(str(adapter_dir / POOLED_HEADS))
    model.norms.load_state_dict({k[len("norms."):]: v for k, v in heads.items() if k.startswith("norms.")})
    model.head.load_state_dict({k[len("head."):]: v for k, v in heads.items() if k.startswith("head.")})
    device = base.embed_tokens.weight.device
    model.norms.to(device)
    model.head.to(device)
    return model.eval()


def pad(ids: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(i) for i in ids)
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for row, i in enumerate(ids):
        input_ids[row, : len(i)] = torch.tensor(i)
        mask[row, : len(i)] = 1
    return input_ids.to(device), mask.to(device)


@torch.no_grad()
def predict_logits(model, ids: list[list[int]], pad_id: int, batch_size: int = 64,
                   dtype: torch.dtype | None = torch.bfloat16) -> np.ndarray:
    """Logits in LABELS order, batched by length and returned in input order.

    `dtype` is the autocast precision (training scores under bf16); None runs in the weights' own fp32,
    which is what a T4 without bf16 should use.
    """
    model.eval()
    order = np.argsort([len(i) for i in ids], kind="stable")
    out = np.empty((len(ids), len(LABELS)), dtype=np.float32)
    device = next(model.parameters()).device
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        input_ids, mask = pad([ids[r] for r in rows], pad_id, device)
        with torch.autocast(device.type, dtype=dtype) if dtype is not None else contextlib.nullcontext():
            logits = model(input_ids=input_ids, attention_mask=mask).logits
        out[rows] = logits.float().cpu().numpy()
    return out
