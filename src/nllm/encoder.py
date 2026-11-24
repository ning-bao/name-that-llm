"""Fine-tuned encoder and decoder classifiers (report section D2, experiments 3 and 4).

Encoders are fully fine-tuned with an 18-way head and plain cross-entropy on the raw
passage string (every candidate tokenizer round-trips the dev passages exactly, see
scripts/audit_tokenizers.py). Decoders (experiment 4) keep bf16 base weights frozen and
train LoRA adapters on every linear projection plus a full fp32 head, pooled at an EOS
appended to the passage. One fit trains on a fixed epoch schedule and records held-out
logits after each requested epoch, so the epoch count is selected on inner folds without
extra fits; the outer refit reuses the schedule and stops at the chosen epoch.

Logit columns follow LABELS order. Requires the `nn` extra; runs on CUDA (CPU only for tests).
"""

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from safetensors.torch import load_file, save_file
from torch.nn.functional import cross_entropy
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from transformers.optimization import Adafactor

from nllm.data import LABEL_TO_INDEX, LABELS, family
from nllm.inference import POOLED_CONFIG, POOLED_HEADS, pad, pooled_features, predict_logits
from nllm.metrics import balanced_accuracy

BACKBONES = {
    "ettin-400m": "jhu-clsp/ettin-encoder-400m",
    "ettin-1b": "jhu-clsp/ettin-encoder-1b",
    "roberta-large": "FacebookAI/roberta-large",
    "qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
    "qwen3-14b-base": "Qwen/Qwen3-14B-Base",
    # A second decoder family for ensemble diversity (Apache 2.0, ungated; 48 layers, <|fim_pad|> in its vocab).
    "qwen2.5-14b": "Qwen/Qwen2.5-14B",
}
# Report D2: lr 1e-5 or 3e-5, effective batch 32-64, 5-10% warmup, 3-10 epochs, wd 0.01. A pilot
# inner fit put roberta-large at lr 5e-5 below 3e-5, so the grid is not extended upward.
LEARNING_RATES = (1e-5, 3e-5)
EPOCHS = 10

# Sharing the machine with other work. All optional; results are unchanged, only wall time grows.
#   NLLM_GPU_DUTY=0.5    pause after each optimizer step so training keeps the GPU busy about half the time
#   NLLM_LOW_PRIORITY=1  below-normal CPU priority for this process
#   NLLM_CPU_THREADS=2   cap torch's CPU threads
#   NLLM_EVAL_BATCH=16   rows per scoring batch (default 64); a pooled 8B keeps 25 hidden states per batch,
#                        which at 64 rows does not fit beside the weights on a 16 GB card
GPU_DUTY = float(os.environ.get("NLLM_GPU_DUTY", "1"))
EVAL_BATCH = int(os.environ.get("NLLM_EVAL_BATCH", "64"))
_machine_shared = False


def share_machine() -> None:
    """Apply NLLM_LOW_PRIORITY and NLLM_CPU_THREADS once per process."""
    global _machine_shared
    if _machine_shared:
        return
    _machine_shared = True
    if os.environ.get("NLLM_CPU_THREADS"):
        torch.set_num_threads(int(os.environ["NLLM_CPU_THREADS"]))
    if os.environ.get("NLLM_LOW_PRIORITY") == "1":
        if os.name == "nt":
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.windll.kernel32
            # Without these the pseudo-handle -1 is truncated to 32 bits and the call silently fails.
            kernel32.GetCurrentProcess.restype = wintypes.HANDLE
            kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            below_normal = 0x00004000
            if not kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), below_normal):
                print("warning: could not lower process priority", flush=True)
        else:
            os.nice(10)

# Experiment 4 decoders (report D2: LoRA r8-16, lr 5e-5 to 2e-4, accumulated batch 32-64).
# qwen3-1.7b-base is the decoder that fits the local 16 GB card with room to train on pool data.
DECODERS = frozenset({"qwen3-1.7b-base", "qwen3-4b-base", "qwen3-8b-base", "qwen3-14b-base", "qwen2.5-14b"})
# The public "LoRA24" recipe (notes/why_public_beats_us_2026-11-05.md): keep the first 24 decoder layers and
# mean-pool these depths, each through its own LayerNorm, into one linear head.
LORA24_LAYERS = (2, 4, 6, 8, 12, 16, 20, 24)
DECODER_LEARNING_RATES = (1e-4, 2e-4)
DECODER_EPOCHS = 8
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# Qwen3 pads with its EOS, which would make the appended EOS look like padding to the last-token
# pooling in *ForSequenceClassification. A distinct unused special token pads instead.
DECODER_PAD_TOKEN = "<|fim_pad|>"


@dataclass(frozen=True)
class TrainConfig:
    backbone: str
    lr: float
    epochs: int = EPOCHS  # schedule length; training may stop earlier at the last evaluated epoch
    batch_size: int = 32
    micro_batch: int = 16
    warmup: float = 0.06
    weight_decay: float = 0.01
    max_length: int = 512
    grad_clip: float = 1.0
    # "adamw", or "adafactor" (factored second moments, no first moment) where AdamW's two fp32 states per
    # weight do not fit: a full fine-tune of ettin-1b needs ~16.5 GB for weights, gradients and AdamW states.
    optimizer: str = "adamw"
    gradient_checkpointing: bool = False
    # Decoders only.
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Pooled decoders (PooledDecoder, the LoRA24 recipe): set pool_layers to mean-pool those hidden-state depths
    # instead of reading the last token; keep_layers drops every decoder layer above it (None: keep max(pool_layers)).
    pool_layers: tuple[int, ...] | None = None
    keep_layers: int | None = None
    # Any model: the head (and the pooled decoder's LayerNorms) train at lr * head_lr_mult.
    head_lr_mult: float = 1.0
    # Cross-entropy weights inversely proportional to class counts (for an unbalanced pool stage 1).
    class_weighted: bool = False
    # > 0 (pooled decoders, with train_groups): batches are units of up to 3 answers to the same prompt, and a
    # second head is trained on unit-centred features with this loss weight.
    centred_weight: float = 0.0
    # > 0 (with centred_weight and init_from): this share of each unit's partners is drawn in proportion to how
    # much the starting model confuses their label with the anchor row's (unit_batches, confusion_affinity), so
    # sibling labels answer the same prompt inside one unit. 0: units are a random cut of the prompt's rows.
    sibling_units: float = 0.0
    schedule: str = "linear"  # or "cosine" (both after the linear warmup)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def length_grouped_batches(lengths: np.ndarray, batch_size: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Shuffled batches of similar-length rows: sort within shuffled chunks of 16 batches, then shuffle batches."""
    order = rng.permutation(len(lengths))
    chunk = batch_size * 16
    batches = []
    for start in range(0, len(order), chunk):
        part = order[start : start + chunk]
        part = part[np.argsort(lengths[part], kind="stable")]
        batches += [part[i : i + batch_size] for i in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]


def unit_batches(groups, batch_size: int, rng: np.random.Generator, unit: int = 3, labels=None, affinity=None,
                 share: float = 0.0) -> list[list[np.ndarray]]:
    """Shuffled batches of whole units, each unit up to `unit` row indices sharing a group (prompt).

    Every row is in exactly one unit. A group's rows are shuffled and cut into units of `unit`, so a group of 18
    gives 6 units and a singleton group a unit of one. Units are shuffled and packed batch_size // unit per batch.

    With share > 0 a group's units are built one at a time instead (the number of units is the same). The first
    row left in the shuffled group is the unit's anchor. Each partner is then drawn from the rows left: with
    probability `share` in proportion to affinity[anchor's label, the row's label], otherwise uniformly. `labels`
    is a label index per row and `affinity` a labels x labels array >= 0 (confusion_affinity).
    """
    codes = pd.factorize(np.asarray(groups))[0]
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    units = []
    for rows in np.split(order, bounds):
        rows = rng.permutation(rows)
        if share > 0:
            units += confusable_units(rows, np.asarray(labels), affinity, share, unit, rng)
        else:
            units += [rows[i : i + unit] for i in range(0, len(rows), unit)]
    units = [units[i] for i in rng.permutation(len(units))]
    per = max(1, batch_size // unit)
    return [units[i : i + per] for i in range(0, len(units), per)]


def confusable_units(rows, labels: np.ndarray, affinity: np.ndarray, share: float, unit: int,
                     rng: np.random.Generator) -> list[np.ndarray]:
    """One group's shuffled `rows` as units of an anchor plus affinity-drawn partners (see unit_batches)."""
    left, units = list(rows), []
    while left:
        members = [left.pop()]
        near = affinity[labels[members[0]]]
        while left and len(members) < unit:
            p = np.full(len(left), 1.0 / len(left))
            weight = near[labels[left]]
            if weight.sum() > 0:
                p = (1 - share) * p + share * weight / weight.sum()
            members.append(left.pop(rng.choice(len(left), p=p)))
        units.append(np.array(members))
    return units


def confusion_affinity(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    """How much a model confuses each pair of labels, from its `logits` on rows with label indices `y`: entry
    (a, b) is the mean probability it gives label b on rows of label a, averaged with (b, a). The diagonal is 0."""
    z = np.asarray(logits, dtype=np.float64)
    prob = np.exp(z - z.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    mean = np.zeros((prob.shape[1], prob.shape[1]))
    for label in np.unique(y):
        mean[label] = prob[y == label].mean(axis=0)
    affinity = (mean + mean.T) / 2
    np.fill_diagonal(affinity, 0.0)
    return affinity


FAMILY_INDEX = np.array([sorted({family(m) for m in LABELS}).index(family(m)) for m in LABELS])


def same_family_pairs(batches, labels: np.ndarray) -> float:
    """Share of same-unit row pairs whose labels belong to one family (0.18 for random units of all 18 labels)."""
    same = total = 0
    for members in (u for batch in batches for u in batch):
        f = FAMILY_INDEX[labels[members]]
        same += int((f[:, None] == f[None, :]).sum() - len(f)) // 2
        total += len(f) * (len(f) - 1) // 2
    return same / max(total, 1)


class PooledDecoder(torch.nn.Module):
    """A truncated decoder read at several depths (the public LoRA24 recipe).

    For each passage the hidden states at `layers` (1-based layer outputs) are mean-pooled over its tokens,
    the first token masked, each depth through its own LayerNorm, and concatenated into `head` (18 logits).
    `centred_head` reads the same features after subtracting the mean of a unit of answers to the same prompt;
    it is trained alongside (TrainConfig.centred_weight) and saved, but predict_logits uses `head` only.
    The base model's final norm is an identity, so the top kept layer is read un-normalised like the others.
    """

    def __init__(self, base, layers, hidden_size: int):
        super().__init__()
        self.base = base
        self.layers = tuple(int(i) for i in layers)
        self.norms = torch.nn.ModuleList(torch.nn.LayerNorm(hidden_size) for _ in self.layers)
        width = hidden_size * len(self.layers)
        self.head = torch.nn.Linear(width, len(LABELS))
        self.centred_head = torch.nn.Linear(width, len(LABELS))

    def head_parameters(self) -> list:
        return [*self.norms.parameters(), *self.head.parameters(), *self.centred_head.parameters()]

    def features(self, input_ids, attention_mask) -> torch.Tensor:
        states = self.base(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True,
                           use_cache=False).hidden_states
        return pooled_features(states, attention_mask, self.layers, self.norms)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.head(self.features(input_ids, attention_mask)))

    def save(self, save_dir) -> None:
        """Adapters (peft's adapter files) plus pooled_heads.safetensors and pooled.json."""
        save_dir = Path(save_dir)
        self.base.save_pretrained(save_dir)
        heads = {f"norms.{k}": v for k, v in self.norms.state_dict().items()}
        heads |= {f"head.{k}": v for k, v in self.head.state_dict().items()}
        heads |= {f"centred_head.{k}": v for k, v in self.centred_head.state_dict().items()}
        save_file({k: v.detach().float().cpu().contiguous() for k, v in heads.items()}, save_dir / POOLED_HEADS)
        (save_dir / POOLED_CONFIG).write_text(json.dumps({"layers": list(self.layers),
                                                          "keep_layers": self.base.config.num_hidden_layers,
                                                          "hidden_size": self.norms[0].normalized_shape[0]}),
                                              encoding="utf-8")

    def load_heads(self, save_dir) -> None:
        state = load_file(Path(save_dir) / POOLED_HEADS)
        for name, module in (("norms", self.norms), ("head", self.head), ("centred_head", self.centred_head)):
            module.load_state_dict({k[len(name) + 1 :]: v for k, v in state.items() if k.startswith(name + ".")})


class CentredView(torch.nn.Module):
    """A PooledDecoder read through its centred head on uncentred features, for predict_logits.

    The head is linear, so its logits on features centred within any group of rows are these outputs minus their
    group mean, plus the head's bias. Saving them lets the twin rule be scored later for any grouping."""

    def __init__(self, model: PooledDecoder):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.model.centred_head(self.model.features(input_ids, attention_mask)))


def load_pooled_decoder(model_path: str, cfg: TrainConfig, device, init_from: str | None = None) -> PooledDecoder:
    """The base model truncated to cfg.keep_layers (weights above it are never loaded), bf16 and frozen, with
    LoRA adapters on LORA_TARGETS (new, or `init_from`'s) and fp32 heads."""
    keep = cfg.keep_layers or max(cfg.pool_layers)
    if max(cfg.pool_layers) > keep:
        raise ValueError(f"pool_layers {cfg.pool_layers} reach above keep_layers {keep}")
    config = AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = keep
    if getattr(config, "layer_types", None):
        config.layer_types = config.layer_types[:keep]
    base = AutoModel.from_pretrained(model_path, config=config, dtype=torch.bfloat16, attn_implementation="sdpa",
                                     device_map={"": device} if device.type == "cuda" else None)
    if len(base.layers) != keep:
        raise RuntimeError(f"expected {keep} layers, loaded {len(base.layers)}")
    base.norm = torch.nn.Identity()
    if cfg.gradient_checkpointing:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if init_from:
        base = PeftModel.from_pretrained(base, init_from, is_trainable=True)
    else:
        base = get_peft_model(base, LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                                               lora_dropout=cfg.lora_dropout, target_modules=list(LORA_TARGETS)))
    model = PooledDecoder(base, cfg.pool_layers, config.hidden_size)
    if init_from:
        saved = json.loads((Path(init_from) / POOLED_CONFIG).read_text(encoding="utf-8"))
        if saved["layers"] != list(model.layers) or saved["keep_layers"] != keep:
            raise ValueError(f"{init_from} was trained with {saved}, not layers {model.layers} / keep {keep}")
        model.load_heads(init_from)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model.to(device)


def load_classifier(model_path: str, cfg: TrainConfig, device, init_from: str | None = None) -> tuple:
    """(tokenizer, model, pad id) for one fit, with an untrained 18-way head unless `init_from` is given.

    Decoders load frozen bf16 base weights with LoRA adapters on LORA_TARGETS; the adapters and
    the fully trained head are kept in fp32, and padding uses DECODER_PAD_TOKEN.
    `init_from` is the save_dir of an earlier fit (a pool stage 1) to continue from: for encoders the
    whole fine-tuned model, for decoders its adapters and head on top of the same base weights.
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    if cfg.pool_layers:
        if cfg.backbone not in DECODERS:
            raise ValueError("pool_layers is for decoders")
        pad_id = tok.convert_tokens_to_ids(DECODER_PAD_TOKEN)
        if pad_id in (None, tok.unk_token_id, tok.eos_token_id):
            raise ValueError(f"{model_path} has no distinct {DECODER_PAD_TOKEN} token")
        return tok, load_pooled_decoder(model_path, cfg, device, init_from), pad_id
    if cfg.backbone not in DECODERS:
        model = AutoModelForSequenceClassification.from_pretrained(
            init_from or model_path, num_labels=len(LABELS), attn_implementation="sdpa"
        )
        if cfg.gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        return tok, model.to(device), tok.pad_token_id
    pad_id = tok.convert_tokens_to_ids(DECODER_PAD_TOKEN)
    if pad_id in (None, tok.unk_token_id, tok.eos_token_id):
        raise ValueError(f"{model_path} has no distinct {DECODER_PAD_TOKEN} token")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(LABELS), attn_implementation="sdpa", dtype=torch.bfloat16, pad_token_id=pad_id
    )
    if model.config.pad_token_id != pad_id:
        raise RuntimeError("pad_token_id did not reach the model config; pooling would read the wrong token")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if init_from:
        model = PeftModel.from_pretrained(model, init_from, is_trainable=True)
    else:
        lora = LoraConfig(
            task_type=TaskType.SEQ_CLS,  # also trains and saves the head ("score") in full
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=list(LORA_TARGETS),
        )
        model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return tok, model.to(device), pad_id


def encode(tok, texts, cfg: TrainConfig) -> list[list[int]]:
    """Token ids for raw passages. Decoders read special-token strings as text and end with EOS."""
    if cfg.backbone not in DECODERS:
        return tok(list(texts), truncation=True, max_length=cfg.max_length)["input_ids"]
    ids = tok(list(texts), truncation=True, max_length=cfg.max_length - 1, split_special_tokens=True)["input_ids"]
    return [i + [tok.eos_token_id] for i in ids]


def fit_predict(
    model_path: str,
    train_texts,
    train_labels,
    eval_texts,
    cfg: TrainConfig,
    seed: int,
    eval_epochs=None,
    save_dir=None,
    init_from=None,
    train_groups=None,
    twin_out: dict | None = None,
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Fine-tune on train rows; return ({epoch: eval logits}, per-epoch history).

    `model_path` is a local snapshot of BACKBONES[cfg.backbone], so every fit loads the same revision.
    `save_dir`, if given, receives the final model (for decoders: the adapters and head) and tokenizer.
    `init_from`, another fit's save_dir, starts this fit from that model (see load_classifier).
    `train_groups` (a prompt id per train row) is needed when cfg.centred_weight > 0: batches are then units of up
    to 3 answers to the same prompt (unit_batches), and the loss adds centred_weight x the cross-entropy of the
    centred head on unit-centred features (units of 2-3 rows; a single row has nothing to centre against).
    With cfg.sibling_units > 0 the units favour labels that the starting model (`init_from`) confuses on these
    training rows. `twin_out`, a dict, receives {epoch: (the centred head's outputs on the eval rows'
    uncentred features, its bias)} (see CentredView).
    """
    eval_epochs = sorted(set(eval_epochs or range(1, cfg.epochs + 1)))
    centred = cfg.centred_weight > 0
    if centred and (train_groups is None or not cfg.pool_layers):
        raise ValueError("centred_weight needs train_groups and a pooled decoder (pool_layers)")
    if cfg.sibling_units > 0 and not (centred and init_from):
        raise ValueError("sibling_units needs centred units (centred_weight) and a trained starting model (init_from)")
    if cfg.head_lr_mult != 1 and not cfg.pool_layers:
        raise ValueError("head_lr_mult is implemented for pooled decoders only")
    share_machine()
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok, model, pad_id = load_classifier(model_path, cfg, device, init_from)
    train_ids = encode(tok, train_texts, cfg)
    eval_ids = encode(tok, eval_texts, cfg)
    y = torch.tensor([LABEL_TO_INDEX[m] for m in train_labels])
    lengths = np.array([len(i) for i in train_ids])
    class_weight = None
    if cfg.class_weighted:
        counts = np.bincount(y.numpy(), minlength=len(LABELS)).astype(float)
        class_weight = torch.tensor(np.where(counts > 0, len(y) / (len(LABELS) * np.maximum(counts, 1)), 0.0),
                                    dtype=torch.float32, device=device)

    head = {id(p) for p in model.head_parameters()} if isinstance(model, PooledDecoder) else set()
    trainable = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in trainable if (p.ndim >= 2) == decay and (id(p) in head) == is_head],
               "weight_decay": cfg.weight_decay if decay else 0.0,
               "lr": cfg.lr * (cfg.head_lr_mult if is_head else 1.0)}
              for decay in (True, False) for is_head in (False, True)]
    groups = [g for g in groups if g["params"]]
    if cfg.optimizer == "adamw":
        optimizer = torch.optim.AdamW(groups, lr=cfg.lr, fused=True)
    elif cfg.optimizer == "adafactor":
        # A fixed lr on Adam's scale (no relative steps or parameter-scale updates), so the same lr grid applies.
        optimizer = Adafactor(groups, lr=cfg.lr, scale_parameter=False, relative_step=False, warmup_init=False)
    else:
        raise ValueError(f"unknown optimizer {cfg.optimizer!r}")
    rng = np.random.default_rng(seed)
    affinity = None
    if cfg.sibling_units > 0:  # the starting model has not been trained on these rows' prompts in a nested run
        affinity = confusion_affinity(predict_logits(model, train_ids, pad_id, batch_size=EVAL_BATCH), y.numpy())
    if centred:
        sizes = pd.Series(np.asarray(train_groups)).value_counts().to_numpy()
        steps_per_epoch = math.ceil(int(np.ceil(sizes / 3).sum()) / max(1, cfg.batch_size // 3))
    else:
        steps_per_epoch = math.ceil(len(train_ids) / cfg.batch_size)
    total_steps = steps_per_epoch * cfg.epochs
    make_schedule = {"linear": get_linear_schedule_with_warmup, "cosine": get_cosine_schedule_with_warmup}
    scheduler = make_schedule[cfg.schedule](optimizer, round(cfg.warmup * total_steps), total_steps)

    def micro_batches(batch):
        """(row indices, unit sizes or None) per micro-batch; units are never split across micro-batches."""
        if not centred:
            return [(batch[s : s + cfg.micro_batch], None) for s in range(0, len(batch), cfg.micro_batch)]
        per = max(1, cfg.micro_batch // 3)
        return [(np.concatenate(batch[s : s + per]), [len(u) for u in batch[s : s + per]])
                for s in range(0, len(batch), per)]

    logits, history = {}, []
    for epoch in range(1, eval_epochs[-1] + 1):
        model.train()
        started, loss_sum = time.time(), 0.0
        batches = (unit_batches(train_groups, cfg.batch_size, rng, labels=y.numpy(), affinity=affinity,
                                share=cfg.sibling_units) if centred
                   else length_grouped_batches(lengths, cfg.batch_size, rng))
        for batch in batches:
            step_started = time.time()
            optimizer.zero_grad(set_to_none=True)
            n_rows = sum(len(u) for u in batch) if centred else len(batch)
            for micro, unit_sizes in micro_batches(batch):
                input_ids, mask = pad([train_ids[i] for i in micro], pad_id, device)
                target = y[micro].to(device)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    if centred:
                        feats = model.features(input_ids, mask)
                        out = model.head(feats)
                    else:
                        out = model(input_ids=input_ids, attention_mask=mask).logits
                loss = cross_entropy(out.float(), target, weight=class_weight, reduction="sum") / n_rows
                if centred:
                    unit = torch.repeat_interleave(torch.arange(len(unit_sizes), device=device),
                                                   torch.tensor(unit_sizes, device=device))
                    sizes = torch.tensor(unit_sizes, device=device, dtype=torch.float32)
                    means = torch.zeros(len(unit_sizes), feats.shape[1], device=device).index_add_(
                        0, unit, feats.float()) / sizes[:, None]
                    keep = sizes[unit] >= 2
                    if keep.any():
                        centred_out = model.centred_head((feats.float() - means[unit])[keep])
                        loss = loss + cfg.centred_weight * cross_entropy(
                            centred_out, target[keep], weight=class_weight, reduction="sum") / n_rows
                loss.backward()
                loss_sum += loss.item() * n_rows
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            if GPU_DUTY < 1 and device.type == "cuda":
                torch.cuda.synchronize()
                time.sleep((time.time() - step_started) * (1 / GPU_DUTY - 1))
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_ids), "train_seconds": time.time() - started}
        if centred:
            record["same_family_pairs"] = same_family_pairs(batches, y.numpy())
        if epoch in eval_epochs:
            logits[epoch] = predict_logits(model, eval_ids, pad_id, batch_size=EVAL_BATCH)
            if twin_out is not None and centred:
                twin_out[epoch] = (predict_logits(CentredView(model), eval_ids, pad_id, batch_size=EVAL_BATCH),
                                      model.centred_head.bias.detach().float().cpu().numpy())
        history.append(record)

    history[-1]["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    if save_dir is not None:
        if isinstance(model, PooledDecoder):
            model.save(save_dir)
        else:
            model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits, history


def predict_saved(model_path: str, cfg: TrainConfig, saved_dir: str, texts) -> np.ndarray:
    """Logits (LABELS order, bf16 autocast as in fit_predict's evaluation) of the model an earlier fit saved to
    `saved_dir`, loaded the way a fit with init_from=saved_dir starts."""
    share_machine()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok, model, pad_id = load_classifier(model_path, cfg, device, saved_dir)
    logits = predict_logits(model, encode(tok, texts, cfg), pad_id, batch_size=EVAL_BATCH)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits


def select(inner_logits: dict[tuple[float, int], np.ndarray], y: np.ndarray) -> tuple[tuple[float, int], pd.DataFrame]:
    """Pick (lr, epoch) by inner out-of-fold balanced accuracy; ties go to fewer epochs, then the smaller lr."""
    table = pd.DataFrame(
        [
            {"lr": lr, "epoch": ep, "inner_ba": balanced_accuracy(y, np.asarray(LABELS)[s.argmax(axis=1)])}
            for (lr, ep), s in inner_logits.items()
        ]
    )
    best = table.sort_values(["inner_ba", "epoch", "lr"], ascending=[False, True, True]).iloc[0]
    return (float(best.lr), int(best.epoch)), table
