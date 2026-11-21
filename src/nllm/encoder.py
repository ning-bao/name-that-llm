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

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.functional import cross_entropy
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from nllm.data import LABEL_TO_INDEX, LABELS
from nllm.metrics import balanced_accuracy

BACKBONES = {
    "ettin-400m": "jhu-clsp/ettin-encoder-400m",
    "roberta-large": "FacebookAI/roberta-large",
    "qwen3-4b-base": "Qwen/Qwen3-4B-Base",
    "qwen3-8b-base": "Qwen/Qwen3-8B-Base",
}
# Report D2: lr 1e-5 or 3e-5, effective batch 32-64, 5-10% warmup, 3-10 epochs, wd 0.01. A pilot
# inner fit put roberta-large at lr 5e-5 below 3e-5, so the grid is not extended upward.
LEARNING_RATES = (1e-5, 3e-5)
EPOCHS = 10

# Experiment 4 decoders (report D2: LoRA r8-16, lr 5e-5 to 2e-4, accumulated batch 32-64).
DECODERS = frozenset({"qwen3-4b-base", "qwen3-8b-base"})
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
    # Decoders only.
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = False


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


def pad(ids: list[list[int]], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(len(i) for i in ids)
    input_ids = torch.full((len(ids), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(ids), width), dtype=torch.long)
    for row, i in enumerate(ids):
        input_ids[row, : len(i)] = torch.tensor(i)
        mask[row, : len(i)] = 1
    return input_ids.to(device), mask.to(device)


@torch.no_grad()
def predict_logits(model, ids: list[list[int]], pad_id: int, batch_size: int = 64) -> np.ndarray:
    """Logits in LABELS order, batched by length and returned in input order."""
    model.eval()
    order = np.argsort([len(i) for i in ids], kind="stable")
    out = np.empty((len(ids), len(LABELS)), dtype=np.float32)
    device = next(model.parameters()).device
    for start in range(0, len(order), batch_size):
        rows = order[start : start + batch_size]
        input_ids, mask = pad([ids[r] for r in rows], pad_id, device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            logits = model(input_ids=input_ids, attention_mask=mask).logits
        out[rows] = logits.float().cpu().numpy()
    return out


def load_classifier(model_path: str, cfg: TrainConfig, device) -> tuple:
    """(tokenizer, model, pad id) for one fit, with an untrained 18-way head.

    Decoders load frozen bf16 base weights with LoRA adapters on LORA_TARGETS; the adapters and
    the fully trained head are kept in fp32, and padding uses DECODER_PAD_TOKEN.
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    if cfg.backbone not in DECODERS:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(LABELS), attn_implementation="sdpa"
        )
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
) -> tuple[dict[int, np.ndarray], list[dict]]:
    """Fine-tune on train rows; return ({epoch: eval logits}, per-epoch history).

    `model_path` is a local snapshot of BACKBONES[cfg.backbone], so every fit loads the same revision.
    `save_dir`, if given, receives the final model (for decoders: the adapters and head) and tokenizer.
    """
    eval_epochs = sorted(set(eval_epochs or range(1, cfg.epochs + 1)))
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok, model, pad_id = load_classifier(model_path, cfg, device)
    train_ids = encode(tok, train_texts, cfg)
    eval_ids = encode(tok, eval_texts, cfg)
    y = torch.tensor([LABEL_TO_INDEX[m] for m in train_labels])
    lengths = np.array([len(i) for i in train_ids])

    trainable = [p for p in model.parameters() if p.requires_grad]
    decay = [p for p in trainable if p.ndim >= 2]
    no_decay = [p for p in trainable if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay, "weight_decay": cfg.weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=cfg.lr,
        fused=True,
    )
    total_steps = math.ceil(len(train_ids) / cfg.batch_size) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(cfg.warmup * total_steps), total_steps)
    rng = np.random.default_rng(seed)

    logits, history = {}, []
    for epoch in range(1, eval_epochs[-1] + 1):
        model.train()
        started, loss_sum = time.time(), 0.0
        for batch in length_grouped_batches(lengths, cfg.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, len(batch), cfg.micro_batch):
                micro = batch[start : start + cfg.micro_batch]
                input_ids, mask = pad([train_ids[i] for i in micro], pad_id, device)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, attention_mask=mask).logits
                loss = cross_entropy(out.float(), y[micro].to(device), reduction="sum") / len(batch)
                loss.backward()
                loss_sum += loss.item() * len(batch)
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
        record = {"epoch": epoch, "train_loss": loss_sum / len(train_ids), "train_seconds": time.time() - started}
        if epoch in eval_epochs:
            logits[epoch] = predict_logits(model, eval_ids, pad_id)
        history.append(record)

    history[-1]["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    if save_dir is not None:
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits, history


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
