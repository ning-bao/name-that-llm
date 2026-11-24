"""Label-conditioned language model: a generative classifier (experiment 7).

One causal language model (frozen bf16 base weights, LoRA adapters on every linear projection) learns to continue
passages written by each label, with the label's name as a prefix: "gpt-5.6-luna\\n" + passage + EOS, the loss
counting passage tokens only. A passage is then scored under all 18 prefixes. The summed log-probability of its
tokens given a prefix approximates log p(passage | label); the prediction is the label with the highest score
plus a per-label offset. The offsets equalise prediction rates on held-out pool rows, which removes the model's
preference for labels whose text it finds easier overall without looking at any official validation row.

This is the offline stand-in for comparing each real model's next-token probabilities, which would require
sending competition text to the providers' APIs. Requires the `nn` extra; runs on CUDA (CPU only for tests).
"""

import math
import time
from dataclasses import dataclass

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.functional import cross_entropy
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from nllm.data import LABELS
from nllm.encoder import LORA_TARGETS, length_grouped_batches, seed_everything, share_machine

BACKBONES = {"qwen3-1.7b-base": "Qwen/Qwen3-1.7B-Base", "qwen3-4b-base": "Qwen/Qwen3-4B-Base"}


@dataclass(frozen=True)
class LMConfig:
    backbone: str = "qwen3-1.7b-base"
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 32
    micro_batch: int = 8
    warmup: float = 0.06
    weight_decay: float = 0.0
    max_length: int = 384
    grad_clip: float = 1.0
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    gradient_checkpointing: bool = True


def load_lm(model_path: str, cfg: LMConfig, device):
    """(tokenizer, model): frozen bf16 base with fp32 LoRA adapters; padding reuses EOS (masked out of the loss)."""
    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, attn_implementation="sdpa")
    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    lora = LoraConfig(task_type=TaskType.CAUSAL_LM, r=cfg.lora_rank, lora_alpha=cfg.lora_alpha,
                      lora_dropout=cfg.lora_dropout, target_modules=list(LORA_TARGETS))
    model = get_peft_model(model, lora)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return tok, model.to(device)


def prefixes(tok) -> list[list[int]]:
    """Token ids of each label's prefix, in LABELS order."""
    return [tok(f"{m}\n", add_special_tokens=False)["input_ids"] for m in LABELS]


def passage_ids(tok, texts, cfg: LMConfig) -> list[list[int]]:
    """Raw passages as ids ending in EOS; special-token strings are read as text."""
    longest = cfg.max_length - max(len(p) for p in prefixes(tok)) - 1
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=longest,
              split_special_tokens=True)["input_ids"]
    return [i + [tok.eos_token_id] for i in ids]


def _batch(seqs: list[tuple[list[int], list[int]]], pad_id: int, device):
    """(input ids, attention mask, targets) for prefix+passage pairs; targets are -100 except passage tokens."""
    width = max(len(p) + len(x) for p, x in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), width), dtype=torch.long)
    target = torch.full((len(seqs), width), -100, dtype=torch.long)
    for row, (p, x) in enumerate(seqs):
        seq = p + x
        ids[row, : len(seq)] = torch.tensor(seq)
        mask[row, : len(seq)] = 1
        target[row, len(p) : len(seq)] = torch.tensor(x)
    return ids.to(device), mask.to(device), target.to(device)


def _chunk_nll(head, hidden, target):
    with torch.autocast(hidden.device.type, dtype=torch.bfloat16):
        logits = head(hidden)
    return cross_entropy(logits.float(), target, reduction="none")


def _token_logprobs(model, ids, mask, target, chunk: int = 1024) -> torch.Tensor:
    """Per-row summed log-probability of the target tokens (next-token prediction, shifted by one).

    The decoder body runs once; only target positions go through the vocabulary head, `chunk` tokens at a time and
    recomputed in the backward pass, so full (rows, length, vocabulary) logits are never held in memory.
    """
    lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    with torch.autocast(ids.device.type, dtype=torch.bfloat16):
        hidden = lm.model(input_ids=ids, attention_mask=mask).last_hidden_state[:, :-1]
    t = target[:, 1:]
    keep = t != -100
    h, y, rows = hidden[keep], t[keep], keep.nonzero()[:, 0]
    parts = []
    for start in range(0, len(y), chunk):
        args = (lm.lm_head, h[start : start + chunk], y[start : start + chunk])
        parts.append(checkpoint(_chunk_nll, *args, use_reentrant=False) if torch.is_grad_enabled()
                     else _chunk_nll(*args))
    nll = torch.cat(parts)
    return -torch.zeros(len(ids), dtype=nll.dtype, device=nll.device).index_add(0, rows, nll)


def fit(model_path: str, texts, labels, cfg: LMConfig, seed: int):
    """Train the adapters on (label prefix, passage) pairs; returns (tokenizer, model, per-epoch history)."""
    share_machine()
    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    tok, model = load_lm(model_path, cfg, device)
    pre = prefixes(tok)
    data = [(pre[LABELS.index(m)], x) for m, x in zip(labels, passage_ids(tok, texts, cfg), strict=True)]
    lengths = np.array([len(p) + len(x) for p, x in data])
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay, fused=True)
    total = math.ceil(len(data) / cfg.batch_size) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, round(cfg.warmup * total), total)
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        started, loss_sum, tokens = time.time(), 0.0, 0
        for batch in length_grouped_batches(lengths, cfg.batch_size, rng):
            optimizer.zero_grad(set_to_none=True)
            n_tokens = sum(len(data[i][1]) for i in batch)
            for start in range(0, len(batch), cfg.micro_batch):
                ids, mask, target = _batch([data[i] for i in batch[start : start + cfg.micro_batch]],
                                           tok.eos_token_id, device)
                loss = -_token_logprobs(model, ids, mask, target).sum() / n_tokens
                loss.backward()
                loss_sum += loss.item() * n_tokens
            tokens += n_tokens
            torch.nn.utils.clip_grad_norm_(trainable, cfg.grad_clip)
            optimizer.step()
            scheduler.step()
        history.append({"epoch": epoch, "train_nll_per_token": loss_sum / tokens,
                        "train_seconds": time.time() - started, "tokens": tokens})
    history[-1]["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    return tok, model, history


@torch.no_grad()
def score(tok, model, texts, cfg: LMConfig, batch_size: int = 32) -> np.ndarray:
    """log p(passage | label) for every passage and label: an (n, 18) array in LABELS order."""
    model.eval()
    device = next(model.parameters()).device
    pre = prefixes(tok)
    xs = passage_ids(tok, texts, cfg)
    pairs = [(i, j) for i in range(len(xs)) for j in range(len(LABELS))]
    pairs.sort(key=lambda ij: len(xs[ij[0]]))
    out = np.empty((len(xs), len(LABELS)), dtype=np.float64)
    for start in range(0, len(pairs), batch_size):
        chunk = pairs[start : start + batch_size]
        ids, mask, target = _batch([(pre[j], xs[i]) for i, j in chunk], tok.eos_token_id, device)
        for (i, j), v in zip(chunk, _token_logprobs(model, ids, mask, target).cpu().numpy(), strict=True):
            out[i, j] = v
    return out


def balancing_offsets(scores: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Per-label offsets that make argmax(scores + offsets) predict every label about equally often.

    Starts from minus each label's mean score relative to the row mean, then nudges over-predicted labels down
    and under-predicted ones up, in steps scaled to the spread of relative scores (log-probabilities in nats).
    """
    relative = scores - scores.mean(axis=1, keepdims=True)
    offsets = -relative.mean(axis=0)
    step = 0.05 * float(relative.std())
    target = 1.0 / scores.shape[1]
    for _ in range(iterations):
        share = np.bincount((scores + offsets).argmax(axis=1), minlength=scores.shape[1]) / len(scores)
        offsets -= step * np.log(np.maximum(share, 0.2 * target) / target)
    return offsets
