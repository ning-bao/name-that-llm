"""Token statistics of passages under a small neutral language model, in candidate_likelihood.py's file format.

A candidate model's likelihood of a passage mixes "this is my own writing" with "this content is easy to predict".
A neutral model that is no candidate (a Qwen3 base model, plain text, no chat turn) measures the second part, so a
stacker can take the difference (scripts/candidate_eval.py reads several feature files).

    uv run --extra nn python scripts/reference_likelihood.py --backbone qwen3-1.7b-base --rows official \
        --out outputs/candidates/ref-qwen3-1.7b-official-001.npz
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from nllm import candidate_lm
from nllm.data import load_train
from nllm.encoder import BACKBONES
from nllm.splits import development_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3-1.7b-base", choices=sorted(k for k in BACKBONES if k.startswith("qwen")))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", choices=["dev", "official"], default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-passage-tokens", type=int, default=320)
    ap.add_argument("--max-tokens", type=int, default=12000, help="padded tokens per batch")
    args = ap.parse_args()

    path = snapshot_download(BACKBONES[args.backbone], allow_patterns=["*.json", "*.txt", "*.safetensors"])
    tok = AutoTokenizer.from_pretrained(path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device).eval()
    rows = development_rows(load_train()) if args.rows == "dev" else load_train()
    rows = rows.head(args.limit) if args.limit else rows
    ids = [tok(t, add_special_tokens=False, split_special_tokens=True)["input_ids"][: args.max_passage_tokens]
           for t in rows.text]
    pad = tok.convert_tokens_to_ids("<|fim_pad|>")
    head = model.get_output_embeddings().weight
    started, stats = time.time(), [None] * len(ids)
    with torch.no_grad():
        for where, padded, mask in candidate_lm.batches_of(ids, pad, args.max_tokens, 64):
            hidden = model.model(input_ids=padded.to(device), attention_mask=mask.to(device)).last_hidden_state
            targets, step = padded[:, 1:].to(device), max(1, 256 // padded.shape[1])
            parts = [candidate_lm.token_stats(hidden[r: r + step, :-1], head, targets[r: r + step])
                     for r in range(0, len(where), step)]
            for r, row in enumerate(where):
                n = int(mask[r].sum()) - 1
                stats[row] = {k: torch.cat([p[k] for p in parts])[r, :n].float().cpu().numpy() for k in parts[0]}
    feats = [candidate_lm.passage_features(st) for st in stats]
    print(f"{args.backbone}: {len(ids)} passages, {sum(map(len, ids))} tokens, {time.time() - started:.0f}s; "
          f"mean log-prob {np.mean([f['lp_mean'] for f in feats]):.3f}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=rows.id.to_numpy(dtype=str), feature_names=np.array(list(feats[0])),
             features=np.array([[f[k] for k in feats[0]] for f in feats], dtype=np.float32),
             lengths=np.array([len(st["lp"]) for st in stats]), model=np.array(f"ref-{args.backbone}"),
             **{f"tok_{k}": np.concatenate([st[k] for st in stats]).astype(np.float32) for k in stats[0]})
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
