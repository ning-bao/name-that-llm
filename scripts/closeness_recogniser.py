"""Train the data study's closeness recogniser (notes/data_study_plan_2026-11-08.md, Phase 0).

A Qwen3-4B LoRA24 (the Exp 4c/4d recipe) on official rows only, no pool stage 1, so it knows nothing but the
official style. Five prompt-grouped folds over all 150 prompts (nllm.closeness.prompt_folds); each fold's model
is saved under OUT/fold<k>/ and scores its 30 held-out prompts. The out-of-fold recall per label is the
denominator of every recognition ratio, and scripts/closeness_instrument.py scores new cells with the same five
models. Resumes: a fold with a saved model and logits is skipped.

    uv run --extra nn python scripts/closeness_recogniser.py [--out outputs/closeness/recogniser-4b]
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

from nllm import closeness, encoder
from nllm.data import LABELS, load_train
from nllm.metrics import balanced_accuracy
from nllm.paths import OUTPUTS

BACKBONE = "qwen3-4b-base"


def config(lr: float, epochs: int, micro_batch: int) -> encoder.TrainConfig:
    return encoder.TrainConfig(BACKBONE, lr, epochs, micro_batch=micro_batch, pool_layers=encoder.LORA24_LAYERS,
                               head_lr_mult=10.0, centred_weight=1.0, schedule="cosine", batch_size=18)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUTPUTS / "closeness" / "recogniser-4b")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(encoder.BACKBONES[BACKBONE], allow_patterns=["*.json", "*.txt", "*.safetensors"])
    cfg = config(args.lr, args.epochs, args.micro_batch)
    official = load_train().reset_index(drop=True)
    folds = closeness.prompt_folds(official.prompt_id, 5, seed=0)
    (args.out / "config.json").write_text(json.dumps(
        {"backbone": BACKBONE, "revision": Path(model_path).name, "train_config": asdict(cfg), "seed": args.seed,
         "folds": [sorted(f) for f in folds], "rows": len(official)}, indent=2), encoding="utf-8")

    parts = []
    for k, val in enumerate(folds):
        save, path = args.out / f"fold{k}", args.out / f"fold{k}.npz"
        v = official.prompt_id.isin(val).to_numpy()
        tr, va = official[~v], official[v]
        if not (path.exists() and (save / "pooled.json").exists()):
            logits, history = encoder.fit_predict(model_path, tr.text, tr.model, va.text, cfg, args.seed,
                                                  save_dir=save, train_groups=tr.prompt_id.to_numpy())
            epochs = sorted(logits)
            np.savez(path, epochs=np.array(epochs), logits=np.stack([logits[e] for e in epochs]),
                     ids=va.id.to_numpy(dtype=str), history=np.array(json.dumps(history)))
            print(f"fold {k}: {len(tr)} train rows, {history[-1]['train_seconds']:.0f}s, "
                  f"peak {history[-1]['peak_gb']:.1f} GB", flush=True)
        with np.load(path) as z:
            for i, e in enumerate(z["epochs"]):
                part = va[["id", "prompt_id", "model"]].assign(fold=k, epoch=int(e))
                parts.append(pd.concat([part.reset_index(drop=True),
                                        pd.DataFrame(z["logits"][i], columns=[f"score:{m}" for m in LABELS])], axis=1))
    oof = pd.concat(parts, ignore_index=True)
    oof["pred"] = np.asarray(LABELS)[oof[[f"score:{m}" for m in LABELS]].to_numpy().argmax(axis=1)]
    oof.to_parquet(args.out / "oof.parquet", index=False)
    by_epoch = oof.groupby("epoch").apply(lambda g: balanced_accuracy(g.model, g.pred), include_groups=False)
    print("out-of-fold BA by epoch:", by_epoch.round(4).to_dict())
    last = oof[oof.epoch == args.epochs]
    recall = closeness.recall_by_label(last.model, last.pred)
    recall.rename("oof_recall").to_csv(args.out / "recall.csv")
    print(recall.round(3).to_string())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
