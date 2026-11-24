"""Score cells of generated passages for closeness to official (data study Phase 0/1, nllm.closeness).

A cell is a batch of passages with claimed labels from one source or generation setup. --calibrate builds the
known cells: our admitted answers by collection interface, and the Open Pool by source dataset (up to
--per-cell rows per label). --cells adds parquet files with columns model, text, cell (Phase 1 pilots).

Per (cell, label): recognition rate and ratio for the linear reference and, if trained
(scripts/closeness_recogniser.py), the official-only Qwen3-4B LoRA24 (mean over its five fold models); the share
put in the right provider family; style-only AUC vs official; word counts. Per (cell, family): within-family
transfer. Official = all 150 prompts.

    uv run --extra nn python -I scripts/closeness_instrument.py --calibrate [--cells pilot.parquet ...]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nllm import closeness, pool
from nllm.data import LABELS, load_train
from nllm.paths import OUTPUTS

NAMES = np.asarray(LABELS)


def calibration_cells(per_cell: int, seed: int = 17) -> pd.DataFrame:
    parts = []
    ours = pd.read_parquet(pool.POOL_DATA / "admitted.parquet")  # unfiltered: every interface is its own cell
    ours = ours[ours.split != "diagnostic"]
    for (m, iface), g in ours.groupby(["model", "interface"]):
        g = g.sample(min(per_cell, len(g)), random_state=seed)
        parts.append(pd.DataFrame({"model": m, "text": g.text.to_numpy(), "cell": f"ours:{iface}"}))
    p = pool.load_pool()
    for (m, src), g in p.groupby(["model", "source"]):
        if len(g) >= 100:
            g = g.sample(min(per_cell, len(g)), random_state=seed)
            parts.append(pd.DataFrame({"model": m, "text": g.text.to_numpy(), "cell": f"pool:{src}"}))
    return pd.concat(parts, ignore_index=True)


def lora_scores(recogniser: Path, texts: pd.Series) -> list[np.ndarray]:
    """Logits of each fold model of the official-only recogniser."""
    import torch
    from huggingface_hub import snapshot_download

    from nllm import encoder

    conf = json.loads((recogniser / "config.json").read_text(encoding="utf-8"))
    tc = conf["train_config"] | {"pool_layers": tuple(conf["train_config"]["pool_layers"])}
    cfg = encoder.TrainConfig(**tc)
    model_path = snapshot_download(encoder.BACKBONES[conf["backbone"]], revision=conf["revision"],
                                   allow_patterns=["*.json", "*.txt", "*.safetensors"])
    out = []
    for k in range(len(conf["folds"])):
        out.append(encoder.predict_saved(model_path, cfg, str(recogniser / f"fold{k}"), texts))
        print(f"  recogniser fold {k}: scored {len(texts)} passages", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibrate", action="store_true", help="score our admitted answers and the pool's sources")
    ap.add_argument("--cells", type=Path, nargs="*", default=[], help="parquet files with model, text, cell")
    ap.add_argument("--per-cell", type=int, default=200, help="rows per (cell, label) at most")
    ap.add_argument("--recogniser", type=Path, default=OUTPUTS / "closeness" / "recogniser-4b")
    ap.add_argument("--no-lora", action="store_true")
    ap.add_argument("--no-transfer", action="store_true")
    ap.add_argument("--out", type=Path, default=OUTPUTS / "closeness" / "instrument")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    parts = [calibration_cells(args.per_cell)] if args.calibrate else []
    for f in args.cells:
        c = pd.read_parquet(f)
        parts.append(c.groupby(["cell", "model"], group_keys=False)
                     .apply(lambda g: g.sample(min(args.per_cell, len(g)), random_state=17))[["model", "text", "cell"]])
    if not parts:
        ap.error("nothing to score: give --calibrate and/or --cells")
    cells = pd.concat(parts, ignore_index=True)
    cells = cells[cells.model.isin(LABELS)].reset_index(drop=True)
    print(f"{len(cells)} passages in {cells.cell.nunique()} cells", flush=True)

    official = load_train().reset_index(drop=True)
    folds = closeness.prompt_folds(official.prompt_id, 5, seed=0)

    # Linear reference: fitted on all official rows for the cells; prompt-grouped CV for its own recall.
    lin_cv = closeness.recall_by_label(official.model, closeness.cv_predictions(official, folds))
    cells["pred_linear"] = NAMES[closeness.linear_scores(official, cells).argmax(axis=1)]
    print("linear reference done", flush=True)

    use_lora = not args.no_lora and (args.recogniser / "recall.csv").exists()
    if use_lora:
        lora_cv = pd.read_csv(args.recogniser / "recall.csv", index_col=0).oof_recall
        fold_logits = lora_scores(args.recogniser, cells.text)
        for k, lg in enumerate(fold_logits):
            cells[f"pred_lora{k}"] = NAMES[lg.argmax(axis=1)]
        mean = np.mean([lg - lg.mean(axis=1, keepdims=True) for lg in fold_logits], axis=0)
        cells["pred_lora"] = NAMES[mean.argmax(axis=1)]
    elif not args.no_lora:
        print(f"no trained recogniser at {args.recogniser}; linear only", flush=True)

    rows = []
    for (cell, m), g in cells.groupby(["cell", "model"]):
        if len(g) < 20:
            continue
        off = official[official.model == m]
        r = {"cell": cell, "model": m, "n": len(g),
             "rec_linear": float((g.pred_linear == m).mean()), "cv_linear": lin_cv[m]}
        r["ratio_linear"] = r["rec_linear"] / r["cv_linear"] if r["cv_linear"] else np.nan
        if use_lora:
            rec = np.mean([(g[f"pred_lora{k}"] == m).mean() for k in range(len(fold_logits))])
            r |= {"rec_lora": float(rec), "cv_lora": float(lora_cv[m]), "ratio_lora": float(rec / lora_cv[m]),
                  "family_lora": float((g.pred_lora.map(closeness.family) == closeness.family(m)).mean())}
        r |= {"style_auc": closeness.style_auc(off.text, g.text)} | closeness.word_stats(off.text, g.text)
        rows.append(r)
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "cells.csv", index=False)

    fam_rows = []
    if not args.no_transfer:
        for cell, g in cells.groupby("cell"):
            fam_rows += [{"cell": cell} | r for r in closeness.family_transfer(g, official, folds)]
        pd.DataFrame(fam_rows).to_csv(args.out / "families.csv", index=False)

    pd.set_option("display.width", 250, "display.max_rows", 500, "display.max_colwidth", 60)
    cols = ["cell", "model", "n", "ratio_linear"] + (["ratio_lora", "family_lora"] if use_lora else []) + [
        "style_auc", "words_med", "words_med_official", "words_ks"]
    print(table[cols].round(3).sort_values(["model", "cell"]).to_string(index=False))
    if fam_rows:
        print(pd.DataFrame(fam_rows).round(3).to_string(index=False))
    (args.out / "config.json").write_text(json.dumps(vars(args) | {"cells": [str(c) for c in args.cells],
                                                                   "recogniser": str(args.recogniser),
                                                                   "out": str(args.out), "lora": use_lora},
                                                     indent=2), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
