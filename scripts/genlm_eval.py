"""Experiment 7: a label-conditioned language model as a classifier (nllm.genlm), on the dev outer folds.

Per outer fold, one LoRA language model learns (label prefix, passage) pairs from a class-balanced pool sample
(pool.training_sample with --recipe; match weights fitted on the fold's 96 official training prompts) plus those
official training rows. A disjoint part of the same pool draw (--calibration-per-label rows per label) sets the
balancing offsets, and the fold's 24 held-out prompts are scored under all 18 labels. Hyperparameters are fixed,
so no official validation row is used for any choice. Each fold's scores are saved as it finishes; --out resumes.

Usage: uv run --extra nn python scripts/genlm_eval.py [--partitions 0] [--per-label 1000] [--recipe wx_own]
           [--baseline outputs/encoder/ettin-400m/<run> --baseline-branch ettin-400m+pool] [--out <run dir>]
Writes outputs/genlm/<backbone>/<UTC timestamp>/ unless --out is given.
"""

import argparse
import json
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download

from nllm import genlm, pool
from nllm.data import LABELS, load_train
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import development_rows, outer_folds, select

SCORE_COLUMNS = [f"score:{m}" for m in LABELS]


def run_fold(args, model_path, cfg, dev, fold, path: Path) -> dict:
    """Train, calibrate and score one outer fold, unless its scores are already on disk."""
    tr, va = select(dev, fold.train), select(dev, fold.validation)
    if not path.exists():
        started = time.time()
        seed = 1000 * fold.partition + fold.outer
        drawn, _ = pool.training_sample(args.recipe, tr, args.per_label + args.calibration_per_label, seed)
        drawn = drawn.sample(frac=1.0, random_state=seed)
        calib = drawn.groupby("model").head(args.calibration_per_label)
        train = pd.concat([drawn.drop(calib.index), tr])
        tok, model, history = genlm.fit(model_path, train.text, train.model, cfg, args.seed)
        s_va = genlm.score(tok, model, va.text, cfg)
        s_cal = genlm.score(tok, model, calib.text, cfg)
        offsets = genlm.balancing_offsets(s_cal)
        del model
        torch.cuda.empty_cache()
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, scores=s_va, calibration=s_cal, calibration_labels=calib.model.to_numpy(dtype=str),
                     offsets=offsets, ids=va.id.to_numpy(dtype=str), history=np.array(json.dumps(history)),
                     train_rows=np.array(len(train)))
        os.replace(tmp, path)
        print(f"  {path.stem}: {len(train)} training rows, {time.time() - started:.0f}s, "
              f"peak {history[-1]['peak_gb']:.1f} GB", flush=True)
    with np.load(path) as z:
        if not (z["ids"] == va.id.to_numpy(dtype=str)).all():
            raise RuntimeError(f"{path} was written for different rows")
        return {"va": va, "scores": z["scores"], "offsets": z["offsets"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="qwen3-1.7b-base", choices=sorted(genlm.BACKBONES))
    ap.add_argument("--partitions", type=int, nargs="+", default=[0])
    ap.add_argument("--per-label", type=int, default=1000)
    ap.add_argument("--calibration-per-label", type=int, default=40)
    ap.add_argument("--recipe", default="wx_own", choices=sorted(pool.RECIPES))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--baseline", type=Path, help="a run dir with oof.parquet, for paired differences")
    ap.add_argument("--baseline-branch")
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, help="run directory: resumed if it exists, otherwise created")
    args = ap.parse_args()
    transformers.logging.set_verbosity_error()

    cfg = genlm.LMConfig(args.backbone, lr=args.lr, epochs=args.epochs, micro_batch=args.micro_batch)
    run = {"backbone": args.backbone, "per_label": args.per_label, "calibration_per_label": args.calibration_per_label,
           "recipe": args.recipe, "train_config": asdict(cfg), "seed": args.seed}
    out = args.out or OUTPUTS / "genlm" / args.backbone / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if (out / "run.json").exists():
        if json.loads((out / "run.json").read_text(encoding="utf-8")) != run:
            raise SystemExit(f"--out was started with different settings: {out / 'run.json'}")
    else:
        (out / "fits").mkdir(parents=True)
        (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    started = time.time()
    model_path = snapshot_download(genlm.BACKBONES[args.backbone], allow_patterns=["*.json", "*.txt", "*.safetensors"])
    dev = development_rows(load_train())
    folds = [f for p in args.partitions for f in outer_folds(p)]
    print(f"{args.backbone}: recipe {args.recipe}, {args.per_label} pool rows per label, {len(folds)} outer folds -> {out}",
          flush=True)

    parts = []
    for fold in folds:
        r = run_fold(args, model_path, cfg, dev, fold, out / "fits" / f"outer_p{fold.partition}_o{fold.outer}.npz")
        va, adjusted = r["va"], r["scores"] + r["offsets"]
        meta = va[["id", "prompt_id", "model"]].reset_index(drop=True).assign(
            partition=fold.partition, outer_fold=fold.outer, branch=f"genlm:{args.backbone}",
            pred=np.asarray(LABELS)[adjusted.argmax(axis=1)], pred_raw=np.asarray(LABELS)[r["scores"].argmax(axis=1)])
        parts.append(pd.concat([meta, pd.DataFrame(adjusted, columns=SCORE_COLUMNS)], axis=1))
        print(f"p{fold.partition} o{fold.outer}: BA {balanced_accuracy(meta.model, meta.pred):.3f} "
              f"(no offsets {balanced_accuracy(meta.model, meta.pred_raw):.3f})  [{(time.time() - started) / 60:.0f} min]",
              flush=True)
    oof = pd.concat(parts, ignore_index=True)
    oof.to_parquet(out / "oof.parquet", index=False)

    baseline = None
    if args.baseline:
        baseline = pd.read_parquet(args.baseline / "oof.parquet")
        if args.baseline_branch:
            baseline = baseline[baseline.branch == args.baseline_branch]
    results, recalls = [], {}
    boot = {"n_draws": args.bootstrap_draws, "seed": args.seed}
    for p in args.partitions:
        rows = oof[oof.partition == p].set_index("id")
        lo, hi = interval(prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, **boot))
        result = {"partition": p, "balanced_accuracy": balanced_accuracy(rows.model, rows.pred), "ci95_lo": lo,
                  "ci95_hi": hi, "ba_without_offsets": balanced_accuracy(rows.model, rows.pred_raw)}
        if baseline is not None:
            ref = baseline[baseline.partition == p].set_index("id").loc[rows.index]
            d = prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, y_pred_baseline=ref.pred, **boot)
            dlo, dhi = interval(d)
            result |= {"baseline_ba": balanced_accuracy(ref.model, ref.pred), "diff_vs_baseline": float(d.mean()),
                       "diff_ci95_lo": dlo, "diff_ci95_hi": dhi}
        results.append(result)
        recalls[f"p{p}"] = per_class_recall(rows.model, rows.pred)
    results = pd.DataFrame(results)
    results.to_csv(out / "results.csv", index=False)
    pd.DataFrame(recalls).to_csv(out / "per_class_recall.csv")
    (out / "config.json").write_text(json.dumps({
        "experiment": "7: label-conditioned language model (generative classifier)", **run,
        "model_repo": genlm.BACKBONES[args.backbone], "revision": Path(model_path).name,
        "baseline": f"{args.baseline}:{args.baseline_branch}" if args.baseline else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "runtime_seconds_this_session": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    print(pd.DataFrame(recalls).round(3).to_string())
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
