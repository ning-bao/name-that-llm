"""Experiment 1: nested-CV TF-IDF / style linear baseline on the 120 development prompts.

For each partition and outer fold, every config is scored on the three inner folds;
each branch's config (and the text+style blend weight) is chosen from inner out-of-fold
balanced accuracy, refitted on the 96 outer-training prompts and scored on the 24 held-out
prompts. Outer predictions are concatenated over all 120 prompts. The confirmation
prompts are never loaded.

Usage: uv run python scripts/baseline_linear.py [--partitions 0 1] [--n-jobs 16]
Writes outputs/baseline_linear/<UTC timestamp>/.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import sklearn
from joblib import Parallel, delayed

from nllm import linear
from nllm.data import LABELS, load_train
from nllm.features import style_features
from nllm.metrics import (
    balanced_accuracy,
    confusion,
    interval,
    per_class_recall,
    prompt_bootstrap,
    within_family_error_share,
)
from nllm.paths import OUTPUTS, data_dir
from nllm.splits import development_rows, initialization_seeds, outer_folds, select

BRANCHES = ("char", "word", "char+word", "style", "text", "text+style")
SCORE_COLUMNS = [f"score:{m}" for m in LABELS]
QUICK_KEYS = {
    "char35|svm|C=1|raw_tf",
    "word12|svm|C=1|raw_tf",
    "char35+word12|svm|C=1|raw_tf",
    "style|lr|C=0.1",
}


def inner_task(dev, style, fold, k, configs, seed):
    inner_train, inner_val = list(fold.inner())[k]
    tr, va = select(dev, inner_train), select(dev, inner_val)
    scores = linear.fit_score(tr, style.loc[tr.index], va, style.loc[va.index], configs, seed)
    return fold.partition, fold.outer, va.index, scores


def outer_task(dev, style, fold, chosen, seed):
    tr, va = select(dev, fold.train), select(dev, fold.validation)
    configs = list({c.key: c for c in chosen.values()}.values())
    scores = linear.fit_score(tr, style.loc[tr.index], va, style.loc[va.index], configs, seed)
    return fold.partition, fold.outer, va.index, scores


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-jobs", type=int, default=16)
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--quick", action="store_true", help="one fast SVM/LR config per branch, for plumbing checks")
    args = ap.parse_args()

    started = time.time()
    seed = initialization_seeds()[0]
    out = OUTPUTS / "baseline_linear" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)

    dev = development_rows(load_train())
    style = style_features(dev.text).set_axis(dev.index)
    configs = linear.all_configs()
    if args.quick:
        configs = [c for c in configs if c.key in QUICK_KEYS]
    folds = {(p, f.outer): f for p in args.partitions for f in outer_folds(p)}
    print(f"{len(dev)} dev rows, {len(configs)} configs, {len(folds)} outer folds -> {out}", flush=True)

    # Phase 1: every config on every inner split, assembled into inner out-of-fold scores.
    inner = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(inner_task)(dev, style, f, k, configs, seed) for f in folds.values() for k in range(3)
    )
    inner_oof = {key: {} for key in folds}
    for p, o, index, scores in inner:
        train_index = select(dev, folds[p, o].train).index
        for key, s in scores.items():
            frame = inner_oof[p, o].setdefault(key, pd.DataFrame(np.nan, index=train_index, columns=LABELS))
            frame.loc[index] = s

    # Selection per outer fold, from inner out-of-fold scores only.
    chosen, weights, inner_tables, inner_parts = {}, {}, [], []
    for (p, o), frames in inner_oof.items():
        train_rows = select(dev, folds[p, o].train)
        y = train_rows.model.to_numpy()
        arrays = {key: frame.to_numpy() for key, frame in frames.items()}
        chosen[p, o], table = linear.select(arrays, y, configs)
        text_key, style_key = chosen[p, o]["text"].key, chosen[p, o]["style"].key
        weights[p, o] = linear.choose_blend_weight(arrays[text_key], arrays[style_key], y)
        inner_tables.append(table.assign(partition=p, outer_fold=o))
        # Inner out-of-fold scores of the chosen branches, for nested blends with later models.
        blended = linear.blend(arrays[text_key], arrays[style_key], *weights[p, o])
        meta = train_rows[["id", "prompt_id", "model"]].reset_index(drop=True).assign(partition=p, outer_fold=o)
        for b, s in {"text": arrays[text_key], "style": arrays[style_key], "text+style": blended}.items():
            inner_parts.append(pd.concat([meta.assign(branch=b), pd.DataFrame(s, columns=SCORE_COLUMNS)], axis=1))
    pd.concat(inner_tables).to_csv(out / "inner_scores.csv", index=False)
    pd.concat(inner_parts, ignore_index=True).to_parquet(out / "inner_oof.parquet", index=False)

    # Phase 2: refit chosen configs on outer-train prompts, score the held-out prompts.
    outer = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(outer_task)(dev, style, folds[key], chosen[key], seed) for key in folds
    )
    oof_parts, selections = [], []
    for p, o, index, scores in outer:
        picks = chosen[p, o]
        w, text_scale, style_scale = weights[p, o]
        branch_scores = {b: scores[c.key] for b, c in picks.items()}
        branch_scores["text+style"] = linear.blend(
            branch_scores["text"], branch_scores["style"], w, text_scale, style_scale
        )
        for b in BRANCHES:
            config = f"{picks['text'].key} + {w:g} * {picks['style'].key}" if b == "text+style" else picks[b].key
            selections.append({"partition": p, "outer_fold": o, "branch": b, "config": config})
            meta = pd.DataFrame({
                "id": dev.loc[index, "id"].to_numpy(),
                "prompt_id": dev.loc[index, "prompt_id"].to_numpy(),
                "model": dev.loc[index, "model"].to_numpy(),
                "partition": p,
                "outer_fold": o,
                "branch": b,
                "pred": linear.predict(branch_scores[b]),
            })
            oof_parts.append(pd.concat([meta, pd.DataFrame(branch_scores[b], columns=SCORE_COLUMNS)], axis=1))
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_parquet(out / "oof.parquet", index=False)
    selections = pd.DataFrame(selections)
    selections.to_csv(out / "selections.csv", index=False)

    # Evaluation over all 120 dev prompts per partition; paired differences against "text".
    results, recalls = [], {}
    for p in args.partitions:
        part = oof[oof.partition == p]
        ref = part[part.branch == "text"].set_index("id")
        for b in BRANCHES:
            rows = part[part.branch == b].set_index("id").loc[ref.index]
            boot = {"n_draws": args.bootstrap_draws, "seed": seed}
            lo, hi = interval(prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, **boot))
            diff = prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, y_pred_baseline=ref.pred, **boot)
            dlo, dhi = interval(diff)
            results.append({
                "partition": p,
                "branch": b,
                "balanced_accuracy": balanced_accuracy(rows.model, rows.pred),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "diff_vs_text": float(diff.mean()),
                "diff_ci95_lo": dlo,
                "diff_ci95_hi": dhi,
                "within_family_error_share": within_family_error_share(rows.model, rows.pred),
            })
            recalls[f"p{p}:{b}"] = per_class_recall(rows.model, rows.pred)
            confusion(rows.model, rows.pred).round(3).to_csv(out / f"confusion_p{p}_{b}.csv")
    results = pd.DataFrame(results)
    results.to_csv(out / "results.csv", index=False)
    recall_table = pd.DataFrame(recalls)
    recall_table.to_csv(out / "per_class_recall.csv")

    (out / "config.json").write_text(json.dumps({
        "experiment": "1: nested-CV TF-IDF/style linear baseline",
        "partitions": args.partitions,
        "seed": seed,
        "train_sha256": hashlib.sha256((data_dir() / "train.csv").read_bytes()).hexdigest(),
        "min_df": linear.MIN_DF,
        "feature_sets": {k: list(v[1]) for k, v in linear.FEATURE_SETS.items()},
        "text_classifiers": linear.TEXT_CLASSIFIERS,
        "style_classifiers": linear.STYLE_CLASSIFIERS,
        "sublinear": linear.SUBLINEAR,
        "blend_weights": linear.BLEND_WEIGHTS,
        "selection": "per outer fold, max BA over concatenated inner out-of-fold predictions",
        "label_order": LABELS,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "runtime_seconds": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nOuter-fold balanced accuracy over 120 dev prompts (95% prompt-bootstrap CI):")
    print(results.round(4).to_string(index=False))
    print("\nSelected configs (count over outer folds):")
    print(selections.groupby(["branch", "config"]).size().to_string())
    print("\nPer-class recall:")
    print(recall_table.round(3).to_string())
    print(f"\nwrote {out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
