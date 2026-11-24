"""Experiment 6: which community-pool training data helps on official held-out prompts?

The model is the fixed linear text+style reference (linear.REFERENCE_*), so only the training data
changes. Every condition is scored on the official outer-validation prompts of the dev folds, and
everything learned from official data (match weights, source multipliers) uses only the fold's 96
outer-train prompts. The pool's own match_weight column, fitted on all 150 official prompts, is never
read. Pool conditions draw the same number of rows per label (class-balanced) with the same seed per fold:

  official              the fold's official training rows only (reference)
  pool_u                pool rows, uniform within each label
  pool_w                pool rows with match weights: marker rates and length matched to the fold's
                        official rows (pool.match_weights)
  pool_w_noproxy        pool_w without Kaggle Benchmarks proxy rows
  pool_w_own            pool plus our admitted answers (pool.own_rows filters), match weights
  pool_wq_own           pool_w_own, with the tilt starting from per-source multipliers: sources whose
                        passages the fold's official-only model recognises more often are drawn more
                        (pool.source_multipliers)
  pool_w+official       pool_w rows plus the fold's official rows, which keep half of each class's weight
  pool_wq_own+official  pool_wq_own rows plus the fold's official rows, likewise

Any pool.RECIPES entry runs as pool_<recipe>, optionally with +official (e.g. pool_w_codex5+official:
our Codex GPT answers drawn ~5x more often than their natural share). Those are the default conditions.

Usage: uv run python -I scripts/pool_eval.py [--partitions 0 1] [--per-label 1000] [--conditions ...]
       [--quick]   (partition 0, outer fold 0 only, 300 rows per label: a timing and smoke run)
Writes outputs/pool_eval/<UTC timestamp>/.
"""

import argparse
import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nllm import linear, pool
from nllm.data import load_train
from nllm.features import style_features
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import development_rows, outer_folds, select

# pool_<recipe> trains on a pool.training_sample(); a +official suffix adds the fold's official rows.
CONDITIONS = ["official", *(f"pool_{r}{s}" for r in pool.RECIPES for s in ("", "+official"))]
DEFAULT_CONDITIONS = ["official", "pool_u", "pool_w", "pool_w_noproxy", "pool_w_own", "pool_wq_own",
                      "pool_w+official", "pool_wq_own+official"]
REFERENCES = ("official", "pool_u", "pool_w")  # paired differences are reported against each of these
SEED = 17
POOL_SHARE = 0.5  # in "+official" conditions, the pool rows' share of each class's training weight
PROBES_PER_CELL = 200  # passages per (label, source) scored by the official-only model for source multipliers


def candidates(style_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool plus own rows indexed by id (cached per worker), and style features for them and the official rows."""
    return pool.candidate_rows(), pd.read_parquet(style_path)


def reference_scores(train, evaluate, style, weights=None) -> np.ndarray:
    scores = linear.fit_score(train, style.loc[train.index], evaluate, style.loc[evaluate.index],
                              [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE], SEED, weights)
    t, s = scores[linear.REFERENCE_TEXT.key], scores[linear.REFERENCE_STYLE.key]
    return linear.blend(t, s, linear.REFERENCE_STYLE_WEIGHT, linear.scale(t), linear.scale(s))


def official_task(dev, fold, style_path):
    """The official-only condition, plus that model's verdict on probe passages of every (label, source)."""
    rows, style = candidates(style_path)
    tr, va = select(dev, fold.train), select(dev, fold.validation)
    probes = rows.sample(frac=1.0, random_state=SEED).groupby(["model", "source"]).head(PROBES_PER_CELL)
    pred = linear.predict(reference_scores(tr, pd.concat([va, probes]), style))
    hit = pd.Series(pred[len(va):] == probes.model.to_numpy(), index=probes.index)
    _, cells = pool.source_multipliers(rows, hit)
    return fold.partition, fold.outer, "official", va, pred[: len(va)], cells, None, None


def pool_task(dev, fold, condition, cells, style_path, per_label):
    rows, style = candidates(style_path)
    recipe, with_official = condition.removeprefix("pool_").removesuffix("+official"), condition.endswith("+official")
    tr, va = select(dev, fold.train), select(dev, fold.validation)
    sample, report = pool.training_sample(recipe, tr, per_label, 1000 * fold.partition + fold.outer, rows, cells)
    if report is not None:
        report = report.assign(partition=fold.partition, outer_fold=fold.outer, condition=condition)
    composition = (sample.assign(proxy=sample.kb_proxy, own=sample.origin == "own")
                   .groupby("model")[["proxy", "own"]].mean()
                   .assign(rows=sample.model.value_counts(), partition=fold.partition, outer_fold=fold.outer,
                           condition=condition).reset_index())
    train, mix = sample, None
    if with_official:
        train = pd.concat([sample, tr])
        mix = linear.mixing_weights(train.model.to_numpy(), np.r_[np.ones(len(sample), bool), np.zeros(len(tr), bool)],
                                    POOL_SHARE)
    pred = linear.predict(reference_scores(train, va, style, mix))
    return fold.partition, fold.outer, condition, va, pred, None, report, composition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--per-label", type=int, default=1000)
    ap.add_argument("--conditions", nargs="+", default=DEFAULT_CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--quick", action="store_true", help="partition 0, outer fold 0, 300 rows per label")
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    args = ap.parse_args()
    conditions = ["official", *(c for c in args.conditions if c != "official")]
    folds = [f for p in args.partitions for f in outer_folds(p)]
    if args.quick:
        folds, args.per_label = [outer_folds(0)[0]], 300

    out = OUTPUTS / "pool_eval" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    started = time.time()
    dev = development_rows(load_train()).set_index("id", drop=False)
    rows = pool.candidate_rows()
    texts = pd.concat([rows.text, dev.text])
    style_path = out / "style.parquet"
    style_features(texts).set_axis(texts.index).to_parquet(style_path)
    print(f"{len(rows)} candidate rows ({(rows.origin == 'own').sum()} own); style features in "
          f"{time.time() - started:.0f}s; {len(folds)} outer folds; {args.per_label} rows per label", flush=True)

    first = Parallel(n_jobs=min(args.n_jobs, len(folds)), verbose=5)(
        delayed(official_task)(dev, f, str(style_path)) for f in folds)
    cells = {(p, o): c for p, o, _, _, _, c, _, _ in first}
    second = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(pool_task)(dev, f, c, cells[f.partition, f.outer], str(style_path), args.per_label)
        for f in folds for c in conditions if c != "official")

    oof, reports, compositions = [], [], []
    for p, o, c, va, pred, _, report, composition in first + second:
        oof.append(pd.DataFrame({"id": va.id.to_numpy(), "prompt_id": va.prompt_id.to_numpy(), "model": va.model.to_numpy(),
                                 "partition": p, "outer_fold": o, "condition": c, "pred": pred}))
        reports += [report] if report is not None else []
        compositions += [composition] if composition is not None else []
    oof = pd.concat(oof, ignore_index=True)
    multipliers = pd.concat([c.assign(partition=p, outer_fold=o) for (p, o), c in cells.items()], ignore_index=True)

    results, recalls = [], {}
    boot = {"n_draws": args.bootstrap_draws, "seed": SEED}
    for p in sorted({f.partition for f in folds}):
        part = oof[oof.partition == p]
        for c in conditions:
            r = part[part.condition == c].set_index("id")
            lo, hi = interval(prompt_bootstrap(r.model, r.pred, r.prompt_id, **boot))
            result = {"partition": p, "condition": c, "rows": len(r),
                      "balanced_accuracy": balanced_accuracy(r.model, r.pred), "ci95_lo": lo, "ci95_hi": hi}
            for ref in REFERENCES:
                if ref == c or ref not in conditions:
                    continue
                b = part[part.condition == ref].set_index("id").loc[r.index]
                d = prompt_bootstrap(r.model, r.pred, r.prompt_id, y_pred_baseline=b.pred, **boot)
                dlo, dhi = interval(d)
                result |= {f"diff_vs_{ref}": float(d.mean()), f"diff_vs_{ref}_lo": dlo, f"diff_vs_{ref}_hi": dhi}
            results.append(result)
            recalls[f"p{p}:{c}"] = per_class_recall(r.model, r.pred)
    results = pd.DataFrame(results)

    results.to_csv(out / "results.csv", index=False)
    oof.to_parquet(out / "oof.parquet", index=False)
    pd.DataFrame(recalls).to_csv(out / "per_class_recall.csv")
    multipliers.to_csv(out / "source_multipliers.csv", index=False)
    if reports:
        pd.concat(reports, ignore_index=True).to_csv(out / "match_weights.csv", index=False)
    if compositions:
        pd.concat(compositions, ignore_index=True).to_csv(out / "sample_composition.csv", index=False)
    style_path.unlink()
    (out / "config.json").write_text(json.dumps({
        "experiment": "6: community pool training data, linear text+style reference",
        "pool": str(pool.POOL_DIR / pool.POOL_FILE), "own_exclude": list(pool.DEFAULT_OWN_EXCLUDE),
        "own_filters": [pool.GEMINI_EM_DASH_FILTER], "conditions": conditions, "per_label": args.per_label,
        "partitions": sorted({f.partition for f in folds}), "folds": [[f.partition, f.outer] for f in folds],
        "pool_share_with_official": POOL_SHARE, "probes_per_cell": PROBES_PER_CELL, "match_ridge": 0.02,
        "model": [linear.REFERENCE_TEXT.key, linear.REFERENCE_STYLE.key, linear.REFERENCE_STYLE_WEIGHT],
        "seed": SEED, "minutes": round((time.time() - started) / 60, 1),
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    print("\nBalanced accuracy on official held-out dev prompts (95% prompt-bootstrap CI), paired differences:")
    print(results.round(4).to_string(index=False))
    print(f"\nwrote {out} in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
