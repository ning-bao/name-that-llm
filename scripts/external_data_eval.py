"""Experiment 5: does chat-collected external data help on official held-out prompts? (report E6)

Every condition is scored on the same official outer-validation prompts of the dev folds,
with the fixed reference text+style model (linear.REFERENCE_*), so only the data changes:
  18 classes: official (outer-train rows only) vs combined (plus admitted external train rows)
  17 classes, without gemini-3.7-flash, which cannot be collected any more: official17,
             external17 (external train rows only) and combined17
In the combined conditions every class gets equal total weight and its external rows a
share s of it, chosen per outer fold on the inner folds from --shares (at most 0.5, so
official data keeps at least half the exposure); a class with no external rows falls back
to its official rows. The external holdout prompts are scored by each condition's five
fold models (scores averaged) as a transfer check. Paired prompt-bootstrap differences
compare combined vs official and external17 / combined17 vs official17.

Usage: uv run python scripts/external_data_eval.py [--external data/external/admitted.parquet]
           [--waves wave1 wave2 ...] [--partitions 0 1] [--shares 0.1 0.25 0.5] [--eligible-only]
Writes outputs/external_eval/<UTC timestamp>/.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nllm import collection, linear
from nllm.data import LABELS, load_train
from nllm.features import style_features
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import development_rows, outer_folds, select

LABELS17 = [m for m in LABELS if m not in collection.UNCOLLECTABLE]
# condition -> (task, official rows, external rows, paired reference)
CONDITIONS = {
    "official": ("18", True, False, None),
    "combined": ("18", True, True, "official"),
    "official17": ("17", True, False, None),
    "external17": ("17", False, True, "official17"),
    "combined17": ("17", True, True, "official17"),
}
SEED = 17


def task_rows(df: pd.DataFrame, task: str) -> pd.DataFrame:
    return df if task == "18" else df[df.model.isin(LABELS17)]


def fit_scores(official, external, condition, share, evaluate, style):
    """Reference text+style scores on each evaluate frame; 17-class scores keep only the 17 label columns."""
    task, use_official, use_external, _ = CONDITIONS[condition]
    parts = ([task_rows(official, task)] if use_official else []) + ([external] if use_external else [])
    train = pd.concat(parts)
    weights = None
    if use_official and use_external:
        weights = linear.mixing_weights(train.model.to_numpy(), train.index.isin(external.index), share)
    keep = [LABELS.index(m) for m in (LABELS if task == "18" else LABELS17)]
    out = []
    for ev in evaluate:
        scores = linear.fit_score(train, style.loc[train.index], ev, style.loc[ev.index],
                                  [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE], SEED, weights)
        text, sty = scores[linear.REFERENCE_TEXT.key][:, keep], scores[linear.REFERENCE_STYLE.key][:, keep]
        out.append(linear.blend(text, sty, linear.REFERENCE_STYLE_WEIGHT, linear.scale(text), linear.scale(sty)))
    return out


def predict(scores: np.ndarray, task: str) -> np.ndarray:
    return np.asarray(LABELS if task == "18" else LABELS17)[scores.argmax(axis=1)]


def inner_task(dev, external, style, fold, k, condition, share):
    inner_train, inner_val = list(fold.inner())[k]
    task = CONDITIONS[condition][0]
    va = task_rows(select(dev, inner_val), task)
    (scores,) = fit_scores(select(dev, inner_train), external, condition, share, [va], style)
    return fold.partition, fold.outer, condition, share, va, scores


def outer_task(dev, external, holdout, style, fold, condition, share):
    task = CONDITIONS[condition][0]
    va = task_rows(select(dev, fold.validation), task)
    evaluate = [va, holdout] if len(holdout) else [va]
    scores = fit_scores(select(dev, fold.train), external, condition, share, evaluate, style)
    return fold.partition, fold.outer, condition, va, scores[0], scores[1] if len(holdout) else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--external", type=Path, default=collection.EXTERNAL_DIR / "admitted.parquet")
    ap.add_argument("--waves", nargs="+", help="only use external answers from these waves")
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--shares", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    ap.add_argument("--eligible-only", action="store_true", help="only answers marked organizer_confirmed")
    ap.add_argument("--n-jobs", type=int, default=12)
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    args = ap.parse_args()
    if max(args.shares) > 0.5:
        ap.error("shares above 0.5 would give external data most of the exposure (report E6)")

    ext = pd.read_parquet(args.external)
    if args.waves:
        ext = ext[ext.wave.isin(args.waves)]
    if args.eligible_only:
        ext = ext[ext.eligibility_status == "organizer_confirmed"]
    ext = ext.set_index("id", drop=False)
    external, holdout = ext[ext.split == "train"], ext[ext.split == "external_holdout"]
    if external.empty:
        raise SystemExit(f"no admitted external training rows in {args.external}; run scripts/import_answers.py")
    if missing := sorted(set(LABELS17) - set(external.model)):
        print(f"warning: no external training rows for {missing}; they fall back to official rows")

    dev = development_rows(load_train()).set_index("id", drop=False)
    style = style_features(pd.concat([dev.text, ext.text])).set_axis(pd.concat([dev.id, ext.id]))
    folds = [f for p in args.partitions for f in outer_folds(p)]
    combined = [c for c, spec in CONDITIONS.items() if spec[1] and spec[2]]
    print(f"{len(external)} external train rows ({external.prompt_id.nunique()} prompts), {len(holdout)} holdout rows; "
          f"{len(folds)} outer folds; shares {args.shares}", flush=True)

    # Phase 1: the external share for each combined condition, from inner out-of-fold BA.
    inner = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(inner_task)(dev, external, style, f, k, c, s)
        for f in folds for k in range(3) for c in combined for s in args.shares
    )
    pooled: dict[tuple, list] = {}
    for p, o, c, s, va, scores in inner:
        pooled.setdefault((p, o, c, s), []).append((va.model.to_numpy(), predict(scores, CONDITIONS[c][0])))
    share_table = pd.DataFrame([
        {"partition": p, "outer_fold": o, "condition": c, "share": s,
         "inner_ba": balanced_accuracy(np.concatenate([y for y, _ in v]), np.concatenate([q for _, q in v]))}
        for (p, o, c, s), v in pooled.items()
    ])
    chosen = (share_table.sort_values(["inner_ba", "share"], ascending=[False, True])
              .groupby(["partition", "outer_fold", "condition"]).share.first().to_dict())

    # Phase 2: every condition on the 96 outer-train prompts, scored on the 24 held-out prompts.
    outer = Parallel(n_jobs=args.n_jobs, verbose=5)(
        delayed(outer_task)(dev, external, holdout, style, f, c, chosen.get((f.partition, f.outer, c), 0.0))
        for f in folds for c in CONDITIONS
    )
    oof_parts, holdout_scores = [], {}
    for p, o, c, va, scores, ho in outer:
        oof_parts.append(pd.DataFrame({"id": va.id.to_numpy(), "prompt_id": va.prompt_id.to_numpy(),
                                       "model": va.model.to_numpy(), "partition": p, "outer_fold": o,
                                       "condition": c, "pred": predict(scores, CONDITIONS[c][0])}))
        if ho is not None:
            holdout_scores.setdefault((p, c), []).append(ho)
    oof = pd.concat(oof_parts, ignore_index=True)

    results, recalls = [], {}
    boot = {"n_draws": args.bootstrap_draws, "seed": SEED}
    for p in args.partitions:
        for c, (task, _, _, reference) in CONDITIONS.items():
            rows = oof[(oof.partition == p) & (oof.condition == c)].set_index("id")
            lo, hi = interval(prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, **boot))
            result = {"partition": p, "condition": c, "classes": int(task), "rows": len(rows),
                      "balanced_accuracy": balanced_accuracy(rows.model, rows.pred), "ci95_lo": lo, "ci95_hi": hi}
            if reference:
                ref = oof[(oof.partition == p) & (oof.condition == reference)].set_index("id").loc[rows.index]
                diff = prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, y_pred_baseline=ref.pred, **boot)
                dlo, dhi = interval(diff)
                result |= {"reference": reference, "diff": float(diff.mean()), "diff_ci95_lo": dlo, "diff_ci95_hi": dhi}
            if (p, c) in holdout_scores:
                ho_pred = predict(np.mean(holdout_scores[p, c], axis=0), task)
                result["external_holdout_ba"] = balanced_accuracy(holdout.model, ho_pred)
            results.append(result)
            recalls[f"p{p}:{c}"] = per_class_recall(rows.model, rows.pred)
    results = pd.DataFrame(results)

    out = OUTPUTS / "external_eval" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    results.to_csv(out / "results.csv", index=False)
    oof.to_parquet(out / "oof.parquet", index=False)
    share_table.to_csv(out / "inner_shares.csv", index=False)
    recall_table = pd.DataFrame(recalls)
    recall_table.to_csv(out / "per_class_recall.csv")
    (out / "config.json").write_text(json.dumps({
        "experiment": "5: official-only vs external-only vs combined (report E6)",
        "external": str(args.external), "waves": args.waves, "eligible_only": args.eligible_only,
        "external_train_rows": len(external), "external_train_prompts": int(external.prompt_id.nunique()),
        "external_holdout_rows": len(holdout), "shares": args.shares, "partitions": args.partitions,
        "model": [linear.REFERENCE_TEXT.key, linear.REFERENCE_STYLE.key, linear.REFERENCE_STYLE_WEIGHT],
        "chosen_shares": {f"p{p}o{o}:{c}": s for (p, o, c), s in chosen.items()}, "seed": SEED,
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 220)
    print("\nBalanced accuracy on official held-out dev prompts (95% prompt-bootstrap CI), paired vs reference:")
    print(results.round(4).to_string(index=False))
    gemini = [m for m in recall_table.index if m.startswith("gemini")]
    print("\nGemini recalls (watch 3.7, which gets no external data):")
    print(recall_table.loc[gemini].round(3).to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
