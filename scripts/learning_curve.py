"""Prompt-count learning curve for the linear text+style model (informs report E6).

For each partition and outer fold, a fixed config is fitted on random subsets of n of the
96 outer-training prompts (all 18 answers per prompt) and scored on the 24 held-out
prompts; BA is concatenated over the five outer folds. The configs and blend weight are
the ones experiment 1 selected most often, so the level is slightly selection-affected;
the slope is what matters. Blend scales are label-free (computed on the scored rows).

Usage: uv run python scripts/learning_curve.py [--sizes 24 48 72 96] [--repeats 3] [--n-jobs 8]
Writes outputs/learning_curve/<UTC timestamp>/.
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from nllm import linear
from nllm.data import FAMILIES, family, load_train
from nllm.features import style_features
from nllm.metrics import balanced_accuracy
from nllm.paths import OUTPUTS
from nllm.splits import development_rows, outer_folds, select

TEXT, STYLE, STYLE_WEIGHT = linear.REFERENCE_TEXT, linear.REFERENCE_STYLE, linear.REFERENCE_STYLE_WEIGHT


def task(dev, style, fold, n, repeat, seed):
    prompts = sorted(fold.train)
    if n < len(prompts):
        prompts = list(np.random.default_rng([seed, fold.partition, fold.outer, n, repeat]).choice(prompts, n, replace=False))
    tr, va = select(dev, prompts), select(dev, fold.validation)
    scores = linear.fit_score(tr, style.loc[tr.index], va, style.loc[va.index], [TEXT, STYLE], seed)
    text, sty = scores[TEXT.key], scores[STYLE.key]
    branches = {
        "text": text,
        "style": sty,
        "text+style": linear.blend(text, sty, STYLE_WEIGHT, linear.scale(text), linear.scale(sty)),
    }
    return [
        pd.DataFrame({"partition": fold.partition, "n_prompts": n, "repeat": repeat, "branch": b,
                      "model": va.model.to_numpy(), "pred": linear.predict(s)})
        for b, s in branches.items()
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[24, 48, 72, 96])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-jobs", type=int, default=8)
    args = ap.parse_args()
    seed = 17

    dev = development_rows(load_train())
    style = style_features(dev.text).set_axis(dev.index)
    jobs = [
        (fold, n, r)
        for p in args.partitions
        for fold in outer_folds(p)
        for n in args.sizes
        for r in range(1 if n >= 96 else args.repeats)
    ]
    parts = Parallel(n_jobs=args.n_jobs, verbose=5)(delayed(task)(dev, style, f, n, r, seed) for f, n, r in jobs)
    rows = pd.concat([df for part in parts for df in part], ignore_index=True)

    def summarise(g):
        return pd.Series({
            "ba": balanced_accuracy(g.model, g.pred),
            "family_ba": balanced_accuracy(g.model.map(family), g.pred.map(family), sorted(set(FAMILIES.values()))),
        })

    curve = rows.groupby(["branch", "partition", "n_prompts", "repeat"]).apply(summarise).reset_index()
    table = curve.groupby(["branch", "n_prompts"])[["ba", "family_ba"]].agg(["mean", "std"]).round(4)
    slopes = {}
    for b, g in curve.groupby("branch"):
        exact = np.polyfit(np.log2(g.n_prompts), g.ba, 1)[0]
        fam = np.polyfit(np.log2(g.n_prompts), g.family_ba, 1)[0]
        slopes[b] = {"ba_per_doubling": round(float(exact), 4), "family_ba_per_doubling": round(float(fam), 4)}

    out = OUTPUTS / "learning_curve" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    curve.to_csv(out / "curve.csv", index=False)
    (out / "config.json").write_text(json.dumps({
        "text": TEXT.key, "style": STYLE.key, "style_weight": STYLE_WEIGHT, "sizes": args.sizes,
        "repeats": args.repeats, "partitions": args.partitions, "seed": seed, "slopes": slopes,
    }, indent=2), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(table.to_string())
    print("\nLeast-squares slope vs log2(prompts), over partitions and repeats:")
    print(pd.DataFrame(slopes).T.to_string())
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
