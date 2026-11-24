"""Compare finished nested runs of pooled decoders on the plain head and on the public pipelines' twin rule.

The twin rule (the public 0.684 notebook's `l24` members): a row's score is log-softmax(plain head) plus
twin x log-softmax(centred head on features minus their group mean), for rows whose group has at least two rows.
There the groups are clusters of the test passages by content. The centred head is linear, so its logits on
centred features are its saved outputs (oof.parquet, centred:<label>) minus their group mean, plus its bias.

Every prompt is cut to m random answers, since the hidden test probably has few answers per prompt. Groups are
either the true prompts, the best any grouping could do, or content clusters of the partition's whole cut
(nllm.centring.content_clusters: our grouping, not the pipeline's). A cluster's rows from different outer folds
were scored by different models, so a row is centred against the rows of its cluster in its own fold only.
The twin weight is fixed (default 0.35, the pipeline's 8B member); nothing is selected here.

Prints and writes, per run, partition, grouping and m: balanced accuracy, and the paired prompt-bootstrap
difference against the first run. Also the plain head's family accuracy and its accuracy where the family is right.

    uv run python scripts/twin_eval.py <baseline run> <run> [<run> ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from nllm import centring
from nllm.data import LABELS, family, load_train
from nllm.metrics import balanced_accuracy, interval, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import development_rows

SCORES = [f"score:{m}" for m in LABELS]
CENTRED = [f"centred:{m}" for m in LABELS]
BIAS = [f"centred_bias:{m}" for m in LABELS]
NAMES = np.asarray(LABELS)
FAMILY = np.asarray([family(m) for m in LABELS])


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def twin_scores(plain: np.ndarray, centred: np.ndarray, bias: np.ndarray, groups: np.ndarray, twin: float) -> np.ndarray:
    """log-softmax(plain) + twin x log-softmax(centred - its group mean + bias) for rows in groups of 2 or more."""
    codes = pd.factorize(groups)[0]
    sizes = np.bincount(codes)
    means = np.zeros((len(sizes), centred.shape[1]))
    np.add.at(means, codes, centred)
    means /= sizes[:, None]
    out = log_softmax(plain.astype(np.float64))
    grouped = sizes[codes] >= 2
    out[grouped] += twin * log_softmax(centred - means[codes] + bias)[grouped]
    return out


def cuts_of(frame: pd.DataFrame, m: int, draws: int, seed: int) -> list[np.ndarray]:
    """Row positions of `draws` random cuts of every prompt to m answers (the same cuts for every run)."""
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("prompt_id").indices.values())
    return [np.sort(np.concatenate([rng.permutation(g)[:m] for g in groups])) for _ in range(draws)]


def load_run(run: Path) -> pd.DataFrame:
    oof = pd.read_parquet(run / "oof.parquet")
    if not set(CENTRED + BIAS) <= set(oof.columns):
        raise SystemExit(f"{run}: oof.parquet has no centred-head outputs (a run from before they were saved)")
    return oof.sort_values(["partition", "id"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+", help="the first is the baseline of the paired differences")
    ap.add_argument("--twin", type=float, default=0.35)
    ap.add_argument("--cut", type=float, default=0.85, help="content-cluster distance cut (Exp 8's setting)")
    ap.add_argument("--eval-m", type=int, nargs="+", default=[2, 3, 6, 18])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    out = args.out or OUTPUTS / "twin_eval" / "+".join(r.name for r in args.runs)
    out.mkdir(parents=True, exist_ok=True)

    runs = {r.name: load_run(r) for r in args.runs}
    if len(runs) != len(args.runs):
        raise SystemExit("run directory names must differ")
    base = next(iter(runs.values()))
    for name, frame in runs.items():
        if not frame[["partition", "id"]].equals(base[["partition", "id"]]):
            raise SystemExit(f"{name} does not cover the baseline's rows")
    text = development_rows(load_train()).set_index("id").text
    boot = {"n_draws": args.bootstrap_draws}

    results, siblings = [], []
    for p, rows_p in base.groupby("partition"):
        at = rows_p.index.to_numpy()
        g = rows_p.reset_index(drop=True)
        y, prompts, folds = g.model.to_numpy(), g.prompt_id.to_numpy(), g.outer_fold.to_numpy()
        texts = text.loc[g.id].to_numpy()
        frames = {name: frame.loc[at].reset_index(drop=True) for name, frame in runs.items()}
        for name, frame in frames.items():
            pred = frame[SCORES].to_numpy().argmax(axis=1)
            right_family = FAMILY[pred] == np.asarray([family(m) for m in y])
            siblings.append({"partition": p, "run": name, "ba_plain": balanced_accuracy(y, NAMES[pred]),
                             "family_accuracy": right_family.mean(),
                             "accuracy_given_family": (NAMES[pred] == y)[right_family].mean()})
        for m in args.eval_m:
            cuts = cuts_of(g, m, args.draws, 7919 + 100 * p + m)
            clusters = [centring.content_clusters(texts[rows], args.cut) for rows in cuts]
            groupings = {"prompt": [prompts[rows] for rows in cuts],
                         "cluster": [np.char.add(np.char.add(c.astype(str), "/"), folds[rows].astype(str))
                                     for c, rows in zip(clusters, cuts, strict=True)]}
            preds = {}
            for name, frame in frames.items():
                plain, cen, bias = frame[SCORES].to_numpy(), frame[CENTRED].to_numpy(), frame[BIAS].to_numpy()
                preds[name, "plain"] = [NAMES[plain[rows].argmax(axis=1)] for rows in cuts]
                for kind, groups in groupings.items():
                    preds[name, kind] = [NAMES[twin_scores(plain[rows], cen[rows], bias[rows], grp, args.twin)
                                               .argmax(axis=1)] for rows, grp in zip(cuts, groups, strict=True)]
            rows_all = np.concatenate(cuts)
            first = next(iter(frames))
            for (name, kind), pred in preds.items():
                row = {"partition": p, "m": m, "scoring": kind if kind == "plain" else f"twin, {kind} groups",
                       "run": name, "ba": np.mean([balanced_accuracy(y[r], q) for r, q in zip(cuts, pred, strict=True)])}
                if name != first:
                    diff = prompt_bootstrap(y[rows_all], np.concatenate(pred), prompts[rows_all],
                                            y_pred_baseline=np.concatenate(preds[first, kind]), **boot)
                    row |= dict(zip(("diff_ci95_lo", "diff_ci95_hi"), interval(diff), strict=True))
                    row["diff_vs_first"] = float(diff.mean())
                results.append(row)

    results, siblings = pd.DataFrame(results), pd.DataFrame(siblings)
    results.to_csv(out / "results.csv", index=False)
    siblings.to_csv(out / "siblings.csv", index=False)
    pd.set_option("display.width", 200)
    print(f"twin weight {args.twin}; runs: {', '.join(runs)} (differences are against the first)\n")
    print(siblings.round(4).to_string(index=False), "\n")
    print(results.round(4).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
