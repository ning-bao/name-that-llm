"""Plan E6: nested check of test-time soft centring inside content clusters (nllm.centring) on finished runs.

Every prompt is cut to m random answers (the hidden test's answers per prompt are unknown and probably few),
passages are clustered by content on that cut, and a share w of each row's cluster mean (cavity: the other rows
only; or plain) is subtracted from its scores. With several --run, the scores are their sum, each divided by the
member's scale as in the Kaggle notebook (--scales; default each run's mean inner scale, nllm.linear.scale).

Per outer fold, (w, cavity, cut) is chosen on the inner OOF scores of that fold's training prompts (at the
selected lr and epoch), averaged over --select-m and the draws; it is then applied to the outer scores. The outer
clustering runs over each partition's whole cut (all 120 dev prompts), the closest we get to the test's ~4,000
passages. Reports BA and the paired prompt-bootstrap difference against no centring for each m.

    uv run python scripts/test_time_centring.py outputs/tc2/qwen3-8b-l24-full-001
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from nllm import centring
from nllm.data import LABELS, load_train
from nllm.linear import scale as member_scale
from nllm.metrics import balanced_accuracy, interval, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import development_rows

SCORES = [f"score:{m}" for m in LABELS]
NAMES = np.asarray(LABELS)


def load_runs(runs: list[Path], scales: list[float] | None) -> tuple[pd.DataFrame, pd.DataFrame, list[float]]:
    """(outer rows, inner rows at each fold's selected lr/epoch) with scores summed over runs / scale."""
    outer, inner, used = None, None, []
    for k, run in enumerate(runs):
        oof = pd.read_parquet(run / "oof.parquet")
        inn = pd.read_parquet(run / "inner_oof.parquet")
        sel = pd.read_csv(run / "selections.csv")
        inn = inn.merge(sel, on=["partition", "outer_fold", "lr", "epoch"])
        s = scales[k] if scales else float(np.mean([member_scale(g[SCORES].to_numpy())
                                                    for _, g in inn.groupby(["partition", "outer_fold"])]))
        used.append(s)
        keys_o, keys_i = ["partition", "id"], ["partition", "outer_fold", "id"]
        o = oof.set_index(keys_o)[SCORES] / s
        i = inn.set_index(keys_i)[SCORES] / s
        if outer is None:
            outer = oof.set_index(keys_o)[["prompt_id", "model", "outer_fold"]].join(o)
            inner = inn.set_index(keys_i)[["prompt_id", "model"]].join(i)
        else:  # members must cover the same rows (a p0-only member drops partition 1)
            outer = outer.loc[outer.index.intersection(o.index)]
            inner = inner.loc[inner.index.intersection(i.index)]
            outer[SCORES] = outer[SCORES].to_numpy() + o.loc[outer.index].to_numpy()
            inner[SCORES] = inner[SCORES].to_numpy() + i.loc[inner.index].to_numpy()
    return outer.reset_index(), inner.reset_index(), used


def cuts_of(frame: pd.DataFrame, m: int, draws: int, seed: int) -> list[np.ndarray]:
    """Row positions of `draws` random cuts of every prompt to m answers (the same cuts for every config)."""
    rng = np.random.default_rng(seed)
    groups = list(frame.groupby("prompt_id").indices.values())
    return [np.sort(np.concatenate([rng.permutation(g)[:m] for g in groups])) for _ in range(draws)]


def pair_scores(clusters: np.ndarray, prompts: np.ndarray) -> tuple[float, float]:
    """Pair precision and recall of clusters against the true prompts."""
    same_c = clusters[:, None] == clusters[None, :]
    same_p = prompts[:, None] == prompts[None, :]
    off = ~np.eye(len(clusters), dtype=bool)
    tp = (same_c & same_p & off).sum()
    return tp / max((same_c & off).sum(), 1), tp / max((same_p & off).sum(), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--scales", type=float, nargs="+")
    ap.add_argument("--weights", type=float, nargs="+", default=[0, 0.1, 0.25, 0.4, 0.6])
    ap.add_argument("--cuts", type=float, nargs="+", default=[0.75, 0.8, 0.85, 0.9])
    ap.add_argument("--select-m", type=int, nargs="+", default=[2, 3, 6],
                    help="answers per prompt the inner selection averages over")
    ap.add_argument("--eval-m", type=int, nargs="+", default=[2, 3, 6, 18])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--bootstrap-draws", type=int, default=1000)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.scales and len(args.scales) != len(args.runs):
        ap.error("one --scales value per run")
    out = args.out or OUTPUTS / "centring" / "+".join(r.name for r in args.runs)
    out.mkdir(parents=True, exist_ok=True)

    text = development_rows(load_train()).set_index("id").text
    outer, inner, scales = load_runs(args.runs, args.scales)
    configs = [(0.0, True, args.cuts[0])] + [(w, cav, c) for w, cav, c in
                                              itertools.product(args.weights, (True, False), args.cuts) if w > 0]

    # Nested selection on inner OOF rows, per outer fold.
    chosen, table = {}, []
    for (p, o), g in inner.groupby(["partition", "outer_fold"]):
        g = g.reset_index(drop=True)
        y, scores, texts = g.model.to_numpy(), g[SCORES].to_numpy(), text.loc[g.id].to_numpy()
        totals = dict.fromkeys(configs, 0.0)
        for m in args.select_m:
            for d, rows in enumerate(cuts_of(g, m, args.draws, 100 * p + 10 * o + m)):
                clusters = {c: centring.content_clusters(texts[rows], c) for c in args.cuts}
                for w, cav, c in configs:
                    adj = centring.centre(scores[rows], clusters[c], w, cav)
                    totals[w, cav, c] += balanced_accuracy(y[rows], NAMES[adj.argmax(axis=1)])
        n = len(args.select_m) * args.draws
        best = max(configs, key=lambda k: (round(totals[k] / n, 6), -k[0]))  # ties: the smaller weight
        chosen[p, o] = best
        table += [{"partition": p, "outer_fold": o, "weight": w, "cavity": cav, "cut": c, "inner_ba": totals[w, cav, c] / n}
                  for w, cav, c in configs]
        print(f"p{p} o{o}: chose w={best[0]} cavity={best[1]} cut={best[2]} "
              f"(inner {totals[best] / n:.4f} vs {totals[configs[0]] / n:.4f} plain)", flush=True)
    pd.DataFrame(table).to_csv(out / "inner_scores.csv", index=False)
    pd.DataFrame([{"partition": p, "outer_fold": o, "weight": w, "cavity": cav, "cut": c}
                  for (p, o), (w, cav, c) in chosen.items()]).to_csv(out / "selections.csv", index=False)

    # Outer: each partition's cut clustered as one set; each row gets its own fold's chosen config.
    results, fixed, quality = [], [], []
    for p, g in outer.groupby("partition"):
        g = g.reset_index(drop=True)
        y, scores, texts = g.model.to_numpy(), g[SCORES].to_numpy(), text.loc[g.id].to_numpy()
        folds, prompts = g.outer_fold.to_numpy(), g.prompt_id.to_numpy()
        for m in args.eval_m:
            plain, nested, ids, fixed_preds = [], [], [], {k: [] for k in configs}
            for d, rows in enumerate(cuts_of(g, m, args.draws, 7919 + 100 * p + m)):
                clusters = {c: centring.content_clusters(texts[rows], c) for c in args.cuts}
                for c in args.cuts:
                    prec, rec = pair_scores(clusters[c], prompts[rows])
                    quality.append({"partition": p, "m": m, "draw": d, "cut": c, "pair_precision": prec,
                                    "pair_recall": rec, "clusters": len(np.unique(clusters[c])), "rows": len(rows)})
                adj = {k: centring.centre(scores[rows], clusters[k[2]], k[0], k[1]) for k in configs}
                pick = np.stack([adj[chosen[p, f]][j] for j, f in enumerate(folds[rows])])
                plain.append(NAMES[scores[rows].argmax(axis=1)])
                nested.append(NAMES[pick.argmax(axis=1)])
                ids.append(rows)
                for k in configs:
                    fixed_preds[k].append(NAMES[adj[k].argmax(axis=1)])
            rows = np.concatenate(ids)
            base, pred = np.concatenate(plain), np.concatenate(nested)
            diff = prompt_bootstrap(y[rows], pred, prompts[rows], y_pred_baseline=base, n_draws=args.bootstrap_draws)
            lo, hi = interval(diff)
            ba_plain = np.mean([balanced_accuracy(y[r], b) for r, b in zip(ids, plain, strict=True)])
            ba_nested = np.mean([balanced_accuracy(y[r], b) for r, b in zip(ids, nested, strict=True)])
            results.append({"partition": p, "m": m, "ba_plain": ba_plain, "ba_centred": ba_nested,
                            "diff": ba_nested - ba_plain, "diff_ci95_lo": lo, "diff_ci95_hi": hi})
            print(f"p{p} m={m:>2}: plain {ba_plain:.4f} -> nested centring {ba_nested:.4f} "
                  f"({ba_nested - ba_plain:+.4f} [{lo:+.4f}, {hi:+.4f}])", flush=True)
            for k in configs:
                fixed.append({"partition": p, "m": m, "weight": k[0], "cavity": k[1], "cut": k[2],
                              "ba": np.mean([balanced_accuracy(y[r], b)
                                             for r, b in zip(ids, fixed_preds[k], strict=True)])})
    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    pd.DataFrame(fixed).to_csv(out / "fixed_configs.csv", index=False)
    q = pd.DataFrame(quality)
    q.to_csv(out / "cluster_quality.csv", index=False)
    print(q.groupby(["m", "cut"])[["pair_precision", "pair_recall"]].mean().round(3).unstack("cut").to_string())
    (out / "config.json").write_text(json.dumps(vars(args) | {"runs": [str(r) for r in args.runs], "scales": scales,
                                                               "out": str(out)}, indent=2), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
