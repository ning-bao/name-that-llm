"""Exp 12: within-provider sibling specialists on top of the Qwen3-8B LoRA24 model (CPU only).

Question: does a classifier that only has to tell siblings apart add information the 18-way 8B lacks?

For each sibling group G (Claude, Gemini, GPT, GLM, and the cross-family fable/GLM triangle) a linear
text+style specialist (the Exp 1 reference config: char 3-5 + word 1-2 TF-IDF LR C=100, plus 51 style
features LR C=0.1, blended at weight 1) is trained on G's labels only: official rows of the fit's training
prompts plus 2,000 uniform pool+extra rows per label (the same rows the 8B's shared stage 1 drew from;
official rows get half of each class's weight, as in Exp 6). It is combined with the 8B so that the
8B's total probability on G is kept and only its split inside G changes:

    p(l) = p8(G) * softmax_G(z8_G + alpha * s_G)    for l in G,  p(l) = p8(l) otherwise,

where s_G is the specialist's row-centred score divided by its inner scale. alpha = 0 is the 8B unchanged.
alpha is chosen per outer fold and group on inner out-of-fold 18-class BA (the 8B's inner OOF at its selected
lr/epoch, the specialist's from fits on the 64 inner-training prompts); outer prompts are only scored.

Runs against the nllm package of commit ae5d01c (pool.py), e.g. from the main checkout:

    NLLM_DATA_DIR=data/raw NLLM_POOL_DATA=data/external \
    PYTHONPATH=.claude/worktrees/leaderboard-architecture-review-a80ffa/src \
    uv run python <this file> --run outputs/tc2/qwen3-8b-l24-full-001 --jobs 10
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

from nllm import linear, pool, splits
from nllm.data import LABELS, load_train
from nllm.features import style_features
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap

GROUPS = {
    "claude": [l for l in LABELS if l.startswith("claude")],
    "gemini": [l for l in LABELS if l.startswith("gemini")],
    "gpt": [l for l in LABELS if l.startswith("gpt")],
    "glm": ["glm-5.3", "glm-5.3-flash"],
    "triangle": ["claude-fable-5", "glm-5.3", "glm-5.3-flash"],
}
ALPHAS = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)
CONFIGS = [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE]
SEED = 17


def fit_task(key, group, train_rows, train_style, train_w, eval_rows, eval_style):
    scores = linear.fit_score(train_rows, train_style, eval_rows, eval_style, CONFIGS, SEED, sample_weight=train_w)
    cols = [LABELS.index(l) for l in GROUPS[group]]
    return key, eval_rows.id.to_numpy(), scores[linear.REFERENCE_TEXT.key][:, cols], scores[linear.REFERENCE_STYLE.key][:, cols]


def combine(z8: np.ndarray, spec: np.ndarray | None, cols: list[int], alpha: float) -> np.ndarray:
    """Log-probabilities after redistributing the 8B's mass inside one group (z8 rows x 18 logits)."""
    logp = z8 - logsumexp(z8, axis=1, keepdims=True)
    if spec is None or alpha == 0:
        return logp
    mass = logsumexp(logp[:, cols], axis=1, keepdims=True)
    inner = z8[:, cols] + alpha * spec
    out = logp.copy()
    out[:, cols] = mass + inner - logsumexp(inner, axis=1, keepdims=True)
    return out


def specialist_score(text, style, ts, ss, s_scale):
    s = linear.blend(text, style, linear.REFERENCE_STYLE_WEIGHT, ts, ss)
    return (s - s.mean(axis=1, keepdims=True)) / s_scale


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True, help="nested 8B run dir (oof.parquet, inner_oof.parquet, selections.csv)")
    ap.add_argument("--out", type=Path, default=Path("outputs/sibling_specialists"))
    ap.add_argument("--per-label", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--groups", nargs="*", default=list(GROUPS))
    ap.add_argument("--partitions", nargs="*", type=int, default=[0, 1])
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    out = args.out / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)
    t0 = time.time()

    train = load_train()
    dev = splits.development_rows(train).reset_index(drop=True)
    dev_style = style_features(dev.text).set_axis(dev.index)

    cand = pool.candidate_rows()
    cand = cand[cand.origin.isin(["pool", "extra"])].reset_index(drop=True)
    samples, sample_style = {}, {}
    for g in args.groups:
        s = pool.balanced_sample(cand[cand.model.isin(GROUPS[g])], args.per_label, SEED).reset_index(drop=True)
        samples[g] = s
        sample_style[g] = style_features(s.text)
        print(f"{g}: pool sample {len(s)} rows {s.model.value_counts().to_dict()}", flush=True)

    # 8B scores: outer OOF, and inner OOF at each outer fold's selected (lr, epoch)
    oof = pd.read_parquet(args.run / "oof.parquet")
    inner = pd.read_parquet(args.run / "inner_oof.parquet")
    sel = pd.read_csv(args.run / "selections.csv")
    inner = inner.merge(sel, on=["partition", "outer_fold", "lr", "epoch"])
    score_cols = [f"score:{l}" for l in LABELS]

    tasks = []
    folds = {(p, f.outer): f for p in args.partitions for f in splits.outer_folds(p) if f.outer in args.folds}
    for g in args.groups:
        s, ss = samples[g], sample_style[g]
        for (p, o), fold in folds.items():
            pieces = [(f"inner{k}", tr, va) for k, (tr, va) in enumerate(fold.inner())]
            pieces.append(("outer", fold.train, fold.validation))
            for name, tr_prompts, ev_prompts in pieces:
                off = dev[dev.prompt_id.isin(tr_prompts) & dev.model.isin(GROUPS[g])]
                tr = pd.concat([off[["id", "text", "model"]], s[["id", "text", "model"]]], ignore_index=True)
                tr_style = pd.concat([dev_style.loc[off.index], ss], ignore_index=True)
                is_ext = np.r_[np.zeros(len(off), bool), np.ones(len(s), bool)]
                w = linear.mixing_weights(tr.model.to_numpy(), is_ext, 0.5)
                ev = dev[dev.prompt_id.isin(ev_prompts)]
                tasks.append(delayed(fit_task)((g, p, o, name), g, tr, tr_style, w, ev[["id", "text", "model"]],
                                               dev_style.loc[ev.index].reset_index(drop=True)))
    print(f"{len(tasks)} fits on {args.jobs} workers", flush=True)
    results = Parallel(n_jobs=args.jobs, verbose=5)(tasks)
    fits = {key: (ids, text, style) for key, ids, text, style in results}
    print(f"fits done in {time.time() - t0:.0f} s", flush=True)

    # per (group, partition, outer fold): scales and alpha from inner OOF, then outer specialist scores
    chosen, outer_spec = [], {}
    for g in args.groups:
        cols = [LABELS.index(l) for l in GROUPS[g]]
        for (p, o), fold in folds.items():
            ids = np.concatenate([fits[g, p, o, f"inner{k}"][0] for k in range(3)])
            it = np.vstack([fits[g, p, o, f"inner{k}"][1] for k in range(3)])
            ist = np.vstack([fits[g, p, o, f"inner{k}"][2] for k in range(3)])
            ts, ss = linear.scale(it), linear.scale(ist)
            raw = linear.blend(it, ist, linear.REFERENCE_STYLE_WEIGHT, ts, ss)
            s_scale = linear.scale(raw)
            ispec = specialist_score(it, ist, ts, ss, s_scale)
            inn = inner[(inner.partition == p) & (inner.outer_fold == o)].set_index("id").loc[ids]
            z8i, yi = inn[score_cols].to_numpy(float), inn.model.to_numpy()
            ba = {a: balanced_accuracy(yi, np.asarray(LABELS)[combine(z8i, ispec, cols, a).argmax(1)]) for a in ALPHAS}
            alpha = max(ALPHAS, key=lambda a: (round(ba[a], 10), -a))
            chosen.append({"group": g, "partition": p, "outer_fold": o, "alpha": alpha,
                           "inner_ba_alpha0": ba[0.0], "inner_ba_chosen": ba[alpha]})
            oids, ot, ost = fits[g, p, o, "outer"]
            outer_spec[g, p, o] = (oids, specialist_score(ot, ost, ts, ss, s_scale), alpha)
    chosen = pd.DataFrame(chosen)
    chosen.to_csv(out / "alphas.csv", index=False)

    # apply to the outer OOF rows; conditions = single groups, the four families, families with the triangle
    conditions = {"8b": []} | {f"+{g}": [g] for g in args.groups}
    fam = [g for g in ("claude", "gemini", "gpt", "glm") if g in args.groups]
    if len(fam) > 1:
        conditions["+families"] = fam
    if "triangle" in args.groups and len(fam) > 1:
        conditions["+families,triangle"] = [g for g in fam if g != "glm"] + ["triangle"]
    rows, preds = [], {}
    for p in args.partitions:
        part = oof[(oof.partition == p) & oof.outer_fold.isin(args.folds)].set_index("id")
        y, groups_ = part.model.to_numpy(), part.prompt_id.to_numpy()
        for cname, glist in conditions.items():
            logp = part[score_cols].to_numpy(float).copy()
            for g in glist:
                cols = [LABELS.index(l) for l in GROUPS[g]]
                for o in args.folds:
                    oids, spec, alpha = outer_spec[g, p, o]
                    pos = part.index.get_indexer(oids)
                    logp[pos] = combine(logp[pos], spec, cols, alpha)  # logp is a valid logit input
            preds[cname, p] = np.asarray(LABELS)[logp.argmax(1)]
        base = preds["8b", p]
        for cname in conditions:
            pr = preds[cname, p]
            d = prompt_bootstrap(y, pr, groups_, y_pred_baseline=base)
            lo, hi = interval(d)
            rows.append({"partition": p, "condition": cname, "ba": balanced_accuracy(y, pr),
                         "delta": balanced_accuracy(y, pr) - balanced_accuracy(y, base), "lo": lo, "hi": hi})
        rec = pd.DataFrame({c: per_class_recall(y, preds[c, p]) for c in conditions})
        rec.to_csv(out / f"recall_p{p}.csv")
    res = pd.DataFrame(rows)
    res.to_csv(out / "results.csv", index=False)

    # specialists alone, forced within their group, against the 8B forced within the group
    alone = []
    for g in args.groups:
        cols = [LABELS.index(l) for l in GROUPS[g]]
        for p in args.partitions:
            part = oof[oof.partition == p].set_index("id")
            ids = np.concatenate([outer_spec[g, p, o][0] for o in args.folds])
            spec = np.vstack([outer_spec[g, p, o][1] for o in args.folds])
            sub = part.loc[ids]
            m = sub.model.isin(GROUPS[g]).to_numpy()
            t = np.array([GROUPS[g].index(v) for v in sub.model[m]])
            z8 = sub[score_cols].to_numpy(float)[m][:, cols]
            sp_ = spec[m]
            row = {"group": g, "partition": p, "rows": int(m.sum()),
                   "8b_forced_acc": float((z8.argmax(1) == t).mean()), "spec_forced_acc": float((sp_.argmax(1) == t).mean())}
            for a in range(len(cols)):
                for b in range(a + 1, len(cols)):
                    mm = (t == a) | (t == b)
                    row[f"auc8b {GROUPS[g][a]}|{GROUPS[g][b]}"] = roc_auc_score(t[mm] == a, z8[mm, a] - z8[mm, b])
                    row[f"aucsp {GROUPS[g][a]}|{GROUPS[g][b]}"] = roc_auc_score(t[mm] == a, sp_[mm, a] - sp_[mm, b])
            alone.append(row)
    pd.DataFrame(alone).to_csv(out / "specialist_alone.csv", index=False)
    (out / "config.json").write_text(json.dumps(vars(args) | {"alphas": ALPHAS, "seed": SEED,
                                                              "seconds": round(time.time() - t0)}, default=str), encoding="utf-8")
    pd.set_option("display.width", 200)
    print(res.to_string(index=False))
    print(chosen.groupby("group").alpha.describe()[["mean", "min", "max"]])
    print(pd.DataFrame(alone)[["group", "partition", "rows", "8b_forced_acc", "spec_forced_acc"]].to_string(index=False))
    print("written to", out, f"in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
