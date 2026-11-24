"""Do the open-weight candidates' likelihoods add to a classifier? A nested, prompt-grouped stacking check.

For each outer fold of a finished nested run (the 8B LoRA24), a correction is fitted on that fold's inner
out-of-fold rows (base scores from models that saw neither those rows nor the outer fold) and applied to the outer
rows. The base log-probabilities are never re-weighted per class (a free logistic regression on them alone loses
0.02, because inner and outer models differ in scale): the model is

    logits = a x base log-probabilities + bias + features @ W

with one scalar a. Compared on the same rows:

    base       the run's own scores, argmax
    recal      a and the 18 biases only (what refitting alone does)
    plus       a, biases and W on the likelihood features (scripts/candidate_likelihood.py; they use no label)
    only       biases and W, no base scores

The L2 penalty on W is chosen by prompt-grouped CV inside each correction's training rows. Random features give
plus = recal within noise (checked when this was written), so a gain here is the features' own.

--pair A B adds token-by-token contrasts between two models that share a tokenizer and a prefix (the two GLM
models): how much likelier each token is under A than under B. Sibling models differ exactly there, and the
passage's content cancels.

    uv run python scripts/candidate_eval.py outputs/tc2/qwen3-8b-l24-full-001 outputs/candidates/flash.npz [glm53.npz]
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from nllm.data import LABELS
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap

SCORES = [f"score:{m}" for m in LABELS]
NAMES = np.asarray(LABELS)
INDEX = {m: i for i, m in enumerate(LABELS)}
WATCH = ("glm-5.3", "glm-5.3-flash", "claude-fable-5", "claude-opus-5", "kimi-k3")
PENALTIES = (0.01, 0.1, 1.0, 10.0, 100.0)


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def fit(base, feats, y, penalty: float, use_base: bool = True):
    """(a, bias, W) minimising the cross-entropy of a x base + bias + feats @ W, plus penalty x mean(W^2)."""
    base, y = torch.tensor(base, dtype=torch.float64), torch.tensor(y)
    feats = torch.tensor(feats, dtype=torch.float64)
    a = torch.ones((), dtype=torch.float64, requires_grad=use_base)
    bias = torch.zeros(base.shape[1], dtype=torch.float64, requires_grad=True)
    w = torch.zeros(feats.shape[1], base.shape[1], dtype=torch.float64, requires_grad=feats.shape[1] > 0)
    opt = torch.optim.LBFGS([t for t in (a, bias, w) if t.requires_grad], max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logits = (a * base if use_base else 0) + bias + feats @ w
        loss = torch.nn.functional.cross_entropy(logits, y) + (penalty * w.square().mean() if w.numel() else 0.0)
        loss.backward()
        return loss

    opt.step(closure)
    return float(a.detach()) if use_base else 0.0, bias.detach().numpy(), w.detach().numpy()


def fit_predict(base_tr, f_tr, y_tr, groups, base_te, f_te, use_base: bool = True) -> tuple[np.ndarray, float]:
    """Predicted labels for the test rows; the penalty is chosen by grouped CV on the training rows."""
    if f_tr.shape[1]:
        scaler = StandardScaler().fit(f_tr)
        f_tr, f_te = scaler.transform(f_tr), scaler.transform(f_te)

    def predict(model, base, feats):
        a, bias, w = model
        return (a * base + bias + feats @ w).argmax(axis=1)

    best = PENALTIES[0]
    if f_tr.shape[1]:
        totals = dict.fromkeys(PENALTIES, 0.0)
        for tr, va in GroupKFold(n_splits=4).split(f_tr, y_tr, groups):
            for pen in PENALTIES:
                pred = predict(fit(base_tr[tr], f_tr[tr], y_tr[tr], pen, use_base), base_tr[va], f_tr[va])
                totals[pen] += balanced_accuracy(NAMES[y_tr[va]], NAMES[pred])
        best = max(PENALTIES, key=lambda k: (round(totals[k], 6), k))  # ties: the stronger penalty
    return NAMES[predict(fit(base_tr, f_tr, y_tr, best, use_base), base_te, f_te)], best


def load_features(paths: list[Path], keep: list[str] | None) -> pd.DataFrame:
    """One row per passage id; columns `<model>:<feature>` from every file."""
    frames = []
    for path in paths:
        z = np.load(path, allow_pickle=False)
        names = [str(n) for n in z["feature_names"]]
        frame = pd.DataFrame(z["features"], index=[str(i) for i in z["ids"]], columns=names)
        # The whole passage's log-probability: comparable between models with different tokenizers, unlike the
        # per-token mean.
        frame["lp_sum"] = frame.lp_mean * frame.n_tokens
        frame = frame[[n for n in frame.columns if keep is None or n in keep]]
        frames.append(frame.add_prefix(f"{str(z['model']) if 'model' in z else path.stem}:"))
    return pd.concat(frames, axis=1, join="inner")


def pair_features(paths: list[Path], a: str, b: str) -> pd.DataFrame:
    """Per passage, summaries of the per-token differences between models `a` and `b` (names as saved in the
    files), which must have scored the same token sequences."""
    files = {str(z["model"]): z for z in (np.load(p, allow_pickle=False) for p in paths) if "model" in z}
    za, zb = files[a], files[b]
    if not (np.array_equal(za["ids"], zb["ids"]) and np.array_equal(za["lengths"], zb["lengths"])):
        raise SystemExit(f"{a} and {b} did not score the same token sequences")
    bounds = np.concatenate([[0], np.cumsum(za["lengths"])])
    d_lp = za["tok_lp"] - zb["tok_lp"]
    d_rank = np.log1p(za["tok_rank"]) - np.log1p(zb["tok_rank"])
    d_ent = za["tok_entropy"] - zb["tok_entropy"]
    top_a, top_b = za["tok_rank"] == 0, zb["tok_rank"] == 0
    rows = []
    for lo, hi in itertools.pairwise(bounds):
        d = d_lp[lo:hi]
        tail = d[8:] if len(d) > 16 else d
        rows.append({"lp_mean": d.mean(), "lp_tail_mean": tail.mean(), "lp_std": d.std(), "lp_positive": (d > 0).mean(),
                     "lp_q10": np.quantile(d, 0.1), "lp_q50": np.quantile(d, 0.5), "lp_q90": np.quantile(d, 0.9),
                     "log_rank": d_rank[lo:hi].mean(), "entropy": d_ent[lo:hi].mean(),
                     "top1_only_first": (top_a[lo:hi] & ~top_b[lo:hi]).mean(),
                     "top1_only_second": (top_b[lo:hi] & ~top_a[lo:hi]).mean()})
    return pd.DataFrame(rows, index=[str(i) for i in za["ids"]]).add_prefix(f"{a}-{b}:")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="a nested run with oof.parquet, inner_oof.parquet and selections.csv")
    ap.add_argument("features", type=Path, nargs="+", help="files written by candidate_likelihood.py")
    ap.add_argument("--keep", nargs="+", help="feature names to use (default: all)")
    ap.add_argument("--pair", nargs=2, action="append", default=[], metavar=("A", "B"),
                    help="add token-by-token contrasts between two models that share a tokenizer")
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, help="where to write the outer predictions (default: next to the first file)")
    args = ap.parse_args()

    feats = load_features(args.features, args.keep)
    for a, b in args.pair:
        feats = feats.join(pair_features(args.features, a, b), how="inner")
    used = list(feats.columns)
    oof = pd.read_parquet(args.run / "oof.parquet")
    inner = pd.read_parquet(args.run / "inner_oof.parquet").merge(
        pd.read_csv(args.run / "selections.csv"), on=["partition", "outer_fold", "lr", "epoch"])
    missing = set(oof.id) - set(feats.index)
    if missing:
        raise SystemExit(f"{len(missing)} rows of the run have no likelihood features")

    # How the features look per label. They use no label, so this describes them; nothing is selected from it.
    first = oof[oof.partition == oof.partition.min()]
    lead = [c for c in used if c.split(":")[1] in ("lp_mean", "top1", "curvature", "lp_positive")]
    pd.set_option("display.width", 200)
    print(feats.loc[first.id, lead].assign(model=first.model.to_numpy()).groupby("model").mean().round(3).to_string(), "\n")
    for label in ("glm-5.3-flash", "glm-5.3"):
        y = (first.model == label).to_numpy()
        aucs = {k: roc_auc_score(y, feats.loc[first.id, k]) for k in used}
        best = sorted(aucs, key=lambda k: -abs(aucs[k] - 0.5))[:5]
        print(f"one-vs-rest AUC for {label}: " + ", ".join(f"{k} {aucs[k]:.3f}" for k in best))
    print()

    preds = []
    for (p, o), val in oof.groupby(["partition", "outer_fold"]):
        tr = inner[(inner.partition == p) & (inner.outer_fold == o)]
        y_tr, g_tr = tr.model.map(INDEX).to_numpy(), tr.prompt_id.to_numpy()
        base_tr, base_va = log_softmax(tr[SCORES].to_numpy()), log_softmax(val[SCORES].to_numpy())
        f_tr, f_va = feats.loc[tr.id].to_numpy(), feats.loc[val.id].to_numpy()
        none_tr, none_va = np.zeros((len(tr), 0)), np.zeros((len(val), 0))
        recal, _ = fit_predict(base_tr, none_tr, y_tr, g_tr, base_va, none_va)
        plus, penalty = fit_predict(base_tr, f_tr, y_tr, g_tr, base_va, f_va)
        only, _ = fit_predict(base_tr, f_tr, y_tr, g_tr, base_va, f_va, use_base=False)
        preds.append(val[["partition", "outer_fold", "id", "prompt_id", "model"]].assign(
            base=NAMES[val[SCORES].to_numpy().argmax(axis=1)], recal=recal, plus=plus, only=only))
        print(f"p{p} o{o}: base {balanced_accuracy(val.model, preds[-1].base):.3f}  recal "
              f"{balanced_accuracy(val.model, recal):.3f}  plus {balanced_accuracy(val.model, plus):.3f} "
              f"(penalty {penalty})  only {balanced_accuracy(val.model, only):.3f}", flush=True)
    preds = pd.concat(preds, ignore_index=True)
    print()
    rows = []
    for p, g in preds.groupby("partition"):
        row = {"partition": p} | {k: balanced_accuracy(g.model, g[k]) for k in ("base", "recal", "plus", "only")}
        for ref in ("base", "recal"):
            lo, hi = interval(prompt_bootstrap(g.model, g.plus, g.prompt_id, y_pred_baseline=g[ref],
                                               n_draws=args.bootstrap_draws))
            row |= {f"plus-{ref}": row["plus"] - row[ref], f"plus-{ref} lo": lo, f"plus-{ref} hi": hi}
        rows.append(row)
        recalls = pd.DataFrame({k: per_class_recall(g.model, g[k]) for k in ("base", "recal", "plus")})
        print(f"partition {p}, recall:\n{recalls.loc[list(WATCH)].round(3).to_string()}")
        gain = (recalls.plus - recalls.recal).sort_values()
        print("largest recall changes (plus minus recal):",
              ", ".join(f"{k} {v:+.3f}" for k, v in pd.concat([gain.head(3), gain.tail(5)]).items()), "\n")
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    out = args.out or args.features[0].with_suffix(".stack.parquet")
    preds.to_parquet(out, index=False)
    print(f"\nfeatures: {len(used)} from {', '.join(p.name for p in args.features)}\nwrote {out}")


if __name__ == "__main__":
    main()
