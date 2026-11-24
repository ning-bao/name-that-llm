"""How official does a batch of generated answers look? Phase 0 of the data study
(notes/data_study_plan_2026-11-08.md).

A "cell" is a batch of passages with claimed labels from one source or generation setup. It is compared with the
official passages (all 150 prompts: the confirmation set has had its single use, Exp 9) in four ways:

- recognition ratio: a recogniser trained on official rows only gives the cell's passages their claimed label at
  some rate; divided by the same recogniser's recall on unseen official prompts (prompt-grouped CV), 1.0 means
  the cell looks as official as official does. Recognisers: the linear text+style reference here, and a
  Qwen3-4B LoRA24 in scripts/closeness_recogniser.py.
- style AUC: a style-features-only discriminator, official vs cell (0.5 = indistinguishable).
- word counts: the cell's median against the label's official median, and the KS distance between them.
- family transfer: a linear model trained only on the cell's rows of one provider family, tested on the official
  rows of those labels, against the same model trained on as many official rows per label (prompt-grouped CV).
  This asks the question training cares about: do the differences between sibling labels carry over?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nllm import linear
from nllm.data import LABELS
from nllm.features import style_features
from nllm.metrics import balanced_accuracy

NAMES = np.asarray(LABELS)


def family(label: str) -> str:
    """Provider family: claude, gemini, glm, gpt, kimi, qwen3.8."""
    return label.split("-")[0]


def prompt_folds(prompt_ids, k: int = 5, seed: int = 0) -> list[frozenset[str]]:
    """k disjoint sets of prompt ids (validation prompts of each fold), fixed by seed."""
    ids = np.array(sorted(set(prompt_ids)))
    order = np.random.default_rng(seed).permutation(len(ids))
    return [frozenset(ids[order[i::k]]) for i in range(k)]


def linear_scores(train: pd.DataFrame, evaluate: pd.DataFrame, seed: int = 17,
                  train_style: pd.DataFrame | None = None, eval_style: pd.DataFrame | None = None) -> np.ndarray:
    """Scores (LABELS order) of the text+style reference fitted on `train` rows (columns text, model). Labels
    absent from `train` score -inf."""
    ts = style_features(train.text) if train_style is None else train_style
    es = style_features(evaluate.text) if eval_style is None else eval_style
    tr = train.reset_index(drop=True)
    ev = evaluate.reset_index(drop=True)
    s = linear.fit_score(tr, ts.reset_index(drop=True), ev, es.reset_index(drop=True),
                         [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE], seed, None)
    t, st = s[linear.REFERENCE_TEXT.key], s[linear.REFERENCE_STYLE.key]
    return linear.blend(t, st, linear.REFERENCE_STYLE_WEIGHT, linear.scale(t), linear.scale(st))


def cv_predictions(official: pd.DataFrame, folds, seed: int = 17) -> pd.Series:
    """Prompt-grouped out-of-fold predictions of the linear reference on official rows (index preserved)."""
    style = style_features(official.text).set_axis(official.index)
    out = pd.Series(index=official.index, dtype=object)
    for val in folds:
        v = official.prompt_id.isin(val)
        sc = linear_scores(official[~v], official[v], seed, style[~v], style[v])
        out[v] = NAMES[sc.argmax(axis=1)]
    return out


def recall_by_label(y, pred) -> pd.Series:
    y, pred = np.asarray(y), np.asarray(pred)
    return pd.Series({m: float((pred[y == m] == m).mean()) for m in LABELS if (y == m).any()})


def style_auc(official_texts, cell_texts, reps: int = 3, n_max: int = 150) -> float:
    """Mean 5-fold AUC of a style-features-only logistic model, official vs cell, on balanced subsamples."""
    off, cell = pd.Series(list(official_texts)), pd.Series(list(cell_texts))
    n = min(len(off), len(cell), n_max)
    if n < 20:
        return float("nan")
    aucs = []
    for rep in range(reps):
        a, b = off.sample(n, random_state=rep), cell.sample(n, random_state=rep)
        x = style_features(pd.concat([a, b])).to_numpy()
        y = np.r_[np.zeros(n), np.ones(n)]
        clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
        p = cross_val_predict(clf, x, y, cv=StratifiedKFold(5, shuffle=True, random_state=rep),
                              method="predict_proba")[:, 1]
        aucs.append(roc_auc_score(y, p))
    return float(np.mean(aucs))


def word_stats(official_texts, cell_texts) -> dict:
    a = pd.Series(list(official_texts)).str.split().str.len()
    b = pd.Series(list(cell_texts)).str.split().str.len()
    return {"words_med": float(b.median()), "words_med_official": float(a.median()),
            "words_ks": float(ks_2samp(a, b).statistic)}


def family_transfer(cell: pd.DataFrame, official: pd.DataFrame, folds, min_rows: int = 20, n_cap: int = 100,
                    seed: int = 17) -> list[dict]:
    """Per provider family with >= 2 labels of >= min_rows rows in the cell: BA on official rows of a model trained
    on the cell's rows only, against one trained on the same number of official rows per label (out of fold).
    n_cap stays below the ~120 official rows per label a 5-fold training split holds, so both sides get n rows.
    transfer_lift = (transfer - chance) / (official - chance): 1.0 means the cell teaches the family as well as
    official rows do."""
    rows = []
    counts = cell.model.value_counts()
    for fam in sorted({family(m) for m in counts.index}):
        labels = sorted(m for m in counts.index if family(m) == fam and counts[m] >= min_rows)
        if len(labels) < 2:
            continue
        n = int(min(n_cap, counts[labels].min()))
        rng = np.random.default_rng(seed)
        train = pd.concat([g.iloc[rng.permutation(len(g))[:n]] for _, g in cell[cell.model.isin(labels)]
                          .groupby("model")])
        test = official[official.model.isin(labels)]
        idx = [LABELS.index(m) for m in labels]

        def predict(tr, ev, labels=labels, idx=idx):
            return np.asarray(labels)[linear_scores(tr, ev, seed)[:, idx].argmax(axis=1)]

        transfer = balanced_accuracy(test.model, predict(train, test))
        oof = pd.Series(index=test.index, dtype=object)
        for val in folds:
            v = test.prompt_id.isin(val)
            pool = test[~v]
            tr = pd.concat([g.iloc[rng.permutation(len(g))[:n]] for _, g in pool.groupby("model")])
            oof[v] = predict(tr, test[v])
        ref = balanced_accuracy(test.model, oof.to_numpy())
        rows.append({"family": fam, "labels": " ".join(labels), "n_per_label": n, "transfer_ba": transfer,
                     "official_ba": ref, "chance": 1 / len(labels),
                     "transfer_lift": (transfer - 1 / len(labels)) / (ref - 1 / len(labels))})
    return rows
