"""Sparse TF-IDF and style-feature linear baselines (report section D1, experiment 1).

Branches: raw character n-grams, case-preserving word n-grams (punctuation kept as
tokens), their concatenation, and ~50 numeric style features. Every learned statistic
(vocabulary, IDF, scaling) is fitted on the rows passed as training rows only.

Scores are decision-function outputs with columns in LABELS order. Argmax of the
scores is the prediction; scores from different branches are blended after scaling.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from nllm.data import LABELS
from nllm.features import WORD_TOKEN_PATTERN, CharNgrams
from nllm.metrics import balanced_accuracy

MIN_DF = 2
VECTORIZERS = {
    "char35": {"analyzer": CharNgrams(3, 5)},
    "char26": {"analyzer": CharNgrams(2, 6)},
    "word12": {"token_pattern": WORD_TOKEN_PATTERN, "ngram_range": (1, 2)},
}
# feature-set name -> (branch, vectorizer blocks hstacked in order)
FEATURE_SETS = {
    "char35": ("char", ("char35",)),
    "char26": ("char", ("char26",)),
    "word12": ("word", ("word12",)),
    "char35+word12": ("char+word", ("char35", "word12")),
    "char26+word12": ("char+word", ("char26", "word12")),
}
# Report D1 proposes C in {0.1, 1, 10}. LR on l2-normalised TF-IDF underfits at the low
# end, so its grid is shifted up a decade; the style LR grid is shifted down.
TEXT_CLASSIFIERS = [("lr", c) for c in (1.0, 10.0, 100.0)] + [("svm", c) for c in (0.1, 1.0, 10.0)]
STYLE_CLASSIFIERS = [("lr", c) for c in (0.01, 0.1, 1.0)] + [("hgb", None)]
SUBLINEAR = (False, True)
BLEND_WEIGHTS = (0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0)


@dataclass(frozen=True)
class Config:
    branch: str
    features: str
    clf: str
    C: float | None = None
    sublinear: bool | None = None

    @property
    def key(self) -> str:
        parts = [self.features, self.clf]
        if self.C is not None:
            parts.append(f"C={self.C:g}")
        if self.sublinear is not None:
            parts.append("sublinear" if self.sublinear else "raw_tf")
        return "|".join(parts)


# A fixed text+style reference for analyses that vary the data rather than the model (learning
# curve, external-data comparison): the configs and blend weight Exp 1 selected most often.
REFERENCE_TEXT = Config("char+word", "char35+word12", "lr", 100.0, False)
REFERENCE_STYLE = Config("style", "style", "lr", 0.1)
REFERENCE_STYLE_WEIGHT = 1.0


def all_configs() -> list[Config]:
    text = [
        Config(branch, name, clf, c, sub)
        for name, (branch, _) in FEATURE_SETS.items()
        for sub in SUBLINEAR
        for clf, c in TEXT_CLASSIFIERS
    ]
    style = [Config("style", "style", clf, c) for clf, c in STYLE_CLASSIFIERS]
    return text + style


def _classifier(cfg: Config, seed: int):
    if cfg.clf == "lr" and cfg.branch == "style":
        return make_pipeline(VarianceThreshold(), StandardScaler(), LogisticRegression(C=cfg.C, max_iter=5000))
    if cfg.clf == "lr":
        return LogisticRegression(C=cfg.C, max_iter=5000)
    if cfg.clf == "svm":
        return LinearSVC(C=cfg.C, max_iter=10000, random_state=seed)
    if cfg.clf == "hgb":
        return HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=seed
        )
    raise ValueError(cfg)


def _label_scores(model, X) -> np.ndarray:
    raw = model.decision_function(X)
    if raw.ndim == 1:  # two classes: sklearn returns the score of classes_[1] only
        raw = np.column_stack([-raw / 2, raw / 2])
    out = np.full((X.shape[0], len(LABELS)), -np.inf)
    out[:, [LABELS.index(c) for c in model.classes_]] = raw
    return out


def fit_score(
    train: pd.DataFrame,
    train_style: pd.DataFrame,
    evaluate: pd.DataFrame,
    eval_style: pd.DataFrame,
    configs: list[Config],
    seed: int,
    sample_weight: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Fit each config on train rows and return its scores on evaluate rows, keyed by Config.key.

    Each (vectorizer, sublinear) pair is fitted once and shared across classifiers.
    `sample_weight` weights training rows in the classifier fit (vocabulary and IDF are unweighted).
    """
    blocks: dict[tuple[str, bool], tuple[sp.csr_matrix, sp.csr_matrix]] = {}

    def block(name: str, sublinear: bool):
        if (name, sublinear) not in blocks:
            vec = TfidfVectorizer(
                min_df=MIN_DF, sublinear_tf=sublinear, lowercase=False, dtype=np.float32, **VECTORIZERS[name]
            )
            blocks[name, sublinear] = (vec.fit_transform(train.text), vec.transform(evaluate.text))
        return blocks[name, sublinear]

    scores = {}
    for cfg in configs:
        if cfg.branch == "style":
            X_train, X_eval = train_style, eval_style
        else:
            parts = [block(name, cfg.sublinear) for name in FEATURE_SETS[cfg.features][1]]
            X_train = sp.hstack([p[0] for p in parts], format="csr")
            X_eval = sp.hstack([p[1] for p in parts], format="csr")
        model = _classifier(cfg, seed)
        if sample_weight is None:
            model.fit(X_train, train.model)
        elif hasattr(model, "steps"):
            model.fit(X_train, train.model, **{f"{model.steps[-1][0]}__sample_weight": sample_weight})
        else:
            model.fit(X_train, train.model, sample_weight=sample_weight)
        scores[cfg.key] = _label_scores(model, X_eval)
    return scores


def mixing_weights(labels: np.ndarray, is_external: np.ndarray, share: float) -> np.ndarray:
    """Equal total weight per class, external rows taking `share` of it where the class has both; mean 1."""
    w = np.zeros(len(labels))
    for c in np.unique(labels):
        off, ext = (labels == c) & ~is_external, (labels == c) & is_external
        if off.any() and ext.any():
            w[off], w[ext] = (1 - share) / off.sum(), share / ext.sum()
        else:
            w[off | ext] = 1 / (off | ext).sum()
    return w * len(w) / w.sum()


def predict(scores: np.ndarray) -> np.ndarray:
    return np.asarray(LABELS)[scores.argmax(axis=1)]


def scale(scores: np.ndarray) -> float:
    """One scalar per branch: std of row-centred scores, so branches blend on comparable scales.

    Labels absent from the training rows score -inf; they are left out here (they would turn every
    row into NaN) and stay -inf in the blend, so they are never predicted.
    """
    finite = np.where(np.isfinite(scores), scores, np.nan)
    centred = finite - np.nanmean(finite, axis=1, keepdims=True)
    return float(np.nanstd(centred)) or 1.0


def blend(text: np.ndarray, style: np.ndarray, weight: float, text_scale: float, style_scale: float) -> np.ndarray:
    return text / text_scale + weight * style / style_scale


def select(
    inner_scores: dict[str, np.ndarray], y: np.ndarray, configs: list[Config]
) -> tuple[dict[str, Config], pd.DataFrame]:
    """Pick the best config per branch, plus 'text' (best non-style), by inner out-of-fold BA."""
    table = pd.DataFrame(
        [
            {"branch": c.branch, "config": c.key, "inner_ba": balanced_accuracy(y, predict(inner_scores[c.key]))}
            for c in configs
        ]
    )
    by_key = {c.key: c for c in configs}
    chosen = {}
    for branch, rows in table.groupby("branch", sort=False):
        chosen[branch] = by_key[rows.loc[rows.inner_ba.idxmax(), "config"]]
    text_rows = table[table.branch != "style"]
    chosen["text"] = by_key[text_rows.loc[text_rows.inner_ba.idxmax(), "config"]]
    return chosen, table


def choose_blend_weight(text: np.ndarray, style: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Inner-OOF choice of the style weight; returns (weight, text_scale, style_scale). Ties go to less style."""
    ts, ss = scale(text), scale(style)
    best = max(BLEND_WEIGHTS, key=lambda w: (balanced_accuracy(y, predict(blend(text, style, w, ts, ss))), -w))
    return best, ts, ss
