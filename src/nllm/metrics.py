"""Official metric and prompt-cluster bootstrap.

Balanced accuracy is the mean per-class recall over classes present in y_true.
Rows answering the same prompt are not independent, so uncertainty comes from
resampling whole prompt groups, keeping all of their rows.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from nllm.data import LABELS, family


def confusion(y_true, y_pred, labels: Sequence[str] = LABELS) -> pd.DataFrame:
    """Row-normalised confusion matrix: rows are true labels, columns predicted."""
    counts = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    counts = counts.reindex(index=labels, columns=labels, fill_value=0)
    return counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0)


def within_family_error_share(y_true, y_pred) -> float:
    """Of the misclassified rows, the share predicted as a sibling from the true provider family."""
    wrong = [(t, p) for t, p in zip(y_true, y_pred) if t != p]
    return float(np.mean([family(t) == family(p) for t, p in wrong])) if wrong else float("nan")


def per_class_recall(y_true, y_pred, labels: Sequence[str] = LABELS) -> pd.Series:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    present = [m for m in labels if (y_true == m).any()]
    return pd.Series(
        {m: float((y_pred[y_true == m] == m).mean()) for m in present}, name="recall"
    )


def balanced_accuracy(y_true, y_pred, labels: Sequence[str] = LABELS) -> float:
    return float(per_class_recall(y_true, y_pred, labels).mean())


def prompt_bootstrap(
    y_true,
    y_pred,
    groups,
    *,
    y_pred_baseline=None,
    n_draws: int = 2000,
    seed: int = 0,
) -> np.ndarray:
    """Balanced accuracy over n_draws prompt-group resamples.

    With y_pred_baseline, returns the paired difference BA(y_pred) - BA(baseline)
    computed on the same resample, which is what promotion decisions should use.
    """
    y_true = np.asarray(y_true)
    group_codes, group_names = pd.factorize(np.asarray(groups))
    class_codes, class_names = pd.factorize(y_true)
    shape = (len(group_names), len(class_names))

    totals = np.zeros(shape)
    np.add.at(totals, (group_codes, class_codes), 1)

    rng = np.random.default_rng(seed)
    n_groups = shape[0]
    weights = rng.multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_draws)
    resampled_totals = weights @ totals

    def resampled_ba(pred) -> np.ndarray:
        correct = np.zeros(shape)
        np.add.at(correct, (group_codes, class_codes), np.asarray(pred) == y_true)
        recall = np.divide(
            weights @ correct,
            resampled_totals,
            out=np.full_like(resampled_totals, np.nan),
            where=resampled_totals > 0,
        )
        return np.nanmean(recall, axis=1)

    draws = resampled_ba(y_pred)
    if y_pred_baseline is not None:
        draws = draws - resampled_ba(y_pred_baseline)
    return draws


def sign_flip_pvalue(differences, n_draws: int = 100_000, seed: int = 0) -> float:
    """One-sided paired permutation p-value for mean(differences) > 0 (exact when n <= 16).

    Under the null both members of each pair come from the same source, so each difference's
    sign is exchangeable.
    """
    d = np.asarray(differences, dtype=float)
    if len(d) <= 16:
        signs = 1 - 2 * ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1)
    else:
        signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=(n_draws, len(d)))
    return float(((signs * d).mean(axis=1) >= d.mean() - 1e-12).mean())


def interval(draws: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    alpha = (1 - level) / 2
    lo, hi = np.quantile(draws, [alpha, 1 - alpha])
    return float(lo), float(hi)
