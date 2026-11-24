"""Nested score blends across experiment runs (report experiments 3 and 11).

A component is one branch of one run directory: its outer out-of-fold scores
(oof.parquet) and the inner out-of-fold scores its outer folds were selected on
(inner_oof.parquet). For each outer fold, every component is divided by a scale computed
on its inner scores, the weights of components after the first are chosen on inner
balanced accuracy only, and the blend is applied to the outer scores.
"""

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from nllm import linear
from nllm.data import LABELS
from nllm.metrics import balanced_accuracy

KEYS = ["partition", "outer_fold", "id"]
SCORE_COLUMNS = [f"score:{m}" for m in LABELS]


def load(run: Path, branch: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(inner, outer) rows of one branch, indexed by partition, outer fold and id."""
    outer = pd.read_parquet(run / "oof.parquet")
    outer = outer[outer.branch == branch]
    inner = pd.read_parquet(run / "inner_oof.parquet")
    if "lr" in inner:  # encoder run: inner logits for every (lr, epoch); keep each fold's selection
        inner = inner.merge(pd.read_csv(run / "selections.csv"), on=["partition", "outer_fold", "lr", "epoch"])
    else:
        inner = inner[inner.branch == branch]
    if outer.empty or inner.empty:
        raise ValueError(f"no rows for branch {branch!r} in {run}")
    return inner.set_index(KEYS).sort_index(), outer.set_index(KEYS).sort_index()


def fold(frame: pd.DataFrame, partition: int, outer_fold: int) -> pd.DataFrame:
    return frame.xs((partition, outer_fold), level=["partition", "outer_fold"])


def choose_weights(inner: list[np.ndarray], y: np.ndarray) -> tuple[tuple[float, ...], list[float]]:
    """Grid-search weights for components 2..n (component 1 fixed at 1); ties go to less total weight."""
    scales = [linear.scale(s) for s in inner]
    z = [s / c for s, c in zip(inner, scales, strict=True)]

    def score(w):
        blended = z[0] + sum(wi * zi for wi, zi in zip(w, z[1:], strict=True))
        return balanced_accuracy(y, linear.predict(blended)), -sum(w)

    return max(product(linear.BLEND_WEIGHTS, repeat=len(z) - 1), key=score), scales


def nested_blend(components: list[tuple[pd.DataFrame, pd.DataFrame]], name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend outer scores fold by fold with inner-chosen weights; returns (oof rows, weights table).

    Only outer folds every component has are blended (a baseline may cover both partitions, an encoder one).
    """
    first_inner, first_outer = components[0]
    shared = set.intersection(*(set(c[1].index.droplevel("id").unique()) for c in components))
    parts, weights = [], []
    for (p, o), outer_rows in first_outer.groupby(level=["partition", "outer_fold"]):
        if (p, o) not in shared:
            continue
        inner_rows = fold(first_inner, p, o)
        ids = outer_rows.index.get_level_values("id")
        inner = [fold(c[0], p, o).loc[inner_rows.index, SCORE_COLUMNS].to_numpy() for c in components]
        outer = [fold(c[1], p, o).loc[ids, SCORE_COLUMNS].to_numpy() for c in components]
        w, scales = choose_weights(inner, inner_rows.model.to_numpy())
        blended = outer[0] / scales[0] + sum(wi * s / c for wi, s, c in zip(w, outer[1:], scales[1:], strict=True))
        rows = outer_rows.reset_index()[KEYS + ["prompt_id", "model"]]
        rows = rows.assign(branch=name, pred=linear.predict(blended))
        parts.append(pd.concat([rows, pd.DataFrame(blended, columns=SCORE_COLUMNS)], axis=1))
        weights.append({"partition": p, "outer_fold": o, **{f"w{i + 2}": wi for i, wi in enumerate(w)}})
    return pd.concat(parts, ignore_index=True), pd.DataFrame(weights)
