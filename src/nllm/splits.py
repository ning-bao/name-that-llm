"""Frozen prompt-group splits from the research kit's validation_plan.json.

- 30 confirmation prompts are locked: score the fully frozen recipe on them once.
- 120 development prompts carry nested CV: 2 partitions x 5 outer folds (96 train /
  24 validation prompts), each outer-train set split into 3 inner folds of 32 for
  tuning, early stopping, calibration and blend weights.

All rows of a prompt stay on one side of every split. prompt_id is for grouping only
and must never reach a model as a feature.
"""

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import pandas as pd

from nllm.paths import KIT_DIR

PLAN_PATH = KIT_DIR / "validation_plan.json"


@dataclass(frozen=True)
class Fold:
    partition: int
    outer: int
    train: frozenset[str]
    validation: frozenset[str]
    inner_validation: tuple[frozenset[str], ...]

    def inner(self) -> Iterator[tuple[frozenset[str], frozenset[str]]]:
        """Yield (inner_train, inner_validation) prompt sets within this fold's training prompts."""
        for val in self.inner_validation:
            yield self.train - val, val


@cache
def load_plan(path: Path = PLAN_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def development_prompts() -> frozenset[str]:
    return frozenset(load_plan()["development_prompts"])


def confirmation_prompts(*, unlock: bool = False) -> frozenset[str]:
    """The locked 30-prompt set. Pass unlock=True only for the single final confirmation run."""
    if not unlock:
        raise PermissionError(
            "confirmation prompts are locked; development work must use development_prompts() "
            "and outer_folds(). Pass unlock=True only to score the frozen final recipe once."
        )
    return frozenset(load_plan()["confirmation_prompts"])


def initialization_seeds() -> list[int]:
    return list(load_plan()["initialization_seeds"])


def partitions() -> list[int]:
    return [r["partition_id"] for r in load_plan()["repetitions"]]


def outer_folds(partition: int = 0) -> list[Fold]:
    rep = next(r for r in load_plan()["repetitions"] if r["partition_id"] == partition)
    return [
        Fold(
            partition=partition,
            outer=f["outer_fold"],
            train=frozenset(f["train"]),
            validation=frozenset(f["validation"]),
            inner_validation=tuple(frozenset(v) for v in f["inner_validation"]),
        )
        for f in rep["folds"]
    ]


def select(df: pd.DataFrame, prompts: Iterable[str]) -> pd.DataFrame:
    """Rows of df whose prompt_id is in prompts."""
    return df[df.prompt_id.isin(frozenset(prompts))]


def development_rows(df: pd.DataFrame) -> pd.DataFrame:
    return select(df, development_prompts())


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(f"split plan check failed: {message}")


def check_plan(train: pd.DataFrame) -> None:
    """Raise if the plan disagrees with train.csv or any fold mixes a prompt across sides."""
    plan = load_plan()
    dev = development_prompts()
    confirm = frozenset(plan["confirmation_prompts"])
    _require(len(dev) == 120 and len(confirm) == 30, "expected 120 development / 30 confirmation")
    _require(not dev & confirm, "development and confirmation prompts overlap")
    _require(dev | confirm == set(train.prompt_id), "plan prompts differ from train.csv prompts")
    for partition in partitions():
        folds = outer_folds(partition)
        seen: set[str] = set()
        for f in folds:
            where = f"partition {partition} outer {f.outer}"
            _require(not f.validation & seen, f"{where}: validation prompt reused across folds")
            seen |= f.validation
            _require(f.train == dev - f.validation, f"{where}: train is not dev minus validation")
            inner_seen: set[str] = set()
            for val in f.inner_validation:
                _require(not val & inner_seen, f"{where}: inner validation sets overlap")
                inner_seen |= val
            _require(inner_seen == f.train, f"{where}: inner folds do not cover outer train")
        _require(seen == dev, f"partition {partition}: outer validation sets do not cover dev")
