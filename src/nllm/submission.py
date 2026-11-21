"""Write and check submission.parquet: columns id and model only, one row per test id."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from nllm.data import LABELS

# Matches the supplied sample_submission.parquet.
SCHEMA = pa.schema([("id", pa.large_string()), ("model", pa.large_string())])


def validate_submission(sub: pd.DataFrame, test_ids: Iterable[str]) -> None:
    test_ids = set(test_ids)
    if list(sub.columns) != ["id", "model"]:
        raise ValueError(f"invalid submission: columns must be ['id', 'model'], got {list(sub.columns)}")
    problems = []
    if sub.isna().any().any():
        problems.append("null values present")
    if sub.id.duplicated().any():
        problems.append(f"{int(sub.id.duplicated().sum())} duplicate ids")
    if missing := test_ids - set(sub.id):
        problems.append(f"{len(missing)} test ids missing")
    if extra := set(sub.id) - test_ids:
        problems.append(f"{len(extra)} ids not in test")
    if bad := set(sub.model.dropna()) - set(LABELS):
        problems.append(f"labels outside the 18 permitted: {sorted(bad)}")
    if problems:
        raise ValueError("invalid submission:\n- " + "\n- ".join(problems))


def write_submission(ids: Iterable[str], preds: Iterable[str], path: Path) -> Path:
    """Write in test order, then re-read the file and validate what actually landed on disk."""
    ids = list(ids)
    sub = pd.DataFrame({"id": ids, "model": list(preds)})
    validate_submission(sub, ids)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(sub, schema=SCHEMA, preserve_index=False), path)
    validate_submission(pd.read_parquet(path), ids)
    return path
