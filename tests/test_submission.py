import pandas as pd
import pytest

from nllm.data import LABELS
from nllm.submission import validate_submission, write_submission


def test_roundtrip_matches_sample_schema(tmp_path):
    ids = ["x1", "x2", "x3"]
    path = write_submission(ids, [LABELS[0], LABELS[5], LABELS[17]], tmp_path / "submission.parquet")
    back = pd.read_parquet(path)
    assert list(back.columns) == ["id", "model"]
    assert list(back.id) == ids


@pytest.mark.parametrize(
    "sub, message",
    [
        (pd.DataFrame({"id": ["x1", "x2"], "model": [LABELS[0], "gpt-4"]}), "labels outside"),
        (pd.DataFrame({"id": ["x1", "x1"], "model": [LABELS[0]] * 2}), "duplicate"),
        (pd.DataFrame({"id": ["x1"], "model": [LABELS[0]]}), "missing"),
        (pd.DataFrame({"id": ["x1", "x2"], "label": [LABELS[0]] * 2}), "columns"),
    ],
)
def test_invalid_submissions_raise(sub, message):
    with pytest.raises(ValueError, match=message):
        validate_submission(sub, ["x1", "x2"])
