import numpy as np
import pandas as pd

from nllm import blending
from nllm.data import LABELS


def write_run(path, scores_for, branch):
    """Two outer folds of 36 rows; each fold's inner rows are the other fold's rows."""
    rng = np.random.default_rng(0)
    ids = {o: [f"r{o}_{i}" for i in range(36)] for o in (0, 1)}
    labels = [LABELS[i % 18] for i in range(36)]
    outer, inner = [], []
    for o in (0, 1):
        for frame, row_ids in ((outer, ids[o]), (inner, ids[1 - o])):
            scores = scores_for(labels, rng)
            meta = pd.DataFrame({"id": row_ids, "prompt_id": [f"p{i // 18}" for i in range(36)], "model": labels,
                                 "partition": 0, "outer_fold": o, "branch": branch})
            frame.append(pd.concat([meta, pd.DataFrame(scores, columns=blending.SCORE_COLUMNS)], axis=1))
    path.mkdir()
    pd.concat(outer).assign(pred="").to_parquet(path / "oof.parquet")
    pd.concat(inner).to_parquet(path / "inner_oof.parquet")
    return blending.load(path, branch)


def noise(labels, rng):
    return rng.normal(size=(len(labels), len(LABELS)))


def oracle(labels, rng):
    return np.eye(len(LABELS))[[LABELS.index(m) for m in labels]] * 5


def test_informative_component_gets_weight(tmp_path):
    components = [write_run(tmp_path / "a", noise, "a"), write_run(tmp_path / "b", oracle, "b")]
    oof, weights = blending.nested_blend(components, "a + b")
    assert (weights.w2 > 0).all()
    assert (oof.pred == oof.model).all()


def test_uninformative_component_gets_zero_weight(tmp_path):
    components = [write_run(tmp_path / "a", oracle, "a"), write_run(tmp_path / "b", noise, "b")]
    oof, weights = blending.nested_blend(components, "a + b")
    assert (weights.w2 == 0).all()
    assert (oof.pred == oof.model).all()
