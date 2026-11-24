import importlib.util

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from nllm.data import LABELS
from nllm.paths import ROOT


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_correction_uses_an_informative_feature_and_ignores_noise():
    """Base scores that cannot tell labels 0 and 1 apart, plus one feature that can: the fitted correction
    separates them; with a pure-noise feature the predictions stay the base's."""
    script = load_script("candidate_eval")
    rng = np.random.default_rng(0)
    n, k = 600, len(LABELS)
    y = rng.integers(0, 2, size=n)
    base = np.full((n, k), -8.0)
    base[:, :2] = np.log(0.5)  # a coin flip between the first two labels
    groups = np.arange(n) // 6
    signal = (2.0 * y - 1 + rng.normal(scale=0.5, size=n))[:, None]
    noise = rng.normal(size=(n, 1))
    tr, te = np.arange(n) < 400, np.arange(n) >= 400
    truth = np.asarray(LABELS)[y[te]]
    pred, _ = script.fit_predict(base[tr], signal[tr], y[tr], groups[tr], base[te], signal[te])
    assert (pred == truth).mean() > 0.9
    pred, _ = script.fit_predict(base[tr], noise[tr], y[tr], groups[tr], base[te], noise[te])
    assert 0.35 < (pred == truth).mean() < 0.65 and set(pred) <= set(LABELS[:2])
    a, bias, w = script.fit(base[tr], np.zeros((int(tr.sum()), 0)), y[tr], 1.0)
    assert w.shape == (0, k) and np.isfinite(bias).all() and a > 0


def test_pair_features_contrast_two_models_token_by_token(tmp_path):
    script = load_script("candidate_eval")
    ids, lengths = np.array(["r0", "r1"]), np.array([3, 20])
    rng = np.random.default_rng(1)
    common = {"ids": ids, "lengths": lengths, "feature_names": np.array(["lp_mean", "n_tokens"]),
              "features": np.zeros((2, 2), dtype=np.float32)}
    lp_a, lp_b = rng.normal(size=23).astype(np.float32), rng.normal(size=23).astype(np.float32)
    rank = rng.integers(0, 5, size=(2, 23)).astype(np.float32)
    for name, lp, r in (("a", lp_a, rank[0]), ("b", lp_b, rank[1])):
        np.savez(tmp_path / f"{name}.npz", model=np.array(name), tok_lp=lp, tok_rank=r,
                 tok_entropy=np.ones(23, dtype=np.float32), **common)
    f = script.pair_features([tmp_path / "a.npz", tmp_path / "b.npz"], "a", "b")
    assert list(f.index) == ["r0", "r1"] and np.isclose(f.loc["r0", "a-b:lp_mean"], (lp_a - lp_b)[:3].mean())
    assert np.isclose(f.loc["r1", "a-b:lp_tail_mean"], (lp_a - lp_b)[3 + 8:].mean())  # long passages skip 8 tokens
    assert np.isclose(f.loc["r1", "a-b:lp_positive"], ((lp_a - lp_b)[3:] > 0).mean()) and (f["a-b:entropy"] == 0).all()
    assert np.isclose(f.loc["r0", "a-b:top1_only_first"], ((rank[0] == 0) & (rank[1] != 0))[:3].mean())
    np.savez(tmp_path / "c.npz", model=np.array("c"), tok_lp=lp_a[:22], tok_rank=rank[0][:22],
             tok_entropy=np.ones(22, dtype=np.float32), **{**common, "lengths": np.array([3, 19])})
    with pytest.raises(SystemExit, match="same token sequences"):
        script.pair_features([tmp_path / "a.npz", tmp_path / "c.npz"], "a", "c")
