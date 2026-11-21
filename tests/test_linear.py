import numpy as np
import pandas as pd

from nllm import linear
from nllm.data import LABELS
from nllm.features import style_features


def test_config_keys_are_unique():
    keys = [c.key for c in linear.all_configs()]
    assert len(keys) == len(set(keys))
    assert {c.branch for c in linear.all_configs()} == {"char", "word", "char+word", "style"}


def test_fit_score_columns_follow_label_order():
    # Each label gets a unique marker word, so a correct column mapping predicts every row.
    texts = [f"passage {i} mentions marker{j}x and more marker{j}x text" for j in range(18) for i in range(3)]
    df = pd.DataFrame({"text": texts, "model": [m for m in LABELS for _ in range(3)]})
    style = style_features(df.text)
    configs = [c for c in linear.all_configs() if c.key in {"word12|svm|C=1|raw_tf", "char35|lr|C=10|raw_tf"}]
    scores = linear.fit_score(df, style, df, style, configs, seed=0)
    for s in scores.values():
        assert s.shape == (len(df), len(LABELS))
        assert (linear.predict(s) == df.model.to_numpy()).all()


def test_select_picks_best_per_branch():
    y = np.array(LABELS[:3])
    perfect = np.eye(len(LABELS))[:3]
    wrong = np.roll(perfect, 1, axis=1)
    configs = [linear.Config("char", "a", "lr", 1.0, False), linear.Config("char", "b", "lr", 1.0, False),
               linear.Config("style", "style", "lr", 0.1)]
    scores = {configs[0].key: wrong, configs[1].key: perfect, configs[2].key: wrong}
    chosen, table = linear.select(scores, y, configs)
    assert chosen["char"] == configs[1] and chosen["text"] == configs[1]
    assert len(table) == 3


def test_blend_weight_ties_go_to_zero():
    rng = np.random.default_rng(0)
    y = np.array([LABELS[i % 18] for i in range(36)])
    text = rng.normal(size=(36, 18))
    weight, _, _ = linear.choose_blend_weight(text, np.zeros_like(text), y)
    assert weight == 0.0


def test_blend_weight_uses_informative_style():
    y = np.array([LABELS[i % 18] for i in range(36)])
    style = np.eye(18)[np.arange(36) % 18]
    weight, _, _ = linear.choose_blend_weight(np.zeros((36, 18)), style, y)
    assert weight > 0


def test_mixing_weights_give_each_class_equal_total_and_external_its_share():
    labels = np.array(["a"] * 6 + ["a"] * 2 + ["b"] * 4)
    external = np.array([False] * 6 + [True] * 2 + [False] * 4)
    w = linear.mixing_weights(labels, external, share=0.25)
    assert np.isclose(w.mean(), 1.0)
    assert np.isclose(w[labels == "a"].sum(), w[labels == "b"].sum())
    assert np.isclose(w[external].sum() / w[labels == "a"].sum(), 0.25)


def test_fit_score_uses_sample_weight():
    # Two conflicting copies of every passage: weighting decides which label wins.
    texts = [f"passage {i} marker{i}x text" for i in range(18)] * 2
    labels = list(LABELS) + list(np.roll(LABELS, 1))
    df = pd.DataFrame({"text": texts, "model": labels})
    style = style_features(df.text)
    cfg = [c for c in linear.all_configs() if c.key == "word12|lr|C=100|raw_tf"]
    weights = np.r_[np.full(18, 10.0), np.full(18, 0.1)]
    scores = linear.fit_score(df, style, df.iloc[:18], style.iloc[:18], cfg, seed=0, sample_weight=weights)
    assert (linear.predict(scores[cfg[0].key]) == np.asarray(LABELS)).all()
