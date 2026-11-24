import numpy as np
import pandas as pd
import pytest

from nllm import closeness
from nllm.data import load_train


def test_family_and_prompt_folds():
    assert [closeness.family(m) for m in ("claude-opus-5", "gpt-6-astra", "glm-5.3-flash", "qwen3.8-max")] == [
        "claude", "gpt", "glm", "qwen3.8"]
    ids = [f"p{i:03d}" for i in range(150)] * 2
    folds = closeness.prompt_folds(ids, 5, seed=0)
    assert sum(len(f) for f in folds) == 150 and len(frozenset().union(*folds)) == 150
    assert folds == closeness.prompt_folds(ids, 5, seed=0)


def test_style_auc_and_words_tell_shifted_text_apart():
    rng = np.random.default_rng(0)
    words = ["light", "river", "stone", "theory", "history", "measure", "signal", "ocean"]
    plain = [" ".join(rng.choice(words, 40)) + "." for _ in range(80)]
    dashed = [" ".join(rng.choice(words, 70)) + " — indeed; quite!" for _ in range(80)]
    assert closeness.style_auc(plain, plain[::-1]) < 0.7
    assert closeness.style_auc(plain, dashed) > 0.9
    w = closeness.word_stats(plain, dashed)
    assert w["words_med_official"] == 40 and w["words_med"] > 70 and w["words_ks"] == 1.0


def test_official_rows_transfer_like_official():
    try:
        official = load_train().reset_index(drop=True)
    except FileNotFoundError:
        pytest.skip("competition data not available locally")
    pair = official[official.model.isin(["gemini-3.1-pro-preview", "gemini-3.8-flash"])]  # an easy pair
    folds = closeness.prompt_folds(official.prompt_id, 5, seed=0)
    held = pair[pair.prompt_id.isin(folds[0])]  # one fold's official rows as the "cell", scored on all official rows
    row = closeness.family_transfer(held, official[~official.prompt_id.isin(folds[0])], folds[1:], n_cap=30)[0]
    assert row["family"] == "gemini" and row["n_per_label"] == 30
    assert row["official_ba"] > 0.7 and 0.5 < row["transfer_lift"] < 1.5
    assert pd.notna(row["transfer_ba"])
