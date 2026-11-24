import numpy as np
import pandas as pd
import pytest

from nllm import pool
from nllm.data import LABELS


def _rows(n_em: int, n_plain: int, model: str = LABELS[0]) -> pd.DataFrame:
    texts = [f"Passage {i} \u2014 with a dash and enough words to count." for i in range(n_em)]
    texts += [f"Passage {i} without one, and enough words to count." for i in range(n_plain)]
    return pd.DataFrame({"text": texts, "model": model})


def test_match_weights_reach_a_feasible_target():
    rows, official = _rows(30, 70), _rows(60, 40)
    w, report = pool.match_weights(rows, official, ridge=1e-4)
    has_em = rows.text.str.contains("\u2014").to_numpy()
    assert abs(w[has_em].sum() / w.sum() - 0.6) < 0.02
    assert abs(w.mean() - 1) < 1e-9
    assert report.loc[0, "mean_gap_after"] < report.loc[0, "mean_gap_before"]


def test_match_weights_leave_labels_without_official_rows_alone():
    rows = pd.concat([_rows(5, 5, LABELS[0]), _rows(5, 5, LABELS[1])], ignore_index=True)
    w, _ = pool.match_weights(rows, _rows(8, 2, LABELS[0]))
    assert (w[rows.model.to_numpy() == LABELS[1]] == 1).all()


def test_balanced_sample_caps_each_label_and_skips_zero_weight():
    rows = pd.concat([_rows(0, 50, LABELS[0]), _rows(0, 5, LABELS[1])], ignore_index=True)
    w = np.ones(len(rows))
    w[:10] = 0
    out = pool.balanced_sample(rows, 20, seed=0, weights=w)
    assert out.model.value_counts().to_dict() == {LABELS[0]: 20, LABELS[1]: 5}
    assert not set(out.index) & set(range(10))
    assert out.equals(pool.balanced_sample(rows, 20, seed=0, weights=w))


def test_own_rows_apply_source_filters_and_join_pool_prompts(tmp_path):
    admitted = pd.DataFrame({
        "id": ["a", "b", "c", "d", "e"],
        "prompt_id": ["xX001", "xX001", "xX002", "xp1234", "xX003"],
        "model": ["gemini-3.8-flash", "gemini-3.8-flash", "qwen3.8-max", "gpt-5.6-sol", "claude-opus-5"],
        "text": ["Plain answer.", "With a dash \u2014 here.", "Qwen answer.", "Codex answer.", "Kit answer."],
        "split": ["train", "train", "train", "train", "diagnostic"],
        "wave": ["wave2", "wave2", "wave2", "pool1", "diagnostic"],
        "collection_method": "api",
        "interface": ["api:agent_platform_generate_content", "api:agent_platform_generate_content",
                      "qoder_cli_cn:print", "codex_cli:exec", "api:anthropic_message_batches"],
        "eligibility_status": "pending_clarification",
    })
    path = tmp_path / "admitted.parquet"
    admitted.to_parquet(path)
    out = pool.own_rows(path, pool_panel=pd.Series({"p1234": 15}))
    assert out.id.tolist() == ["own:a", "own:d"]
    assert out.group.tolist() == ["xX001", "p1234"]
    assert out.panel_labels.tolist() == [18, 15]
    kept = pool.own_rows(path, exclude=(), gemini_em_dash_filter=False)
    assert set(kept.id) == {"own:a", "own:b", "own:c", "own:d"}


def test_training_sample_selects_and_boosts_own_sources():
    texts = [f"Passage number {i} with enough plain words in it." for i in range(400)]
    rows = pd.DataFrame({
        "id": [f"r{i}" for i in range(400)], "text": texts, "model": LABELS[0], "group": "g",
        "source": ["pool src"] * 300 + ["own codex_cli:exec"] * 50 + ["own api:other"] * 50,
        "kb_proxy": False, "panel_labels": 18, "origin": ["pool"] * 300 + ["own"] * 100,
    }).set_index("id", drop=False)
    official = rows.iloc[:50][["text", "model"]]
    pool_only, _ = pool.training_sample("w", official, 100, seed=0, rows=rows)
    assert (pool_only.origin == "pool").all()
    plain, _ = pool.training_sample("w_own", official, 100, seed=0, rows=rows)
    boosted, _ = pool.training_sample("w_codex5", official, 100, seed=0, rows=rows)
    assert "own api:other" not in set(boosted.source)
    assert (boosted.source == "own codex_cli:exec").sum() > 2 * (plain.source == "own codex_cli:exec").sum()


def test_extra_rows_join_only_extra_recipes():
    texts = [f"Passage number {i} with enough plain words in it." for i in range(200)]
    rows = pd.DataFrame({
        "id": [f"r{i}" for i in range(200)], "text": texts, "model": LABELS[0], "group": "g",
        "source": ["pool src"] * 150 + ["kilupy/ntl-18-model-corpus"] * 50, "kb_proxy": False, "panel_labels": 18,
        "origin": ["pool"] * 150 + ["extra"] * 50,
    }).set_index("id", drop=False)
    official = rows.iloc[:40][["text", "model"]]
    without, _ = pool.training_sample("w", official, 200, seed=0, rows=rows)
    with_extra, _ = pool.training_sample("wx", official, 200, seed=0, rows=rows)
    assert (without.origin == "pool").all() and len(without) == 150
    assert (with_extra.origin == "extra").sum() == 50


def test_balanced_sample_with_replacement_tops_short_labels_up():
    rows = pd.concat([_rows(0, 50, LABELS[0]), _rows(0, 5, LABELS[1])], ignore_index=True)
    w = np.ones(len(rows))
    w[50] = 0  # a zero-weight row is never drawn, not even in the top-up
    out = pool.balanced_sample(rows, 20, seed=0, weights=w, replace=True)
    assert out.model.value_counts().to_dict() == {LABELS[0]: 20, LABELS[1]: 20}
    short = out[out.model == LABELS[1]]
    assert set(short.index) == set(range(51, 55))  # every usable row at least once, nothing else
    assert out.equals(pool.balanced_sample(rows, 20, seed=0, weights=w, replace=True))


def test_uniform_recipe_needs_no_official_rows():
    rows = pd.concat([_rows(0, 30, LABELS[0]), _rows(0, 30, LABELS[1])], ignore_index=True)
    rows = rows.assign(origin="pool", source="s", kb_proxy=False)
    sample, report = pool.training_sample("u", None, 10, seed=0, rows=rows)
    assert report is None and sample.model.value_counts().tolist() == [10, 10]
    with pytest.raises(ValueError):
        pool.training_sample("w", None, 10, seed=0, rows=rows)
