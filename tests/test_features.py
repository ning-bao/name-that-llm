import re

import pytest

from nllm.features import WORD_TOKEN_PATTERN, CharNgrams, style_features


def test_char_ngrams_keep_raw_whitespace():
    assert CharNgrams(2, 2)("a  b") == ["a ", "  ", " b"]
    assert CharNgrams(1, 2)("ab") == ["a", "b", "ab"]


def test_word_tokens_keep_typography():
    assert re.findall(WORD_TOKEN_PATTERN, "Don’t—stop") == ["Don", "’", "t", "—", "stop"]


def test_style_features_detect_punctuation():
    f = style_features(["It’s late — very late; go.", "Plain words only here."])
    assert f.shape[0] == 2
    assert f.loc[0, "has_em_dash"] == 1 and f.loc[1, "has_em_dash"] == 0
    assert f.loc[0, "has_semicolon"] == 1
    assert f.loc[0, "spaced_em_dash_per_1k"] > 0
    assert f.loc[0, "curly_apostrophe_per_1k"] > 0 and f.loc[0, "straight_apostrophe_per_1k"] == 0
    assert f.notna().all().all()


@pytest.mark.parametrize("text", ["", "word", "...", "Ünïcödé — ²³"])
def test_style_features_handle_edge_cases(text):
    assert style_features([text]).notna().all().all()
