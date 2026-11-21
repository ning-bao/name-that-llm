"""Stateless feature extraction from the raw passage string.

Nothing here is fitted, so it is safe to apply to any rows. Learned statistics
(vocabulary, IDF, scaling) belong in the fold-fitted pipelines in nllm.linear.
"""

import math
import re

import numpy as np
import pandas as pd


class CharNgrams:
    """Raw character n-grams. Unlike sklearn's analyzer="char", whitespace is not collapsed."""

    def __init__(self, lo: int, hi: int):
        self.lo, self.hi = lo, hi

    def __call__(self, text: str) -> list[str]:
        return [text[i : i + n] for n in range(self.lo, self.hi + 1) for i in range(len(text) - n + 1)]

    def __repr__(self) -> str:
        return f"CharNgrams({self.lo}, {self.hi})"


# Case-preserving words plus every punctuation mark as its own token, so the word
# branch still sees em dashes and curly versus straight apostrophes.
WORD_TOKEN_PATTERN = r"(?u)\w+|[^\w\s]"

_LEXICAL = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?")
_SENTENCE_END = re.compile(r"[.!?]+[\"”’)]*(?:\s+|$)")

_PER_1K_CHARS = {
    "em_dash": "—",
    "spaced_em_dash": " — ",
    "en_dash": "–",
    "hyphen": "-",
    "curly_apostrophe": "’",
    "straight_apostrophe": "'",
    "curly_double_quote": "“",
    "straight_double_quote": '"',
    "comma": ",",
    "semicolon": ";",
    "colon": ":",
    "question": "?",
    "exclamation": "!",
    "parenthesis": "(",
    "asterisk": "*",
    "slash": "/",
}
_HAS = ("em_dash", "curly_apostrophe", "semicolon", "colon", "curly_double_quote", "parenthesis")

_PER_100_WORDS = {
    "first_person": r"\b(?:I|me|my|mine|we|us|our|ours)\b",
    "second_person": r"\b(?:you|your|yours)\b",
    "negation": r"\b(?:not|never|no|nor)\b|n['’]t\b",
    "contraction": r"\w['’](?:s|t|re|ve|ll|d|m)\b",
    "the": r"\b[Tt]he\b",
    "of": r"\bof\b",
    "and": r"\band\b",
    "to": r"\bto\b",
    "a_an": r"\b(?:a|an)\b",
    "in": r"\bin\b",
    "is_are": r"\b(?:is|are)\b",
    "that": r"\bthat\b",
    "it": r"\b[Ii]t\b",
    "as": r"\bas\b",
    "but_yet": r"\b(?:but|yet)\b",
    "this_these": r"\b(?:[Tt]his|[Tt]hese)\b",
}
_PER_100_WORDS = {k: re.compile(v) for k, v in _PER_100_WORDS.items()}


def _row(text: str) -> dict[str, float]:
    n_chars = max(1, len(text))
    words = text.split()
    n_words = max(1, len(words))
    lex = _LEXICAL.findall(text)
    lex_lens = np.array([len(w) for w in lex]) if lex else np.zeros(1)
    types = len({w.lower() for w in lex})
    sentences = [s for s in _SENTENCE_END.split(text) if s.strip()]
    sent_words = np.array([len(s.split()) for s in sentences]) if sentences else np.array([n_words])

    r = {
        "log_words": math.log1p(len(words)),
        "chars_per_word": len(text) / n_words,
        "lex_word_chars_mean": float(lex_lens.mean()),
        "lex_word_chars_std": float(lex_lens.std()),
        "long_word_share": float((lex_lens >= 9).mean()),
        "herdan_c": math.log(max(types, 1)) / math.log(max(len(lex), 2)),
        "sentences": float(len(sentences)),
        "sentence_words_mean": float(sent_words.mean()),
        "sentence_words_std": float(sent_words.std()),
        "ends_with_period": float(text.rstrip().endswith(".")),
        "digit_per_1k": sum(c.isdigit() for c in text) * 1000 / n_chars,
        "uppercase_per_1k": sum(c.isupper() for c in text) * 1000 / n_chars,
        "nonascii_other_per_1k": sum(ord(c) > 127 and c not in "—–’‘“”" for c in text) * 1000 / n_chars,
    }
    for name, mark in _PER_1K_CHARS.items():
        r[f"{name}_per_1k"] = text.count(mark) * 1000 / n_chars
    for name in _HAS:
        r[f"has_{name}"] = float(_PER_1K_CHARS[name] in text)
    for name, pattern in _PER_100_WORDS.items():
        r[f"{name}_per_100w"] = len(pattern.findall(text)) * 100 / n_words
    return r


def style_features(texts) -> pd.DataFrame:
    """About 50 numeric style features per passage: length, lexis, punctuation, function words."""
    return pd.DataFrame([_row(t) for t in texts])
