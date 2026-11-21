"""Loading the supplied files without altering a single character of the passages.

Text is read as raw UTF-8 strings: no NA parsing, no stripping, no Unicode or
punctuation normalisation. Typography (em dashes, curly apostrophes) is one of
the strongest source signals, so any transform belongs in an explicit,
versioned feature step, never here.
"""

from pathlib import Path

import pandas as pd

from nllm.paths import data_dir

# Frozen label order. Model outputs, logits and saved label maps all use it.
LABELS: tuple[str, ...] = (
    "claude-fable-5",
    "claude-opus-4-6",
    "claude-opus-4-8",
    "claude-opus-5",
    "claude-sonnet-5",
    "gemini-3.1-pro-preview",
    "gemini-3.6-flash",
    "gemini-3.7-flash",
    "gemini-3.8-flash",
    "glm-5.3",
    "glm-5.3-flash",
    "gpt-5.5",
    "gpt-5.6-luna",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-6-astra",
    "kimi-k3",
    "qwen3.8-max",
)
LABEL_TO_INDEX = {m: i for i, m in enumerate(LABELS)}

FAMILIES = {
    "claude": "Anthropic",
    "gemini": "Google",
    "glm": "ZAI",
    "gpt": "OpenAI",
    "kimi": "Moonshot",
    "qwen": "Qwen",
}


def family(label: str) -> str:
    for prefix, name in FAMILIES.items():
        if label.startswith(prefix):
            return name
    raise ValueError(f"unknown label {label!r}")


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


def load_train(directory: Path | None = None) -> pd.DataFrame:
    """Columns id, prompt_id, text, model. prompt_id is for grouping only, never a feature."""
    df = _read_csv((directory or data_dir()) / "train.csv")
    unknown = set(df.model) - set(LABELS)
    if unknown:
        raise ValueError(f"train.csv has labels outside LABELS: {sorted(unknown)}")
    return df


def load_test(directory: Path | None = None) -> pd.DataFrame:
    """Columns id, text. The supplied file has 3 rows; the hidden rerun has ~4,000."""
    return _read_csv((directory or data_dir()) / "test.csv")


def load_sample_submission(directory: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet((directory or data_dir()) / "sample_submission.parquet")
