"""Filesystem locations. Data resolves to $NLLM_DATA_DIR, then Kaggle's input mount, then data/raw."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT_DIR = ROOT / "Name_That_LLM_Research_Kit"
OUTPUTS = ROOT / "outputs"
KAGGLE_INPUT = Path("/kaggle/input/competitions/name-that-llm")


def data_dir() -> Path:
    if env := os.environ.get("NLLM_DATA_DIR"):
        return Path(env)
    if KAGGLE_INPUT.exists():
        return KAGGLE_INPUT
    return ROOT / "data" / "raw"
