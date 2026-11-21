"""Tokenizer audit for encoder candidates (report D2): lengths and raw-text fidelity.

For each tokenizer, over the development passages: token-length percentiles, the share
over common caps, the [UNK] rate, and whether decode(encode(text)) reproduces the raw
string, including the typography the linear baseline relies on (em dashes, curly
quotes, repeated whitespace). Only tokenizer files are downloaded.

Usage: uv run python scripts/audit_tokenizers.py [repo ...]
Writes outputs/audit/tokenizers.csv.
"""

import argparse

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from nllm.data import load_train
from nllm.paths import OUTPUTS
from nllm.splits import development_rows

CANDIDATES = (
    "jhu-clsp/ettin-encoder-150m",
    "jhu-clsp/ettin-encoder-400m",
    "FacebookAI/roberta-large",
    "microsoft/deberta-v3-large",
    "Qwen/Qwen3-8B-Base",  # experiment 4; Qwen3-4B-Base ships the same tokenizer files
)
PROBES = {"em_dash": "—", "curly_apostrophe": "’", "curly_double_quote": "“", "double_space": "  "}


def audit(repo: str, texts: pd.Series) -> dict:
    tok = AutoTokenizer.from_pretrained(repo)
    ids = tok(list(texts), add_special_tokens=True)["input_ids"]
    lengths = np.array([len(i) for i in ids])
    decoded = [tok.decode(i, skip_special_tokens=True, clean_up_tokenization_spaces=False) for i in ids]
    row = {
        "tokenizer": repo,
        "class": type(tok).__name__,
        "vocab": len(tok),
        "median": float(np.median(lengths)),
        "p95": float(np.percentile(lengths, 95)),
        "max": int(lengths.max()),
        **{f"over_{cap}": float((lengths > cap).mean()) for cap in (256, 384, 512)},
        "unk_rate": float(np.mean([i.count(tok.unk_token_id) / len(i) for i in ids])) if tok.unk_token_id is not None else 0.0,
        "exact_roundtrip": float(np.mean([d == t for d, t in zip(decoded, texts)])),
    }
    for name, probe in PROBES.items():
        has = texts.str.contains(probe, regex=False).to_numpy()
        kept = np.array([d.count(probe) == t.count(probe) for d, t in zip(decoded, texts)])
        row[f"keeps_{name}"] = float(kept[has].mean()) if has.any() else float("nan")
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("repos", nargs="*", default=list(CANDIDATES))
    args = ap.parse_args()
    texts = development_rows(load_train()).text.reset_index(drop=True)
    table = pd.DataFrame([audit(r, texts) for r in args.repos])
    out = OUTPUTS / "audit"
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "tokenizers.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
