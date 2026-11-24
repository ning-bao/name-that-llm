"""Turn community datasets published after Open Pool v13 into extra pool rows (data/external/extra_sources.parquet).

Sources, vetted on 3 Nov 2026 (EXPERIMENTS.md, "More gemini-3.7 data"), each in its own download folder:
  zeminglii/ntl-own-exact-route-generations v2 (CC BY 4.0), passages.csv: gemini-3.7 through OpenRouter pinned to
      Vertex, other labels through the Kaggle Benchmarks proxy or first-party OpenRouter providers. prompt_canon
      is the pool's, so these rows join the pool's prompt groups.
  kilupy/ntl-18-model-corpus v1 (CC BY 4.0), ntl_18model_passages.parquet: 3,338 prompts answered by all five
      Claude models (Anthropic API) and 300 prompts answered by the other 13 labels (OpenRouter).
Rows keep the pool's cleaning rules: a competition label; one paragraph without Markdown; 36-275 words; not a
duplicate of a pool passage or of another row; word TF-IDF cosine below 0.3 to every official train passage and
below 0.4 to every official prompt's passage centroid. typography-source-filter-v1 drops Gemini rows with an em dash.
The pool's own "typography a label never uses in train" rule judged all 150 official prompts; it is not repeated.

Usage: uv run python -I scripts/prepare_extra_sources.py
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from nllm import collection, pool
from nllm.data import LABELS, load_train

ROOT = collection.EXTERNAL_DIR / "third_party"
OUT = collection.EXTERNAL_DIR / "extra_sources.parquet"
MARKDOWN = r"\*\*|^\s*#|^\s*[-*] "


def zeminglii() -> pd.DataFrame:
    d = pd.read_csv(ROOT / "zeminglii__ntl-own-exact-route-generations" / "passages.csv", dtype=str,
                    keep_default_na=False, encoding="utf-8")
    return pd.DataFrame({
        "id": "zeminglii:" + d.index.astype(str), "text": d.text, "model": d.model,
        "group": d.prompt_canon.where(d.prompt_canon != "", "zeminglii:" + d.index.astype(str)),
        "source": "zeminglii/ntl-own-exact-route-generations", "kb_proxy": d.route == "kaggle_proxy",
    })


def kilupy() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "kilupy__ntl-18-model-corpus" / "ntl_18model_passages.parquet")
    return pd.DataFrame({
        "id": "kilupy:" + d.prompt_id.astype(str) + ":" + d.model, "text": d.text, "model": d.model,
        "group": "kilupy:" + d.prompt_id.astype(str), "source": "kilupy/ntl-18-model-corpus", "kb_proxy": False,
    })


def main() -> None:
    rows = pd.concat([zeminglii(), kilupy()], ignore_index=True)
    rows["text"] = rows.text.astype(str)
    steps = [("downloaded", len(rows))]
    rows = rows[rows.model.isin(LABELS)]
    steps.append(("competition label", len(rows)))
    words = rows.text.str.split().str.len()
    rows = rows[~rows.text.str.strip().str.contains("\n") & ~rows.text.str.contains(MARKDOWN, regex=True)
                & words.between(36, 275)]
    steps.append(("one paragraph, no Markdown, 36-275 words", len(rows)))
    existing = set(pool.load_pool().text.str.strip())
    rows = rows[~rows.text.str.strip().isin(existing)].drop_duplicates("text")
    steps.append(("not in the pool, no duplicates", len(rows)))

    train = load_train()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english").fit(train.text)
    passages = vec.transform(train.text)
    centroids = normalize(np.asarray([passages[(train.prompt_id == g).to_numpy()].mean(axis=0)
                                      for g in train.prompt_id.unique()]).squeeze())
    X = vec.transform(rows.text)
    near = ((X @ passages.T).max(axis=1).toarray().ravel() >= 0.3) | ((X @ centroids.T).max(axis=1) >= 0.4)
    rows = rows[~near]
    steps.append(("TF-IDF cosine < 0.3 to official passages, < 0.4 to prompt centroids", len(rows)))
    rows = rows[~(rows.model.str.startswith("gemini") & rows.text.str.contains("—"))]
    steps.append((pool.GEMINI_EM_DASH_FILTER, len(rows)))

    panel = pool.load_pool().groupby("group").panel_labels.max()
    own_panel = rows.groupby("group").model.nunique()
    rows["panel_labels"] = rows.group.map(panel).fillna(rows.group.map(own_panel)).astype(int)
    rows["origin"] = "extra"
    rows[pool.COLUMNS].to_parquet(OUT, index=False)
    for name, n in steps:
        print(f"{n:7}  {name}")
    print(rows.groupby(["source", "model"]).size().unstack(0).fillna(0).astype(int).to_string())
    claude = rows[rows.model.str.startswith("claude")]
    print(f"Claude rows with a curly apostrophe (official: none): {claude.text.str.contains(chr(0x2019)).mean():.1%}")
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
