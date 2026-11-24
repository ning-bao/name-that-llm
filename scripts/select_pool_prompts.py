"""Pick Open Pool prompts for a new collection wave (appends to data/external/pool_prompts.tsv).

Of our collected sources, only the Codex GPT answers came out at least as close to official passages
as the pool's own rows (closeness check, 3 Nov 2026), so more of them are worth collecting. New waves
reuse the pool's recovered prompts instead of new hand-written ones: the answers then join the pool's
passages for the same prompt (grouped by prompt_canon). A prompt is eligible when it
  - belongs to a prompt group answered by at least --min-labels labels in the pool,
  - asks for one paragraph of about N words (30 <= N <= 300), like the official single-paragraph passages,
  - names no model, vendor or AI, asks nothing about identity, and says nothing about punctuation,
    typography or imitating a style,
  - is not close to competition text: word TF-IDF cosine below --max-cos to every official train
    passage and to every official prompt's passage centroid,
  - was not picked for an earlier pool wave.
The wave is sampled to the bank's official length mix (38/33/19/10% by requested length). The prompt
text is kept verbatim; collection.pool_bank() appends the no-tools line.

Usage: uv run python -I scripts/select_pool_prompts.py --pool <open pool dir> --wave pool1 --n 400
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from nllm import collection
from nllm.data import load_train

LENGTH = re.compile(r"\b(?:about|around|roughly|approximately)\s+(\d{2,3})\s+words\b", re.IGNORECASE)
PARAGRAPH = re.compile(r"\bparagraph\b", re.IGNORECASE)
BANNED = re.compile(
    r"\b(?:claude|anthropic|gpt|chatgpt|openai|gemini|google|bard|kimi|moonshot|qwen|alibaba|glm|zhipu|deepseek|llama"
    r"|mistral|llm|language model|ai model|chatbot|assistant|which model|who (?:are|made) you|your (?:name|creator)"
    r"|dash(?:es)?|em-dash|hyphens?|apostrophes?|quotation marks?|punctuation|semicolons?|colons?|typograph\w*"
    r"|in the style of|exaggerat\w*|imitat\w*)\b",
    re.IGNORECASE,
)
# Requested-length bins and the share of the wave in each (the bank's official length mix).
BINS = [(30, 80, 0.38), (81, 150, 0.33), (151, 215, 0.19), (216, 300, 0.10)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True, help="the downloaded Open Pool directory")
    ap.add_argument("--wave", required=True, help="wave name, e.g. pool1")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--min-labels", type=int, default=12)
    ap.add_argument("--max-cos", type=float, default=0.25)
    args = ap.parse_args()

    prompts = pd.read_parquet(args.pool / "prompts.parquet")
    earlier = (pd.read_csv(collection.POOL_PROMPTS_PATH, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
               if collection.POOL_PROMPTS_PATH.exists() else pd.DataFrame(columns=["prompt_canon", "wave"]))
    if args.wave in set(earlier.wave):
        raise SystemExit(f"wave {args.wave} already exists in {collection.POOL_PROMPTS_PATH}")

    p = prompts[prompts.prompt_canon.str.startswith("p") & ~prompts.prompt_canon.isin(earlier.prompt_canon)].copy()
    steps = [("recovered, shared prompt ids", len(p))]
    p = p[p.panel_labels >= args.min_labels]
    steps.append((f"answered by >= {args.min_labels} labels", len(p)))
    p["requested_words"] = p.prompt_text.str.extract(LENGTH, expand=False)
    p = p[p.requested_words.notna() & p.prompt_text.str.contains(PARAGRAPH)]
    p = p[p.requested_words.astype(int).between(30, 300) & (p.prompt_text.str.count(LENGTH) == 1)]
    steps.append(("one paragraph of about N words", len(p)))
    p = p[~p.prompt_text.str.contains(BANNED)]
    steps.append(("no model, identity, punctuation or style words", len(p)))

    train = load_train()
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words="english")
    vec.fit(pd.concat([train.text, p.prompt_text]))
    passages = vec.transform(train.text)
    centroids = normalize(np.asarray([passages[(train.prompt_id == g).to_numpy()].mean(axis=0)
                                      for g in train.prompt_id.unique()]).squeeze())
    candidates = vec.transform(p.prompt_text)
    p["max_cos_passage"] = (candidates @ passages.T).max(axis=1).toarray().ravel()
    p["max_cos_centroid"] = (candidates @ centroids.T).max(axis=1)
    p = p[(p.max_cos_passage < args.max_cos) & (p.max_cos_centroid < args.max_cos)]
    steps.append((f"TF-IDF cosine to competition text < {args.max_cos}", len(p)))

    rng = np.random.default_rng(sum(map(ord, args.wave)))
    words = p.requested_words.astype(int)
    picked = []
    for lo, hi, share in BINS:
        pool_bin = p[words.between(lo, hi)]
        k = min(len(pool_bin), round(args.n * share))
        picked.append(pool_bin.iloc[rng.choice(len(pool_bin), k, replace=False)])
    wave = pd.concat(picked).assign(wave=args.wave).sort_values("prompt_canon")
    for name, n in steps:
        print(f"{n:6}  {name}")
    print(f"{len(wave):6}  sampled for {args.wave} (requested words by bin: "
          + ", ".join(f"{lo}-{hi}: {words.loc[wave.index].between(lo, hi).sum()}" for lo, hi, _ in BINS) + ")")
    print(f"max cosine to competition text in the wave: passage {wave.max_cos_passage.max():.3f}, "
          f"centroid {wave.max_cos_centroid.max():.3f}")

    cols = ["prompt_canon", "wave", "requested_words", "panel_labels", "max_cos_passage", "max_cos_centroid",
            "prompt_text"]
    out = pd.concat([earlier, wave[cols].round({"max_cos_passage": 4, "max_cos_centroid": 4}).astype(str)])
    out[cols].to_csv(collection.POOL_PROMPTS_PATH, sep="\t", index=False, encoding="utf-8")
    print(f"wrote {collection.POOL_PROMPTS_PATH} ({len(out)} prompts)")


if __name__ == "__main__":
    main()
