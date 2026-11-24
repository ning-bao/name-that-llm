"""Within-family word choice, prompt-paired: does model A word things differently from its sibling B
when both answer the SAME prompt?

Official data: every prompt is answered by every label, so each sibling pair gives up to 150 matched pairs and
topic cancels exactly. For each pair:
  1. Words/phrases/openers: McNemar exact test on presence (A uses it and B does not, vs the reverse),
     Benjamini-Hochberg over tokens with >= 8 discordant prompts.
  2. Split-half replication: rank tokens by paired effect on a random half of the prompts, check the top 20
     on the other half; compared with a null where A/B are swapped at random within each prompt (200 reps).
  3. Style measures (length, sentence length, word length, vocabulary richness, adverbs, punctuation...):
     paired Wilcoxon test.
  4. "Twin test": given the two answers to one prompt, which one is A? A logistic regression on the difference
     of their features (selected inside each fold), 10-fold CV over prompts, 5 seeds. The 8B's pairwise AUC
     on the same rows is the reference.
Pool check: pairs answered by both labels within ONE pool source (same prompt, same source), same tests,
and whether each official-significant token / style measure points the same way.

    uv run --no-project --with pandas --with pyarrow --with scikit-learn --with scipy python -I sibling_words.py \
        TRAIN.csv POOL.parquet RUN_DIR OUT_DIR
"""

from __future__ import annotations

import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

TRAIN, POOL, RUN, OUT = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
off = pd.read_csv(TRAIN, dtype=str, keep_default_na=False, encoding="utf-8")
LABELS = sorted(off.model.unique())
FAM = {l: l.split("-")[0].replace("qwen3.8", "qwen") for l in LABELS}
PAIRS = [(a, b) for a, b in itertools.combinations(LABELS, 2) if FAM[a] == FAM[b]] + [
    ("claude-fable-5", "glm-5.3"), ("claude-fable-5", "glm-5.3-flash")]
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"(?<=[.!?])[\"”’')\]]*\s+")
rng = np.random.default_rng(17)


def tokens(text):
    t = text.replace("’", "'").replace("‘", "'")
    words = [w.lower() for w in WORD.findall(t)]
    out = {f"w:{w}" for w in words} | {f"b:{a} {b}" for a, b in zip(words, words[1:])}
    firsts = [f[0].lower() for f in (WORD.findall(s)[:1] for s in SENT.split(t) if s.strip()) if f]
    out |= {f"S:{w}" for w in firsts}
    if firsts:
        out |= {f"F:{firsts[0]}", f"L:{firsts[-1]}"}
    return list(out)


def style(text):
    t = text.replace("’", "'")
    words = WORD.findall(t)
    n = max(len(words), 1)
    sents = [s for s in SENT.split(text) if s.strip()]
    lw = [w.lower() for w in words]
    first60 = lw[:60]
    return {
        "words": len(words),
        "sentences": len(sents),
        "words per sentence": n / max(len(sents), 1),
        "mean word length": np.mean([len(w) for w in words]) if words else 0,
        "long words (>=9 letters) %": 100 * np.mean([len(w) >= 9 for w in words]) if words else 0,
        "type-token ratio (first 60 words)": len(set(first60)) / max(len(first60), 1),
        "-ly adverbs %": 100 * np.mean([w.endswith("ly") and len(w) > 4 for w in lw]) if words else 0,
        "-ing words %": 100 * np.mean([w.endswith("ing") and len(w) > 4 for w in lw]) if words else 0,
        "commas per 100 words": 100 * text.count(",") / n,
        "semicolons per 100 words": 100 * text.count(";") / n,
        "colons per 100 words": 100 * text.count(":") / n,
        "em dashes per 100 words": 100 * text.count("—") / n,
        "curly apostrophe (any)": float("’" in text),
        "contractions per 100 words": 100 * sum("'" in w for w in words) / n,
        "'the' per 100 words": 100 * lw.count("the") / n,
        "'and' per 100 words": 100 * lw.count("and") / n,
        "pronouns we/our/us %": 100 * np.mean([w in ("we", "our", "us") for w in lw]) if words else 0,
        "pronouns you/your %": 100 * np.mean([w in ("you", "your") for w in lw]) if words else 0,
        "questions": text.count("?"),
    }


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    q = np.empty_like(p)
    q[order] = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    return np.minimum(q, 1)


vec = CountVectorizer(analyzer=tokens, binary=True, min_df=10, dtype=np.int8)
X = vec.fit_transform(off.text).tocsr()
vocab = vec.get_feature_names_out()
ST = pd.DataFrame([style(t) for t in off.text])
key = {(m, p): i for i, (m, p) in enumerate(zip(off.model, off.prompt_id))}

oof = pd.read_parquet(RUN / "oof.parquet")
oof_scores = {(r.partition, r.id): r for r in oof.itertuples()}


def paired(a, b, X, ST, rows_a, rows_b):
    xa, xb = X[rows_a].toarray().astype(int), X[rows_b].toarray().astype(int)
    n10, n01 = ((xa == 1) & (xb == 0)).sum(0), ((xb == 1) & (xa == 0)).sum(0)
    return xa, xb, n10, n01


summary, word_rows, style_rows = [], [], []
for a, b in PAIRS:
    prompts = sorted(set(off.prompt_id[off.model == a]) & set(off.prompt_id[off.model == b]))
    ra, rb = [key[a, p] for p in prompts], [key[b, p] for p in prompts]
    xa, xb, n10, n01 = paired(a, b, X, ST, ra, rb)
    disc = n10 + n01
    test = np.where(disc >= 8)[0]
    pv = np.array([binomtest(int(n10[i]), int(disc[i]), 0.5).pvalue for i in test])
    q = bh(pv) if len(pv) else pv
    for i, p_, q_ in zip(test, pv, q):
        if q_ < 0.10:
            word_rows.append({"pair": f"{a} vs {b}", "token": vocab[i], "a_only": int(n10[i]), "b_only": int(n01[i]),
                              "a_rate": xa[:, i].mean(), "b_rate": xb[:, i].mean(), "p": p_, "q": q_})
    # split-half replication of the top 20 tokens, against a within-prompt label-swap null
    def split_rep(xa, xb, reps=200):
        hits = []
        for _ in range(reps):
            m = rng.random(len(prompts)) < 0.5
            d1 = (xa[m] - xb[m]).mean(0)
            d2 = (xa[~m] - xb[~m]).mean(0)
            top = np.argsort(-np.abs(d1))[:20]
            hits.append(np.mean(np.sign(d1[top]) == np.sign(d2[top])))
        return float(np.mean(hits))
    rep = split_rep(xa, xb)
    swap = rng.random(len(prompts)) < 0.5
    null = []
    for _ in range(20):
        s = rng.random(len(prompts)) < 0.5
        ya, yb = np.where(s[:, None], xb, xa), np.where(s[:, None], xa, xb)
        null.append(split_rep(ya, yb, reps=20))
    # style measures, paired Wilcoxon
    sa, sb = ST.iloc[ra].reset_index(drop=True), ST.iloc[rb].reset_index(drop=True)
    n_style_sig = 0
    for c in ST.columns:
        d = (sa[c] - sb[c]).to_numpy()
        p_ = wilcoxon(d[d != 0]).pvalue if (d != 0).sum() >= 10 else 1.0
        n_style_sig += p_ < 0.01
        style_rows.append({"pair": f"{a} vs {b}", "measure": c, "a_median": sa[c].median(), "b_median": sb[c].median(),
                           "a_mean": sa[c].mean(), "b_mean": sb[c].mean(), "p": p_})
    # twin test: which of the two answers is A? difference features, CV over prompts
    D = np.hstack([xa - xb, ((sa - sb) / (ST.std() + 1e-9)).to_numpy()])
    aucs = []
    for seed in range(5):
        folds = np.random.default_rng(seed).permutation(len(prompts)) % 10
        score = np.zeros(len(prompts))
        for k in range(10):
            tr, te = folds != k, folds == k
            # token selection inside the fold: most discordant tokens on training prompts
            dtr = np.abs(xa[tr] - xb[tr]).sum(0)
            sel = np.r_[np.argsort(-dtr)[:300], X.shape[1] + np.arange(ST.shape[1])]
            Xtr = np.vstack([D[tr][:, sel], -D[tr][:, sel]])
            ytr = np.r_[np.ones(tr.sum()), np.zeros(tr.sum())]
            clf = LogisticRegression(C=0.05, max_iter=2000).fit(Xtr, ytr)
            score[te] = clf.decision_function(D[te][:, sel])
        flip = np.random.default_rng(seed + 100).random(len(prompts)) < 0.5
        y = np.where(flip, 0, 1)
        aucs.append(roc_auc_score(y, np.where(flip, -score, score)))
    # 8B on the same twins (dev prompts only): which of the two has the higher A-vs-B score?
    sc_a = f"score:{a}"; sc_b = f"score:{b}"
    tw = []
    for p_ in range(2):
        part = oof[oof.partition == p_].set_index(["prompt_id", "model"])
        for pr in prompts:
            if (pr, a) in part.index and (pr, b) in part.index:
                ga, gb = part.loc[(pr, a)], part.loc[(pr, b)]
                tw.append((ga[sc_a] - ga[sc_b]) > (gb[sc_a] - gb[sc_b]))
    summary.append({"pair": f"{a} vs {b}", "prompts": len(prompts), "tokens_q<0.10": int((q < 0.10).sum()) if len(q) else 0,
                    "split_half_sign_agree": rep, "null_sign_agree": float(np.mean(null)),
                    "style_measures_p<0.01": int(n_style_sig), "twin_auc_words+style": float(np.mean(aucs)),
                    "twin_acc_8b": float(np.mean(tw)) if tw else np.nan})
    print(summary[-1], flush=True)

S = pd.DataFrame(summary)
W = pd.DataFrame(word_rows)
STY = pd.DataFrame(style_rows)
S.to_csv(OUT / "pair_summary.csv", index=False)
W.to_csv(OUT / "pair_words_q10.csv", index=False)
STY.to_csv(OUT / "pair_style.csv", index=False)

# ---------------------------------------------------------------- pool check: same prompt, same source
pool = pd.read_parquet(POOL, columns=["text", "model", "prompt_canon", "source_dataset"])
pool = pool[pool.prompt_canon.fillna("") != ""]
checks = []
for a, b in [("gemini-3.7-flash", "gemini-3.8-flash"), ("gemini-3.6-flash", "gemini-3.7-flash"),
             ("gemini-3.6-flash", "gemini-3.8-flash"), ("glm-5.3", "glm-5.3-flash"),
             ("gpt-5.6-luna", "gpt-5.6-sol"), ("gpt-5.6-sol", "gpt-5.6-terra"), ("gpt-5.6-luna", "gpt-5.6-terra"),
             ("claude-fable-5", "glm-5.3")]:
    sub = pool[pool.model.isin([a, b])].drop_duplicates(["model", "prompt_canon", "source_dataset"])
    piv = sub.pivot_table(index=["source_dataset", "prompt_canon"], columns="model", values="text", aggfunc="first").dropna()
    if len(piv) < 30:
        continue
    pa = vec.transform(piv[a]).toarray().astype(int)
    pb = vec.transform(piv[b]).toarray().astype(int)
    psa = pd.DataFrame([style(t) for t in piv[a]])
    psb = pd.DataFrame([style(t) for t in piv[b]])
    sig = W[W.pair == f"{a} vs {b}"]
    agree_tok = [np.sign((pa[:, list(vocab).index(t)] - pb[:, list(vocab).index(t)]).mean()) == np.sign(r.a_only - r.b_only)
                 for t, r in zip(sig.token, sig.itertuples())]
    st = STY[(STY.pair == f"{a} vs {b}") & (STY.p < 0.01)]
    agree_sty = [np.sign((psa[m] - psb[m]).mean()) == np.sign(r.a_mean - r.b_mean) for m, r in zip(st.measure, st.itertuples())]
    pool_style = {m: (psa[m].median(), psb[m].median()) for m in ["words", "words per sentence", "mean word length",
                                                                 "type-token ratio (first 60 words)", "commas per 100 words"]}
    checks.append({"pair": f"{a} vs {b}", "pool_matched_pairs": len(piv), "sources": piv.index.get_level_values(0).nunique(),
                   "official_sig_tokens": len(sig), "pool_agrees_tokens": float(np.mean(agree_tok)) if agree_tok else np.nan,
                   "official_sig_style": len(st), "pool_agrees_style": float(np.mean(agree_sty)) if agree_sty else np.nan,
                   **{f"pool median {k} (a/b)": f"{v[0]:.2f} / {v[1]:.2f}" for k, v in pool_style.items()}})
    print(checks[-1], flush=True)
pd.DataFrame(checks).to_csv(OUT / "pool_check.csv", index=False)
pd.set_option("display.width", 250)
print(S.round(3).to_string(index=False))
print("written to", OUT)
