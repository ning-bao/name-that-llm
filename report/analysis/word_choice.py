"""Word-choice fingerprints of the 18 labels: discover on the 18 Nov Open Pool, confirm on official text.

Tokens per passage (document presence, so a word counts once per passage):
  w:<word>   lowercase word (apostrophes folded to ' so typography does not split words)
  b:<w1 w2>  lowercase bigram
  S:<word>   first word of any sentence
  F:<word>   first word of the passage
  L:<word>   first word of the last sentence

Discovery: pool passages from prompt groups answered by >= 6 labels (topic roughly balanced across labels).
Confirmation: all 150 official prompts, where every label answers every prompt (topic balanced by design).
For each contrast (label vs all other labels; label vs its siblings; named sibling pairs) a token's
smoothed log-odds ratio of presence and its z-score are computed in both corpora. A token is "replicated"
when it is among the pool's top 40 in its direction and official z >= 3.0 with the same sign.

Predictive part: a logistic regression on label-free style vocabulary (the pool's 1,000 most common words plus
sentence/passage-position tokens), trained on pool rows only, is scored on official rows (never trained on),
alone and combined with the Qwen3-8B LoRA24 out-of-fold scores (weight chosen on the 8B's inner OOF rows).

    uv run --no-project --with pandas --with pyarrow --with scikit-learn --with scipy python -I word_choice.py \
        POOL.parquet TRAIN.csv RUN_DIR OUT_DIR
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

POOL, TRAIN, RUN, OUT = map(Path, sys.argv[1:5])
OUT.mkdir(parents=True, exist_ok=True)
LABELS = sorted(pd.read_csv(TRAIN, dtype=str, keep_default_na=False, encoding="utf-8").model.unique())
FAMILY = {l: l.split("-")[0].replace("qwen3.8", "qwen") for l in LABELS}
PAIRS = [("gemini-3.7-flash", "gemini-3.8-flash"), ("glm-5.3", "glm-5.3-flash"), ("claude-fable-5", "glm-5.3"),
         ("gpt-5.6-luna", "gpt-5.6-sol"), ("gpt-5.6-sol", "gpt-5.6-terra"), ("gpt-5.6-luna", "gpt-5.6-terra"),
         ("gemini-3.6-flash", "gemini-3.7-flash"), ("kimi-k3", "gpt-5.5")]
WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")
SENT = re.compile(r"(?<=[.!?])[\"”’')\]]*\s+")
rng = np.random.default_rng(17)


def tokens(text: str) -> list[str]:
    t = text.replace("’", "'").replace("‘", "'")
    words = [w.lower() for w in WORD.findall(t)]
    out = {f"w:{w}" for w in words} | {f"b:{a} {b}" for a, b in zip(words, words[1:])}
    sents = [s for s in SENT.split(t) if s.strip()]
    firsts = [WORD.findall(s)[:1] for s in sents]
    firsts = [f[0].lower() for f in firsts if f]
    out |= {f"S:{w}" for w in firsts}
    if firsts:
        out.add(f"F:{firsts[0]}")
        out.add(f"L:{firsts[-1]}")
    return list(out)


def presence(texts, vocab=None, min_df=1):
    vec = CountVectorizer(analyzer=tokens, binary=True, vocabulary=vocab, min_df=min_df, dtype=np.int8)
    X = vec.fit_transform(texts) if vocab is None else vec.transform(texts)
    return sparse.csr_matrix(X), np.asarray(vec.get_feature_names_out())


def log_odds(X, in_a, in_b):
    ya, yb = np.asarray(X[in_a].sum(0)).ravel(), np.asarray(X[in_b].sum(0)).ravel()
    na, nb = in_a.sum(), in_b.sum()
    d = np.log((ya + .5) / (na - ya + .5)) - np.log((yb + .5) / (nb - yb + .5))
    se = np.sqrt(1 / (ya + .5) + 1 / (na - ya + .5) + 1 / (yb + .5) + 1 / (nb - yb + .5))
    return d, d / se, ya / na, yb / nb


# ---------------------------------------------------------------- data
off = pd.read_csv(TRAIN, dtype=str, keep_default_na=False, encoding="utf-8")
pool = pd.read_parquet(POOL, columns=["id", "text", "model", "prompt_canon", "prompt_group", "panel_labels",
                                      "source_dataset", "n_words"])
pool = pool[pool.model.isin(LABELS)].reset_index(drop=True)
disc = pool[pool.panel_labels >= 6].reset_index(drop=True)
print(f"official {len(off)} rows; pool {len(pool)} rows, discovery subset (panels >= 6 labels) {len(disc)}", flush=True)

Xd, vocab = presence(disc.text, min_df=50)
Xo, _ = presence(off.text, vocab=vocab)
print(f"vocabulary {len(vocab)} tokens (pool df >= 50)", flush=True)
yd, yo = disc.model.to_numpy(), off.model.to_numpy()
keep_o = np.asarray(Xo.sum(0)).ravel() >= 15  # enough official passages to test


def contrast(name, a_pool, b_pool, a_off, b_off, top=40):
    d1, z1, pa1, pb1 = log_odds(Xd, a_pool, b_pool)
    d2, z2, pa2, pb2 = log_odds(Xo, a_off, b_off)
    rows = []
    for sign in (1, -1):
        cand = np.where(keep_o & (pa1 + pb1 > 0.002))[0]
        cand = cand[np.argsort(-sign * z1[cand])][:top]
        for i in cand:
            rows.append({"contrast": name, "direction": "over" if sign > 0 else "under", "token": vocab[i],
                         "pool_z": z1[i], "pool_rate_a": pa1[i], "pool_rate_b": pb1[i],
                         "off_z": z2[i], "off_rate_a": pa2[i], "off_rate_b": pb2[i], "off_logodds": d2[i],
                         "replicated": bool(sign * z2[i] >= 3.0)})
    return rows


rows = []
for l in LABELS:
    rows += contrast(f"{l} vs all", yd == l, yd != l, yo == l, yo != l)
    sib = [s for s in LABELS if FAMILY[s] == FAMILY[l] and s != l]
    if sib:
        rows += contrast(f"{l} vs siblings", yd == l, np.isin(yd, sib), yo == l, np.isin(yo, sib))
for a, b in PAIRS:
    rows += contrast(f"{a} vs {b}", yd == a, yd == b, yo == a, yo == b)
res = pd.DataFrame(rows)
res.to_csv(OUT / "contrasts.csv", index=False)
rep = res.groupby("contrast").replicated.agg(["sum", "count"])
rep["rate"] = rep["sum"] / rep["count"]
rep.to_csv(OUT / "replication.csv")
print(rep.to_string(), flush=True)

# ---------------------------------------------------------------- predictive tests (pool-trained, official-scored)
df_pool = np.asarray(Xd.sum(0)).ravel() / Xd.shape[0]
is_word = np.char.startswith(vocab.astype(str), "w:")
is_pos = np.array([v[:2] in ("S:", "F:", "L:") for v in vocab])
top_words = np.where(is_word)[0][np.argsort(-df_pool[is_word])][:1000]
pos_tokens = np.where(is_pos & (df_pool >= 0.002))[0]
style_idx = np.r_[top_words, pos_tokens]
all_words = np.where(is_word | np.array([v[:2] == "b:" for v in vocab]))[0]

# class-balanced training sample from the whole pool (any panel size), 3,000 per label at most
pick = np.concatenate([rng.choice(g.index.to_numpy(), min(len(g), 3000), replace=False) for _, g in pool.groupby("model")])
train_ids = pool.loc[np.sort(pick)]
Xp, _ = presence(train_ids.text, vocab=vocab)
yp = train_ids.model.to_numpy()

MARKERS = ["—", "–", "’", "‘", "“", "”", '"', "'", ";", ":", "(", "!", "?", ",", " - ", "…"]


def typo(texts):
    t = pd.Series(list(texts))
    n = t.str.split().str.len().clip(lower=1).to_numpy()
    cols = [t.str.count(re.escape(m)).to_numpy() / n for m in MARKERS] + [t.str.contains(re.escape(m)).to_numpy() for m in MARKERS]
    return np.column_stack(cols + [np.log(n)]).astype(float)


def fit_eval(name, Xtr, Xte, C):
    clf = LogisticRegression(C=C, max_iter=3000)
    clf.fit(Xtr, yp)
    s = clf.decision_function(Xte)
    s = s[:, [list(clf.classes_).index(l) for l in LABELS]]
    pred = np.asarray(LABELS)[s.argmax(1)]
    rec = pd.Series({l: (pred[yo == l] == l).mean() for l in LABELS})
    fam = np.mean([FAMILY[p] == FAMILY[t] for p, t in zip(pred, yo)])
    aucs = {}
    for a, b in PAIRS:
        m = np.isin(yo, [a, b])
        aucs[f"{a}|{b}"] = roc_auc_score(yo[m] == a, s[m, LABELS.index(a)] - s[m, LABELS.index(b)])
    print(f"{name}: official BA {rec.mean():.3f}, family acc {fam:.3f}", flush=True)
    return s, rec, aucs


from sklearn.preprocessing import StandardScaler  # noqa: E402

tp, to = typo(train_ids.text), typo(off.text)
sc = StandardScaler().fit(tp)
models = {
    "style words (1,000 common words + positions)": (Xp[:, style_idx], Xo[:, style_idx], 0.05),
    "all words + bigrams (pool df >= 50)": (Xp[:, all_words], Xo[:, all_words], 0.05),
    "typography only (16 markers)": (sc.transform(tp), sc.transform(to), 1.0),
    "style words + typography": (sparse.hstack([Xp[:, style_idx], sparse.csr_matrix(sc.transform(tp))]).tocsr(),
                                 sparse.hstack([Xo[:, style_idx], sparse.csr_matrix(sc.transform(to))]).tocsr(), 0.05),
}
scores, recalls, aucs = {}, {}, {}
for name, (a, b, C) in models.items():
    scores[name], recalls[name], aucs[name] = fit_eval(name, a, b, C)
pd.DataFrame(recalls).to_csv(OUT / "recall_pool_trained.csv")
pd.DataFrame(aucs).to_csv(OUT / "pair_auc_pool_trained.csv")

# ---------------------------------------------------------------- does word choice add to the 8B? (nested alpha)
oof = pd.read_parquet(RUN / "oof.parquet")
inner = pd.read_parquet(RUN / "inner_oof.parquet").merge(pd.read_csv(RUN / "selections.csv"),
                                                         on=["partition", "outer_fold", "lr", "epoch"])
SC = [f"score:{l}" for l in LABELS]
pos = {i: k for k, i in enumerate(off.id)}
ALPHAS = (0, .05, .1, .2, .3, .5, .75, 1.0, 1.5)


def ba(y, p):
    return np.mean([(p[y == l] == l).mean() for l in LABELS])


def mix(z8, lex, a, family_only):
    lp8 = z8 - logsumexp(z8, 1, keepdims=True)
    lx = lex - logsumexp(lex, 1, keepdims=True)
    if not family_only:
        return lp8 + a * lx
    out = lp8.copy()
    for f in set(FAMILY.values()):
        cols = [i for i, l in enumerate(LABELS) if FAMILY[l] == f]
        if len(cols) < 2:
            continue
        inner_ = lp8[:, cols] + a * lx[:, cols]
        out[:, cols] = logsumexp(lp8[:, cols], 1, keepdims=True) + inner_ - logsumexp(inner_, 1, keepdims=True)
    return out


def bootstrap(y, p, base, groups, n=2000):
    g, gi = np.unique(groups, return_inverse=True)
    w = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=n)
    out = []
    for pred in (p, base):
        rec = []
        for l in LABELS:
            m = y == l
            num = np.zeros(len(g)); den = np.zeros(len(g))
            np.add.at(num, gi[m], pred[m] == l); np.add.at(den, gi[m], 1)
            rec.append((w @ num) / np.maximum(w @ den, 1e-9))
        out.append(np.mean(rec, axis=0))
    d = out[0] - out[1]
    return np.quantile(d, [.025, .975])


lines = []
for name in ["style words (1,000 common words + positions)", "style words + typography", "typography only (16 markers)"]:
    s_all = scores[name]
    for fam_only in (False, True):
        for p in (0, 1):
            part = oof[oof.partition == p]
            preds, chosen = [], []
            y_all, base_all, grp_all = [], [], []
            for o in range(5):
                inn = inner[(inner.partition == p) & (inner.outer_fold == o)]
                zi, yi = inn[SC].to_numpy(float), inn.model.to_numpy()
                li = s_all[[pos[i] for i in inn.id]]
                a = max(ALPHAS, key=lambda a: (ba(yi, np.asarray(LABELS)[mix(zi, li, a, fam_only).argmax(1)]), -a))
                chosen.append(a)
                ou = part[part.outer_fold == o]
                zo, lo = ou[SC].to_numpy(float), s_all[[pos[i] for i in ou.id]]
                preds.append(np.asarray(LABELS)[mix(zo, lo, a, fam_only).argmax(1)])
                base_all.append(np.asarray(LABELS)[zo.argmax(1)]); y_all.append(ou.model.to_numpy()); grp_all.append(ou.prompt_id.to_numpy())
            pr, bs, yy, gg = map(np.concatenate, (preds, base_all, y_all, grp_all))
            lo_, hi_ = bootstrap(yy, pr, bs, gg)
            lines.append({"lexical model": name, "family-only": fam_only, "partition": p, "8b": ba(yy, bs),
                          "combined": ba(yy, pr), "delta": ba(yy, pr) - ba(yy, bs), "lo": lo_, "hi": hi_,
                          "alphas": chosen})
comb = pd.DataFrame(lines)
comb.to_csv(OUT / "combined_with_8b.csv", index=False)
pd.set_option("display.width", 220)
print(comb.round(4).to_string(index=False))
print("written to", OUT)
