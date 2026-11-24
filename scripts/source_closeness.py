"""How close is each external source to official passages of the same label? (official = 120 dev prompts only)

Three measures per (label, source):
  dist   mean absolute gap, in points, between marker rates (typography, pronouns, format) and official
  auc    style-only discriminator AUC, official vs source (0.5 = indistinguishable), 5-fold CV, 3 subsamples
  recall share of the source's rows that the text+style reference model fitted on all official dev rows
         gives the claimed label; fam = share given the right provider family
Usage: uv run python -I scripts/source_closeness.py <admitted.parquet> <open pool parquet> <exp5 run dir> <out csv>
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nllm import linear
from nllm.data import LABELS, load_train
from nllm.features import style_features
from nllm.metrics import per_class_recall
from nllm.splits import development_rows

ours = pd.read_parquet(sys.argv[1])
pool = pd.read_parquet(sys.argv[2])
exp5 = pd.read_parquet(Path(sys.argv[3]) / "oof.parquet")
OUT = Path(sys.argv[4])
rng = np.random.default_rng(17)

dev = development_rows(load_train()).reset_index(drop=True)
dev["source"] = "official"
fam = lambda m: m.split("-")[0].replace("qwen3.8", "qwen")

parts = [dev[["id", "model", "text", "source"]]]
o = ours[ours.split != "diagnostic"].copy()
o["source"] = "ours " + o.interface
parts.append(o[["id", "model", "text", "source"]])
for m, g in pool.groupby("model"):
    def take(df, name, n, m=m):
        if len(df) >= 100:
            s = df.sample(min(n, len(df)), random_state=17)
            parts.append(pd.DataFrame({"id": s.id.astype(str), "model": m, "text": s.text, "source": name}))
    take(g, "pool all", 500)
    take(g[g.flag_kb_proxy], "pool proxy", 300)
    take(g[~g.flag_kb_proxy], "pool non-proxy", 300)
    for src in g.source_dataset.value_counts().head(4).index:
        take(g[g.source_dataset == src], "pool " + src, 300)
df = pd.concat(parts, ignore_index=True)
df["uid"] = df.source + "|" + df.id.astype(str)
df = df.drop_duplicates("uid").set_index("uid", drop=False)
print(f"{len(df)} passages, {df.source.nunique()} sources", flush=True)

HY = re.compile(r"[A-Za-z]-[A-Za-z]")
MARKERS = {
    "em": "\u2014", "en": "\u2013", "hyph": HY, "curly'": "\u2019", "str'": "'", "curly\"": "[\u201c\u201d]",
    "str\"": '"', "colon": ":", "semi": ";", "paren": r"\(", "digit": r"\d", "excl": "!", "quest": r"\?",
    "we": r"(?i)\b(?:we|our|us)\b", "I": r"\b(?:I|me|my)\b", "format": r"\*\*|\n|^#",
}
for k, p in MARKERS.items():
    df[k] = df.text.str.contains(p, regex=True)
df["words"] = df.text.str.split().str.len()

style = style_features(df.text).set_axis(df.index)
print("style features done", flush=True)

ref = dev.set_index(dev.id, drop=False)
sty_ref = style_features(ref.text).set_axis(ref.index)
scores = linear.fit_score(ref, sty_ref, df, style, [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE], 17, None)
t, s = scores[linear.REFERENCE_TEXT.key], scores[linear.REFERENCE_STYLE.key]
df["pred"] = np.asarray(LABELS)[linear.blend(t, s, linear.REFERENCE_STYLE_WEIGHT, linear.scale(t), linear.scale(s)).argmax(1)]
print("reference predictions done", flush=True)

off_cv = exp5[exp5.condition == "official"]
cv_recall = per_class_recall(off_cv.model, off_cv.pred)

rows = []
for m in LABELS:
    off = df[(df.model == m) & (df.source == "official")]
    for src, g in df[(df.model == m) & (df.source != "official")].groupby("source"):
        dist = float(np.mean([abs(g[k].mean() - off[k].mean()) for k in MARKERS]) * 100)
        aucs = []
        for rep in range(3):
            n = min(len(off), len(g), 150)
            a = off.sample(n, random_state=rep)
            b = g.sample(n, random_state=rep)
            X = pd.concat([style.loc[a.index], style.loc[b.index]]).to_numpy()
            y = np.r_[np.zeros(n), np.ones(n)]
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000))
            p = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=rep), method="predict_proba")[:, 1]
            aucs.append(roc_auc_score(y, p))
        rows.append({"model": m, "source": src, "n": len(g), "dist": dist, "auc": float(np.mean(aucs)),
                     "recall": float((g.pred == m).mean()), "fam": float((g.pred.map(fam) == fam(m)).mean()),
                     "cv_recall_official": cv_recall.get(m, np.nan),
                     "words_med": float(g.words.median()), "words_med_official": float(off.words.median()),
                     **{f"{k}_gap": float((g[k].mean() - off[k].mean()) * 100) for k in MARKERS}})
res = pd.DataFrame(rows)
res.to_csv(OUT, index=False)
pd.set_option("display.width", 250, "display.max_rows", 500)
print(res[["model", "source", "n", "dist", "auc", "recall", "fam", "cv_recall_official", "words_med", "words_med_official"]]
      .round(3).to_string(index=False))
