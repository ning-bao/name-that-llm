"""Community pool training data (experiment 6): the Open Pool plus our own admitted answers.

load_pool() reads the Open Pool main file (crtproduction/name-that-llm-open-pool v13, CC BY-SA 4.0) as
training rows. own_rows() adds our admitted external answers with the named source filters applied.
Both carry the same columns: id, text, model, group, source, kb_proxy, panel_labels, origin.
`group` is the prompt a passage answers: the pool's recovered prompt_canon (shared across sources when
it starts with "p"), our pool-wave answers join that same canon, and bank answers keep our prompt id.

The pool ships a match_weight column fitted on all 150 official train prompts. Using it would leak the
validation prompts' typography into every fold, so it is never read here: match_weights() refits the
same idea (per-label weights that make marker rates and length match official passages) on whatever
official rows it is given, which in nested validation is the training fold only.

The pool has also already dropped 1,278 rows whose typography a label "never uses in train", judged on
all 150 official prompts. Those rows are gone from the file, so that small leak cannot be undone; it
makes any score on official prompts, the confirmation set included, read slightly high.
"""

import fnmatch
import functools
import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from nllm import collection, linear
from nllm.data import LABELS
from nllm.features import style_features

# $NLLM_POOL_DATA holds the same layout elsewhere (e.g. on TC2): third_party/<pool dir>/, extra_sources.parquet and
# admitted.parquet.
POOL_DATA = Path(os.environ.get("NLLM_POOL_DATA", collection.EXTERNAL_DIR))
POOL_DIR = POOL_DATA / "third_party" / "crtproduction__name-that-llm-open-pool"
POOL_FILE = "ntl_open_pool.parquet"
COLUMNS = ["id", "text", "model", "group", "source", "kb_proxy", "panel_labels", "origin"]
EXTRA_PATH = POOL_DATA / "extra_sources.parquet"

# Named source filters for our own answers (see EXPERIMENTS.md, 3 Nov 2026).
# typography-source-filter-v1: official Gemini passages contain an em dash once in 600, ours in 21-35%,
# so a Gemini answer with an em dash teaches a cue that never appears in Gemini test passages.
GEMINI_EM_DASH_FILTER = "typography-source-filter-v1"
# Qoder's qwen3.8-max answers look like Claude/GLM text (em dashes in 62%, no curly apostrophes) and an
# official-trained model recognises 4% of them as Qwen; the pool's Qwen rows match official far better.
DEFAULT_OWN_EXCLUDE = ("qwen3.8-max@qoder*",)

# Binary markers whose rates match_weights() aligns with official passages, plus word-count bins.
MARKERS = {
    "em_dash": "\u2014",
    "en_dash": "\u2013",
    "hyphenated": r"[A-Za-z]-[A-Za-z]",
    "curly_apostrophe": "\u2019",
    "straight_apostrophe": "'",
    "curly_quote": "[\u201c\u201d]",
    "straight_quote": '"',
    "colon": ":",
    "semicolon": ";",
    "digit": r"\d",
    "parenthesis": r"\(",
    "exclamation": "!",
    "question": r"\?",
    "we_our_us": r"(?i)\b(?:we|our|us)\b",
    "i_me_my": r"\b(?:I|me|my)\b",
}
LENGTH_BINS = (60, 100, 150)  # words: <=60, 61-100, 101-150, >150 (the last bin is implied)


def load_pool(path: Path | None = None) -> pd.DataFrame:
    """The Open Pool main file as training rows (its match_weight and style_score columns are dropped)."""
    raw = pd.read_parquet(path or POOL_DIR / POOL_FILE)
    if unknown := set(raw.model) - set(LABELS):
        raise ValueError(f"pool labels outside LABELS: {sorted(unknown)}")
    canon = raw.prompt_canon.fillna("")
    group = canon.where(canon != "", "src:" + raw.prompt_group.fillna(raw.id.astype(str)).astype(str))
    return pd.DataFrame({
        "id": "pool:" + raw.id.astype(str),
        "text": raw.text,
        "model": raw.model,
        "group": group,
        "source": raw.source_dataset,
        "kb_proxy": raw.flag_kb_proxy.astype(bool),
        "panel_labels": raw.panel_labels.astype(int),
        "origin": "pool",
    })


def own_rows(path: Path | None = None, exclude=DEFAULT_OWN_EXCLUDE, gemini_em_dash_filter: bool = True,
             pool_panel: pd.Series | None = None) -> pd.DataFrame:
    """Our admitted answers (diagnostic prompts left out) with the named source filters applied.

    `exclude` holds MODEL@INTERFACE wildcard patterns. `pool_panel` (prompt_canon -> labels answering it in
    the pool) fills panel_labels for pool-wave answers; bank prompts were asked of all 18 labels.
    """
    ext = pd.read_parquet(path or POOL_DATA / "admitted.parquet")
    ext = ext[ext.split != "diagnostic"]
    tag = ext.model + "@" + ext.interface
    ext = ext[~tag.map(lambda t: any(fnmatch.fnmatchcase(t, p) for p in exclude))]
    if gemini_em_dash_filter:
        ext = ext[~(ext.model.str.startswith("gemini") & ext.text.str.contains("\u2014"))]
    from_pool = ext.wave.str.startswith("pool")
    canon = ext.prompt_id.str.removeprefix("x")
    panel = canon.map(pool_panel) if pool_panel is not None else pd.Series(np.nan, index=ext.index)
    return pd.DataFrame({
        "id": "own:" + ext.id.astype(str),
        "text": ext.text,
        "model": ext.model,
        "group": canon.where(from_pool, ext.prompt_id),
        "source": "own " + ext.interface,
        "kb_proxy": False,
        "panel_labels": np.where(from_pool, panel.fillna(1), 18).astype(int),
        "origin": "own",
    })


def load_extra(path: Path | None = None) -> pd.DataFrame:
    """Community rows published after Open Pool v13, cleaned by scripts/prepare_extra_sources.py (origin "extra");
    empty if that file has not been built."""
    path = path or EXTRA_PATH
    return pd.read_parquet(path)[COLUMNS] if path.exists() else pd.DataFrame(columns=COLUMNS)


@functools.lru_cache(maxsize=1)
def candidate_rows() -> pd.DataFrame:
    """Pool, extra and own rows (default filters), indexed by id; loaded once per process."""
    rows = load_pool()
    own = own_rows(pool_panel=rows.groupby("group").panel_labels.max())
    return pd.concat([rows, load_extra(), own], ignore_index=True).set_index("id", drop=False)


def marker_matrix(texts: pd.Series) -> np.ndarray:
    """Rows x (markers + length bins) 0/1 matrix; the last length bin is left out as the baseline."""
    cols = [texts.str.contains(p, regex=True).to_numpy() for p in MARKERS.values()]
    words = texts.str.split().str.len().to_numpy()
    edges = (0, *LENGTH_BINS)
    cols += [(words > lo) & (words <= hi) for lo, hi in itertools.pairwise(edges)]
    return np.column_stack(cols).astype(np.float64)


def marker_names() -> list[str]:
    edges = (0, *LENGTH_BINS)
    return [*MARKERS, *(f"words_{lo + 1}_{hi}" for lo, hi in itertools.pairwise(edges))]


def _tilt(X: np.ndarray, target: np.ndarray, ridge: float, base: np.ndarray) -> np.ndarray:
    """Exponential-tilting weights w ~ base * exp(X @ lam) whose weighted column means approach `target`."""
    log_base = np.log(base)

    def objective(lam):
        z = log_base + X @ lam
        p = np.exp(z - logsumexp(z))
        return (logsumexp(z) - np.log(len(z)) - lam @ target + ridge / 2 * lam @ lam,
                p @ X - target + ridge * lam)

    lam = minimize(objective, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x
    z = log_base + X @ lam
    return np.exp(z - logsumexp(z)) * len(z)


def match_weights(rows: pd.DataFrame, official: pd.DataFrame, ridge: float = 0.02, max_weight: float = 20.0,
                  base: np.ndarray | None = None) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-label sampling weights (mean 1 within each label) matching marker rates and length to `official`.

    Fit only on official rows of the training fold. Targets the rows cannot reach (a rate the source never
    shows) are approached as far as the ridge allows; weights are capped at `max_weight` times the mean.
    `base` (positive, in `rows` order) is a prior weight the tilt starts from, e.g. source multipliers.
    Returns the weights in `rows` order and a per-label table of effective sample share and marker gaps.
    """
    X_all = marker_matrix(rows.text)
    X_off = marker_matrix(official.text)
    base = np.ones(len(rows)) if base is None else np.asarray(base, dtype=float)
    names = marker_names()
    weights = np.zeros(len(rows))
    report = []
    for m in LABELS:
        sel = (rows.model == m).to_numpy()
        off = (official.model == m).to_numpy()
        if not sel.any() or not off.any():
            weights[sel] = base[sel] * sel.sum() / base[sel].sum()
            continue
        X, target = X_all[sel], X_off[off].mean(axis=0)
        w = np.minimum(_tilt(X, target, ridge, base[sel]), max_weight)
        w *= len(w) / w.sum()
        weights[sel] = w
        before, after = X.mean(axis=0) - target, w @ X / len(w) - target
        worst = int(np.argmax(np.abs(after)))
        report.append({"model": m, "rows": int(sel.sum()), "official_rows": int(off.sum()),
                       "effective_share": float(w.sum() ** 2 / (w ** 2).sum() / len(w)),
                       "mean_gap_before": float(np.abs(before).mean() * 100),
                       "mean_gap_after": float(np.abs(after).mean() * 100),
                       "worst_marker_after": names[worst], "worst_gap_after": float(after[worst] * 100)})
    return weights, pd.DataFrame(report)


def source_multipliers(rows: pd.DataFrame, recognised: pd.Series, min_rows: int = 30,
                       clip: tuple[float, float] = (0.25, 4.0)) -> tuple[np.ndarray, pd.DataFrame]:
    """Per-row multipliers that favour sources whose passages look more like official ones.

    `recognised` (indexed like a probe subset of `rows`) is True where a model trained on the fold's
    official rows gives a passage its claimed label. A (label, source) cell's multiplier is its
    recognition rate over the label's row-weighted mean rate (both +0.02, so a 0% source is not zeroed),
    clipped; cells with fewer than `min_rows` probed passages get 1.
    """
    probe = pd.DataFrame({"model": rows.loc[recognised.index, "model"], "source": rows.loc[recognised.index, "source"],
                          "hit": recognised.astype(float)})
    cells = probe.groupby(["model", "source"]).hit.agg(["mean", "size"]).reset_index()
    sizes = rows.groupby(["model", "source"]).size().rename("rows").reset_index()
    cells = cells.merge(sizes, on=["model", "source"])
    label_rate = cells.assign(w=cells["mean"] * cells.rows).groupby("model").apply(
        lambda d: d.w.sum() / d.rows.sum(), include_groups=False)
    cells["multiplier"] = ((cells["mean"] + 0.02) / (cells.model.map(label_rate) + 0.02)).clip(*clip)
    cells.loc[cells["size"] < min_rows, "multiplier"] = 1.0
    lookup = cells.set_index(["model", "source"]).multiplier
    keys = pd.MultiIndex.from_arrays([rows.model, rows.source])
    return lookup.reindex(keys).fillna(1.0).to_numpy(), cells.rename(columns={"mean": "recognised", "size": "probed"})


def balanced_sample(rows: pd.DataFrame, n_per_label: int, seed: int, weights: np.ndarray | None = None,
                    replace: bool = False) -> pd.DataFrame:
    """Up to n_per_label rows of each label, drawn without replacement with probability ~ weight
    (Efraimidis-Spirakis keys), so every label gets the same exposure however uneven the pool is.

    With replace=True a label with fewer than n_per_label usable rows keeps all of them and is topped up to
    n_per_label by a with-replacement draw (probability ~ weight), so every label has exactly n_per_label rows:
    some passages then appear more than once."""
    rng = np.random.default_rng(seed)
    w = np.ones(len(rows)) if weights is None else np.asarray(weights, dtype=float)
    keys = np.full(len(rows), -np.inf)
    positive = w > 0
    keys[positive] = np.log(rng.random(positive.sum())) / w[positive]
    picked = (pd.DataFrame({"model": rows.model.to_numpy(), "key": keys})
              .loc[positive].sort_values("key", ascending=False).groupby("model").head(n_per_label))
    positions = np.sort(picked.index.to_numpy())
    if replace:
        labels = rows.model.to_numpy()
        extra = []
        for label, count in picked.model.value_counts().items():
            if count < n_per_label:
                idx = np.flatnonzero((labels == label) & positive)
                extra.append(rng.choice(idx, size=n_per_label - count, replace=True, p=w[idx] / w[idx].sum()))
        positions = np.sort(np.concatenate([positions, *extra]))
    return rows.iloc[positions]


@dataclass(frozen=True)
class Recipe:
    """How training_sample() draws rows. `own` is a wildcard on our own rows' `source` ("own *": all of
    them; None: pool rows only). `weighting` is "none" (uniform within label), "match" (match_weights) or
    "quality" (match_weights starting from source_quality multipliers). `own_boost` multiplies our own
    rows' starting weight before the tilt. `extra` adds the rows of load_extra() to the pool's.
    Experiment 6 (scripts/pool_eval.py) compares the recipes."""
    own: str | None = None
    drop_proxy: bool = False
    weighting: str = "match"
    own_boost: float = 1.0
    extra: bool = False


RECIPES = {
    "u": Recipe(weighting="none"),
    # Uniform, so nothing about the sample depends on official rows: the recipe for a stage 1 shared by every
    # outer fold (encoder_finetune.py --shared-stage1).
    "ux": Recipe(weighting="none", extra=True),
    "w": Recipe(),
    "w_noproxy": Recipe(drop_proxy=True),
    "w_own": Recipe(own="own *"),
    "wq_own": Recipe(own="own *", weighting="quality"),
    # Our Codex GPT answers measured closer to official than the pool's GPT rows (EXPERIMENTS.md, source
    # closeness), but at their natural share they are 1-2% of a GPT label's sample; these draw them ~5x more.
    "w_codex5": Recipe(own="own codex*", own_boost=5.0),
    "wq_codex5": Recipe(own="own codex*", weighting="quality", own_boost=5.0),
    # Plus zeminglii's Vertex-pinned and kilupy's prompt-paired corpora (EXPERIMENTS.md, "More gemini-3.7 data").
    "wx": Recipe(extra=True),
    "wx_own": Recipe(own="own *", extra=True),
}


def source_quality(official: pd.DataFrame, rows: pd.DataFrame | None = None, seed: int = 17,
                   probes_per_cell: int = 200) -> pd.DataFrame:
    """Per-(label, source) multiplier table (source_multipliers): the linear text+style reference, fitted
    on `official` rows only, scores up to `probes_per_cell` passages of every cell."""
    rows = candidate_rows() if rows is None else rows
    probes = rows.sample(frac=1.0, random_state=seed).groupby(["model", "source"]).head(probes_per_cell)
    style = style_features(pd.concat([official.text, probes.text], ignore_index=True))
    scores = linear.fit_score(official, style.iloc[: len(official)], probes, style.iloc[len(official):],
                              [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE], seed)
    t, s = scores[linear.REFERENCE_TEXT.key], scores[linear.REFERENCE_STYLE.key]
    pred = linear.predict(linear.blend(t, s, linear.REFERENCE_STYLE_WEIGHT, linear.scale(t), linear.scale(s)))
    return source_multipliers(rows, pd.Series(pred == probes.model.to_numpy(), index=probes.index))[1]


def training_sample(recipe: str, official: pd.DataFrame | None, per_label: int, seed: int,
                    rows: pd.DataFrame | None = None, cells: pd.DataFrame | None = None,
                    replace: bool = False) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """A class-balanced training sample for one recipe; `official` must be the fold's training rows only
    (None for a uniform recipe, which reads no official rows). `cells` is a source_quality() table already
    computed from those rows (else computed when needed). `replace` tops short labels up to per_label
    (balanced_sample). Returns the sample and the match-weight report (None for uniform sampling)."""
    r = RECIPES[recipe]
    if official is None and r.weighting != "none":
        raise ValueError(f"recipe {recipe} fits weights on official rows, so it needs them")
    rows = candidate_rows() if rows is None else rows
    own = (rows.origin == "own") & rows.source.map(lambda s: r.own is not None and fnmatch.fnmatchcase(s, r.own))
    cand = rows[(rows.origin == "pool") | ((rows.origin == "extra") & r.extra) | own]
    if r.drop_proxy:
        cand = cand[~cand.kb_proxy]
    base = np.where(cand.origin == "own", r.own_boost, 1.0)
    if r.weighting == "quality":
        cells = source_quality(official, rows, seed) if cells is None else cells
        lookup = cells.set_index(["model", "source"]).multiplier
        base = base * lookup.reindex(pd.MultiIndex.from_arrays([cand.model, cand.source])).fillna(1.0).to_numpy()
    weights = report = None
    if r.weighting != "none":
        weights, report = match_weights(cand, official, base=base)
    elif r.own_boost != 1.0:
        weights = base
    return balanced_sample(cand, per_label, seed, weights, replace), report
