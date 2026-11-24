"""Check whether answers collected under a held label come from that model (report E4, E6).

Google routes gemini-3.7-flash API requests to 3.8, so an app showing "3.7" may be served by
3.8. Every bank prompt goes to both models, so collected answers form matched pairs: the
held label's answer and its sibling's answer to the same prompt. The reference text+style
model is fitted on official passages in each dev outer fold (10 folds over both partitions)
and scores every answer by its held-minus-sibling gap, with blend scales from that fold's
official held-out rows. On official held-out pairs the held model's answer beats its
sibling's by a known margin. If the app really serves the held model, collected pairs show
a similar margin; if it serves the sibling under the held label, the margin is near zero.
Pairing removes prompt topic and any interface effect shared by both models.

Verdict (collection.held_verdict, fixed before any collected data was scored): at least
collection.MIN_PAIRS pairs, and a positive collected margin (one-sided sign-flip p below
collection.HELD_ALPHA) of at least half the official margin -> consistent; the upper 95%
bound below half the official margin -> looks like the sibling; otherwise inconclusive,
and the answers stay held. The official separation is weak, so expect a verdict only after
several waves; the script prints how many matched prompts the official margin implies.
With --release, a
consistent verdict lists the scored answers whose model label was checked in the app in
data/external/held_release.json, and the next import admits them.

Usage: uv run python scripts/import_answers.py      (first: rebuilds the records)
       uv run python scripts/verify_held.py [--release]
Writes outputs/held_check/<UTC timestamp>/.
"""

import argparse
import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from nllm import collection, linear
from nllm.data import LABELS, load_train
from nllm.features import style_features
from nllm.metrics import sign_flip_pvalue
from nllm.paths import OUTPUTS
from nllm.splits import development_rows, outer_folds, select

SEED = 17
REFERENCE = [linear.REFERENCE_TEXT, linear.REFERENCE_STYLE]


def usable_answers(records: list[dict], labels: set[str]) -> pd.DataFrame:
    """First usable chat answer per (label, prompt): no quarantine flag, fresh chat, no contradicting version.

    API answers are left out: their responses name the served model, which already verifies them.
    """
    rows = []
    for r in sorted(records, key=lambda r: r["provenance"]["attempt_index"]):
        flags = {i.split(":")[0] for i in r["quality"]["issues"]}
        if (r["target_label"] not in labels or flags & collection.QUARANTINE_FLAGS
                or r["provenance"]["collection_method"] != "manual_chat"
                or not r["session"]["fresh_conversation"]
                or r["source_verification"]["status"] == "rejected_exact_label"):
            continue
        rows.append({"id": r["record_id"], "model": r["target_label"], "prompt_id": r["prompt_id"],
                     "text": r["raw_response"], "status": r["source_verification"]["status"]})
    frame = pd.DataFrame(rows, columns=["id", "model", "prompt_id", "text", "status"])
    return frame.drop_duplicates(["model", "prompt_id"], keep="first").reset_index(drop=True)


def fold_gaps(dev, style, collected, collected_style, fold, held, sibling):
    """Held-minus-sibling blended score for the fold's official held-out rows and every collected answer."""
    tr, va = select(dev, fold.train), select(dev, fold.validation)
    evaluate = pd.concat([va[["text"]], collected[["text"]]], ignore_index=True)
    eval_style = pd.concat([style.loc[va.index], collected_style], ignore_index=True)
    scores = linear.fit_score(tr, style.loc[tr.index], evaluate, eval_style, REFERENCE, SEED)
    text, sty = scores[linear.REFERENCE_TEXT.key], scores[linear.REFERENCE_STYLE.key]
    n = len(va)
    blended = linear.blend(text, sty, linear.REFERENCE_STYLE_WEIGHT, linear.scale(text[:n]), linear.scale(sty[:n]))
    gap = blended[:, LABELS.index(held)] - blended[:, LABELS.index(sibling)]
    return va.index, gap[:n], gap[n:]


def paired(frame: pd.DataFrame, held: str, sibling: str) -> pd.DataFrame:
    """One row per prompt answered by both: each answer's gap and their difference."""
    wide = frame.pivot_table(index="prompt_id", columns="model", values="gap", aggfunc="first")
    wide = wide.dropna(subset=[held, sibling]) if {held, sibling} <= set(wide.columns) else wide.iloc[:0]
    out = pd.DataFrame({"gap_held": wide.get(held, pd.Series(dtype=float)),
                        "gap_sibling": wide.get(sibling, pd.Series(dtype=float))})
    return out.assign(difference=out.gap_held - out.gap_sibling)


def check(held: str, sibling: str, records: list[dict], dev, style, args, out) -> dict:
    collected = usable_answers(records, {held, sibling})
    counts = collected.model.value_counts()
    print(f"\n=== {held} vs {sibling}: {counts.get(held, 0)} usable {held} answers, {counts.get(sibling, 0)} {sibling}")
    if counts.get(held, 0) == 0:
        print("nothing to check yet")
        return {"label": held, "verdict": "no_answers"}
    collected_style = style_features(collected.text)
    folds = [f for p in args.partitions for f in outer_folds(p)]
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(fold_gaps)(dev, style, collected, collected_style, f, held, sibling) for f in folds
    )
    official_gap = pd.concat([pd.Series(g, index=idx) for idx, g, _ in results]).groupby(level=0).mean()
    official = dev.loc[official_gap.index, ["prompt_id", "model"]].assign(gap=official_gap)
    official = official[official.model.isin([held, sibling])]
    collected["gap"] = np.mean([c for _, _, c in results], axis=0)

    official_pairs = paired(official, held, sibling)
    margin = float(official_pairs.difference.mean())
    if margin <= 0:
        raise SystemExit(f"the reference model does not separate {held} from {sibling} on official data")
    pairs = paired(collected, held, sibling)
    n = len(pairs)
    held_mean, sibling_mean = (official.loc[official.model == m, "gap"].mean() for m in (held, sibling))

    def position(model):  # 0 = like official sibling answers, 1 = like official held answers
        return float((collected.loc[collected.model == model, "gap"].mean() - sibling_mean) / (held_mean - sibling_mean))

    effect = margin / float(official_pairs.difference.std())
    summary = {
        "label": held,
        "sibling": sibling,
        "official_pairs": len(official_pairs),
        "official_auc": float(roc_auc_score(official.model == held, official.gap)),
        "official_margin": margin,
        "official_effect_size": effect,
        # matched prompts for 80% power to confirm a held model that matches its official margin
        "pairs_needed": int(np.ceil(((norm.ppf(1 - collection.HELD_ALPHA) + norm.ppf(0.8)) / effect) ** 2)),
        "official_pairs_held_ahead": float((official_pairs.difference > 0).mean()),
        "collected_pairs": n,
        "collected_held_position": position(held),
        "collected_sibling_position": position(sibling) if counts.get(sibling, 0) else None,
    }
    if n:
        boot = np.random.default_rng(SEED).choice(pairs.difference.to_numpy(), size=(10_000, n)).mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        summary |= {
            "collected_margin": float(pairs.difference.mean()),
            "collected_pairs_held_ahead": float((pairs.difference > 0).mean()),
            "p_value": sign_flip_pvalue(pairs.difference, seed=SEED),
            "margin_ratio": float(pairs.difference.mean() / margin),
            "margin_ratio_ci95": [float(lo / margin), float(hi / margin)],
        }
        summary["verdict"] = collection.held_verdict(n, summary["p_value"], summary["margin_ratio"],
                                                     summary["margin_ratio_ci95"][1])
    else:
        summary["verdict"] = "inconclusive"

    pairs.to_csv(out / f"pairs_{held}.csv")
    collected.drop(columns="text").to_csv(out / f"answers_{held}.csv", index=False)
    print(f"official held-out pairs: {held} ahead of {sibling} in {summary['official_pairs_held_ahead']:.0%} "
          f"of {len(official_pairs)} prompts, mean margin {margin:.3f}, AUC {summary['official_auc']:.3f}; "
          f"about {summary['pairs_needed']} matched prompts give 80% power to confirm genuine {held}")
    if n:
        lo, hi = summary["margin_ratio_ci95"]
        print(f"collected pairs: {n} prompts, held ahead in {summary['collected_pairs_held_ahead']:.0%}, "
              f"margin {summary['collected_margin']:.3f} = {summary['margin_ratio']:.2f}x official "
              f"[95% CI {lo:.2f}, {hi:.2f}], one-sided p = {summary['p_value']:.4f}")
    print(f"position on official scale (0 = {sibling}-like, 1 = {held}-like): collected {held} "
          f"{summary['collected_held_position']:.2f}"
          + (f", collected {sibling} {summary['collected_sibling_position']:.2f}"
             if summary["collected_sibling_position"] is not None else ""))
    explain = {
        "consistent": f"consistent with genuine {held}",
        "looks_like_sibling": f"looks like {sibling} served under the {held} label: do not release",
        "inconclusive": f"inconclusive, answers stay held ({n} matched prompts; at least {collection.MIN_PAIRS} "
                        "needed, and the margin must be clearly near official or clearly near zero)",
    }
    print(f"VERDICT: {explain[summary['verdict']]}")
    summary["releasable_ids"] = collected.loc[(collected.model == held)
                                              & (collected.status == "ui_label_verified"), "id"].tolist()
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=[m for m in collection.HELD if m in collection.SIBLINGS])
    ap.add_argument("--partitions", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--n-jobs", type=int, default=10)
    ap.add_argument("--release", action="store_true", help="on a consistent verdict, release the checked answers")
    args = ap.parse_args()
    if not collection.RECORDS_PATH.exists():
        raise SystemExit("no records yet: run scripts/import_answers.py first")
    records = [json.loads(line) for line in collection.RECORDS_PATH.read_text(encoding="utf-8").splitlines() if line]

    dev = development_rows(load_train())
    style = style_features(dev.text).set_axis(dev.index)
    out = OUTPUTS / "held_check" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    summaries = [check(m, collection.SIBLINGS[m], records, dev, style, args, out) for m in args.labels]
    (out / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")

    if args.release:
        released = json.loads(collection.RELEASE_PATH.read_text(encoding="utf-8")) if collection.RELEASE_PATH.exists() else {}
        for s in summaries:
            if s["verdict"] != "consistent":
                print(f"{s['label']}: not released (verdict {s['verdict']})")
            elif not s["releasable_ids"]:
                print(f"{s['label']}: not released: set model_label_checked_in_app: yes in its answer files first")
            else:
                released[s["label"]] = {"record_ids": sorted(s["releasable_ids"]),
                                        "released_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                        "check": str(out),
                                        **{k: v for k, v in s.items() if k != "releasable_ids"}}
                print(f"{s['label']}: released {len(s['releasable_ids'])} answers; rerun scripts/import_answers.py")
        collection.RELEASE_PATH.write_text(json.dumps(released, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
