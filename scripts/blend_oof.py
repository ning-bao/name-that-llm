"""Nested blend of branches from finished runs, scored against the first component.

Each --component names a run directory (with oof.parquet and inner_oof.parquet) and one
of its branches. Per outer fold, the weights of the second and later components are
chosen on inner out-of-fold balanced accuracy only (src/nllm/blending.py). Reports every
component alone and the blend, with paired prompt-bootstrap differences vs the first.

Usage: uv run python scripts/blend_oof.py \
           --component outputs/baseline_linear/<run> text+style \
           --component outputs/encoder/ettin-400m/<run> ettin-400m
Writes outputs/blend/<UTC timestamp>/.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nllm import blending
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import initialization_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", nargs=2, action="append", metavar=("RUN", "BRANCH"), required=True)
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    args = ap.parse_args()
    if len(args.component) < 2:
        ap.error("need at least two components")

    seed = initialization_seeds()[0]
    components = [blending.load(Path(run), branch) for run, branch in args.component]
    names = [branch for _, branch in args.component]
    name = " + ".join(names)
    oof, weights = blending.nested_blend(components, name)
    out = OUTPUTS / "blend" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True)

    results, recalls = [], {}
    boot = {"n_draws": args.bootstrap_draws, "seed": seed}
    for p, blend_rows in oof.groupby("partition"):
        ref = blend_rows.set_index("id")
        first = components[0][1].xs(p, level="partition").droplevel("outer_fold").loc[ref.index]
        candidates = {
            n: outer.xs(p, level="partition").droplevel("outer_fold").loc[ref.index]
            for n, (_, outer) in zip(names, components, strict=True)
        } | {name: ref}
        for branch, rows in candidates.items():
            lo, hi = interval(prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, **boot))
            diff = prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, y_pred_baseline=first.pred, **boot)
            dlo, dhi = interval(diff)
            results.append({
                "partition": p,
                "branch": branch,
                "balanced_accuracy": balanced_accuracy(rows.model, rows.pred),
                "ci95_lo": lo,
                "ci95_hi": hi,
                f"diff_vs_{names[0]}": float(diff.mean()),
                "diff_ci95_lo": dlo,
                "diff_ci95_hi": dhi,
            })
            recalls[f"p{p}:{branch}"] = per_class_recall(rows.model, rows.pred)
    results = pd.DataFrame(results)
    results.to_csv(out / "results.csv", index=False)
    weights.to_csv(out / "weights.csv", index=False)
    oof.to_parquet(out / "oof.parquet", index=False)
    recall_table = pd.DataFrame(recalls)
    recall_table.to_csv(out / "per_class_recall.csv")
    (out / "config.json").write_text(json.dumps({
        "components": [{"run": run, "branch": branch} for run, branch in args.component],
        "weights": "component 1 fixed at 1; later ones from linear.BLEND_WEIGHTS, chosen per outer fold on inner OOF BA",
        "bootstrap_draws": args.bootstrap_draws,
        "seed": seed,
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(results.round(4).to_string(index=False))
    print("\nWeights per outer fold:")
    print(weights.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
