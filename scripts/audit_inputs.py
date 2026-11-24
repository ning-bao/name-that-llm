"""Experiment 0: confirm the supplied files and the frozen split plan before any training.

Checks file hashes against the research kit's record, the train/test/sample schemas,
the crossed prompt x model design, and every partition/fold of validation_plan.json.
Read-only: inputs are never modified. Exits non-zero on any failure.

Usage: uv run python scripts/audit_inputs.py [--data-dir DIR]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from nllm.data import LABELS, load_sample_submission, load_test, load_train
from nllm.paths import KIT_DIR, OUTPUTS, data_dir
from nllm.splits import check_plan

DATA_FILES = ("train.csv", "test.csv", "sample_submission.parquet")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=None)
    args = ap.parse_args()
    src = args.data_dir or data_dir()

    recorded = json.loads((KIT_DIR / "forensic_summary.json").read_text(encoding="utf-8"))["files"]
    checks: dict[str, bool] = {}
    hashes = {name: sha256(src / name) for name in DATA_FILES}
    for name in DATA_FILES:
        checks[f"sha256 {name} matches kit record"] = hashes[name] == recorded[name]["sha256"]

    train, test, sample = load_train(src), load_test(src), load_sample_submission(src)
    counts = train.model.value_counts()
    cells = train.groupby("prompt_id").model.apply(set)
    missing_cells = {p: sorted(set(LABELS) - s) for p, s in cells.items() if s != set(LABELS)}

    checks.update({
        "train columns id,prompt_id,text,model": list(train.columns) == ["id", "prompt_id", "text", "model"],
        "train has 2699 rows": len(train) == 2699,
        "train has 150 prompts": train.prompt_id.nunique() == 150,
        "train labels equal LABELS": set(train.model) == set(LABELS),
        "150 rows per label except fable 149": all(
            counts[m] == (149 if m == "claude-fable-5" else 150) for m in LABELS
        ),
        "only missing cell is p002/claude-fable-5": missing_cells == {"p002": ["claude-fable-5"]},
        "no empty fields": not (train == "").any().any(),
        "no duplicate ids": not train.id.duplicated().any(),
        "no duplicate prompt/model cells": not train.duplicated(["prompt_id", "model"]).any(),
        "test columns id,text": list(test.columns) == ["id", "text"],
        "sample columns id,model": list(sample.columns) == ["id", "model"],
        "sample ids equal test ids": list(sample.id) == list(test.id),
        "no train/test id overlap": not set(train.id) & set(test.id),
    })
    try:
        check_plan(train)
        checks["validation plan integrity (both partitions, outer+inner)"] = True
    except ValueError as e:
        print(e, file=sys.stderr)
        checks["validation plan integrity (both partitions, outer+inner)"] = False

    width = max(map(len, checks))
    for name, ok in checks.items():
        print(f"{name:<{width}}  {'PASS' if ok else 'FAIL'}")

    out = OUTPUTS / "audit"
    out.mkdir(parents=True, exist_ok=True)
    report = {"data_dir": str(src), "sha256": hashes, "checks": checks, "missing_cells": missing_cells}
    (out / "input_audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out / 'input_audit.json'}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
