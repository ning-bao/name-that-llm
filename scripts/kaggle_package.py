"""Build the Kaggle dataset and notebook for one or more --final encoder runs (report section H3).

<out>/dataset/ is uploaded as a private Kaggle dataset: nllm/ (only the modules the notebook imports),
ensemble.json (members, their scales, label order), files.json (sha256 of every file), and one directory per member
holding model/ (weights and tokenizer), reference.npz (ids and fp32 scores of the reference training rows) and
manifest.json. No competition text is packaged; the notebook reads the reference rows from the competition's own
train.csv. The notebook sums the members' logits, each divided by its --scales value (1 by default), and takes the
argmax; with --centring W CUT it first subtracts W x each test row's cavity mean inside content clusters of the test
passages cut at cosine distance CUT (nllm.centring; validated by scripts/test_time_centring.py). <out>/nllm_infer.ipynb is built from kaggle/nllm_infer.py. --dry-run then runs the notebook source on the
CPU against the package and the local data (3 test rows).

Usage: uv run python scripts/kaggle_package.py outputs/tc2/<final run> [--out outputs/kaggle/<name>] [--dry-run]
       uv run python scripts/kaggle_package.py <run> <run> ... --scales <s> <s> ... --out outputs/kaggle/<name>
"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from nllm.data import LABELS
from nllm.paths import OUTPUTS, ROOT, data_dir

MODULES = ("__init__.py", "data.py", "paths.py", "metrics.py", "submission.py", "inference.py", "centring.py")
NOTEBOOK_SOURCE = ROOT / "kaggle" / "nllm_infer.py"


def notebook(source: str) -> dict:
    """Jupyter cells from `# %%` / `# %% [markdown]` markers."""
    cells = []
    for chunk in source.split("# %%")[1:]:
        header, _, body = chunk.partition("\n")
        if "[markdown]" in header:
            lines = [line.removeprefix("#").removeprefix(" ") for line in body.strip().splitlines()]
            cells.append({"cell_type": "markdown", "metadata": {}, "source": "\n".join(lines)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.strip("\n")})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


def legacy_rope(config: dict) -> dict:
    """Add transformers 4.x's ModernBERT rope keys next to 5.x's rope_parameters. Without them 4.x silently uses
    local_rope_theta 10,000 instead of Ettin's 160,000 (reference agreement 0.81 under transformers 4.57.1)."""
    rope = config.get("rope_parameters")
    if config.get("model_type") == "modernbert" and rope:
        config.setdefault("global_rope_theta", rope["full_attention"]["rope_theta"])
        config.setdefault("local_rope_theta", rope["sliding_attention"]["rope_theta"])
    return config


def is_decoder(run: Path) -> bool:
    """A LoRA decoder's model/ holds only its adapters, head and tokenizer; the base weights ship separately."""
    return (run / "model" / "adapter_config.json").exists()


def is_pooled(run: Path) -> bool:
    """A pooled (LoRA24-style) decoder also saves pooled.json and its LayerNorms and head separately."""
    return (run / "model" / "pooled.json").exists()


def check_run(run: Path) -> dict:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["labels"] != list(LABELS):
        raise SystemExit(f"{run}: label order differs from nllm.data.LABELS")
    if manifest.get("limit_rows"):
        print(f"warning: {run} is a smoke-test run (--limit-rows {manifest['limit_rows']})")
    needed = (("adapter_config.json", "adapter_model.safetensors", "tokenizer.json", "tokenizer_config.json")
              if is_decoder(run) else ("config.json", "model.safetensors", "tokenizer.json"))
    if is_pooled(run):
        needed += ("pooled.json", "pooled_heads.safetensors")
    for name in needed:
        if not (run / "model" / name).exists():
            raise SystemExit(f"{run}/model has no {name}")
    return manifest


def add_member(run: Path, dest: Path) -> None:
    shutil.copytree(run / "model", dest / "model")
    config = dest / "model" / "config.json"
    if config.exists():
        config.write_text(json.dumps(legacy_rope(json.loads(config.read_text(encoding="utf-8"))), indent=2),
                          encoding="utf-8")
    for name in ("reference.npz", "manifest.json"):
        shutil.copy2(run / name, dest / name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", type=Path, nargs="+", help="scripts/encoder_finetune.py --final run directories")
    ap.add_argument("--scales", type=float, nargs="+",
                    help="one per run: each member's logits are divided by it before summing (default 1)")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--centring", type=float, nargs=2, metavar=("WEIGHT", "CUT"),
                    help="test-time soft centring: weight of the cavity cluster mean, clustering cut (e.g. 0.25 0.85)")
    ap.add_argument("--dry-run", action="store_true", help="run the notebook source locally on the CPU")
    ap.add_argument("--dry-run-base", type=Path, nargs="+", default=[],
                    help="base-weight directories (each with a REVISION file) linked in for a decoder member's dry run")
    args = ap.parse_args()
    scales = args.scales or [1.0] * len(args.runs)
    if len(scales) != len(args.runs):
        ap.error("--scales needs one value per run")
    if len(args.runs) > 1 and args.out is None:
        ap.error("--out is required for more than one run")
    manifests = [check_run(run) for run in args.runs]

    out = (args.out or OUTPUTS / "kaggle" / args.runs[0].name).resolve()  # the dry run executes from inside it
    dataset = out / "dataset"
    if dataset.exists():
        shutil.rmtree(dataset)
    members = []
    for i, (run, manifest, scale) in enumerate(zip(args.runs, manifests, scales, strict=True)):
        name = f"m{i}-{run.name}"
        add_member(run, dataset / name)
        member = {"dir": name, "scale": scale, "source_run": str(run), "backbone": manifest["backbone"],
                  "seed": manifest["seed"], "epoch": manifest["epoch"], "code_id": manifest.get("code_id")}
        if is_decoder(run):  # fp16 on the base weights at this revision, which the notebook finds by REVISION
            member |= {"kind": "pooled_decoder" if is_pooled(run) else "decoder",
                       "base_revision": manifest["revision"], "reference": "fp16", "batch_size": 16}
        else:
            member |= {"kind": "encoder", "reference": "fp32", "batch_size": 32}
        members.append(member)
    (dataset / "nllm").mkdir()
    for name in MODULES:
        shutil.copy2(ROOT / "src" / "nllm" / name, dataset / "nllm" / name)
    code_ids = sorted({str(m["code_id"]) for m in members})
    (dataset / "ensemble.json").write_text(json.dumps(
        {"labels": list(LABELS), "combine": "sum over members of fp32 logits / scale, then argmax",
         "centring": {"weight": args.centring[0], "cut": args.centring[1], "cavity": True} if args.centring else None,
         "code_id": ",".join(code_ids), "members": members}, indent=2), encoding="utf-8")
    files = {str(p.relative_to(dataset)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(dataset.rglob("*")) if p.is_file()}
    (dataset / "files.json").write_text(json.dumps({"source_runs": [str(r) for r in args.runs], "files": files},
                                                   indent=2), encoding="utf-8")
    (out / "nllm_infer.ipynb").write_text(json.dumps(notebook(NOTEBOOK_SOURCE.read_text(encoding="utf-8")),
                                                     indent=1), encoding="utf-8")
    size = sum(p.stat().st_size for p in dataset.rglob("*") if p.is_file())
    print(f"wrote {dataset} ({len(members)} member(s), {len(files)} files, {size / 2**30:.2f} GiB) and "
          f"{out / 'nllm_infer.ipynb'}", flush=True)

    if args.dry_run:
        bases = out / "dry-run-bases"
        shutil.rmtree(bases, ignore_errors=True)
        for i, base in enumerate(args.dry_run_base):
            bases.mkdir(exist_ok=True)
            (bases / str(i)).symlink_to(base.resolve(), target_is_directory=True)
        env = os.environ | {"NLLM_KAGGLE_INPUT": str(out), "NLLM_KAGGLE_WORKING": str(out / "working"),
                            "NLLM_STAGE_DIR": str(out / "stage"),  # on the project disk, not the /tmp RAM disk
                            "NLLM_DATA_DIR": str(data_dir()), "CUDA_VISIBLE_DEVICES": ""}
        subprocess.run([sys.executable, str(NOTEBOOK_SOURCE)], env=env, check=True, cwd=out)


if __name__ == "__main__":
    main()
