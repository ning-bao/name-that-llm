"""Experiments 3 and 4: nested-CV fine-tuned encoder / LoRA decoder screening on the 120 development prompts.

For one backbone, each learning rate is fitted on each outer fold's three inner splits,
with logits recorded after every epoch. (lr, epoch) is chosen per outer fold from inner
out-of-fold balanced accuracy, refitted on the 96 outer-training prompts (same schedule,
stopped at the chosen epoch) and scored on the 24 held-out prompts. Every fit's logits are
written as soon as it finishes, so an interrupted run resumes with --out (a new path starts
a run there). --time-budget-minutes stops before a fit that might not finish in time, so a
Slurm job can end cleanly and the next job resumes. The confirmation prompts are never loaded.

Usage: uv run python scripts/encoder_finetune.py --backbone ettin-400m [--partitions 0]
           [--baseline outputs/baseline_linear/<run>] [--out <existing run dir to resume>]
       uv run python scripts/encoder_finetune.py --backbone ettin-400m --pilot
       uv run python scripts/encoder_finetune.py --backbone qwen3-8b-base --micro-batch 4 \
           --out <run dir> --time-budget-minutes 340          # experiment 4 on TC2 (scripts/tc2/)
Writes outputs/encoder/<backbone>/<UTC timestamp>/ unless --out is given.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download

from nllm import encoder
from nllm.data import LABELS, load_train
from nllm.metrics import (
    balanced_accuracy,
    confusion,
    interval,
    per_class_recall,
    prompt_bootstrap,
    within_family_error_share,
)
from nllm.paths import OUTPUTS, data_dir
from nllm.splits import development_rows, initialization_seeds, outer_folds, select

SCORE_COLUMNS = [f"score:{m}" for m in LABELS]
RUN_KEYS = ("backbone", "lrs", "epochs", "seed", "micro_batch", "lora_rank")
OUT_OF_TIME = 75  # exit status when the time budget stops a run before it finishes (EX_TEMPFAIL)


class OutOfTime(Exception):
    pass


def train_config(args, lr: float) -> encoder.TrainConfig:
    lora = {"lora_rank": args.lora_rank} if args.backbone in encoder.DECODERS else {}
    return encoder.TrainConfig(args.backbone, lr, args.epochs, micro_batch=args.micro_batch,
                               gradient_checkpointing=args.gradient_checkpointing, **lora)


def download(repo: str, attempts: int = 5) -> str:
    """Local snapshot path. Retries, since large downloads on the cluster can drop mid-file (they resume)."""
    for attempt in range(1, attempts + 1):
        try:
            return snapshot_download(repo, allow_patterns=["*.json", "*.txt", "*.safetensors", "*.bin"])
        except Exception as e:  # network errors come from several libraries
            if attempt == attempts:
                raise
            print(f"download attempt {attempt} failed ({type(e).__name__}: {e}); retrying", flush=True)
            time.sleep(30 * attempt)


def cached_fit(path: Path, model_path, tr, va, cfg, seed, eval_epochs, stop_at=None) -> dict[int, np.ndarray]:
    """Run one fit unless its logits are already on disk; returns {epoch: logits for va rows}.

    Raises OutOfTime instead of starting a fit after `stop_at` (a time.time() value).
    """
    if not path.exists():
        if stop_at is not None and time.time() > stop_at:
            raise OutOfTime(path.stem)
        started = time.time()
        logits, history = encoder.fit_predict(model_path, tr.text, tr.model, va.text, cfg, seed, eval_epochs)
        epochs = sorted(logits)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(
                f,
                epochs=np.array(epochs),
                logits=np.stack([logits[e] for e in epochs]),
                ids=va.id.to_numpy(dtype=str),
                history=np.array(json.dumps(history)),
            )
        os.replace(tmp, path)
        print(f"  {path.stem}: {time.time() - started:.0f}s, peak {history[-1]['peak_gb']:.1f} GB", flush=True)
    with np.load(path) as z:
        if not (z["ids"] == va.id.to_numpy(dtype=str)).all():
            raise RuntimeError(f"{path} was written for different rows")
        return {int(e): z["logits"][i] for i, e in enumerate(z["epochs"])}


def histories(fits_dir: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(fits_dir.glob("*.npz")):
        with np.load(path) as z:
            rows += [{"fit": path.stem, **h} for h in json.loads(str(z["history"]))]
    return pd.DataFrame(rows)


def pilot(args, dev, model_path, seed) -> None:
    """One inner fit (partition 0, outer fold 0, inner fold 0) at the largest lr: timing, memory, sanity.

    With --out, also writes pilot.npz (per-epoch logits for the inner-val rows) and pilot.json, and with
    --save-model the final model under model/ (the reference for the Kaggle fp16 agreement check).
    """
    fold = outer_folds(0)[0]
    inner_train, inner_val = next(fold.inner())
    tr, va = select(dev, inner_train), select(dev, inner_val)
    cfg = train_config(args, max(args.lrs))
    save_dir = args.out / "model" if args.out and args.save_model else None
    started = time.time()
    logits, history = encoder.fit_predict(model_path, tr.text, tr.model, va.text, cfg, seed, save_dir=save_dir)
    for h in history:
        h["inner_val_ba"] = balanced_accuracy(va.model, np.asarray(LABELS)[logits[h["epoch"]].argmax(axis=1)])
        print(f"epoch {h['epoch']}: loss {h['train_loss']:.3f}, {h['train_seconds']:.0f}s, "
              f"inner-val BA {h['inner_val_ba']:.3f}")
    print(f"{len(tr)} train / {len(va)} eval rows, {time.time() - started:.0f}s, peak {history[-1]['peak_gb']:.1f} GB")
    if args.out:
        epochs = sorted(logits)
        np.savez(args.out / "pilot.npz", epochs=np.array(epochs), logits=np.stack([logits[e] for e in epochs]),
                 ids=va.id.to_numpy(dtype=str), history=np.array(json.dumps(history)))
        (args.out / "pilot.json").write_text(json.dumps({
            "train_config": asdict(cfg), "seed": seed, "revision": Path(model_path).name,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch": torch.__version__, "transformers": transformers.__version__, "history": history,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, choices=sorted(encoder.BACKBONES))
    ap.add_argument("--lrs", type=float, nargs="+", help="default: LEARNING_RATES, or DECODER_LEARNING_RATES")
    ap.add_argument("--epochs", type=int, help="default: EPOCHS, or DECODER_EPOCHS")
    ap.add_argument("--partitions", type=int, nargs="+", default=[0])
    ap.add_argument("--seed", type=int, default=initialization_seeds()[0])
    # 16 peaks near 12 GB on roberta-large, which spills into shared memory on a 16 GB WDDM card.
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--lora-rank", type=int, default=encoder.TrainConfig.lora_rank, help="decoders only")
    ap.add_argument("--gradient-checkpointing", action="store_true")
    ap.add_argument("--baseline", type=Path, help="baseline run dir with oof.parquet, for paired differences")
    ap.add_argument("--baseline-branch", default="text+style")
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, help="run directory: resumed if it exists, otherwise created")
    ap.add_argument("--time-budget-minutes", type=float,
                    help="start no new fit after this many minutes minus --fit-minutes; exits 75 to be resumed")
    ap.add_argument("--fit-minutes", type=float, default=35, help="time reserved for the longest single fit")
    ap.add_argument("--pilot", action="store_true", help="time one inner fit and exit")
    ap.add_argument("--save-model", action="store_true", help="with --pilot and --out: also save the fitted model")
    args = ap.parse_args()
    decoder = args.backbone in encoder.DECODERS
    args.lrs = args.lrs or list(encoder.DECODER_LEARNING_RATES if decoder else encoder.LEARNING_RATES)
    args.epochs = args.epochs or (encoder.DECODER_EPOCHS if decoder else encoder.EPOCHS)
    args.lora_rank = args.lora_rank if decoder else None
    transformers.logging.set_verbosity_error()  # the per-fit head-initialisation load report

    started = time.time()
    stop_at = started + 60 * (args.time_budget_minutes - args.fit_minutes) if args.time_budget_minutes else None
    dev = development_rows(load_train())
    model_path = download(encoder.BACKBONES[args.backbone])
    if args.pilot:
        if args.out:
            args.out.mkdir(parents=True)
        pilot(args, dev, model_path, args.seed)
        return

    run = {k: getattr(args, k) for k in RUN_KEYS} | {"revision": Path(model_path).name}
    out = args.out or OUTPUTS / "encoder" / args.backbone / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if (out / "run.json").exists():
        saved = json.loads((out / "run.json").read_text(encoding="utf-8"))
        if {k: saved.get(k) for k in run} != run:  # runs started before lora_rank existed saved no key for it
            raise SystemExit(f"--out was started with different settings: {saved}")
    else:
        (out / "fits").mkdir(parents=True)
        (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    folds = [f for p in args.partitions for f in outer_folds(p)]
    print(f"{args.backbone} @ {run['revision'][:10]}: {len(dev)} dev rows, lrs {args.lrs}, "
          f"{args.epochs} epochs, {len(folds)} outer folds -> {out}", flush=True)

    try:
        fit_and_report(args, dev, model_path, folds, out, run, started, stop_at)
    except OutOfTime as e:
        done = len(list((out / "fits").glob("*.npz")))
        print(f"time budget reached before {e}; {done} fits on disk. Resume with --out {out}", flush=True)
        raise SystemExit(OUT_OF_TIME) from None


def fit_and_report(args, dev, model_path, folds, out, run, started, stop_at) -> None:
    inner_parts, selection_tables, oof_parts, selections = [], [], [], []
    for fold in folds:
        p, o = fold.partition, fold.outer
        outer_train = select(dev, fold.train)
        inner_logits = {}
        for k, (inner_train, inner_val) in enumerate(fold.inner()):
            tr, va = select(dev, inner_train), select(dev, inner_val)
            for lr in args.lrs:
                cfg = train_config(args, lr)
                path = out / "fits" / f"inner_p{p}_o{o}_k{k}_lr{lr:g}.npz"
                for epoch, s in cached_fit(path, model_path, tr, va, cfg, args.seed, None, stop_at).items():
                    inner_logits.setdefault((lr, epoch), pd.DataFrame(np.nan, index=outer_train.index, columns=LABELS))
                    inner_logits[lr, epoch].loc[va.index] = s

        # Selection from inner out-of-fold logits only.
        arrays = {key: frame.to_numpy() for key, frame in inner_logits.items()}
        (lr, epoch), table = encoder.select(arrays, outer_train.model.to_numpy())
        selection_tables.append(table.assign(partition=p, outer_fold=o))
        selections.append({"partition": p, "outer_fold": o, "lr": lr, "epoch": epoch})
        meta = outer_train[["id", "prompt_id", "model"]].assign(partition=p, outer_fold=o)
        for (key_lr, key_ep), frame in inner_logits.items():
            scores = pd.DataFrame(frame.to_numpy(), columns=SCORE_COLUMNS, index=frame.index)
            inner_parts.append(pd.concat([meta.assign(lr=key_lr, epoch=key_ep), scores], axis=1))

        # Outer refit on the 96 training prompts with the chosen schedule, stopped at the chosen epoch.
        va = select(dev, fold.validation)
        cfg = train_config(args, lr)
        path = out / "fits" / f"outer_p{p}_o{o}_lr{lr:g}_ep{epoch}.npz"
        scores = cached_fit(path, model_path, outer_train, va, cfg, args.seed, [epoch], stop_at)[epoch]
        meta = va[["id", "prompt_id", "model"]].reset_index(drop=True)
        meta = meta.assign(partition=p, outer_fold=o, branch=args.backbone, pred=np.asarray(LABELS)[scores.argmax(axis=1)])
        oof_parts.append(pd.concat([meta, pd.DataFrame(scores, columns=SCORE_COLUMNS)], axis=1))
        fold_ba = balanced_accuracy(meta.model, meta.pred)
        print(f"p{p} o{o}: chose lr={lr:g} epoch={epoch} (inner BA {table.inner_ba.max():.3f}), "
              f"outer BA {fold_ba:.3f}  [{(time.time() - started) / 60:.0f} min]", flush=True)

    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_parquet(out / "oof.parquet", index=False)
    pd.concat(inner_parts, ignore_index=True).to_parquet(out / "inner_oof.parquet", index=False)
    pd.concat(selection_tables).to_csv(out / "inner_scores.csv", index=False)
    selections = pd.DataFrame(selections)
    selections.to_csv(out / "selections.csv", index=False)
    histories(out / "fits").to_csv(out / "history.csv", index=False)

    baseline = None
    if args.baseline:
        baseline = pd.read_parquet(args.baseline / "oof.parquet")
        baseline = baseline[baseline.branch == args.baseline_branch]
    results, recalls = [], {}
    boot = {"n_draws": args.bootstrap_draws, "seed": args.seed}
    for p in args.partitions:
        rows = oof[oof.partition == p].set_index("id")
        lo, hi = interval(prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, **boot))
        result = {
            "partition": p,
            "branch": args.backbone,
            "balanced_accuracy": balanced_accuracy(rows.model, rows.pred),
            "ci95_lo": lo,
            "ci95_hi": hi,
            "within_family_error_share": within_family_error_share(rows.model, rows.pred),
        }
        if baseline is not None:
            ref = baseline[baseline.partition == p].set_index("id").loc[rows.index]
            diff = prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, y_pred_baseline=ref.pred, **boot)
            dlo, dhi = interval(diff)
            result |= {
                "baseline_ba": balanced_accuracy(ref.model, ref.pred),
                "diff_vs_baseline": float(diff.mean()),
                "diff_ci95_lo": dlo,
                "diff_ci95_hi": dhi,
            }
        results.append(result)
        recalls[f"p{p}:{args.backbone}"] = per_class_recall(rows.model, rows.pred)
        confusion(rows.model, rows.pred).round(3).to_csv(out / f"confusion_p{p}.csv")
    results = pd.DataFrame(results)
    results.to_csv(out / "results.csv", index=False)
    recall_table = pd.DataFrame(recalls)
    recall_table.to_csv(out / "per_class_recall.csv")

    (out / "config.json").write_text(json.dumps({
        "experiment": "4: nested-CV LoRA decoder screening" if args.backbone in encoder.DECODERS
        else "3: nested-CV fine-tuned encoder screening",
        "model_repo": encoder.BACKBONES[args.backbone],
        **run,
        "partitions": args.partitions,
        "train_config": asdict(train_config(args, args.lrs[0])),
        "selection": "per outer fold, max BA of (lr, epoch) over concatenated inner out-of-fold logits; ties -> fewer epochs, smaller lr",
        "baseline": f"{args.baseline}:{args.baseline_branch}" if args.baseline else None,
        "train_sha256": hashlib.sha256((data_dir() / "train.csv").read_bytes()).hexdigest(),
        "label_order": LABELS,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "runtime_seconds_this_session": round(time.time() - started, 1),
    }, indent=2), encoding="utf-8")

    pd.set_option("display.width", 200)
    print("\nOuter-fold balanced accuracy over 120 dev prompts (95% prompt-bootstrap CI):")
    print(results.round(4).to_string(index=False))
    print("\nSelections:")
    print(selections.to_string(index=False))
    print("\nPer-class recall:")
    print(recall_table.round(3).to_string())
    print(f"\nwrote {out}  ({time.time() - started:.0f}s)")


if __name__ == "__main__":
    main()
