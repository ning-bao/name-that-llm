"""Experiments 3 and 4: nested-CV fine-tuned encoder / LoRA decoder screening on the 120 development prompts.

For one backbone, each learning rate is fitted on each outer fold's three inner splits,
with logits recorded after every epoch. (lr, epoch) is chosen per outer fold from inner
out-of-fold balanced accuracy, refitted on the 96 outer-training prompts (same schedule,
stopped at the chosen epoch) and scored on the 24 held-out prompts. Every fit's logits are
written as soon as it finishes, so an interrupted run resumes with --out (a new path starts
a run there). --time-budget-minutes stops before a fit that might not finish in time, so a
Slurm job can end cleanly and the next job resumes. The confirmation prompts are never loaded.

With --pool-per-label N, each outer fold first trains a pool stage 1: a class-balanced
sample of N community-pool rows per label (nllm.pool.training_sample, recipe --pool-recipe), with match
weights and source multipliers fitted on that fold's 96 official training prompts only. The stage-1 model
is saved under stage1/ and scored on the held-out prompts (oof_stage1.parquet, a pool-only result); the
inner and outer fits then start from it instead of the pretrained backbone (decoders: they keep
training the stage-1 adapters and head on the same frozen base weights). Inner fits reuse the outer
fold's stage 1, whose weights saw the inner-validation prompts' marker rates: a small leak into the
(lr, epoch) choice only, never into outer scores.

With --fixed-epoch N there are no inner fits: every outer fold is refitted at the single --lrs value and stopped
at epoch N, a schedule fixed in advance (a screen: nothing is selected, so nothing can be selected on outer rows).
With --sibling-units S (and --centred-weight), stage 2's same-prompt units favour labels that the stage-1 model
confuses on the fit's own training rows (nllm.encoder.unit_batches). Fits with a centred head also save its
outputs on the held-out rows (oof.parquet columns centred:<label> and centred_bias:<label>), from which
scripts/twin_eval.py scores the public pipelines' twin rule.

With --final there is no cross-validation: one model is trained on all 120 development prompts with a fixed
(lr, epoch), after the same pool stage 1, and saved under model/ with reference.npz and manifest.json for the
Kaggle notebook (scripts/kaggle_package.py). The schedule comes from the nested runs' selections.

Usage: uv run python scripts/encoder_finetune.py --backbone ettin-400m [--partitions 0]
           [--baseline outputs/baseline_linear/<run>] [--out <existing run dir to resume>]
       uv run python scripts/encoder_finetune.py --backbone ettin-400m --pilot
       uv run python scripts/encoder_finetune.py --backbone qwen3-8b-base --micro-batch 4 \
           --out <run dir> --time-budget-minutes 340          # experiment 4 on TC2 (scripts/tc2/)
       uv run python scripts/encoder_finetune.py --backbone ettin-400m --pool-per-label 2500 --epochs 6
                                                              # pool stage 1, then official stage 2
       uv run python scripts/encoder_finetune.py --backbone ettin-400m --final --pool-per-label 2500 \
           --pool-recipe w --pool-epochs 2 --lrs 1e-5 --epochs 6 --final-epoch 3   # model for Kaggle
Writes outputs/encoder/<backbone>/<UTC timestamp>/ unless --out is given.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from huggingface_hub import snapshot_download
from peft import PeftModel

from nllm import encoder, inference, pool
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
from nllm.splits import (
    confirmation_prompts,
    development_prompts,
    development_rows,
    initialization_seeds,
    outer_folds,
    select,
)

SCORE_COLUMNS = [f"score:{m}" for m in LABELS]
CENTRED_COLUMNS = [f"centred:{m}" for m in LABELS]
CENTRED_BIAS_COLUMNS = [f"centred_bias:{m}" for m in LABELS]
RUN_KEYS = ("backbone", "lrs", "epochs", "seed", "micro_batch", "lora_rank", "pool_per_label", "pool_recipe",
            "pool_epochs", "pool_lr", "optimizer",
            # Added for the LoRA24 recipe; None in runs started before them, so those still resume.
            "pool_layers", "keep_layers", "head_lr_mult", "centred_weight", "schedule", "batch_size",
            "pool_class_weighted", "pool_replace", "shared_stage1",
            # Reusing or continuing a saved stage 1 (--stage1-from, --stage1-init); pool_seed also offsets the
            # shared stage 1's training seed. Runs started before them saved none, i.e. RUN_DEFAULTS.
            "stage1_from", "stage1_init", "pool_seed",
            # Sibling-focused stage 2 and the no-selection screen; None in runs started before them.
            "sibling_units", "fixed_epoch")
RUN_DEFAULTS = {"pool_seed": 0}
OUT_OF_TIME = 75  # exit status when the time budget stops a run before it finishes (EX_TEMPFAIL)


class OutOfTime(Exception):
    pass


def opt(args, name: str):
    """An option added for the LoRA24 recipe; None when `args` predates it (tests build args by hand)."""
    return getattr(args, name, None)


def train_config(args, lr: float) -> encoder.TrainConfig:
    lora = {"lora_rank": args.lora_rank} if args.backbone in encoder.DECODERS else {}
    layers = opt(args, "pool_layers")
    pooled = {"pool_layers": tuple(layers), "keep_layers": opt(args, "keep_layers")} if layers else {}
    optional = {k: opt(args, k) for k in ("head_lr_mult", "centred_weight", "schedule", "batch_size", "sibling_units")
                if opt(args, k) is not None}
    return encoder.TrainConfig(args.backbone, lr, args.epochs, micro_batch=args.micro_batch,
                               gradient_checkpointing=args.gradient_checkpointing,
                               optimizer=args.optimizer or "adamw", **lora, **pooled, **optional)


def stage1_config(args) -> encoder.TrainConfig:
    # Stage 1 starts from the pretrained backbone, which has no confusions to read yet: its units stay random.
    return replace(train_config(args, args.pool_lr), epochs=args.pool_epochs,
                   class_weighted=bool(opt(args, "pool_class_weighted")), sibling_units=0.0)


def stage1_source(args, src: str, nested: bool) -> dict:
    """What trained the saved stage 1 at `src`, checked against this run: a --final run's out/stage1 (described
    by its manifest.json) or a nested run's shared stage 1, out/stage1/shared (its run.json). The backbone must
    match (the adapters' shapes and pooled depths are checked when they load). For a nested run (`nested`) it must also have read no official rows, so it must come
    from a uniform pool recipe (a per-fold stage 1 never qualifies). For --stage1-from, which trains no stage 1,
    this run's pool settings must equal the source's, so they describe the model that is actually used.
    """
    d = Path(src)
    if (d.parent / "manifest.json").exists() and d.name == "stage1":
        saved = json.loads((d.parent / "manifest.json").read_text(encoding="utf-8"))
    elif (d.parent.parent / "run.json").exists() and d.name == "shared":
        saved = json.loads((d.parent.parent / "run.json").read_text(encoding="utf-8"))
        if not saved.get("shared_stage1"):
            raise SystemExit(f"{src}: its run did not use --shared-stage1")
    else:
        raise SystemExit(f"{src} is neither a --final run's stage1/ nor a nested run's stage1/shared/")
    if not (d / "adapter_config.json").exists() and not (d / "config.json").exists():
        raise SystemExit(f"{src} holds no saved model")
    if saved.get("backbone") != args.backbone:
        raise SystemExit(f"{src} is a {saved.get('backbone')} stage 1, not {args.backbone}")
    if nested and pool.RECIPES[saved["pool_recipe"]].weighting != "none":
        raise SystemExit(f"{src} used pool recipe {saved['pool_recipe']}, which reads official rows")
    if src == opt(args, "stage1_from"):
        keys = ["pool_recipe", "pool_epochs", "pool_lr", "pool_class_weighted", "pool_replace"]
        keys += ["pool_per_label"] if "pool_per_label" in saved else []
        differ = {k: (opt(args, k), saved.get(k)) for k in keys
                  if bool(opt(args, k)) != bool(saved.get(k)) or (opt(args, k) and opt(args, k) != saved.get(k))}
        if differ:
            raise SystemExit(f"--stage1-from {src} was trained with other pool settings (ours, its): {differ}")
    return {k: saved.get(k) for k in ("backbone", "revision", "pool_recipe", "pool_per_label", "pool_epochs",
                                      "pool_lr", "pool_seed", "pool_class_weighted", "pool_replace", "pool_rows",
                                      "stage1_from", "stage1_init")}


def groups_for(cfg: encoder.TrainConfig, rows: pd.DataFrame, column: str):
    """rows[column] (prompt groups) for fit_predict when the centred loss needs them, else None."""
    return rows[column].to_numpy() if cfg.centred_weight > 0 else None


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


def cached_fit(path: Path, model_path, tr, va, cfg, seed, eval_epochs, stop_at=None,
               init_from=None) -> dict[int, np.ndarray]:
    """Run one fit unless its logits are already on disk; returns {epoch: logits for va rows}.

    Raises OutOfTime instead of starting a fit after `stop_at` (a time.time() value).
    """
    if not path.exists():
        if stop_at is not None and time.time() > stop_at:
            raise OutOfTime(path.stem)
        started = time.time()
        centred = {}
        logits, history = encoder.fit_predict(model_path, tr.text, tr.model, va.text, cfg, seed, eval_epochs,
                                              init_from=init_from, train_groups=groups_for(cfg, tr, "prompt_id"),
                                              twin_out=centred)
        epochs = sorted(logits)
        extra = {"centred": np.stack([centred[e][0] for e in epochs]),
                 "centred_bias": np.stack([centred[e][1] for e in epochs])} if centred else {}
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(
                f,
                epochs=np.array(epochs),
                logits=np.stack([logits[e] for e in epochs]),
                ids=va.id.to_numpy(dtype=str),
                history=np.array(json.dumps(history)),
                **extra,
            )
        os.replace(tmp, path)
        print(f"  {path.stem}: {time.time() - started:.0f}s, peak {history[-1]['peak_gb']:.1f} GB", flush=True)
    with np.load(path) as z:
        if not (z["ids"] == va.id.to_numpy(dtype=str)).all():
            raise RuntimeError(f"{path} was written for different rows")
        return {int(e): z["logits"][i] for i, e in enumerate(z["epochs"])}


def centred_columns(path: Path, epoch: int) -> pd.DataFrame | None:
    """A fit's saved centred-head outputs at `epoch`, one row per held-out row (CENTRED_COLUMNS, and the head's bias
    repeated in CENTRED_BIAS_COLUMNS); None for a fit without a centred head or saved before they were kept."""
    with np.load(path) as z:
        if "centred" not in z:
            return None
        i = list(z["epochs"]).index(epoch)
        bias = np.tile(z["centred_bias"][i], (len(z["ids"]), 1))
        return pd.DataFrame(np.hstack([z["centred"][i], bias]), columns=CENTRED_COLUMNS + CENTRED_BIAS_COLUMNS)


def stage1(args, model_path, fold, outer_train, va, out, stop_at) -> tuple[str, np.ndarray]:
    """Pool stage 1 for one outer fold; returns the saved model's path and its logits on the held-out rows."""
    p, o = fold.partition, fold.outer
    model_dir, path = out / "stage1" / f"p{p}_o{o}", out / "fits" / f"stage1_p{p}_o{o}.npz"
    if not path.exists():
        if stop_at is not None and time.time() > stop_at:
            raise OutOfTime(path.stem)
        started = time.time()
        sample, report = pool.training_sample(args.pool_recipe, outer_train, args.pool_per_label, 1000 * p + o,
                                              replace=bool(opt(args, "pool_replace")))
        model_dir.mkdir(parents=True, exist_ok=True)
        sample[["id", "model", "source", "origin", "kb_proxy"]].to_parquet(model_dir.parent / f"p{p}_o{o}_sample.parquet")
        if report is not None:
            report.to_csv(model_dir.parent / f"p{p}_o{o}_match_weights.csv", index=False)
        cfg = stage1_config(args)
        logits, history = encoder.fit_predict(model_path, sample.text, sample.model, va.text, cfg, args.seed,
                                              save_dir=model_dir, train_groups=groups_for(cfg, sample, "group"))
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, epochs=np.array([args.pool_epochs]), logits=logits[args.pool_epochs][None],
                     ids=va.id.to_numpy(dtype=str), history=np.array(json.dumps(history)))
        os.replace(tmp, path)
        print(f"  {path.stem}: {len(sample)} pool rows, {time.time() - started:.0f}s, "
              f"peak {history[-1]['peak_gb']:.1f} GB", flush=True)
    with np.load(path) as z:
        if not (z["ids"] == va.id.to_numpy(dtype=str)).all():
            raise RuntimeError(f"{path} was written for different rows")
        return str(model_dir), z["logits"][0]


def shared_stage1(args, model_path, dev, out, stop_at) -> tuple[str, pd.DataFrame]:
    """One pool stage 1 for every outer fold (--shared-stage1); returns the saved model's path and its logits on
    all development rows, indexed by id.

    The recipe must be uniform (nllm.pool.training_sample refuses otherwise), so nothing about the sample or the
    fit depends on official rows: every development prompt is unseen by this stage 1, and each fold's stage 1
    scores (oof_stage1.parquet) are held-out scores as with a per-fold stage 1.
    """
    reuse, init = opt(args, "stage1_from"), opt(args, "stage1_init")
    model_dir, path = Path(reuse) if reuse else out / "stage1" / "shared", out / "fits" / "stage1_shared.npz"
    if not path.exists():
        if stop_at is not None and time.time() > stop_at:
            raise OutOfTime(path.stem)
        started = time.time()
        cfg = stage1_config(args)
        if reuse:  # --stage1-from: no stage-1 training, only its scores on the dev rows (for oof_stage1)
            stage1_source(args, reuse, nested=True)
            logits, history = {args.pool_epochs: encoder.predict_saved(model_path, cfg, reuse, dev.text)}, []
            print(f"  {path.stem}: reused {reuse}, {time.time() - started:.0f}s", flush=True)
        else:
            if init:  # --stage1-init: continue training a saved stage 1
                stage1_source(args, init, nested=True)
            sample, _ = pool.training_sample(args.pool_recipe, None, args.pool_per_label, args.pool_seed,
                                             replace=bool(opt(args, "pool_replace")))
            model_dir.mkdir(parents=True, exist_ok=True)
            sample[["id", "model", "source", "origin", "kb_proxy"]].to_parquet(
                model_dir.parent / "shared_sample.parquet")
            logits, history = encoder.fit_predict(model_path, sample.text, sample.model, dev.text, cfg,
                                                  args.seed + args.pool_seed, save_dir=model_dir, init_from=init,
                                                  train_groups=groups_for(cfg, sample, "group"))
            print(f"  {path.stem}: {len(sample)} pool rows ({sample.model.value_counts().min()}-"
                  f"{sample.model.value_counts().max()} per label){f', from {init}' if init else ''}, "
                  f"{time.time() - started:.0f}s, peak {history[-1]['peak_gb']:.1f} GB", flush=True)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "wb") as f:
            np.savez(f, epochs=np.array([args.pool_epochs]), logits=logits[args.pool_epochs][None],
                     ids=dev.id.to_numpy(dtype=str), history=np.array(json.dumps(history)))
        os.replace(tmp, path)
    with np.load(path) as z:
        if not (z["ids"] == dev.id.to_numpy(dtype=str)).all():
            raise RuntimeError(f"{path} was written for different rows")
        return str(model_dir), pd.DataFrame(z["logits"][0], index=z["ids"], columns=LABELS)


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
    logits, history = encoder.fit_predict(model_path, tr.text, tr.model, va.text, cfg, seed, save_dir=save_dir,
                                          train_groups=groups_for(cfg, tr, "prompt_id"))
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


def decoder_reference(model_dir: Path, model_path: str, cfg, texts) -> tuple[int, dict[str, np.ndarray]]:
    """(pad id, reference logits) from a decoder's saved adapters and head, merged into a fresh copy of the base
    weights: in bf16 as trained, and in fp16, which is what a Kaggle T4 runs (it has no bf16)."""
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    ids = encoder.encode(tok, texts, cfg)
    pad_id = tok.convert_tokens_to_ids(encoder.DECODER_PAD_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        base = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=len(LABELS), attn_implementation="sdpa", dtype=dtype, pad_token_id=pad_id)
        model = PeftModel.from_pretrained(base, str(model_dir)).merge_and_unload().to(device).eval()
        scores[name] = inference.predict_logits(model, ids, pad_id, dtype=dtype)
        del model, base
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pad_id, scores


def pooled_reference(model_dir: Path, model_path: str, cfg, texts) -> tuple[int, dict[str, np.ndarray]]:
    """(pad id, reference logits) for a pooled decoder, through the Kaggle loader (nllm.inference: adapters merged
    by hand into a fresh truncated base, no peft): in bf16 as trained, and in fp16 as a T4 runs it."""
    tok = transformers.AutoTokenizer.from_pretrained(model_dir)
    ids = encoder.encode(tok, texts, cfg)
    if ids != inference.tokenize_decoder(model_dir, texts, cfg.max_length):
        raise RuntimeError("nllm.inference.tokenize_decoder differs from the training tokenizer")
    pad_id = tok.convert_tokens_to_ids(encoder.DECODER_PAD_TOKEN)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scores = {}
    for name, dtype in (("bf16", torch.bfloat16), ("fp16", torch.float16)):
        model = inference.load_pooled_decoder(model_path, model_dir, dtype=dtype,
                                              device_map={"": device} if device.type == "cuda" else None)
        scores[name] = inference.predict_logits(model, ids, pad_id, dtype=dtype)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return pad_id, scores


def final(args, dev, model_path, out, started) -> None:
    """One model on all development rows with a fixed schedule, saved under out/model for Kaggle inference.

    Stage 2 uses the single --lrs value on the --epochs schedule, stopped at --final-epoch like an outer
    refit. Reference rows (4 per label, training rows) are scored from the saved files: encoders with
    nllm.inference under bf16 autocast and in fp32 (the Kaggle notebook checks its own scores against the fp32
    ones); decoders, whose out/model holds only the adapters and head, with decoder_reference in bf16 and fp16.
    reference.npz also keeps the in-fit logits.
    """
    if (out / "manifest.json").exists():
        print(f"{out} is already finished")
        return
    out.mkdir(parents=True, exist_ok=True)
    if args.limit_rows:
        dev = dev.groupby("model", group_keys=False).head(args.limit_rows)
    ref = dev.groupby("model", group_keys=False).sample(min(4, dev.model.value_counts().min()), random_state=0)
    ref = ref.sort_values("id")
    manifest = {"backbone": args.backbone, "model_repo": encoder.BACKBONES[args.backbone],
                "revision": Path(model_path).name, "labels": list(LABELS), "train_rows": len(dev),
                "train_prompts": dev.prompt_id.nunique(), "lr": args.lrs[0], "schedule_epochs": args.epochs,
                "epoch": args.final_epoch, "seed": args.seed, "limit_rows": args.limit_rows}
    init = None
    reuse = opt(args, "stage1_from")
    if reuse:  # --stage1-from: stage 2 starts from that saved stage 1, which is never deleted
        init = Path(reuse)
        manifest |= {"stage1_from": reuse, "stage1_source": stage1_source(args, reuse, nested=False),
                     "pool_recipe": args.pool_recipe, "pool_epochs": args.pool_epochs, "pool_lr": args.pool_lr,
                     "pool_class_weighted": bool(opt(args, "pool_class_weighted")),
                     "pool_replace": bool(opt(args, "pool_replace"))}
    elif args.pool_per_label:
        per_label = args.limit_rows or args.pool_per_label
        sample, report = pool.training_sample(args.pool_recipe, dev, per_label, args.pool_seed,
                                              replace=bool(opt(args, "pool_replace")))
        sample[["id", "model", "source", "origin", "kb_proxy"]].to_parquet(out / "stage1_sample.parquet")
        if report is not None:
            report.to_csv(out / "stage1_match_weights.csv", index=False)
        cfg1 = stage1_config(args)
        init, start = out / "stage1", opt(args, "stage1_init")
        if start:
            manifest |= {"stage1_init": start, "stage1_source": stage1_source(args, start, nested=False)}
        _, history = encoder.fit_predict(model_path, sample.text, sample.model, ref.text, cfg1,
                                         args.seed + args.pool_seed, save_dir=init, init_from=start,
                                         train_groups=groups_for(cfg1, sample, "group"))
        manifest |= {"pool_class_weighted": bool(opt(args, "pool_class_weighted")),
                     "pool_replace": bool(opt(args, "pool_replace"))}
        manifest |= {"pool_recipe": args.pool_recipe, "pool_rows": len(sample), "pool_epochs": args.pool_epochs,
                     "pool_lr": args.pool_lr, "pool_seed": args.pool_seed, "stage1_history": history}
        print(f"  stage 1: {len(sample)} pool rows, {time.time() - started:.0f}s, "
              f"peak {history[-1]['peak_gb']:.1f} GB", flush=True)
    cfg = train_config(args, args.lrs[0])
    fit_logits, history = encoder.fit_predict(model_path, dev.text, dev.model, ref.text, cfg, args.seed,
                                              [args.final_epoch], save_dir=out / "model",
                                              init_from=str(init) if init else None,
                                              train_groups=groups_for(cfg, dev, "prompt_id"))
    print(f"  stage 2: {len(dev)} rows, {time.time() - started:.0f}s, peak {history[-1]['peak_gb']:.1f} GB",
          flush=True)

    if cfg.pool_layers:
        pad_id, scores = pooled_reference(out / "model", model_path, cfg, ref.text)
        check = ("fp16", "bf16")
        manifest |= {"pooled": json.loads((out / "model" / inference.POOLED_CONFIG).read_text(encoding="utf-8"))}
    elif args.backbone in encoder.DECODERS:
        pad_id, scores = decoder_reference(out / "model", model_path, cfg, ref.text)
        check = ("fp16", "bf16")
    else:
        # Reload from the saved files the way the notebook does, and check the tokenizer path.
        tok = transformers.AutoTokenizer.from_pretrained(out / "model")
        ids = inference.tokenize(out / "model", ref.text, cfg.max_length)
        if ids != encoder.encode(tok, ref.text, cfg):
            raise RuntimeError("nllm.inference.tokenize differs from the training tokenizer")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = inference.load(out / "model", device)
        pad_id = tok.pad_token_id
        scores = {"bf16": inference.predict_logits(model, ids, pad_id),
                  "fp32": inference.predict_logits(model, ids, pad_id, dtype=None)}
        check = ("fp32", "bf16")
    a, b = scores[check[0]], scores[check[1]]
    agree = float((a.argmax(axis=1) == b.argmax(axis=1)).mean())
    print(f"  reference rows: {check[0]} vs {check[1]} argmax agreement {agree:.3f}, max |diff| "
          f"{np.abs(a - b).max():.3f}; reloaded vs in-fit max |diff| "
          f"{np.abs(scores['bf16'] - fit_logits[args.final_epoch]).max():.4f}", flush=True)
    np.savez(out / "reference.npz", ids=ref.id.to_numpy(dtype=str), in_fit=fit_logits[args.final_epoch], **scores)
    code_id = Path("CODE_ID")
    manifest |= {
        "train_config": asdict(cfg), "history": history, "pad_id": pad_id, "max_length": cfg.max_length,
        "reference_rows": len(ref), f"reference_{check[0]}_{check[1]}_agreement": agree,
        "train_sha256": hashlib.sha256((data_dir() / "train.csv").read_bytes()).hexdigest(),
        "code_id": code_id.read_text().strip() if code_id.exists() else None,
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "runtime_seconds": round(time.time() - started, 1),
    }
    if init and not args.keep_stage1 and not reuse:
        shutil.rmtree(init, ignore_errors=True)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {out}")


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
    ap.add_argument("--optimizer", choices=["adamw", "adafactor"],
                    help="default adamw; adafactor where AdamW states do not fit (ettin-1b on 16 GB)")
    ap.add_argument("--baseline", type=Path, help="baseline run dir with oof.parquet, for paired differences")
    ap.add_argument("--baseline-branch", default="text+style")
    ap.add_argument("--bootstrap-draws", type=int, default=2000)
    ap.add_argument("--out", type=Path, help="run directory: resumed if it exists, otherwise created")
    ap.add_argument("--time-budget-minutes", type=float,
                    help="start no new fit after this many minutes minus --fit-minutes; exits 75 to be resumed")
    ap.add_argument("--fit-minutes", type=float, default=35, help="time reserved for the longest single fit")
    ap.add_argument("--stage1-minutes", type=float, help="time reserved for a pool stage-1 fit (default --fit-minutes)")
    ap.add_argument("--pilot", action="store_true", help="time one inner fit and exit")
    ap.add_argument("--save-model", action="store_true", help="with --pilot and --out: also save the fitted model")
    ap.add_argument("--pool-per-label", type=int, help="train a pool stage 1 per outer fold on this many rows per label")
    ap.add_argument("--pool-recipe", choices=sorted(pool.RECIPES), help="default wq_own (with --pool-per-label)")
    ap.add_argument("--pool-epochs", type=int, help="stage-1 epochs, default 2 (with --pool-per-label)")
    ap.add_argument("--pool-lr", type=float, help="stage-1 learning rate, default 3e-5 (with --pool-per-label)")
    ap.add_argument("--keep-stage1", action="store_true", help="keep each fold's stage-1 model after its fits are done")
    ap.add_argument("--final", action="store_true", help="train one model on all dev rows and save it (no CV)")
    ap.add_argument("--final-epoch", type=int, help="with --final: stop the --epochs schedule after this epoch")
    ap.add_argument("--pool-seed", type=int, default=0,
                    help="final and shared stage 1: seed of the pool sample, also added to the stage-1 training seed")
    ap.add_argument("--limit-rows", type=int, help="with --final: N dev and N pool rows per label (a smoke test)")
    ap.add_argument("--all-prompts", action="store_true",
                    help="with --final: train on all 150 prompts, the 30 confirmation prompts included. Only after "
                         "their single use on the frozen recipe (scripts/confirm_locked.py, Exp 9, 2026-11-08)")
    # The public LoRA24 recipe (notes/why_public_beats_us_2026-11-05.md); all off by default.
    ap.add_argument("--pool-layers", type=int, nargs="+",
                    help=f"decoders: mean-pool these layer depths (e.g. {' '.join(map(str, encoder.LORA24_LAYERS))})")
    ap.add_argument("--keep-layers", type=int, help="with --pool-layers: keep this many layers (default max depth)")
    ap.add_argument("--head-lr-mult", type=float, help="head (and pooled LayerNorm) lr = lr x this")
    ap.add_argument("--centred-weight", type=float,
                    help="with --pool-layers: batches of same-prompt units plus a centred head with this loss weight")
    ap.add_argument("--sibling-units", type=float,
                    help="with --centred-weight, stage 2 only: this share (0-1) of each unit's partners is drawn in "
                         "proportion to the stage-1 model's confusion with the anchor row's label")
    ap.add_argument("--fixed-epoch", type=int,
                    help="nested runs: no inner fits; refit every outer fold at the single --lrs value and stop at "
                         "this epoch (a screen with a schedule fixed in advance)")
    ap.add_argument("--schedule", choices=["linear", "cosine"], help="lr decay after warmup (default linear)")
    ap.add_argument("--batch-size", type=int, help="effective batch (default 32)")
    ap.add_argument("--pool-class-weighted", action="store_true", default=None,
                    help="stage 1: cross-entropy weighted by inverse class counts")
    ap.add_argument("--pool-replace", action="store_true", default=None,
                    help="stage 1: top labels with fewer than --pool-per-label rows up by drawing with replacement")
    ap.add_argument("--shared-stage1", action="store_true", default=None,
                    help="one stage 1 for every fold; needs a uniform --pool-recipe (u, ux), which reads no official rows")
    ap.add_argument("--stage1-from", metavar="DIR",
                    help="use this saved stage 1 (a --final run's stage1/ or a nested run's stage1/shared/) and train "
                         "none; nested runs need --shared-stage1. The --pool-* options must be the ones it used")
    ap.add_argument("--stage1-init", metavar="DIR",
                    help="start the stage-1 fit from this saved stage 1 (as --stage1-from) and train --pool-epochs "
                         "more on a new sample; nested runs need --shared-stage1")
    args = ap.parse_args()
    decoder = args.backbone in encoder.DECODERS
    args.lrs = args.lrs or list(encoder.DECODER_LEARNING_RATES if decoder else encoder.LEARNING_RATES)
    args.epochs = args.epochs or (encoder.DECODER_EPOCHS if decoder else encoder.EPOCHS)
    args.lora_rank = args.lora_rank if decoder else None
    if args.pool_per_label:
        args.pool_recipe, args.pool_epochs = args.pool_recipe or "wq_own", args.pool_epochs or 2
        args.pool_lr = args.pool_lr or (1e-4 if decoder else 3e-5)
    if opt(args, "shared_stage1") and (not args.pool_per_label or pool.RECIPES[args.pool_recipe].weighting != "none"):
        ap.error("--shared-stage1 needs --pool-per-label and a uniform --pool-recipe (u or ux)")
    if opt(args, "stage1_from") and opt(args, "stage1_init"):
        ap.error("--stage1-from and --stage1-init exclude each other")
    if (args.stage1_from or args.stage1_init) and not (args.pool_per_label and (args.final or args.shared_stage1)):
        ap.error("--stage1-from/--stage1-init need --pool-per-label, and --shared-stage1 unless --final")
    if args.all_prompts and not args.final:
        ap.error("--all-prompts is for --final runs only (nested runs never see the confirmation prompts)")
    if args.final and (args.final_epoch is None or len(args.lrs) != 1 or not 1 <= args.final_epoch <= args.epochs):
        ap.error("--final needs one --lrs value and --final-epoch within --epochs")
    if args.fixed_epoch is not None and (args.final or args.pilot or len(args.lrs) != 1
                                         or not 1 <= args.fixed_epoch <= args.epochs):
        ap.error("--fixed-epoch is for nested runs, with one --lrs value and an epoch within --epochs")
    if args.sibling_units is not None and not (0 <= args.sibling_units <= 1 and args.centred_weight
                                               and args.pool_per_label and not args.pilot):
        ap.error("--sibling-units is a share in 0-1 and needs --centred-weight and a pool stage 1 to start from")
    transformers.logging.set_verbosity_error()  # the per-fit head-initialisation load report

    started = time.time()
    stop_at = started + 60 * (args.time_budget_minutes - args.fit_minutes) if args.time_budget_minutes else None
    stop_s1 = (started + 60 * (args.time_budget_minutes - (args.stage1_minutes or args.fit_minutes))
               if args.time_budget_minutes else None)
    dev = (select(load_train(), development_prompts() | confirmation_prompts(unlock=True)) if args.all_prompts
           else development_rows(load_train()))
    model_path = download(encoder.BACKBONES[args.backbone])
    if args.pilot:
        if args.out:
            args.out.mkdir(parents=True)
        pilot(args, dev, model_path, args.seed)
        return
    if args.final:
        final(args, dev, model_path, args.out or OUTPUTS / "final" / args.backbone
              / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"), started)
        return

    run = {k: getattr(args, k) for k in RUN_KEYS} | {"revision": Path(model_path).name}
    out = args.out or OUTPUTS / "encoder" / args.backbone / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    if (out / "run.json").exists():
        saved = json.loads((out / "run.json").read_text(encoding="utf-8"))
        if {k: saved.get(k, RUN_DEFAULTS.get(k)) for k in run} != run:  # runs started before lora_rank existed saved no key for it
            raise SystemExit(f"--out was started with different settings: {saved}")
    else:
        (out / "fits").mkdir(parents=True)
        (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    folds = [f for p in args.partitions for f in outer_folds(p)]
    print(f"{args.backbone} @ {run['revision'][:10]}: {len(dev)} dev rows, lrs {args.lrs}, "
          f"{args.epochs} epochs, {len(folds)} outer folds -> {out}", flush=True)

    try:
        fit_and_report(args, dev, model_path, folds, out, run, started, stop_at, stop_s1)
    except OutOfTime as e:
        done = len(list((out / "fits").glob("*.npz")))
        print(f"time budget reached before {e}; {done} fits on disk. Resume with --out {out}", flush=True)
        raise SystemExit(OUT_OF_TIME) from None


def fit_and_report(args, dev, model_path, folds, out, run, started, stop_at, stop_s1) -> None:
    inner_parts, selection_tables, oof_parts, selections, stage1_parts = [], [], [], [], []
    branch = f"{args.backbone}+pool" if args.pool_per_label else args.backbone
    init = shared = None
    if opt(args, "shared_stage1"):
        init, shared = shared_stage1(args, model_path, dev, out, stop_s1)
    for fold in folds:
        p, o = fold.partition, fold.outer
        outer_train = select(dev, fold.train)
        if args.pool_per_label:
            va = select(dev, fold.validation)
            if shared is not None:
                s1 = shared.loc[va.id.to_numpy(dtype=str)].to_numpy()
            else:
                init, s1 = stage1(args, model_path, fold, outer_train, va, out, stop_s1)
            meta = va[["id", "prompt_id", "model"]].reset_index(drop=True)
            meta = meta.assign(partition=p, outer_fold=o, branch=f"{args.backbone}:pool-stage1",
                               pred=np.asarray(LABELS)[s1.argmax(axis=1)])
            stage1_parts.append(pd.concat([meta, pd.DataFrame(s1, columns=SCORE_COLUMNS)], axis=1))
        if opt(args, "fixed_epoch"):  # a schedule fixed in advance: no inner fits and no selection
            lr, epoch, chose = args.lrs[0], args.fixed_epoch, "fixed"
        else:
            inner_logits = {}
            for k, (inner_train, inner_val) in enumerate(fold.inner()):
                tr, va = select(dev, inner_train), select(dev, inner_val)
                for lr in args.lrs:
                    cfg = train_config(args, lr)
                    path = out / "fits" / f"inner_p{p}_o{o}_k{k}_lr{lr:g}.npz"
                    for epoch, s in cached_fit(path, model_path, tr, va, cfg, args.seed, None, stop_at, init).items():
                        inner_logits.setdefault((lr, epoch),
                                                pd.DataFrame(np.nan, index=outer_train.index, columns=LABELS))
                        inner_logits[lr, epoch].loc[va.index] = s

            # Selection from inner out-of-fold logits only.
            arrays = {key: frame.to_numpy() for key, frame in inner_logits.items()}
            (lr, epoch), table = encoder.select(arrays, outer_train.model.to_numpy())
            chose = f"chose (inner BA {table.inner_ba.max():.3f})"
            selection_tables.append(table.assign(partition=p, outer_fold=o))
            meta = outer_train[["id", "prompt_id", "model"]].assign(partition=p, outer_fold=o)
            for (key_lr, key_ep), frame in inner_logits.items():
                scores = pd.DataFrame(frame.to_numpy(), columns=SCORE_COLUMNS, index=frame.index)
                inner_parts.append(pd.concat([meta.assign(lr=key_lr, epoch=key_ep), scores], axis=1))
        selections.append({"partition": p, "outer_fold": o, "lr": lr, "epoch": epoch})

        # Outer refit on the 96 training prompts with the chosen schedule, stopped at the chosen epoch.
        va = select(dev, fold.validation)
        cfg = train_config(args, lr)
        path = out / "fits" / f"outer_p{p}_o{o}_lr{lr:g}_ep{epoch}.npz"
        scores = cached_fit(path, model_path, outer_train, va, cfg, args.seed, [epoch], stop_at, init)[epoch]
        meta = va[["id", "prompt_id", "model"]].reset_index(drop=True)
        meta = meta.assign(partition=p, outer_fold=o, branch=branch, pred=np.asarray(LABELS)[scores.argmax(axis=1)])
        centred = centred_columns(path, epoch)
        oof_parts.append(pd.concat([meta, pd.DataFrame(scores, columns=SCORE_COLUMNS)]
                                   + ([centred] if centred is not None else []), axis=1))
        fold_ba = balanced_accuracy(meta.model, meta.pred)
        print(f"p{p} o{o}: lr={lr:g} epoch={epoch} {chose}, "
              f"outer BA {fold_ba:.3f}  [{(time.time() - started) / 60:.0f} min]", flush=True)
        if args.pool_per_label and not args.keep_stage1 and not opt(args, "shared_stage1"):
            shutil.rmtree(init, ignore_errors=True)  # every fit of this fold is on disk; 1.5 GB for Ettin

    if shared is not None and not args.keep_stage1 and not opt(args, "stage1_from"):
        shutil.rmtree(init, ignore_errors=True)  # every fold's fits are on disk
    oof = pd.concat(oof_parts, ignore_index=True)
    oof.to_parquet(out / "oof.parquet", index=False)
    if stage1_parts:
        pd.concat(stage1_parts, ignore_index=True).to_parquet(out / "oof_stage1.parquet", index=False)
    if inner_parts:  # none with --fixed-epoch
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
    scored = [oof] + ([pd.concat(stage1_parts, ignore_index=True)] if stage1_parts else [])
    for frame, p in ((f, p) for f in scored for p in args.partitions):
        rows = frame[frame.partition == p].set_index("id")
        name = rows.branch.iloc[0]
        lo, hi = interval(prompt_bootstrap(rows.model, rows.pred, rows.prompt_id, **boot))
        result = {
            "partition": p,
            "branch": name,
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
        recalls[f"p{p}:{name}"] = per_class_recall(rows.model, rows.pred)
        if name == branch:
            confusion(rows.model, rows.pred).round(3).to_csv(out / f"confusion_p{p}.csv")
    results = pd.DataFrame(results)
    results.to_csv(out / "results.csv", index=False)
    recall_table = pd.DataFrame(recalls)
    recall_table.to_csv(out / "per_class_recall.csv")

    (out / "config.json").write_text(json.dumps({
        "experiment": "4: nested-CV LoRA decoder screening" if args.backbone in encoder.DECODERS
        else "6: pool stage 1, then nested-CV encoder stage 2" if args.pool_per_label
        else "3: nested-CV fine-tuned encoder screening",
        "model_repo": encoder.BACKBONES[args.backbone],
        **run,
        "partitions": args.partitions,
        "train_config": asdict(train_config(args, args.lrs[0])),
        "selection": "none: every outer fold at the single lr, stopped at --fixed-epoch" if opt(args, "fixed_epoch")
        else "per outer fold, max BA of (lr, epoch) over concatenated inner out-of-fold logits; ties -> fewer epochs, smaller lr",
        "baseline": f"{args.baseline}:{args.baseline_branch}" if args.baseline else None,
        "train_sha256": hashlib.sha256((data_dir() / "train.csv").read_bytes()).hexdigest(),
        "label_order": LABELS,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
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
