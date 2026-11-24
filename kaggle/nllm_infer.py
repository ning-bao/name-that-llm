# %% [markdown]
# # Name That LLM: fine-tuned encoders, offline
#
# Inputs: the competition data and the private dataset built by `scripts/kaggle_package.py` (`ensemble.json`,
# `nllm/`, and per member `model/`, `reference.npz`, `manifest.json`), plus, for a LoRA decoder member, a dataset
# with its base weights and a `REVISION` file. Internet off; GPU T4 x2. For each member the notebook
# first rescores a few training rows and stops if they disagree with the scores saved when the model was trained,
# then scores `test.csv`. The members' logits, each divided by its scale, are summed. If `ensemble.json` sets
# `centring`, each test row then loses a share of the mean score of the other test rows in its content cluster
# (TF-IDF topic clusters of the test passages; `nllm.centring`). The argmax is written to
# `/kaggle/working/submission.parquet` in input order. Source: `kaggle/nllm_infer.py` in the project repository.

# %%
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

started = time.time()
# All overridable for a local dry run (scripts/kaggle_package.py --dry-run).
INPUT = Path(os.environ.get("NLLM_KAGGLE_INPUT", "/kaggle/input"))
WORKING = Path(os.environ.get("NLLM_KAGGLE_WORKING", "/kaggle/working"))
# Model files are copied to local disk before loading. from_pretrained reading safetensors in place from the
# /kaggle/input mount (random access) took 4 h 15 min for the 16 GB 8B base in nllm-ensemble-5 version 1.
STAGE = Path(os.environ.get("NLLM_STAGE_DIR", "/tmp/nllm_stage"))


def find(predicate, what):
    for top, dirs, _ in os.walk(INPUT, followlinks=True):  # a local dry run links large inputs in
        dirs.sort()
        if predicate(Path(top)):
            return Path(top)
    raise FileNotFoundError(f"no {what} under {INPUT}")


PACKAGE = find(lambda p: (p / "ensemble.json").exists() and (p / "nllm" / "inference.py").exists(), "model package")
if "NLLM_DATA_DIR" not in os.environ:
    os.environ["NLLM_DATA_DIR"] = str(find(lambda p: (p / "test.csv").exists() and (p / "train.csv").exists(),
                                           "competition data"))
sys.path.insert(0, str(PACKAGE))

import tokenizers
import transformers

from nllm import inference
from nllm.data import LABELS, load_test, load_train
from nllm.submission import write_submission

ensemble = json.loads((PACKAGE / "ensemble.json").read_text(encoding="utf-8"))
if ensemble["labels"] != list(LABELS):
    raise RuntimeError("the package's label order differs from nllm.data.LABELS")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"nllm from {Path(inference.__file__).parent}")
print(f"package {PACKAGE}: {len(ensemble['members'])} member(s), code {ensemble.get('code_id')}")
print(f"data {os.environ['NLLM_DATA_DIR']}; python {sys.version.split()[0]}, torch {torch.__version__}, "
      f"transformers {transformers.__version__}, tokenizers {tokenizers.__version__}, "
      f"device {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'}")

# %%
# One member at a time. Encoders run in fp32 (a T4 has no bf16, and fp32 is what their reference scores were
# computed in). LoRA decoders run in fp16 on the base weights from their own dataset (found by its REVISION
# file), split across the GPUs, and are checked against fp16 reference scores. The agreement check rescores
# training rows read from the competition data (the package stores only their ids).
def staged(path, copies):
    """A copy of the directory `path` on local disk (its parent is added to `copies` for deletion), or `path`
    itself when the disk is too small."""
    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    STAGE.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(STAGE).free
    if free < size + 2 * 2**30:
        print(f"  {path.name}: {size / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free on {STAGE}; loading in place")
        return path
    t0 = time.time()
    parent = Path(tempfile.mkdtemp(dir=STAGE))
    copies.append(parent)
    shutil.copytree(path, parent / path.name)
    print(f"  staged {path}: {size / 2**30:.2f} GiB in {time.time() - t0:.0f}s")
    return parent / path.name


def load_member(member, root, manifest):
    """(model, score, copies): score(texts) returns the member's logits; copies are staged directories."""
    pad_id, max_length, batch = manifest["pad_id"], manifest["max_length"], member.get("batch_size", 32)
    copies = []
    model_dir = staged(root / "model", copies)
    if member.get("kind") in ("decoder", "pooled_decoder"):
        rev = member["base_revision"]
        base = find(lambda p: (p / "REVISION").is_file() and (p / "REVISION").read_text().strip() == rev,
                    f"base weights at revision {rev}")
        if member["kind"] == "pooled_decoder":  # truncated base, adapters merged, per-depth LayerNorms + head
            model = inference.load_pooled_decoder(staged(base, copies), model_dir, dtype=torch.float16,
                                                  device_map="auto" if device.type == "cuda" else None)
        else:
            model = inference.load_decoder(staged(base, copies), model_dir, pad_id, dtype=torch.float16,
                                           device_map="auto" if device.type == "cuda" else None)

        def score(texts):
            ids = inference.tokenize_decoder(model_dir, texts, max_length)
            return inference.predict_logits(model, ids, pad_id, batch_size=batch, dtype=torch.float16)
    else:
        model = inference.load(model_dir, device)

        def score(texts):
            ids = inference.tokenize(model_dir, texts, max_length)
            return inference.predict_logits(model, ids, pad_id, batch_size=batch, dtype=None)
    return model, score, copies


train = load_train().set_index("id")
test = load_test()
total = np.zeros((len(test), len(LABELS)))
for member in ensemble["members"]:
    t0 = time.time()
    root = PACKAGE / member["dir"]
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest["labels"] != list(LABELS):
        raise RuntimeError(f"{member['dir']}: label order differs from nllm.data.LABELS")
    model, score, copies = load_member(member, root, manifest)
    print(f"  loaded  [{time.time() - t0:.0f}s]")
    ref = np.load(root / "reference.npz")
    expected = ref[member.get("reference", "fp32")]
    check = score(train.loc[ref["ids"], "text"])
    agree = float((check.argmax(axis=1) == expected.argmax(axis=1)).mean())
    print(f"{member['dir']}: {manifest['backbone']} seed {manifest['seed']} epoch {manifest['epoch']}, scale "
          f"{member['scale']:.3f}; reference rows: {len(check)}, argmax agreement {agree:.3f}, "
          f"max |diff| {np.abs(check - expected).max():.4f}  [{time.time() - t0:.0f}s]")
    if agree < 0.95:
        raise RuntimeError(f"{member['dir']}: reference scores disagree with training; not writing a submission")
    total += score(test.text) / member["scale"]
    print(f"  {len(test)} test rows scored  [{time.time() - t0:.0f}s]")
    del model, score
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    for copy in copies:
        shutil.rmtree(copy, ignore_errors=True)

# %%
if ensemble.get("centring"):
    from nllm import centring

    c = ensemble["centring"]
    clusters = centring.content_clusters(test.text, c["cut"])
    sizes = pd.Series(clusters).value_counts()
    before = total.argmax(axis=1)
    total = centring.centre(total, clusters, c["weight"], c["cavity"])
    print(f"centring (weight {c['weight']}, cut {c['cut']}, cavity {c['cavity']}): {len(sizes)} clusters over "
          f"{len(test)} rows, {int((sizes >= 2).sum())} with 2+ rows (largest {sizes.max()}); "
          f"{int((total.argmax(axis=1) != before).sum())} predictions changed")
preds = np.asarray(LABELS)[total.argmax(axis=1)]
path = write_submission(test.id, preds, WORKING / "submission.parquet")
print(f"wrote {path}  [{time.time() - started:.0f}s total]")
print(pd.Series(preds).value_counts().to_string())
