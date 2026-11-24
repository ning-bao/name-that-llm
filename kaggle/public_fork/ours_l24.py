"""Our Qwen3-8B LoRA24 model as a native `l24` member of the public blend (written into the notebook by %%writefile).

The public pipeline's `l24` members have the architecture of our pooled decoder (`nllm.encoder.PooledDecoder`): a
decoder truncated to `keep` layers, LoRA adapters merged as W += B @ A * alpha / r, the hidden states at `layers`
mean-pooled with the first token masked, one LayerNorm per depth, then a plain head and a centred head ("twin")
that reads features minus their prompt-cluster mean. So our model needs no process of its own: build_member()
rewrites its saved files into the layout infer.py's `l24_predict` reads and returns the member spec for MODELS.
The pipeline then applies its own test-time grouping and twin rule to it, like to every other member.

    <out>/adapter/adapter_config.json, adapter_model.safetensors   copied unchanged
    <out>/heads.safetensors   norms.<i>.weight|bias, plain.weight|bias (our `head`), centered.* (our `centred_head`)
    <out>/meta.json           our label order (infer.py permutes the heads to its own) and the spec's settings

reference_check() runs infer.py itself on the 72 training rows whose scores were saved at training time, with our
member alone, and stops unless infer.py's log-probabilities agree with them. It runs in the commit (the 3-row
public test), so a version that committed has shown that the pipeline reads our model the way training did.
Nothing here reads test rows.
"""
import json
import os
import re
import shutil
import sys

import numpy as np

HEAD_KEYS = (("head.", "plain."), ("centred_head.", "centered."), ("norms.", "norms."))
MEMBER_NAME = "ours_q8"


def eos_id(model_dir) -> int:
    """The id of the saved tokenizer's EOS token, which training appended to every passage."""
    with open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        eos = json.load(f)["eos_token"]
    eos = eos["content"] if isinstance(eos, dict) else eos
    with open(os.path.join(model_dir, "tokenizer.json"), encoding="utf-8") as f:
        tok = json.load(f)
    ids = [t["id"] for t in tok.get("added_tokens", []) if t["content"] == eos]
    if not ids and eos in tok.get("model", {}).get("vocab", {}):
        ids = [tok["model"]["vocab"][eos]]
    if len(ids) != 1:
        raise ValueError(f"{model_dir}: EOS token {eos!r} has {len(ids)} ids")
    return int(ids[0])


def convert_heads(pooled: dict, n_layers: int, hidden: int, n_labels: int) -> dict:
    """Our pooled_heads tensors under the names infer.py reads; shapes are checked, values untouched."""
    out = {}
    for key, value in pooled.items():
        for ours, theirs in HEAD_KEYS:
            if key.startswith(ours):
                out[theirs + key[len(ours):]] = np.ascontiguousarray(value, dtype=np.float32)
                break
        else:
            raise ValueError(f"unexpected tensor {key} in the pooled heads")
    want = {f"norms.{i}.{p}": (hidden,) for i in range(n_layers) for p in ("weight", "bias")}
    for head in ("plain", "centered"):
        want |= {f"{head}.weight": (n_labels, hidden * n_layers), f"{head}.bias": (n_labels,)}
    got = {k: v.shape for k, v in out.items()}
    if got != want:
        wrong = sorted(set(got.items()) ^ set(want.items()))
        raise ValueError(f"pooled heads do not match {n_layers} depths x {hidden}: {wrong[:6]}")
    return out


def build_member(package, out, base, weight, twin, batch_size=16) -> dict:
    """Write our model as an infer.py l24 adapter directory under `out` and return its MODELS spec.

    `package` is the member directory of our Kaggle dataset (manifest.json, reference.npz, model/); `base` is the
    Qwen3-8B-Base directory the pipeline already uses for its own 8B member.
    """
    from safetensors.numpy import load_file, save_file

    model = os.path.join(package, "model")
    with open(os.path.join(package, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    pooled = manifest["pooled"]
    layers, keep, hidden = [int(i) for i in pooled["layers"]], int(pooled["keep_layers"]), int(pooled["hidden_size"])
    labels = list(manifest["labels"])
    with open(os.path.join(model, "adapter_config.json"), encoding="utf-8") as f:
        adapter = json.load(f)
    if adapter.get("use_dora") or adapter.get("use_rslora") or adapter.get("fan_in_fan_out"):
        raise ValueError(f"{model}: infer.py merges plain LoRA adapters only")
    if max(layers) > keep:
        raise ValueError(f"{model}: pooled depths {layers} reach above the {keep} kept layers")
    heads = convert_heads(load_file(os.path.join(model, "pooled_heads.safetensors")), len(layers), hidden, len(labels))

    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(os.path.join(out, "adapter"))
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        shutil.copyfile(os.path.join(model, name), os.path.join(out, "adapter", name))
    save_file(heads, os.path.join(out, "heads.safetensors"))
    cfg = {"keep": keep, "layers": layers, "max_len": int(manifest["max_length"]), "eos_id": eos_id(model)}
    with open(os.path.join(out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "cfg": cfg, "backbone": manifest.get("backbone"), "seed": manifest.get("seed"),
                   "epoch": manifest.get("epoch"), "train_prompts": manifest.get("train_prompts")}, f)
    # no_final_norm: training replaced the base's final norm with an identity. dp: one copy per T4, as for the
    # pipeline's own 8B member (it falls back to one split copy if a GPU has under 1 GiB to spare).
    return {"name": MEMBER_NAME, "type": "l24", "path": base, "adapters": [out], "w": float(weight), "bs": batch_size,
            "twin": float(twin), "no_final_norm": True, "dp": True, **cfg}


def log_softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def compare(got: np.ndarray, expected_logits: np.ndarray, got_labels, expected_labels) -> tuple[float, float]:
    """(argmax agreement, max |difference|) between infer.py's log-probabilities (columns `got_labels`) and the
    log-softmax of our saved logits (columns `expected_labels`)."""
    if sorted(got_labels) != sorted(expected_labels):
        raise ValueError("label sets differ")
    expected = log_softmax(np.asarray(expected_logits)[:, [list(expected_labels).index(m) for m in got_labels]])
    got = np.asarray(got, dtype=np.float64)
    if got.shape != expected.shape or not np.isfinite(got).all():
        raise ValueError(f"infer.py scores have shape {got.shape}, expected {expected.shape}, or are not finite")
    return float((got.argmax(axis=1) == expected.argmax(axis=1)).mean()), float(np.abs(got - expected).max())


def infer_labels(infer_py="infer.py") -> list:
    """The LABELS list of infer.py, read from its source (the order of the scores it saves)."""
    with open(infer_py, encoding="utf-8") as f:
        match = re.search(r"^LABELS = (\[.*?\])", f.read(), re.DOTALL | re.MULTILINE)
    if not match:
        raise ValueError(f"no LABELS list in {infer_py}")
    return json.loads(match.group(1).replace("'", '"'))


def reference_check(package, spec, data, run, work="/kaggle/temp/ours_check", min_agree=0.95, max_diff=1.0,
                    infer_py="infer.py") -> dict:
    """Score the reference training rows with infer.py, our member alone, and compare with the fp16 scores saved at
    training time. `run(cmd)` runs a command and returns its exit code (the notebook's own helper). Raises unless
    argmax agreement >= min_agree and every log-probability is within max_diff."""
    import pandas as pd

    ref = np.load(os.path.join(package, "reference.npz"), allow_pickle=False)
    ids = [str(i) for i in ref["ids"]]
    with open(os.path.join(package, "manifest.json"), encoding="utf-8") as f:
        our_labels = list(json.load(f)["labels"])
    train = pd.read_csv(os.path.join(data, "train.csv"), dtype=str, keep_default_na=False, encoding="utf-8")
    rows = train.set_index("id").loc[ids].reset_index()
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(os.path.join(work, "data"))
    shutil.copyfile(os.path.join(data, "train.csv"), os.path.join(work, "data", "train.csv"))
    rows[["id", "text"]].to_csv(os.path.join(work, "data", "test.csv"), index=False, encoding="utf-8")
    lp_path = os.path.join(work, "reference_lp.npy")
    # Weight 1 and every other term off, so the scores infer.py saves are this member's plain log-probabilities.
    cmd = [sys.executable, infer_py, "--data", os.path.join(work, "data"), "--models", json.dumps([dict(spec, w=1.0)]),
           "--tfidf_w", "0", "--distort_w", "0", "--svc_w", "0", "--adapt_w", "0", "--sinkhorn", "0",
           "--save_lp", lp_path, "--out", os.path.join(work, "reference_pred.parquet")]
    code = run(cmd)
    if code != 0 or not os.path.exists(lp_path):
        raise RuntimeError(f"infer.py did not score the reference rows (exit code {code})")
    agree, diff = compare(np.load(lp_path), ref["fp16"], infer_labels(infer_py), our_labels)
    result = {"rows": len(ids), "agreement": agree, "max_abs_diff": diff}
    print(f"ours reference check through infer.py: {len(ids)} rows, argmax agreement {agree:.3f}, "
          f"max |log-prob diff| {diff:.4f}", flush=True)
    if agree < min_agree or diff > max_diff:
        raise RuntimeError(f"infer.py does not reproduce our model's training-time scores: {result}")
    shutil.rmtree(work, ignore_errors=True)
    return result
