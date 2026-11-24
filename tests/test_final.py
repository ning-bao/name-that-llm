"""End to end on the CPU: encoder_finetune.py --final with a tiny model, then kaggle_package.py --dry-run."""

import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
from nllm import encoder
from nllm.data import LABELS, load_test, load_train
from nllm.paths import ROOT
from nllm.splits import development_rows


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tiny_ettin(tmp_path_factory):
    """A randomly initialised 2-layer ModernBERT saved with the real Ettin tokenizer (skipped if not cached)."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["ettin-400m"], local_files_only=True)
    except OSError:
        pytest.skip("Ettin tokenizer not in the local Hugging Face cache")
    config = transformers.ModernBertConfig(
        vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
        pad_token_id=tok.pad_token_id,
    )
    path = tmp_path_factory.mktemp("tiny_ettin")
    torch.manual_seed(0)
    transformers.ModernBertForMaskedLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="module")
def tiny_qwen3(tmp_path_factory):
    """A randomly initialised 2-layer Qwen3 saved with the real Qwen3 tokenizer (skipped if not cached)."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["qwen3-8b-base"], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    config = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    path = tmp_path_factory.mktemp("tiny_qwen3")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def stage1_sample(recipe, official, per_label, seed, **_):  # stands in for the community pool
    rows = official.groupby("model", group_keys=False).tail(per_label)
    return rows.assign(source="test", origin="pool", kb_proxy=False, group=rows.prompt_id), None


def dev_rows():
    try:
        return development_rows(load_train())
    except FileNotFoundError:
        pytest.skip("competition data not available locally")


def test_final_decoder_run(tiny_qwen3, tmp_path, monkeypatch):
    """A decoder's --final run saves its adapters and head, and scores the reference rows in bf16 and fp16."""
    script = load_script("encoder_finetune")
    monkeypatch.setattr(script.pool, "training_sample", stage1_sample)
    args = SimpleNamespace(backbone="qwen3-8b-base", lrs=[1e-3], epochs=2, final_epoch=1, seed=0, micro_batch=8,
                           gradient_checkpointing=False, optimizer=None, lora_rank=4, pool_per_label=2000,
                           pool_recipe="wx_own", pool_epochs=1, pool_lr=1e-3, pool_seed=0, limit_rows=2,
                           keep_stage1=False)
    run = tmp_path / "run"
    script.final(args, dev_rows(), tiny_qwen3, run, time.time())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["train_rows"], manifest["reference_rows"]) == (36, 36)
    assert 0 <= manifest["reference_fp16_bf16_agreement"] <= 1
    assert (run / "model" / "adapter_model.safetensors").exists() and not (run / "stage1").exists()
    ref = np.load(run / "reference.npz")
    assert set(ref.files) == {"ids", "in_fit", "bf16", "fp16"} and ref["fp16"].shape == (36, len(LABELS))
    assert np.abs(ref["bf16"] - ref["in_fit"]).max() < 0.5  # merged bf16 weights vs the unmerged in-fit model

    # The Kaggle path: the adapters merged by hand (no peft) reproduce the peft-merged fp16 scores.
    from nllm import inference
    model = inference.load_decoder(tiny_qwen3, run / "model", manifest["pad_id"], dtype=torch.float16)
    texts = load_train().set_index("id").loc[ref["ids"], "text"]
    fp16 = inference.predict_logits(model, inference.tokenize_decoder(run / "model", texts, manifest["max_length"]),
                                    manifest["pad_id"], dtype=torch.float16)
    assert (fp16.argmax(axis=1) == ref["fp16"].argmax(axis=1)).all() and np.abs(fp16 - ref["fp16"]).max() < 0.01

    # Packaged and run like the notebook, with the base found through its REVISION file.
    (Path(tiny_qwen3) / "REVISION").write_text(manifest["revision"], encoding="utf-8")
    out = tmp_path / "kaggle"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_package.py"), str(run), "--out", str(out),
                           "--dry-run", "--dry-run-base", tiny_qwen3],
                          check=False, capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stdout + done.stderr[-3000:]
    assert "argmax agreement 1.000" in done.stdout and done.stdout.count("  staged ") == 2  # adapter and base
    ensemble = json.loads((out / "dataset" / "ensemble.json").read_text(encoding="utf-8"))
    assert ensemble["members"][0]["kind"] == "decoder" and (out / "working" / "submission.parquet").exists()


def test_final_run_packages_and_scores(tiny_ettin, tmp_path, monkeypatch):
    dev = dev_rows()
    try:
        test = load_test()
    except FileNotFoundError:
        pytest.skip("competition data not available locally")
    script = load_script("encoder_finetune")
    monkeypatch.setattr(script.pool, "training_sample", stage1_sample)
    args = SimpleNamespace(backbone="ettin-400m", lrs=[1e-3], epochs=2, final_epoch=1, seed=0, micro_batch=8,
                           gradient_checkpointing=False, optimizer=None, lora_rank=None, pool_per_label=2500,
                           pool_recipe="w", pool_epochs=1, pool_lr=1e-3, pool_seed=0, limit_rows=4,
                           keep_stage1=False)
    run = tmp_path / "run"
    script.final(args, dev, tiny_ettin, run, time.time())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert (manifest["train_rows"], manifest["pool_rows"], manifest["reference_rows"]) == (72, 72, 72)
    assert manifest["epoch"] == 1 and not (run / "stage1").exists()

    out = tmp_path / "kaggle"  # the same run twice stands in for a two-member ensemble
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_package.py"), str(run), str(run),
                           "--scales", "1", "2", "--out", str(out), "--dry-run"],
                          check=False, capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stdout + done.stderr[-3000:]
    assert f"nllm from {out / 'dataset' / 'nllm'}" in done.stdout  # the packaged modules, not the repo's
    assert done.stdout.count("argmax agreement 1.000") == 2
    assert done.stdout.count("  staged ") == 2 and not any((out / "stage").iterdir())  # copies removed
    sub = pd.read_parquet(out / "working" / "submission.parquet")
    assert list(sub.id) == list(test.id)
    ensemble = json.loads((out / "dataset" / "ensemble.json").read_text(encoding="utf-8"))
    assert [(m["dir"], m["scale"]) for m in ensemble["members"]] == [("m0-run", 1.0), ("m1-run", 2.0)]
    config = json.loads((out / "dataset" / "m0-run" / "model" / "config.json").read_text(encoding="utf-8"))
    assert config["local_rope_theta"] == config["rope_parameters"]["sliding_attention"]["rope_theta"]
    notebook = json.loads((out / "nllm_infer.ipynb").read_text(encoding="utf-8"))
    assert [c["cell_type"] for c in notebook["cells"]] == ["markdown", "code", "code", "code"]


@pytest.fixture(scope="module")
def tiny_qwen3_deep(tmp_path_factory):
    """A randomly initialised 4-layer Qwen3 with the real Qwen3 tokenizer, for the pooled (LoRA24) decoder."""
    try:
        tok = transformers.AutoTokenizer.from_pretrained(encoder.BACKBONES["qwen3-8b-base"], local_files_only=True)
    except OSError:
        pytest.skip("Qwen3 tokenizer not in the local Hugging Face cache")
    config = transformers.Qwen3Config(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=4,
                                      num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                                      eos_token_id=tok.eos_token_id)
    path = tmp_path_factory.mktemp("tiny_qwen3_deep")
    torch.manual_seed(0)
    transformers.Qwen3ForCausalLM(config).save_pretrained(path)
    tok.save_pretrained(path)
    return str(path)


def test_final_pooled_decoder_run_packages_and_scores(tiny_qwen3_deep, tmp_path, monkeypatch):
    """A pooled decoder's --final run (LoRA24 options) saves adapters and heads; the peft-free Kaggle loader
    reproduces its scores; and the packaged notebook passes its reference check."""
    script = load_script("encoder_finetune")
    monkeypatch.setattr(script.pool, "training_sample", stage1_sample)
    args = SimpleNamespace(backbone="qwen3-8b-base", lrs=[1e-3], epochs=2, final_epoch=2, seed=0, micro_batch=6,
                           gradient_checkpointing=False, optimizer=None, lora_rank=4, pool_per_label=2000,
                           pool_recipe="ux", pool_epochs=1, pool_lr=1e-3, pool_seed=0, limit_rows=2,
                           keep_stage1=False, pool_layers=[1, 2, 3], keep_layers=None, head_lr_mult=10.0,
                           centred_weight=1.0, schedule="cosine", batch_size=9, pool_class_weighted=True,
                           pool_replace=None)
    run = tmp_path / "run"
    script.final(args, dev_rows(), tiny_qwen3_deep, run, time.time())
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["pooled"] == {"layers": [1, 2, 3], "keep_layers": 3, "hidden_size": 32}
    assert manifest["pool_class_weighted"] is True and manifest["train_config"]["centred_weight"] == 1.0
    assert (run / "model" / "pooled_heads.safetensors").exists() and not (run / "stage1").exists()
    ref = np.load(run / "reference.npz")
    assert set(ref.files) == {"ids", "in_fit", "bf16", "fp16"}
    assert np.abs(ref["bf16"] - ref["in_fit"]).max() < 0.5  # merged bf16 weights vs the unmerged in-fit model

    # The hand-merged fp32 model reproduces the in-fit logits closely (peft is not involved).
    from nllm import inference
    model = inference.load_pooled_decoder(tiny_qwen3_deep, run / "model", dtype=torch.float32)
    texts = load_train().set_index("id").loc[ref["ids"], "text"]
    ids = inference.tokenize_decoder(run / "model", texts, manifest["max_length"])
    fp32 = inference.predict_logits(model, ids, manifest["pad_id"], dtype=None)
    assert (fp32.argmax(axis=1) == ref["in_fit"].argmax(axis=1)).mean() >= 0.95
    assert np.abs(fp32 - ref["in_fit"]).max() < 0.1

    (Path(tiny_qwen3_deep) / "REVISION").write_text(manifest["revision"], encoding="utf-8")
    out = tmp_path / "kaggle"
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "kaggle_package.py"), str(run), "--out", str(out),
                           "--dry-run", "--dry-run-base", tiny_qwen3_deep],
                          check=False, capture_output=True, text=True, encoding="utf-8")
    assert done.returncode == 0, done.stdout + done.stderr[-3000:]
    assert "argmax agreement 1.000" in done.stdout
    ensemble = json.loads((out / "dataset" / "ensemble.json").read_text(encoding="utf-8"))
    assert ensemble["members"][0]["kind"] == "pooled_decoder" and (out / "working" / "submission.parquet").exists()


def test_saved_stage1_is_reused_continued_and_checked(tiny_qwen3_deep, tmp_path, monkeypatch):
    """--stage1-from skips stage 1 and never deletes its source; --stage1-init trains on from it; both refuse a
    source with another backbone or (for nested runs) a recipe that read official rows, and --stage1-from refuses
    pool settings that differ from the source's."""
    script = load_script("encoder_finetune")
    monkeypatch.setattr(script.pool, "training_sample", stage1_sample)
    base = {"backbone": "qwen3-8b-base", "lrs": [1e-3], "epochs": 1, "final_epoch": 1, "seed": 0, "micro_batch": 6,
            "gradient_checkpointing": False, "optimizer": None, "lora_rank": 4, "pool_per_label": 2000,
            "pool_recipe": "ux", "pool_epochs": 1, "pool_lr": 1e-3, "pool_seed": 0, "limit_rows": 2,
            "keep_stage1": True, "pool_layers": [1, 2, 3], "keep_layers": None, "head_lr_mult": 10.0,
            "centred_weight": 1.0, "schedule": "cosine", "batch_size": 9, "pool_class_weighted": True,
            "pool_replace": None, "stage1_from": None, "stage1_init": None}
    dev = dev_rows()
    first = tmp_path / "first"
    script.final(SimpleNamespace(**base), dev, tiny_qwen3_deep, first, time.time())
    source = str(first / "stage1")

    reused = tmp_path / "reused"
    script.final(SimpleNamespace(**base | {"stage1_from": source, "keep_stage1": False}), dev, tiny_qwen3_deep,
                 reused, time.time())
    manifest = json.loads((reused / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage1_from"] == source and manifest["stage1_source"]["pool_recipe"] == "ux"
    assert "stage1_history" not in manifest and not (reused / "stage1").exists()
    assert (first / "stage1" / "pooled_heads.safetensors").exists()  # the source is never deleted

    with pytest.raises(SystemExit, match="other pool settings"):
        script.final(SimpleNamespace(**base | {"stage1_from": source, "pool_lr": 5e-4}), dev, tiny_qwen3_deep,
                     tmp_path / "mismatch", time.time())

    continued = tmp_path / "continued"
    script.final(SimpleNamespace(**base | {"stage1_init": source, "pool_lr": 5e-4, "pool_seed": 1}), dev,
                 tiny_qwen3_deep, continued, time.time())
    manifest = json.loads((continued / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage1_init"] == source and len(manifest["stage1_history"]) == 1
    assert (continued / "stage1" / "pooled_heads.safetensors").exists()

    # A nested run's shared stage 1 from the saved one: only dev-row scores are computed, nothing is trained.
    nested = tmp_path / "nested"
    (nested / "fits").mkdir(parents=True)
    args = SimpleNamespace(**base | {"stage1_from": source, "shared_stage1": True})
    model_dir, scores = script.shared_stage1(args, tiny_qwen3_deep, dev.head(20), nested, None)
    assert model_dir == source and list(scores.index) == list(dev.head(20).id) and not (nested / "stage1").exists()

    # Refusals: another backbone, and a source whose recipe read official rows (a nested run cannot use it).
    with pytest.raises(SystemExit, match="not qwen3-4b-base"):
        script.stage1_source(SimpleNamespace(**base | {"backbone": "qwen3-4b-base"}), source, nested=False)
    saved = json.loads((first / "manifest.json").read_text(encoding="utf-8"))
    (first / "manifest.json").write_text(json.dumps(saved | {"pool_recipe": "wx_own"}), encoding="utf-8")
    with pytest.raises(SystemExit, match="reads official rows"):
        script.stage1_source(SimpleNamespace(**base | {"stage1_init": source}), source, nested=True)
    with pytest.raises(SystemExit, match="neither"):
        script.stage1_source(SimpleNamespace(**base), str(first / "model"), nested=False)


def test_fixed_epoch_screen_with_sibling_units_keeps_centred_outputs(tiny_qwen3_deep, tmp_path, monkeypatch):
    """A nested run with --fixed-epoch fits no inner folds; with --sibling-units its stage 2 starts from the saved
    stage 1 with confusion-drawn units, and oof.parquet keeps the centred head's outputs for the twin rule."""
    from nllm.splits import outer_folds

    script = load_script("encoder_finetune")
    monkeypatch.setattr(script.pool, "training_sample", stage1_sample)
    base = {"backbone": "qwen3-8b-base", "lrs": [1e-3], "epochs": 2, "final_epoch": 1, "seed": 0, "micro_batch": 6,
            "gradient_checkpointing": False, "optimizer": None, "lora_rank": 4, "pool_per_label": 2000,
            "pool_recipe": "ux", "pool_epochs": 1, "pool_lr": 1e-3, "pool_seed": 0, "limit_rows": 2,
            "keep_stage1": True, "pool_layers": [1, 2, 3], "keep_layers": None, "head_lr_mult": 10.0,
            "centred_weight": 1.0, "schedule": "cosine", "batch_size": 9, "pool_class_weighted": True,
            "pool_replace": None, "stage1_from": None, "stage1_init": None}
    dev = dev_rows()
    first = tmp_path / "first"
    script.final(SimpleNamespace(**base), dev, tiny_qwen3_deep, first, time.time())

    # Three answers per prompt keep the CPU fits short; one outer fold.
    small = dev.groupby("prompt_id", group_keys=False).head(3)
    fold = outer_folds(0)[0]
    args = SimpleNamespace(**base | {"stage1_from": str(first / "stage1"), "shared_stage1": True, "sibling_units": 1.0,
                                     "fixed_epoch": 2, "partitions": [0], "baseline": None, "bootstrap_draws": 20})
    out = tmp_path / "screen"
    (out / "fits").mkdir(parents=True)
    script.fit_and_report(args, small, tiny_qwen3_deep, [fold], out, {"backbone": args.backbone}, time.time(), None,
                          None)
    assert sorted(p.name for p in (out / "fits").glob("*.npz")) == ["outer_p0_o0_lr0.001_ep2.npz", "stage1_shared.npz"]
    assert not (out / "inner_oof.parquet").exists()
    oof = pd.read_parquet(out / "oof.parquet")
    held_out = small[small.prompt_id.isin(fold.validation)]
    assert list(oof.id) == list(held_out.id)
    assert np.isfinite(oof[script.SCORE_COLUMNS + script.CENTRED_COLUMNS].to_numpy()).all()
    assert (oof[script.CENTRED_BIAS_COLUMNS].nunique() == 1).all()  # one fold, one head
    history = pd.read_csv(out / "history.csv")
    assert history[history.fit.str.startswith("outer")].same_family_pairs.between(0, 1).all()
    assert json.loads((out / "config.json").read_text(encoding="utf-8"))["selection"].startswith("none")
    assert list(pd.read_csv(out / "selections.csv").epoch) == [2]
    assert script.stage1_config(args).sibling_units == 0 and script.train_config(args, 1e-3).sibling_units == 1.0

