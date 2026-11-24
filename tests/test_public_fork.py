"""The public-fork notebook pieces we wrote: our member's file conversion, the commit-time reference check and the
builder's patching. No Kaggle code is run; infer.py is replaced by a stand-in that writes scores."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ours_l24 = load("ours_l24", ROOT / "kaggle" / "public_fork" / "ours_l24.py")
build_public_fork = load("build_public_fork", ROOT / "scripts" / "build_public_fork.py")
LABELS = ["a", "b", "c"]
HIDDEN, LAYERS = 4, [1, 3]


def write_package(root, adapter_extra=None, heads=None):
    """A tiny member directory with our saved-file layout; returns (package dir, the heads written)."""
    save_file = pytest.importorskip("safetensors.numpy").save_file
    rng = np.random.default_rng(0)
    model = root / "pkg" / "model"
    model.mkdir(parents=True)
    if heads is None:
        heads = {f"norms.{i}.{p}": rng.normal(size=HIDDEN).astype(np.float32) for i in range(len(LAYERS))
                 for p in ("weight", "bias")}
        for name in ("head", "centred_head"):
            heads[f"{name}.weight"] = rng.normal(size=(len(LABELS), HIDDEN * len(LAYERS))).astype(np.float32)
            heads[f"{name}.bias"] = rng.normal(size=len(LABELS)).astype(np.float32)
    save_file(heads, str(model / "pooled_heads.safetensors"))
    save_file({"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": np.ones((2, HIDDEN), dtype=np.float32)},
              str(model / "adapter_model.safetensors"))
    (model / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 4, **(adapter_extra or {})}))
    (model / "tokenizer_config.json").write_text(json.dumps({"eos_token": "<eos>"}))
    (model / "tokenizer.json").write_text(json.dumps({"added_tokens": [{"id": 7, "content": "<pad>"},
                                                                      {"id": 9, "content": "<eos>"}]}))
    (root / "pkg" / "manifest.json").write_text(json.dumps({
        "labels": LABELS, "max_length": 512, "backbone": "tiny", "seed": 1, "epoch": 3, "train_prompts": 120,
        "pooled": {"layers": LAYERS, "keep_layers": 3, "hidden_size": HIDDEN}}))
    return root / "pkg", heads


def test_build_member_writes_the_layout_infer_reads(tmp_path):
    load_file = pytest.importorskip("safetensors.numpy").load_file
    package, heads = write_package(tmp_path)
    out = tmp_path / "member"
    spec = ours_l24.build_member(str(package), str(out), base="/base/8b", weight=2.0, twin=0.35)
    assert spec == {"name": "ours_q8", "type": "l24", "path": "/base/8b", "adapters": [str(out)], "w": 2.0, "bs": 16,
                    "twin": 0.35, "no_final_norm": True, "dp": True, "keep": 3, "layers": LAYERS, "max_len": 512,
                    "eos_id": 9}
    json.dumps(spec)  # infer.py receives MODELS as JSON
    written = load_file(str(out / "heads.safetensors"))
    assert set(written) == ({f"norms.{i}.{p}" for i in range(2) for p in ("weight", "bias")}
                            | {f"{h}.{p}" for h in ("plain", "centered") for p in ("weight", "bias")})
    np.testing.assert_array_equal(written["plain.weight"], heads["head.weight"])
    np.testing.assert_array_equal(written["centered.bias"], heads["centred_head.bias"])
    np.testing.assert_array_equal(written["norms.1.weight"], heads["norms.1.weight"])
    assert (out / "adapter" / "adapter_config.json").exists() and (out / "adapter" / "adapter_model.safetensors").exists()
    assert not (out / "heads.pt").exists()  # infer.py would read heads.pt first
    meta = json.loads((out / "meta.json").read_text())
    assert meta["labels"] == LABELS and meta["cfg"] == {"keep": 3, "layers": LAYERS, "max_len": 512, "eos_id": 9}


def test_build_member_refuses_adapters_infer_cannot_merge(tmp_path):
    package, _ = write_package(tmp_path, adapter_extra={"use_rslora": True})
    with pytest.raises(ValueError):
        ours_l24.build_member(str(package), str(tmp_path / "member"), base="/b", weight=1.0, twin=0.35)


def test_build_member_refuses_heads_of_another_shape(tmp_path):
    bad = {"norms.0.weight": np.ones(HIDDEN, dtype=np.float32), "head.weight": np.ones((3, HIDDEN), dtype=np.float32)}
    package, _ = write_package(tmp_path, heads=bad)
    with pytest.raises(ValueError):
        ours_l24.build_member(str(package), str(tmp_path / "member"), base="/b", weight=1.0, twin=0.35)


def test_compare_aligns_label_orders():
    logits = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]])  # columns a, b, c
    theirs = ["c", "a", "b"]
    got = ours_l24.log_softmax(logits[:, [2, 0, 1]])
    agree, diff = ours_l24.compare(got, logits, theirs, LABELS)
    assert agree == 1.0 and diff < 1e-12
    agree, _ = ours_l24.compare(got[:, ::-1], logits, theirs, LABELS)
    assert agree < 1.0
    with pytest.raises(ValueError):
        ours_l24.compare(got, logits, ["a", "b", "d"], LABELS)


def reference_setup(tmp_path):
    """A package with reference scores, a data directory and a stand-in infer.py source."""
    package, _ = write_package(tmp_path)
    expected = np.array([[2.0, 0.0, -1.0], [0.0, 1.0, 3.0]], dtype=np.float32)
    np.savez(package / "reference.npz", ids=np.array(["r2", "r0"]), fp16=expected)
    data = tmp_path / "data"
    data.mkdir()
    pd.DataFrame({"id": ["r0", "r1", "r2"], "prompt_id": ["p1", "p1", "p2"], "text": ["zero, \"quoted\"", "one", "two"],
                  "model": ["a", "b", "c"]}).to_csv(data / "train.csv", index=False)
    infer_py = tmp_path / "infer.py"
    infer_py.write_text("import os\nLABELS = ['c','a',\n 'b']\nL2I = {}\n")
    return package, data, infer_py, expected


def test_reference_check_passes_when_infer_reproduces_our_scores(tmp_path):
    package, data, infer_py, expected = reference_setup(tmp_path)
    seen = {}

    def run(cmd):
        arg = lambda name: cmd[cmd.index(name) + 1]
        test = pd.read_csv(Path(arg("--data")) / "test.csv")
        seen.update(ids=test.id.tolist(), texts=test.text.tolist(), models=json.loads(arg("--models")),
                    train=(Path(arg("--data")) / "train.csv").exists(), adapt_w=arg("--adapt_w"))
        np.save(arg("--save_lp"), ours_l24.log_softmax(expected[:, [2, 0, 1]]))  # in infer.py's label order
        return 0

    spec = {"name": "ours_q8", "type": "l24", "w": 2.0, "twin": 0.35}
    result = ours_l24.reference_check(str(package), spec, str(data), run, work=str(tmp_path / "work"),
                                      infer_py=str(infer_py))
    assert result["rows"] == 2 and result["agreement"] == 1.0 and result["max_abs_diff"] < 1e-6
    assert seen["ids"] == ["r2", "r0"] and seen["texts"] == ["two", "zero, \"quoted\""] and seen["train"]
    assert seen["models"] == [dict(spec, w=1.0)] and seen["adapt_w"] == "0"  # our member alone, plain head only


def test_reference_check_stops_when_scores_differ_or_infer_fails(tmp_path):
    package, data, infer_py, expected = reference_setup(tmp_path)
    spec = {"name": "ours_q8", "type": "l24", "w": 2.0}

    def zeros(cmd):  # what infer.py saves when the member failed inside it
        np.save(cmd[cmd.index("--save_lp") + 1], np.zeros_like(expected))
        return 0

    for run in (zeros, lambda cmd: 1):
        with pytest.raises(RuntimeError):
            ours_l24.reference_check(str(package), spec, str(data), run, work=str(tmp_path / "work"),
                                     infer_py=str(infer_py))


def fake_notebook():
    run = ("LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]\nMODELS = [\n"
           '    {"name": "q8v5", "type": "l24", "path": B8, "w": 2.0},\n'
           '    {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n]\n'
           "cmd = ['infer.py', '--models', json.dumps(MODELS)]\n"
           "if FAST_COMMIT:\n"
           "    smoke = {}\n"
           "    print('prefit core votes smoke ok', {k: v.shape for k, v in smoke.items()})\n"
           "else:\n    print('infer exit code', run(cmd))\n")
    cell = build_public_fork.code
    return {"cells": [build_public_fork.markdown("intro"), cell("B8 = DEB = 'x'\nFAST_COMMIT = True\n"),
                      cell("%%writefile infer.py\nLABELS = ['a']\n"), cell("def run(cmd):\n    return 0\n"), cell(run)],
            "metadata": {"kaggle": {}, "papermill": {}}, "nbformat": 4}


def test_base_keeps_every_code_cell():
    nb = fake_notebook()
    out = build_public_fork.build(nb, "base", 2.0, 0.35)
    assert [c["source"] for c in out["cells"] if c["cell_type"] == "code"] == \
           [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    assert build_public_fork.code_hash(out) == build_public_fork.code_hash(nb)
    assert out["cells"][0]["cell_type"] == "markdown" and "kaggle" not in out["metadata"]


def test_plus_adds_our_member_before_the_run_cell_and_patches_it_once():
    nb = fake_notebook()
    out = build_public_fork.build(nb, "plus", 1.5, 0.35)
    code = [c["source"] for c in out["cells"] if c["cell_type"] == "code"]
    before = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    assert code[:3] == before[:3]  # everything before our section is untouched
    assert code[3].startswith("%%writefile ours_l24.py\n") and "def build_member(" in code[3]
    assert "weight=1.5, twin=0.35" in code[4] and "m0-qwen3-8b-l24-full-final-001/manifest.json" in code[4]
    run = code[5]
    member = "    OURS_SPEC,   # fork"
    assert run.count(member) == 1 and run.index('"name": "deb"') < run.index(member) < run.index("]\ncmd = [")
    assert run.count("ours_l24.reference_check(OURS_PACKAGE, OURS_SPEC, DATA, run)") == 1
    assert run.index("smoke ok") < run.index("ours_l24.reference_check") < run.index("else:")
    for text in (code[3].split("\n", 1)[1], code[4], run):
        compile(text, "<cell>", "exec")


@pytest.mark.parametrize("old", ['{"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n]', "smoke ok"])
def test_patch_refuses_a_missing_anchor(old):
    nb = fake_notebook()
    nb["cells"][4]["source"] = nb["cells"][4]["source"].replace(old, "changed")
    with pytest.raises(ValueError):
        build_public_fork.build(nb, "plus", 2.0, 0.35)


def test_metadata_attaches_our_dataset_only_in_plus_and_pins_versions():
    base, plus = build_public_fork.metadata("base"), build_public_fork.metadata("plus")
    assert base["is_private"] and plus["is_private"] and not base["enable_internet"]
    assert build_public_fork.OURS_DATASET not in base["dataset_sources"]
    assert plus["dataset_sources"] == base["dataset_sources"] + [build_public_fork.OURS_DATASET]
    assert all(d.count("/") == 2 and d.rsplit("/", 1)[1].isdigit() for d in base["dataset_sources"])
    unpinned = build_public_fork.metadata("base", pinned=False)["dataset_sources"]
    assert unpinned == [d.rsplit("/", 1)[0] for d in base["dataset_sources"]] and len(set(unpinned)) == 17
    assert base["id"] != plus["id"] and base["code_file"].endswith(".ipynb")
