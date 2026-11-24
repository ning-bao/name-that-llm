"""Build private Kaggle notebooks from the best public Name That LLM pipeline, with and without our own member.

The base is alexchilton's "Name That LLM 0.684: All Public + What Failed", version 25 (kernel session 368561683,
public LB 0.684 on the author's account; the same pipeline is public under five other names and 278 teams sit at
exactly 0.684). Pull it first; it is downloaded code, read but never run locally:

    uvx --with six kaggle kernels pull alexchilton/name-that-llm-0-684-all-public-what-failed -p <dir> -m

    uv run python scripts/build_public_fork.py <dir>/name-that-llm-0-684-all-public-what-failed.ipynb base
    uv run python scripts/build_public_fork.py <same ipynb> plus --ours-w 2.0

Each call writes outputs/kaggle/public-fork-684/<name>/ with the notebook and a kernel-metadata.json. Datasets are
pinned to the versions that session ran with (read from its Kaggle run record on 2026-11-12; each was then also the
latest version); `--unpinned` drops the version suffixes. Notebook-output sources carry no version in that record
and are attached as in the public notebook's own metadata.

`base` leaves every code cell byte for byte as version 25 and only prepends a markdown note.

`plus` also adds our Qwen3-8B LoRA24 model (private dataset rhysbao/nllm-q8-l24-c25-001, public LB 0.599 alone and
0.619 with our own centring) as one more native `l24` member. The pipeline's l24 members have our model's
architecture, so kaggle/public_fork/ours_l24.py only rewrites our saved files into the layout infer.py reads; the
pipeline then scores it with its own code and applies its own test-time grouping and twin rule to our centred head.
Changes, all marked `fork (rhysbao)`: one new section before the run cell, and in the run cell our spec at the end
of MODELS plus a commit-time check that infer.py reproduces our model's training-time scores on 72 training rows.
Every patch is an exact string that must occur once, and the source's code cells must hash to version 25's.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / "kaggle" / "public_fork"
SOURCE_SHA256 = "66079ff458808866bb331ccd5ffbda243ad0fca7595e0d00c7fb94e43cdf07a8"  # code cells of version 25
SOURCE = "alexchilton/name-that-llm-0-684-all-public-what-failed, version 25 (kernel session 368561683)"
SOURCE_LB = 0.684

# Inputs of that session. Dataset versions are from its run record; kernels and models as in its kernel-metadata.json.
DOCKER_IMAGE = "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01"
DATASETS = [
    "dhruvtibarewal/nlm-l24-adapters/7",
    "lightningv08/ntl-olmo2-native24-final-inference-assets/22",
    "crtproduction/name-that-llm-open-pool/16",
    "seyitkaangunes/ntl-extra-api-passages/2",
    "abdullah1afify/ntl-generations/11",
    "dophuchung/ntl-proxy-generations/1",
    "zbigniewkirchner/ntl-qwen3-8b-lora-claude-head/1",
    "dhruvtibarewal/nlm-matched-clean/3",
    "dhruvtibarewal/nlm-offline-wheels/2",
    "lucasboesen/ntl-free-proxy-generations/12",
    "nuridudak/ntl-q14-lora/1",
    "nuridudak/ntl-q14-lora-v4/1",
    "nishantdahal/ntl-two-stage-lora/1",
    "nishantdahal/ntl-handwriting-blend/1",
    "dhruvtibarewal/nlm-l24-adapters-v5/1",
    "prvsiyan/prvsiyans-extended-name-that-llm-dataset/13",
    "pjmathematician/ultrabert-380m-ntl/1",
]
KERNELS = [
    "lucasboesen/name-that-llm-glm4-9b-trunc25",
    "lucasboesen/name-that-llm-extract-encoders2",
    "lucasboesen/name-that-llm-electra-large-weights",
    "lucasboesen/name-that-llm-desklib-weights",
    "lucasboesen/name-that-llm-train-lora24-s7",
    "lucasboesen/name-that-llm-train-tokattn",
    "lucasboesen/name-that-llm-train-tokattn2",
    "vnton37nguyn/ntl-train-lora-qwen25-instyle",
]
MODELS = [
    "alexchilton/llama-3-1-8b-base/Transformers/default/1",
    "qwen-lm/qwen-3/Transformers/14b/1",
    "qwen-lm/qwen-3/Transformers/8b/1",
    "qwen-lm/qwen-3/Transformers/4b-base/1",
    "qwen-lm/qwen-3/Transformers/14b-base/1",
    "qwen-lm/qwen-3/Transformers/8b-base/1",
    "qwen-lm/qwen-3-embedding/Transformers/8b/1",
    "qwen-lm/qwen2.5/Transformers/14b/1",
]
OURS_DATASET = "rhysbao/nllm-q8-l24-c25-001"
OURS_MEMBER_DIR = "m0-qwen3-8b-l24-full-final-001"

# ---- the plus variant's patches to the run cell: (old, new); each old string must occur exactly once
RUN_MARKER = "LAYERS = [2, 4, 6, 8, 12, 16, 20, 24]\nMODELS = ["
MODELS_END_OLD = '    {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n]\n'
MODELS_END_NEW = ('    {"name": "deb", "type": "enc", "ckpt": DEB, "w": 1.0},\n'
                  "    OURS_SPEC,   # fork (rhysbao): our Qwen3-8B LoRA24 as one more l24 member\n]\n")
COMMIT_OLD = "    print('prefit core votes smoke ok', {k: v.shape for k, v in smoke.items()})\n"
COMMIT_NEW = (COMMIT_OLD
              + "    # fork (rhysbao): the commit scores our member's 72 reference training rows through infer.py itself\n"
              "    ours_l24.reference_check(OURS_PACKAGE, OURS_SPEC, DATA, run)\n")

OURS_CELL = """# fork (rhysbao): our Qwen3-8B LoRA24 in the layout infer.py's l24 members use; B8 is the pipeline's own 8B base
sys.path.insert(0, os.getcwd())
import ours_l24
OURS_PACKAGE = os.path.dirname(first('/kaggle/input/**/{member_dir}/manifest.json'))
OURS_SPEC = ours_l24.build_member(OURS_PACKAGE, '/kaggle/temp/ours_l24', base=B8, weight={weight!r}, twin={twin!r})
print('ours member', {{k: v for k, v in OURS_SPEC.items() if k != 'adapters'}}, 'from', OURS_PACKAGE)
"""


def source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def markdown(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def code_hash(nb) -> str:
    return hashlib.sha256("\x00".join(source(c) for c in nb["cells"] if c["cell_type"] == "code").encode()).hexdigest()


def find(cells, marker) -> int:
    hits = [i for i, c in enumerate(cells) if c["cell_type"] == "code" and source(c).startswith(marker)]
    if len(hits) != 1:
        raise ValueError(f"{len(hits)} code cells start with {marker!r}")
    return hits[0]


def patch(cells, marker, old, new):
    cell = cells[find(cells, marker)]
    text = source(cell)
    if text.count(old) != 1:
        raise ValueError(f"{old!r} occurs {text.count(old)} times in the {marker!r} cell")
    cell["source"] = text.replace(old, new)


def build(nb, variant, weight, twin):
    nb = copy.deepcopy(nb)
    cells = nb["cells"]
    for c in cells:
        c["source"] = source(c)
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
    nb["metadata"].pop("kaggle", None)  # stale editor settings; kernel-metadata.json sets the inputs
    nb["metadata"].pop("papermill", None)
    if variant == "base":
        cells.insert(0, markdown(f"**Private copy (rhysbao) of {SOURCE}, public LB {SOURCE_LB}.** Every code cell is "
                                 "unchanged; datasets are pinned to the versions that session ran with."))
        return nb
    patch(cells, RUN_MARKER, MODELS_END_OLD, MODELS_END_NEW)
    patch(cells, RUN_MARKER, COMMIT_OLD, COMMIT_NEW)
    ours = [
        markdown("## 0. Our member (fork, rhysbao)\n"
                 "Qwen3-8B-Base truncated to 24 layers, LoRA r16, hidden states mean-pooled at 8 depths, a plain and "
                 "a centred head; stage 1 on the 212k-row public pool, stage 2 on 120 official prompts (private "
                 f"dataset {OURS_DATASET}; public LB 0.599 alone, 0.619 with our own cluster centring). It has the "
                 "architecture of this pipeline's `l24` members, so `ours_l24.py` only rewrites its files into their "
                 f"layout and it joins MODELS at weight {weight} with twin {twin}, scored by `infer.py` like the "
                 "others. On a commit, the run cell first checks that `infer.py` reproduces its training-time scores "
                 "on 72 training rows."),
        code("%%writefile ours_l24.py\n" + (FORK / "ours_l24.py").read_text(encoding="utf-8")),
        code(OURS_CELL.format(member_dir=OURS_MEMBER_DIR, weight=weight, twin=twin)),
    ]
    at = find(cells, RUN_MARKER)
    cells[at:at] = ours
    cells.insert(0, markdown(
        f"**Private fork (rhysbao) of {SOURCE}, public LB {SOURCE_LB}, plus our Qwen3-8B LoRA24 as one more `l24` "
        f"member (weight {weight}, twin {twin}).** Changes, all marked `fork (rhysbao)`: section 0 before the run "
        "cell; in the run cell, `OURS_SPEC` at the end of `MODELS` and one commit-only line that checks our member "
        "through `infer.py`. Datasets are pinned to the versions that session ran with."))
    return nb


def metadata(variant, pinned=True):
    slug = "nllm-public684-base" if variant == "base" else "nllm-public684-plus-ours"
    datasets = DATASETS if pinned else [d.rsplit("/", 1)[0] for d in DATASETS]
    return {
        "id": f"rhysbao/{slug}",
        "title": slug,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "machine_shape": "NvidiaTeslaT4",
        "docker_image": DOCKER_IMAGE,
        "docker_image_pinning_type": "original",
        "dataset_sources": datasets + ([OURS_DATASET] if variant == "plus" else []),
        "kernel_sources": KERNELS,
        "competition_sources": ["name-that-llm"],
        "model_sources": MODELS,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="version 25 of the public notebook (.ipynb)")
    ap.add_argument("variant", choices=["base", "plus"])
    ap.add_argument("--ours-w", type=float, default=2.0, help="our member's weight (the pipeline's own 8B member has 2.0)")
    ap.add_argument("--ours-twin", type=float, default=0.35, help="our centred head's twin weight (0.35 for every public member)")
    ap.add_argument("--unpinned", action="store_true", help="attach the latest version of every dataset")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    nb = json.loads(args.source.read_text(encoding="utf-8"))
    if (digest := code_hash(nb)) != SOURCE_SHA256:
        raise SystemExit(f"{args.source}: code cells hash to {digest}, not version 25's {SOURCE_SHA256}")
    meta = metadata(args.variant, pinned=not args.unpinned)
    out = args.out or ROOT / "outputs" / "kaggle" / "public-fork-684" / (
        "base" if args.variant == "base" else f"plus-w{args.ours_w:g}-t{args.ours_twin:g}")
    out.mkdir(parents=True, exist_ok=True)
    built = build(nb, args.variant, args.ours_w, args.ours_twin)
    (out / meta["code_file"]).write_text(json.dumps(built, indent=1, ensure_ascii=False), encoding="utf-8")
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out / meta['code_file']} and kernel-metadata.json ({meta['id']})")


if __name__ == "__main__":
    main()
