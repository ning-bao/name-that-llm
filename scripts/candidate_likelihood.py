"""Score passages under an open-weight candidate model (GLM-5.3-Flash or GLM-5.3) read from GGUF layer segments
(nllm.candidate_lm), and save per-token statistics and per-passage features.

Each passage is presented as a finished assistant answer, the way the model's chat template renders one:
    [gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{prompt}<|assistant|><think></think>{passage}
The user prompt is unknown at test time, so one generic instruction stands in for it. Only passage tokens are
scored. Four fixed texts are always scored too and printed: fluent English must be far more likely than the same
words shuffled, which is the end-to-end check that every tensor was read correctly.

--follow makes each layer wait until its segment file has finished downloading (a `<file>.done` marker next to
it), and --delete-used deletes a segment once its last layer has run, so a model larger than the free disk can
be scored while it downloads (GLM-5.3 is 278 GB). Before a segment is deleted the hidden states are written to
`<out>.checkpoint`, and a rerun with the same arguments resumes from that file.

    uv run --extra nn --extra candidates python scripts/candidate_likelihood.py --model flash \
        --segments data/external/third_party/jinghengwu__ntl-flash-iq3xxs-segs --model-dir <config + tokenizer> \
        --rows official --out outputs/candidates/flash.npz

Model directories hold config.json and the tokenizer files of zai-org/GLM-5.3-Flash or zai-org/GLM-5.3.
"""

import argparse
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from nllm import candidate_lm
from nllm.gguf_segments import Segments, read_header

PROMPT = "Write a single paragraph."
SANITY = (
    ("The tide comes in twice a day because the Moon pulls on the oceans, and the Earth turns beneath the bulge of "
    "water that this pull creates. Along most coasts the sea therefore rises and falls on a schedule that sailors "
    "have relied on for thousands of years."),
    ("Bread rises because yeast eats the sugars in the dough and gives off carbon dioxide, which is trapped by the "
    "stretchy network of gluten. When the loaf goes into the oven the gas expands, the crust sets, and the crumb "
    "keeps the shape of all those small bubbles."),
)


def sanity_texts() -> tuple[list[str], list[str]]:
    rng = random.Random(0)
    texts, names = [], []
    for k, text in enumerate(SANITY):
        shuffled = text.split()
        rng.shuffle(shuffled)
        texts += [text, " ".join(shuffled)]
        names += [f"sanity:english {k}", f"sanity:shuffled {k}"]
    return texts, names


class SegmentFiles:
    """Finds the segment file of a tensor, optionally waiting for its download and deleting files after use."""

    def __init__(self, directory: Path, dtype, follow: bool, delete_used: bool):
        self.directory, self.dtype, self.follow, self.delete_used = directory, dtype, follow, delete_used

    def holding(self, name: str) -> Segments:
        while True:
            segments = Segments(self.directory, self.dtype)
            if name in segments.where:
                if not self.follow or Path(str(segments.where[name][0]) + ".done").exists():
                    return segments
            elif not self.follow:
                raise KeyError(f"{name} is in no segment under {self.directory}")
            time.sleep(15)

    def release(self, i: int, save) -> None:
        """Delete the segment whose last decoder layer is i (never the one that holds the output head), after
        `save()` has written a checkpoint that makes those layers unnecessary."""
        if not self.delete_used:
            return
        for path in self.directory.glob("*.gguf"):
            m = re.search(r"-L(\d+)-(\d+)\.gguf$", path.name)
            if m and int(m.group(2)) - 1 == i and "output.weight" not in read_header(path)[1]:
                save()
                path.unlink()
                print(f"  checkpoint written, deleted {path.name}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=sorted(candidate_lm.ARCHITECTURES), required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True, help="config.json and tokenizer from the Hub")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--rows", choices=["sanity", "dev", "official"], default="dev",
                    help="official: all of train.csv (the 30 confirmation prompts are only ever final training data)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--max-passage-tokens", type=int, default=320)
    ap.add_argument("--max-tokens", type=int, default=12000, help="padded tokens per batch")
    ap.add_argument("--max-rows", type=int, default=64)
    ap.add_argument("--group-tokens", type=int, default=300_000, help="tokens on the GPU at once in the expert sweep")
    ap.add_argument("--chunk", type=int, default=8, help="experts unpacked at once")
    ap.add_argument("--pool-layers", type=int, nargs="*",
                    help="also save each passage's mean hidden state after these layers (default: about every "
                         "eighth layer and the last); pass no value to save none")
    ap.add_argument("--follow", action="store_true")
    ap.add_argument("--delete-used", action="store_true")
    ap.add_argument("--cpu", action="store_true", help="float32 on the CPU (a slow reference)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    prefix_text = f"[gMASK]<sop><|system|>Reasoning Effort: Max<|user|>{args.prompt}<|assistant|><think></think>"
    prefix = tok(prefix_text, add_special_tokens=False)["input_ids"]
    if prefix[:2] != tok.convert_tokens_to_ids(["[gMASK]", "<sop>"]):
        raise SystemExit("the chat prefix was not tokenised with its special tokens")
    texts, names = sanity_texts()
    if args.rows != "sanity":
        from nllm.data import load_train
        from nllm.splits import development_rows

        rows = development_rows(load_train()) if args.rows == "dev" else load_train()
        rows = rows.head(args.limit) if args.limit else rows
        texts, names = texts + rows.text.tolist(), names + rows.id.astype(str).tolist()
    ids = [prefix + tok(t, add_special_tokens=False, split_special_tokens=True)["input_ids"][: args.max_passage_tokens]
           for t in texts]
    print(f"{args.model}: {len(ids)} sequences, {sum(map(len, ids))} tokens (prefix {len(prefix)}), longest "
          f"{max(map(len, ids))}", flush=True)

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    arch_class = candidate_lm.ARCHITECTURES[args.model]
    arch = arch_class(arch_class.config(args.model_dir))
    files = SegmentFiles(args.segments, dtype, args.follow, args.delete_used)
    n_layers = arch.cfg.num_hidden_layers
    checkpoint = args.out.with_suffix(".checkpoint")  # written before a segment is deleted; a rerun resumes from it
    pool_layers = sorted(set(range(8, n_layers, 8)) | {n_layers}) if args.pool_layers is None else args.pool_layers
    started, pooled = time.time(), {}
    stats = candidate_lm.score(arch, ids, device, *candidate_lm.segment_model(arch, files.holding, device, dtype),
                               dtype=dtype, max_tokens=args.max_tokens, max_rows=args.max_rows,
                               after_layer=files.release, group_tokens=args.group_tokens, chunk=args.chunk,
                               pooled=pooled, pool_layers=pool_layers, pool_from=len(prefix),
                               checkpoint=checkpoint if args.delete_used or checkpoint.exists() else None)
    checkpoint.unlink(missing_ok=True)
    # stats[t] predicts token t + 1, so the first passage token (index len(prefix)) is stats[len(prefix) - 1].
    stats = [{k: v[len(prefix) - 1:] for k, v in st.items()} for st in stats]
    feats = [candidate_lm.passage_features(st) for st in stats]
    print(f"scored in {time.time() - started:.0f}s", flush=True)
    for name, f in zip(names, feats, strict=True):
        if name.startswith("sanity:"):
            print(f"{name:20s} tokens {f['n_tokens']:4.0f}  mean log-prob {f['lp_mean']:7.3f}  perplexity "
                  f"{np.exp(-f['lp_mean']):10.1f}  top-1 {f['top1']:.2f}  entropy {f['entropy']:.2f}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, ids=np.array(names), feature_names=np.array(list(feats[0])),
             features=np.array([[f[k] for k in feats[0]] for f in feats], dtype=np.float32),
             lengths=np.array([len(st["lp"]) for st in stats]), prompt=np.array(args.prompt), model=np.array(args.model),
             pool_layers=np.array(sorted(pooled)),
             pooled=np.stack([pooled[k] for k in sorted(pooled)], axis=1).astype(np.float16) if pooled else np.zeros(0),
             **{f"tok_{k}": np.concatenate([st[k] for st in stats]).astype(np.float32) for k in stats[0]})
    print(f"wrote {args.out}\nSCORING DONE", flush=True)


if __name__ == "__main__":
    main()
