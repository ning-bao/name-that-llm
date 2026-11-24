"""The single use of the 30 locked confirmation prompts (report section F), on the frozen recipe.

The recipe is the Kaggle submission's: one --final pooled-decoder run (trained on the 120 development prompts
only, so every confirmation prompt is unseen), scored the way kaggle/nllm_infer.py scores it (adapters merged by
hand into the fp16 base, fp16 autocast), then test-time centring (nllm.centring) at the packaged setting. The
confirmation rows are scored whole and also cut to m answers per prompt (as scripts/test_time_centring.py does),
with clusters recovered on each cut. Writes outputs/confirmation/<run>/ and refuses to run again once
outputs/confirmation/USED.json exists: the set is used once, then it becomes training data for the finals.

    uv run --extra nn python scripts/confirm_locked.py outputs/tc2/qwen3-8b-l24-full-final-001 \
        --base outputs/kaggle/qwen3-8b-base-49e3418f/model --centring 0.25 0.85
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from nllm import centring, inference
from nllm.data import LABELS, load_train
from nllm.metrics import balanced_accuracy, interval, per_class_recall, prompt_bootstrap
from nllm.paths import OUTPUTS
from nllm.splits import confirmation_prompts, select

NAMES = np.asarray(LABELS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path, help="the frozen --final run (pooled decoder)")
    ap.add_argument("--base", type=Path, required=True, help="the base model snapshot it was trained on")
    ap.add_argument("--centring", type=float, nargs=2, metavar=("WEIGHT", "CUT"), required=True)
    ap.add_argument("--m", type=int, nargs="+", default=[2, 3, 6, 18])
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--outputs", type=Path, default=OUTPUTS, help="outputs root holding confirmation/ (default: the repo's)")
    args = ap.parse_args()
    used = args.outputs / "confirmation" / "USED.json"
    if used.exists():
        raise SystemExit(f"the confirmation set was already used: {used.read_text(encoding='utf-8')}")
    manifest = json.loads((args.run / "manifest.json").read_text(encoding="utf-8"))
    if manifest["train_prompts"] != 120:
        raise SystemExit(f"{args.run} trained on {manifest['train_prompts']} prompts, not the 120 dev prompts")

    rows = select(load_train(), confirmation_prompts(unlock=True)).reset_index(drop=True)
    print(f"{len(rows)} confirmation rows, {rows.prompt_id.nunique()} prompts; model {args.run.name}", flush=True)
    model = inference.load_pooled_decoder(args.base, args.run / "model", dtype=torch.float16,
                                          device_map={"": torch.device("cuda")} if torch.cuda.is_available() else None)
    ids = inference.tokenize_decoder(args.run / "model", rows.text, manifest["max_length"])
    scores = inference.predict_logits(model, ids, manifest["pad_id"], batch_size=args.batch_size, dtype=torch.float16)
    del model
    y, prompts = rows.model.to_numpy(), rows.prompt_id.to_numpy()
    weight, cut = args.centring

    out = args.outputs / "confirmation" / args.run.name
    out.mkdir(parents=True, exist_ok=True)
    plain = NAMES[scores.argmax(axis=1)]
    results = []

    def report(name, m, idx_draws, base_preds, preds):
        r = np.concatenate(idx_draws)
        b, p = np.concatenate(base_preds), np.concatenate(preds)
        lo, hi = interval(prompt_bootstrap(y[r], p, prompts[r]))
        d = prompt_bootstrap(y[r], p, prompts[r], y_pred_baseline=b)
        dlo, dhi = interval(d)
        ba_b = np.mean([balanced_accuracy(y[i], x) for i, x in zip(idx_draws, base_preds, strict=True)])
        ba_p = np.mean([balanced_accuracy(y[i], x) for i, x in zip(idx_draws, preds, strict=True)])
        results.append({"setting": name, "m": m, "ba_plain": ba_b, "ba": ba_p, "ci95_lo": lo, "ci95_hi": hi,
                        "diff": ba_p - ba_b, "diff_ci95_lo": dlo, "diff_ci95_hi": dhi})
        print(f"{name:>9} m={m!s:>4}: plain {ba_b:.4f}, centred {ba_p:.4f} [{lo:.4f}, {hi:.4f}], "
              f"diff {ba_p - ba_b:+.4f} [{dlo:+.4f}, {dhi:+.4f}]", flush=True)

    whole = np.arange(len(rows))
    centred = centring.centre(scores, centring.content_clusters(rows.text, cut), weight)
    report("whole", "all", [whole], [plain], [NAMES[centred.argmax(axis=1)]])
    rng = np.random.default_rng(20261108)
    groups = list(rows.groupby("prompt_id").indices.values())
    for m in args.m:
        cuts, base_preds, preds = [], [], []
        for _ in range(args.draws):
            r = np.sort(np.concatenate([rng.permutation(g)[:m] for g in groups]))
            c = centring.centre(scores[r], centring.content_clusters(rows.text.to_numpy()[r], cut), weight)
            cuts.append(r)
            base_preds.append(plain[r])
            preds.append(NAMES[c.argmax(axis=1)])
        report("cut", m, cuts, base_preds, preds)

    frame = rows[["id", "prompt_id", "model"]].assign(pred=plain, pred_centred=NAMES[centred.argmax(axis=1)])
    pd.concat([frame, pd.DataFrame(scores, columns=[f"score:{m}" for m in LABELS])], axis=1).to_parquet(
        out / "scores.parquet", index=False)
    pd.DataFrame(results).to_csv(out / "results.csv", index=False)
    pd.DataFrame({"plain": per_class_recall(y, plain),
                  "centred": per_class_recall(y, NAMES[centred.argmax(axis=1)])}).to_csv(out / "per_class_recall.csv")
    record = {"used_at": datetime.now(UTC).isoformat(timespec="seconds"), "run": str(args.run),
              "code_id": manifest.get("code_id"), "centring": {"weight": weight, "cut": cut, "cavity": True},
              "rows": len(rows), "prompts": int(rows.prompt_id.nunique()),
              "ba_plain": balanced_accuracy(y, plain), "ba_centred": balanced_accuracy(y, frame.pred_centred)}
    used.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"wrote {out}; marked the confirmation set used ({used})")


if __name__ == "__main__":
    main()
