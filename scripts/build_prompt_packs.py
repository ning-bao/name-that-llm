"""Build the external-data prompt bank and the copy-ready collection files (report E1, E2, E6).

Writes, from the kit's prompt bank plus collection/prompts_v2.tsv:
  collection/prompt_bank.json          every prompt with id, family, split, wave and wrapper
  collection/prompts/<wave>.md         the prompts of one wave, ready to copy into a chat
  data/external/answers/<wave>/<model>.txt   one answer file per collectable model and wave

wave1 is the kit's 24 core + 6 holdout prompts; wave2-wave5 add 40 prompts each (32 train,
8 external holdout chosen by hash within length band); the kit's 6 diagnostic prompts are
a separate optional pack. Existing answer files are never overwritten.

Usage: uv run python scripts/build_prompt_packs.py
"""

import json

import pandas as pd

from nllm import collection


def main() -> None:
    bank = collection.build_bank()
    collection.BANK_PATH.write_text(json.dumps(bank, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    frame = pd.DataFrame(bank)
    md_dir = collection.COLLECTION_DIR / "prompts"
    md_dir.mkdir(parents=True, exist_ok=True)

    created = kept = 0
    for wave, group in frame.groupby("wave", sort=True):
        prompts = collection.wave_order(group.to_dict("records"), wave)
        lines = [
            f"# {wave}: {len(prompts)} prompts",
            "",
            f"Ask each of the {len(collection.TARGETS)} models every prompt below, one fresh chat per prompt, and paste "
            f"the first complete answer into `data/external/answers/{wave}/<model>.txt` (same order as here). "
            "Do not edit or regenerate answers. Not collected: "
            + "; ".join(f"{m} ({why})" for m, why in collection.UNCOLLECTABLE.items()) + ".",
        ]
        for p in prompts:
            lines += ["", f"## {p['prompt_id']}", "", "```text", p["prompt_text"], "```"]
        (md_dir / f"{wave}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        answer_dir = collection.ANSWERS_DIR / wave
        answer_dir.mkdir(parents=True, exist_ok=True)
        for label in collection.TARGETS:
            path = answer_dir / f"{label}.txt"
            if path.exists():
                kept += 1
                continue
            path.write_text(collection.answer_template(label, wave, prompts), encoding="utf-8")
            created += 1

    summary = frame.pivot_table(index="wave", columns="split", values="prompt_id", aggfunc="count", fill_value=0)
    summary["responses_to_collect"] = summary.sum(axis=1) * len(collection.TARGETS)
    new = frame[frame.bank_version == collection.BANK_VERSION]
    print(summary.to_string())
    print(f"\nnew prompts by length band:\n{new.requested_words.value_counts().sort_index().to_string()}")
    print(f"\nnew prompts by genre:\n{new.genre.value_counts().to_string()}")
    print(f"\nnew prompts by domain:\n{new.domain.value_counts().to_string()}")
    print(f"\nwrote {collection.BANK_PATH}, {md_dir}/*.md; answer files: {created} created, {kept} already existed")


if __name__ == "__main__":
    main()
