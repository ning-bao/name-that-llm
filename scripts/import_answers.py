"""Import pasted chat answers, rebuild the collection records and decide admission (report E3, E5, E6).

Reads every answer file data/external/answers/<wave>/<model>.txt. Each filled answer not
seen before is appended to data/external/import_log.jsonl, which is never edited. Records
are rebuilt from that log into data/external/records.jsonl (kit schema v1.0), validated,
and admitted or quarantined; admitted rows go to data/external/admitted.parquet in the
competition's row format. An answer deleted or replaced in its file counts as withdrawn.
Programmatic answers (data/external/api_log*.jsonl from collect_agent_platform.py,
collect_anthropic.py and collect_codex.py) become records too, admitted as their own collection method.

Finally each model's admitted text is compared with its official passages on length and
typography. A large gap in curly vs straight quotes or em dashes usually means the copy
path converted characters (smart quotes, autocorrect), though it can also be a genuine
serving difference; either way, check before training on that model's answers.

Usage: uv run python scripts/import_answers.py      (rerun any time; it is idempotent)
       uv run python scripts/import_answers.py --eligibility organizer_confirmed   (once organizers approve)
Writes outputs/external_check/<UTC timestamp>/.
"""

import argparse
import hashlib
import json
from datetime import UTC, datetime

import pandas as pd
from jsonschema import Draft202012Validator, FormatChecker

from nllm import collection
from nllm.data import load_train
from nllm.paths import OUTPUTS
from nllm.splits import development_rows

TYPOGRAPHY = {"em dash": "—", "curly apostrophe": "’", "straight apostrophe": "'",
              "curly double quote": "“", "straight double quote": '"'}
MIN_ROWS_FOR_WARNING = 5


def read_jsonl(path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def style_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per model: rows, median words, and the share of passages containing each typography mark."""
    shares = pd.DataFrame({name: df.text.str.contains(ch, regex=False) for name, ch in TYPOGRAPHY.items()})
    shares["multi_paragraph"] = df.text.str.strip().str.contains(r"\n\s*\n", regex=True)
    table = shares.groupby(df.model.to_numpy()).mean()
    table.insert(0, "median_words", df.text.str.split().str.len().groupby(df.model.to_numpy()).median())
    table.insert(0, "n", df.groupby(df.model.to_numpy()).size())
    return table


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eligibility", default="pending_clarification",
                    choices=["pending_clarification", "organizer_confirmed", "not_eligible"],
                    help="competition-eligibility status of the collection plan (report I1), applied to every record")
    args = ap.parse_args()
    bank = collection.load_bank()
    log = read_jsonl(collection.LOG_PATH)
    seen = {collection.log_key(e) for e in log}
    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    headers, current, problems, new = {}, set(), [], []
    files = sorted(collection.ANSWERS_DIR.glob("*/*.txt"))
    for path in files:
        ref = path.relative_to(collection.EXTERNAL_DIR).as_posix()
        try:
            header, blocks = collection.parse_answer_file(path.read_text(encoding="utf-8-sig"))
        except ValueError as e:
            problems.append(f"{ref}: {e}; file skipped")
            continue
        label = header["model"]
        if label != path.stem or label not in collection.TARGETS:
            problems.append(f"{ref}: header model {label!r} does not match the file name; file skipped")
            continue
        headers[ref] = header
        for pid, prompt_text, answer in blocks:
            if pid not in bank:
                problems.append(f"{ref}: unknown prompt id {pid}; block skipped")
                continue
            if not answer.strip():
                continue
            if not collection.same_prompt(prompt_text, bank[pid]):
                problems.append(f"{ref}: prompt {pid} text was edited; the text in the file is recorded as asked")
            entry = {"label": label, "prompt_id": pid, "sha256": hashlib.sha256(answer.strip().encode("utf-8")).hexdigest(),
                     "answer": answer, "prompt_text": prompt_text.strip(), "first_seen_utc": now, "file": ref}
            current.add(collection.log_key(entry))
            if collection.log_key(entry) not in seen:
                seen.add(collection.log_key(entry))
                new.append(entry)
    if new:
        collection.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(collection.LOG_PATH, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(e, ensure_ascii=False) + "\n" for e in new)
    log += new

    records = collection.build_records(log, headers, bank, current, args.eligibility)
    api_log = collection.read_api_logs()
    records += collection.api_records(api_log, bank, args.eligibility)
    collection.RECORDS_PATH.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records),
                                       encoding="utf-8")
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    schema_errors = [(r["record_id"], e.message) for r in records for e in validator.iter_errors(r)]
    decisions = collection.admission(records, collection.load_released())
    bad = decisions.record_id.isin({rid for rid, _ in schema_errors})
    decisions.loc[bad, "admitted"] = False
    decisions.loc[bad, "reasons"] = (decisions.loc[bad, "reasons"] + "; schema").str.lstrip("; ")
    admitted = collection.admitted_frame(records, decisions, bank)
    admitted.to_parquet(collection.EXTERNAL_DIR / "admitted.parquet", index=False)

    out = OUTPUTS / "external_check" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(out / "admission.csv", index=False)
    print(f"{len(files)} answer files, {len(new)} new answers imported, {len(api_log)} API answers, "
          f"{len(records)} records, {int(decisions.admitted.sum())} admitted, {len(schema_errors)} schema errors")
    for message in problems:
        print(f"  ! {message}")
    for rid, message in schema_errors[:20]:
        print(f"  ! schema {rid}: {message}")
    if not records:
        print("No answers pasted yet. Fill blocks in data/external/answers/<wave>/<model>.txt and rerun.")
        return

    decisions["wave"] = decisions.prompt_id.map(lambda p: bank[p]["wave"])
    decisions["source"] = decisions.interface.str.split(":").str[-1]
    counts = decisions.groupby(["wave", "target_label", "source"]).agg(
        answers=("record_id", "size"), admitted=("admitted", "sum"))
    counts.to_csv(out / "counts.csv")
    pd.set_option("display.width", 220)
    print("\nAnswers and admitted rows by wave, model and interface:")
    print(counts.unstack("wave").fillna(0).astype(int).to_string())
    rejected = decisions.loc[~decisions.admitted, "reasons"].str.split("; ").explode()
    if len(rejected):
        print("\nRejection reasons:")
        print(rejected.str.split(":").str[0].value_counts().to_string())
    kept = pd.Series([i.split(":")[0] for r, ok in zip(records, decisions.set_index("record_id").loc[
        [r["record_id"] for r in records], "admitted"], strict=True) if ok for i in r["quality"]["issues"]], dtype=str)
    if len(kept):
        print("\nFlags on admitted answers (kept, worth a look):")
        print(kept.value_counts().to_string())

    official = style_table(development_rows(load_train()))
    external = style_table(admitted) if len(admitted) else official.iloc[:0]
    shift = pd.concat({"official": official, "external": external.reindex(official.index)}, axis=1)
    shift.to_csv(out / "style_shift.csv")
    warnings = []
    for model, row in external.iterrows():
        if row.n < MIN_ROWS_FOR_WARNING:
            continue
        for mark in TYPOGRAPHY:
            off, ext = official.loc[model, mark], row[mark]
            # A large gap, or a mark one side almost never uses that the other uses often (a fingerprint flip,
            # such as em dashes in 40% of external Gemini answers when official Gemini has almost none).
            if abs(ext - off) >= 0.5 or (min(off, ext) < 0.03 and max(off, ext) >= 0.15):
                warnings.append(f"{model}: {mark} in {row[mark]:.0%} of external answers vs "
                                f"{official.loc[model, mark]:.0%} of official passages")
    columns = [(side, c) for c in ["n", "median_words", *TYPOGRAPHY] for side in ("official", "external")]
    view = shift.loc[external.index, columns]
    view.columns = [f"{c} ({side[:3]})" for side, c in columns]
    print("\nModels with answers: share of passages containing each mark (off = official, ext = external):")
    print(view.round(2).to_string())
    if warnings:
        print("\nTypography gaps (pasted answers: check the copy path for smart-quote or autocorrect conversion; "
              "API answers: a serving difference from the competition data):")
        print("\n".join(f"  ! {w}" for w in warnings))
    print(f"\nwrote {out}, {collection.RECORDS_PATH} and {collection.EXTERNAL_DIR / 'admitted.parquet'}")


if __name__ == "__main__":
    main()
