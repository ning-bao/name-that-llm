"""Manual external-data collection (report section E; the experiment-5 pilot).

The owner asks every prompt-bank prompt in a fresh chat with each target model and pastes
the first complete answer into a per-model answer file. This module builds the prompt
bank, writes those files, parses them back into the kit's collection records
(collection_record.schema.json, v1.0), flags copy problems and decides admission for
exact-label training. There are no API clients, and competition passages never appear
in a prompt.

Every distinct pasted answer is appended once to an import log, which is never edited;
records are rebuilt from that log on each import, so an answer that is later edited or
replaced in its file keeps its place as the first attempt.
"""

import hashlib
import json
import random
import re

import pandas as pd

from nllm.data import LABELS
from nllm.paths import KIT_DIR, ROOT

SCHEMA_PATH = KIT_DIR / "collection_record.schema.json"
KIT_BANK_PATH = KIT_DIR / "collection_prompt_bank.json"
COLLECTION_DIR = ROOT / "collection"
NEW_PROMPTS_PATH = COLLECTION_DIR / "prompts_v2.tsv"
BANK_PATH = COLLECTION_DIR / "prompt_bank.json"
EXTERNAL_DIR = ROOT / "data" / "external"
ANSWERS_DIR = EXTERNAL_DIR / "answers"
LOG_PATH = EXTERNAL_DIR / "import_log.jsonl"
RECORDS_PATH = EXTERNAL_DIR / "records.jsonl"

UNCOLLECTABLE = {
    "gemini-3.7-flash": "Google routes 3.7 requests to 3.8 since 8 Oct 2026; never label 3.8 output as 3.7",
}
TARGETS = tuple(m for m in LABELS if m not in UNCOLLECTABLE)

BANK_VERSION = "2.0"
HOLDOUT_SALT = "name-that-llm-external-holdout-v1|"
# Per wave of 40 new prompts, by requested length: 8 external-holdout families (20%), in
# proportion to the official length mix (38% / 33% / 19% / 10% across the four bands).
HOLDOUT_PER_BAND = {"40-70": 3, "90-130": 3, "160-210": 1, "220-250": 1}
WRAPPERS = {
    "paragraph-length-v1": "Write one paragraph of about {band} words.",
    "paragraph-length-v2": "Answer in a single paragraph of roughly {band} words.",
    "paragraph-length-v3": "Keep it to one paragraph of around {band} words.",
}
SCENE_WRAPPER = ("scene-length-v1", "Use one paragraph of about {band} words.")

ADMISSIBLE_STATUS = ("snapshot_verified", "nominal_target_verified", "ui_label_verified")
# Flags that send a record to quarantine. The first group are copy problems: a corrected
# re-paste of the same prompt may still be admitted. A refusal is the model's answer, so a
# re-asked prompt after it is not (no answer shopping, report E3 step 5).
COPY_PROBLEMS = {"empty", "app_clutter", "prompt_echo", "possible_truncation", "withdrawn"}
QUARANTINE_FLAGS = COPY_PROBLEMS | {"possible_refusal"}

HEADER_DEFAULTS = {
    "model": "",
    "displayed_model_label": "",
    "model_label_checked_in_app": "no",
    "fresh_chat_per_prompt": "yes",
    "memory_and_custom_instructions": "unknown",
    "mode_or_reasoning_setting": "",
    "collected_on": "",
    "collector": "",
}
BLOCK = re.compile(
    r"<<<PROMPT (?P<id>[^>\s]+)>>>\n(?P<prompt>.*?)\n<<<ANSWER (?P=id)>>>\n(?P<answer>.*?)<<<END (?P=id)>>>", re.DOTALL
)
# Interface text that copying a whole chat turn tends to drag along. Only short standalone lines
# are tested, so ordinary prose that happens to start with one of these words is not flagged.
CLUTTER = re.compile(
    r"(copy|copied!?|copy code|share|edit|retry|regenerate|read aloud|like|dislike|good response|bad response"
    r"|show (thinking|drafts|more)|hide thinking|thinking( \.\.\.|…)?|thought for .+|reasoned for .+|you said:?"
    r"|[\w .-]{1,30} said:|sources?|searched .+|\d+ ?/ ?\d+|was this (response|answer) (better|helpful).*"
    r"|(chatgpt|claude|gemini|kimi|qwen|glm|z\.ai)( [\w.-]+){0,3})",
    re.IGNORECASE,
)
REFUSAL = re.compile(r"\s*(I'm sorry|I’m sorry|I am sorry|I can't|I can’t|I cannot|I won't|I won’t|Sorry,)")
ODD_WHITESPACE = re.compile("[\u00a0\u200b\u200c\u200d\ufeff\t]")


def _ranked(ids, salt: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode("utf-8")).hexdigest())


def build_bank() -> list[dict]:
    """The kit's 36 prompts (wave1 plus a separate diagnostic pack) and the new waves from prompts_v2.tsv."""
    bank = []
    for p in json.loads(KIT_BANK_PATH.read_text(encoding="utf-8")):
        wave = "diagnostic" if p["split"] == "diagnostic" else "wave1"
        bank.append({**p, "wave": wave})
    new = pd.read_csv(NEW_PROMPTS_PATH, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    holdout = set()
    for (_, band), group in new.groupby(["wave", "band"]):
        holdout |= set(_ranked(group.prompt_id, HOLDOUT_SALT)[: HOLDOUT_PER_BAND[band]])
    for i, row in enumerate(new.itertuples()):
        wrapper_id, wrapper = (
            SCENE_WRAPPER if row.genre == "creative scene" else list(WRAPPERS.items())[i % len(WRAPPERS)]
        )
        bank.append({
            "prompt_id": row.prompt_id,
            "split": "external_holdout" if row.prompt_id in holdout else "train",
            "domain": row.domain,
            "genre": row.genre,
            "requested_words": row.band,
            "prompt_text": f"{row.content} {wrapper.format(band=row.band)}",
            "bank_version": BANK_VERSION,
            "origin": "independently authored for this project; contains no competition text",
            "prompt_family_id": row.prompt_id,
            "template_family_id": row.prompt_id,
            "wrapper_id": wrapper_id,
            "wave": f"wave{row.wave}",
        })
    if len({p["prompt_id"] for p in bank}) != len(bank):
        raise ValueError("duplicate prompt ids in the bank")
    return bank


def load_bank() -> dict[str, dict]:
    return {p["prompt_id"]: p for p in json.loads(BANK_PATH.read_text(encoding="utf-8"))}


def wave_order(prompts: list[dict], wave: str) -> list[dict]:
    """A fixed shuffle per wave, so consecutive prompts vary in length and domain."""
    order = sorted(prompts, key=lambda p: p["prompt_id"])
    random.Random(f"order|{wave}").shuffle(order)
    return order


def answer_template(label: str, wave: str, prompts: list[dict]) -> str:
    lines = [
        f"# Name That LLM answers: {label}, {wave} ({len(prompts)} prompts).",
        "# For each prompt, open a NEW chat with this model (memory and custom instructions off if the app",
        "# allows), paste the prompt exactly as written, and paste the model's FIRST complete answer between",
        "# its ANSWER and END lines. Do not edit the answer or regenerate it for a nicer one. Copy only the",
        "# answer text, not buttons, model names or 'thought for' lines. Leave a block empty to skip it; the",
        "# importer reads only filled blocks and can be rerun at any time.",
        "# Header: set model_label_checked_in_app to yes once the app's model picker shows exactly this",
        "# model. Until then its answers stay out of exact-label training. Other fields are optional.",
        "",
        *(f"{k}: {label if k == 'model' else v}" for k, v in HEADER_DEFAULTS.items()),
    ]
    for p in prompts:
        pid = p["prompt_id"]
        lines += ["", f"<<<PROMPT {pid}>>>", p["prompt_text"], f"<<<ANSWER {pid}>>>", "", f"<<<END {pid}>>>"]
    return "\n".join(lines) + "\n"


def parse_answer_file(text: str) -> tuple[dict, list[tuple[str, str, str]]]:
    """(header, [(prompt_id, prompt text as in the file, raw answer)]) for every block, filled or not."""
    text = text.replace("\r\n", "\n").lstrip("\ufeff")
    header = dict(HEADER_DEFAULTS)
    for line in text.split("<<<PROMPT", 1)[0].splitlines():
        if not line.startswith("#") and ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    blocks = [(m["id"], m["prompt"], m["answer"]) for m in BLOCK.finditer(text)]
    if text.count("<<<PROMPT ") != len(blocks):
        found = {b[0] for b in blocks}
        broken = [i for i in re.findall(r"<<<PROMPT (\S+?)>>>", text) if i not in found]
        raise ValueError(f"malformed answer blocks (markers edited or missing): {broken}")
    return header, blocks


def quality(answer: str, prompt_text: str, requested_words: str | None) -> dict:
    words = len(answer.split())
    issues = []
    if not answer:
        issues.append("empty")
    if any(len(line.strip()) <= 40 and CLUTTER.fullmatch(line.strip()) for line in answer.splitlines()):
        issues.append("app_clutter")
    if prompt_text and answer.startswith(prompt_text[:40]):
        issues.append("prompt_echo")
    if answer and answer[-1] not in ".!?\"'”’)…*":
        issues.append("possible_truncation")
    refusal = bool(REFUSAL.match(answer))
    if refusal:
        issues.append("possible_refusal")
    if re.search(r"^\s*(#{1,6} |[-*•] |\d+\. )|\*\*|__", answer, flags=re.MULTILINE):
        issues.append("markdown")
    if re.search(r"\n\s*\n", answer):
        issues.append("multi_paragraph")
    if ODD_WHITESPACE.search(answer):
        issues.append("unusual_whitespace")
    if requested_words and (m := re.fullmatch(r"(\d+)-(\d+)", requested_words)):
        lo, hi = int(m[1]), int(m[2])
        if not lo <= words <= hi:
            issues.append(f"length_outside_request:{requested_words}")
        if words < lo / 2 or words > hi * 1.5:
            issues.append("length_far_outside_request")
    return {
        "word_count": words,
        "tokenizer_probe": None,
        "token_count": None,
        "complete": bool(answer) and "possible_truncation" not in issues,
        "refusal": refusal,
        "tool_use_observed": False,
        "duplicate_of": None,
        "issues": issues,
    }


def _yes(value: str) -> bool:
    return value.strip().lower() in ("yes", "y", "true", "1")


def make_record(entry: dict, attempt: int, parent: str | None, header: dict, prompt: dict,
                withdrawn: bool = False, eligibility: str = "pending_clarification") -> dict:
    """One schema-v1.0 record for one logged answer; header fields come from its answer file.

    `withdrawn` marks an answer that has since been deleted or replaced in its file (a mis-paste).
    `eligibility` is the competition-eligibility status of the whole collection plan (report I1).
    """
    answer = entry["answer"].strip()  # the paste's surrounding blank lines are not part of the answer
    raw_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    label = entry["label"]
    checked = _yes(header.get("model_label_checked_in_app", ""))
    shown = header.get("displayed_model_label") or None
    setting = {"off": "off", "on": "on"}.get(header.get("memory_and_custom_instructions", "").lower(), "unknown")
    date = header.get("collected_on", "")
    collected = f"{date}T00:00:00Z" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else entry["first_seen_utc"]
    checks = quality(answer, prompt["prompt_text"], prompt.get("requested_words"))
    if withdrawn:
        checks["issues"].append("withdrawn")
    fresh = _yes(header.get("fresh_chat_per_prompt", ""))
    quarantine = (bool(QUARANTINE_FLAGS & {i.split(":")[0] for i in checks["issues"]})
                  or not checked or not fresh)
    return {
        "record_id": "x-" + hashlib.sha256(f"{label}|{entry['prompt_id']}|{raw_sha}".encode()).hexdigest()[:16],
        "schema_version": "1.0",
        "target_label": label,
        "served_model_id": None,
        "displayed_model_label": shown,
        "snapshot_id": None,
        "provider": "unspecified",
        "interface": "manual_chat:unspecified",
        "interface_version": None,
        "endpoint_or_product_url": None,
        "region": None,
        "collected_at_utc": collected,
        "prompt_id": prompt["prompt_id"],
        "prompt_family_id": prompt["prompt_family_id"],
        "template_family_id": prompt["template_family_id"],
        "wrapper_id": prompt.get("wrapper_id"),
        "prompt_bank_version": prompt["bank_version"],
        "prompt_text": prompt["prompt_text"],
        "raw_response": answer,
        "rendered_text_export": None,
        "raw_response_sha256": raw_sha,
        "visible_reasoning": None,
        "final_answer_span": None,
        "source_verification": {
            "status": "ui_label_verified" if checked else "family_proxy",
            "evidence": [f"operator confirmed the app's model picker showed {shown or label}"] if checked
            else ["model label not yet confirmed by the operator"],
            "routing_uncertainty": "consumer chat: the provider's routing, system instructions and served snapshot "
                                   "are not observable",
            "competition_snapshot_match": "unknown",
            "verification_date": None,
        },
        "session": {
            "fresh_conversation": fresh,
            "prior_turn_count": 0 if fresh else 1,
            "session_id_hash": None,
            "memory": setting,
            "personalization": setting,
            "custom_instructions": "empty" if setting == "off" else "unknown",
            "system_instructions_status": "provider_default_unknown",
            "known_system_instructions": None,
            "tools": [],
            "model_fallback_observed": None,
        },
        "settings": {
            "temperature": None,
            "top_p": None,
            "seed": None,
            "max_output_tokens": None,
            "reasoning_effort": header.get("mode_or_reasoning_setting") or None,
            "settings_source": "consumer chat defaults; values unknown",
        },
        "provenance": {
            "collection_method": "manual_chat",
            "copy_method": "pasted into a plain-text answer file; CRLF read as LF",
            "collector_id": header.get("collector") or "owner",
            "original_artifact_reference": entry["file"],
            "license_or_terms_url": None,
            "source_dataset_url": None,
            "eligibility_status": eligibility,
            "eligibility_evidence": None,
            "transforms": ["outer_whitespace_strip"] if answer != entry["answer"] else [],
            "attempt_index": attempt,
            "parent_record_id": parent,
            "request_id": None,
        },
        "quality": checks,
        "cost": {"input_tokens": None, "output_tokens": None, "reasoning_tokens": None, "usd": None,
                 "operator_minutes": None},
        "split": "quarantine" if quarantine else prompt["split"],
    }


def log_key(entry: dict) -> tuple[str, str, str]:
    return entry["label"], entry["prompt_id"], entry["sha256"]


def build_records(log: list[dict], headers: dict[str, dict], bank: dict[str, dict],
                  current: set[tuple[str, str, str]], eligibility: str = "pending_clarification") -> list[dict]:
    """Records for every logged answer: attempts numbered by first-seen order within each (model, prompt).

    `headers` maps answer-file references to their headers; `current` holds the log keys of answers
    still present in their files, and any other logged answer is marked withdrawn.
    """
    records = []
    by_pair: dict[tuple[str, str], list[dict]] = {}
    for entry in sorted(log, key=lambda e: e["first_seen_utc"]):
        by_pair.setdefault((entry["label"], entry["prompt_id"]), []).append(entry)
    for (label, pid), entries in sorted(by_pair.items()):
        parent = None
        for attempt, entry in enumerate(entries, start=1):
            record = make_record(entry, attempt, parent, headers.get(entry["file"], HEADER_DEFAULTS), bank[pid],
                                 withdrawn=log_key(entry) not in current, eligibility=eligibility)
            parent = parent or record["record_id"]
            records.append(record)
    return records


def admission(records: list[dict]) -> pd.DataFrame:
    """Per-record admission for exact-label training (report E3 step 11).

    Within each (model, prompt) the first passing attempt is admitted; a later attempt is only
    considered when every earlier one failed on a copy problem. Text that occurs more than once
    (exactly, or after lowercasing and whitespace collapsing) is rejected everywhere, since the
    paste cannot be attributed.
    """
    norm = {r["record_id"]: re.sub(r"\s+", " ", r["raw_response"].lower()).strip() for r in records}
    text_count = pd.Series(list(norm.values())).value_counts()
    rows, pair_state = [], {}
    for r in sorted(records, key=lambda r: (r["target_label"], r["prompt_id"], r["provenance"]["attempt_index"])):
        flags = {i.split(":")[0] for i in r["quality"]["issues"]}
        reasons = sorted(QUARANTINE_FLAGS & flags)
        if r["source_verification"]["status"] not in ADMISSIBLE_STATUS:
            reasons.append(f"verification:{r['source_verification']['status']}")
        if not r["session"]["fresh_conversation"]:
            reasons.append("not a fresh chat")
        if r["target_label"] not in TARGETS:
            reasons.append("not a collectable label")
        if r["provenance"]["eligibility_status"] == "not_eligible":
            reasons.append("not eligible")
        if norm[r["record_id"]] and text_count[norm[r["record_id"]]] > 1:
            reasons.append("duplicate_text")
        pair = r["target_label"], r["prompt_id"]
        state = pair_state.get(pair)
        if state == "admitted":
            reasons.append("later_answer_for_pair")
        elif state == "blocked":
            reasons.append("follows_a_rejected_answer")
        if not reasons:
            pair_state[pair] = "admitted"
        elif state is None and not set(reasons) <= COPY_PROBLEMS:
            pair_state[pair] = "blocked"
        rows.append({
            "record_id": r["record_id"],
            "target_label": r["target_label"],
            "prompt_id": r["prompt_id"],
            "attempt_index": r["provenance"]["attempt_index"],
            "split": r["split"],
            "admitted": not reasons,
            "reasons": "; ".join(reasons),
        })
    return pd.DataFrame(rows, columns=["record_id", "target_label", "prompt_id", "attempt_index", "split",
                                       "admitted", "reasons"])


def admitted_frame(records: list[dict], decisions: pd.DataFrame, bank: dict[str, dict]) -> pd.DataFrame:
    """Admitted records in the competition's row format. Prompt groups get an 'x' prefix so they never
    collide with official prompt ids; the other columns are for grouping and audit, never features."""
    by_id = {r["record_id"]: r for r in records}
    keep = [by_id[i] for i in decisions.loc[decisions.admitted, "record_id"]]
    return pd.DataFrame({
        "id": [r["record_id"] for r in keep],
        "prompt_id": ["x" + r["prompt_family_id"] for r in keep],
        "model": [r["target_label"] for r in keep],
        "text": [r["raw_response"] for r in keep],
        "split": [bank[r["prompt_id"]]["split"] for r in keep],
        "wave": [bank[r["prompt_id"]]["wave"] for r in keep],
        "eligibility_status": [r["provenance"]["eligibility_status"] for r in keep],
    }, columns=["id", "prompt_id", "model", "text", "split", "wave", "eligibility_status"])
