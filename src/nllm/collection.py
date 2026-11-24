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
# Community prompts from the Open Pool, screened by scripts/select_pool_prompts.py. They are third-party
# (CC BY-SA 4.0), so they live under data/ and join the bank in load_bank(), not in prompt_bank.json.
POOL_PROMPTS_PATH = EXTERNAL_DIR / "pool_prompts.tsv"
POOL_PROMPTS_ORIGIN = ("community prompt from crtproduction/name-that-llm-open-pool v13 prompts.parquet (CC BY-SA 4.0); "
                       "screened for competition text, model names and style instructions")
ANSWERS_DIR = EXTERNAL_DIR / "answers"
LOG_PATH = EXTERNAL_DIR / "import_log.jsonl"
RECORDS_PATH = EXTERNAL_DIR / "records.jsonl"
# Written by scripts/verify_held.py --release: held answers whose source passed the matched-pair check.
RELEASE_PATH = EXTERNAL_DIR / "held_release.json"
# Append-only logs of programmatic answers, one line per answer: Agent Platform (collect_agent_platform.py),
# Claude Message Batches (collect_anthropic.py), Codex CLI (collect_codex.py), the WorkBuddy gateway
# (collect_workbuddy.py), Qoder CLI CN (collect_qoder.py), Moonshot's API (collect_moonshot.py) and
# Antigravity's agent (collect_antigravity.py). Entries carry a "source".
API_LOG_PATH = EXTERNAL_DIR / "api_log.jsonl"
ANTHROPIC_LOG_PATH = EXTERNAL_DIR / "api_log_anthropic.jsonl"
CODEX_LOG_PATH = EXTERNAL_DIR / "api_log_codex.jsonl"
WORKBUDDY_LOG_PATH = EXTERNAL_DIR / "api_log_workbuddy.jsonl"  # first GLM/Kimi run; later runs write one log per model
QODER_LOG_PATH = EXTERNAL_DIR / "api_log_qoder.jsonl"
MOONSHOT_LOG_PATH = EXTERNAL_DIR / "api_log_moonshot.jsonl"
ANTIGRAVITY_LOG_PATH = EXTERNAL_DIR / "api_log_antigravity.jsonl"
API_LOG_GLOB = "api_log*.jsonl"

UNCOLLECTABLE: dict[str, str] = {}
# Collected, but kept out of training until the served model is verified (snapshot_verified).
HELD = {
    "gemini-3.7-flash": "Google's API routes 3.7 requests to 3.8 since 8 Oct 2026, so an app showing 3.7 may be "
                        "served by 3.8; held until verified or until its answers resemble official 3.7 text",
}
TARGETS = tuple(m for m in LABELS if m not in UNCOLLECTABLE)
# The sibling a held label is most likely served as, and against which scripts/verify_held.py
# compares it on matched prompts. On official held-out pairs the reference model separates
# 3.7 from 3.8 only weakly (3.7 ahead in about 60% of prompts, effect size about 0.28):
# confirming genuine 3.7 at p < 0.01 takes about 120 matched prompts for 80% power, and ruling
# it out about 80-200 depending on the margin. A release therefore needs
# MIN_PAIRS prompts answered by both and p < HELD_ALPHA, which keeps the false-release risk
# near 5% even if the check is rerun after each of five waves.
SIBLINGS = {"gemini-3.7-flash": "gemini-3.8-flash"}
MIN_PAIRS = 70
HELD_ALPHA = 0.01

# Some apps (e.g. Gemini in agent studios) cannot switch web search off, and searched answers
# carry citation markers and web-summary prose. Every prompt, for every model, ends with the same
# line, so the instruction is not a per-app confound; the text actually asked is recorded per answer.
NO_TOOLS_LINE = "Answer from your own knowledge, without searching the web or using any tools."
NO_TOOLS_ID = "no-tools-v1"

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
# A searched answer is the model's answer (summarised web text), so it cannot be re-asked either;
# the same holds for an answer during which an agent client ran tools.
QUARANTINE_FLAGS = COPY_PROBLEMS | {"possible_refusal", "citations_or_search", "tool_use"}

HEADER_DEFAULTS = {
    "model": "",
    "displayed_model_label": "",
    "model_label_checked_in_app": "no",
    "fresh_chat_per_prompt": "yes",
    "memory_and_custom_instructions": "unknown",
    "mode_or_reasoning_setting": "",
    "collected_on": "",
    "collector": "",
    "served_model_version": "",
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
# Search grounding leaves [1] / [1, 2] / \u30103\u3011 markers, citation digits glued to a word ("model12."),
# superscript references, bare URLs or a sources footer. The glued-digit form only follows lowercase
# words, so acronyms such as POP3 or MP3 are not taken for citations.
CITATION = re.compile(
    r"\[\d{1,3}(?:\s*[,\-\u2013]\s*\d{1,3})*\]|\u3010[^\u3011]{0,40}\u3011"
    r"|(?-i:[a-z]{3,})(?:\d{1,3}|[\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u2070]+)(?=[.,;:!?)]|\s|$)"
    r"|https?://|www\.|^\s*(?:sources?|references|citations)\s*:?\s*$",
    re.MULTILINE | re.IGNORECASE,
)


def _ranked(ids, salt: str) -> list[str]:
    return sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode("utf-8")).hexdigest())


def build_bank() -> list[dict]:
    """The kit's 36 prompts (wave1 plus a separate diagnostic pack) and the new waves from prompts_v2.tsv."""
    bank = []
    for p in json.loads(KIT_BANK_PATH.read_text(encoding="utf-8")):
        wave = "diagnostic" if p["split"] == "diagnostic" else "wave1"
        bank.append({**p, "prompt_text": f"{p['prompt_text']} {NO_TOOLS_LINE}",
                     "wrapper_id": f"{p['wrapper_id']}+{NO_TOOLS_ID}", "wave": wave})
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
            "prompt_text": f"{row.content} {wrapper.format(band=row.band)} {NO_TOOLS_LINE}",
            "bank_version": BANK_VERSION,
            "origin": "independently authored for this project; contains no competition text",
            "prompt_family_id": row.prompt_id,
            "template_family_id": row.prompt_id,
            "wrapper_id": f"{wrapper_id}+{NO_TOOLS_ID}",
            "wave": f"wave{row.wave}",
        })
    if len({p["prompt_id"] for p in bank}) != len(bank):
        raise ValueError("duplicate prompt ids in the bank")
    return bank


def pool_bank() -> list[dict]:
    """Screened Open Pool prompts, asked verbatim plus the no-tools line. The prompt id is the pool's
    prompt_canon, so answers join the pool's passages for the same prompt in grouped validation."""
    if not POOL_PROMPTS_PATH.exists():
        return []
    rows = pd.read_csv(POOL_PROMPTS_PATH, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    return [{
        "prompt_id": row.prompt_canon,
        "split": "train",
        "domain": "",
        "genre": "",
        "requested_words": row.requested_words,
        "prompt_text": f"{row.prompt_text} {NO_TOOLS_LINE}",
        "bank_version": "pool-v13",
        "origin": POOL_PROMPTS_ORIGIN,
        "prompt_family_id": row.prompt_canon,
        "template_family_id": row.prompt_canon,
        "wrapper_id": f"pool-verbatim+{NO_TOOLS_ID}",
        "wave": row.wave,
    } for row in rows.itertuples()]


def load_bank() -> dict[str, dict]:
    bank = {p["prompt_id"]: p for p in json.loads(BANK_PATH.read_text(encoding="utf-8"))}
    for p in pool_bank():
        if p["prompt_id"] in bank:
            raise ValueError(f"pool prompt id {p['prompt_id']} clashes with the bank")
        bank[p["prompt_id"]] = p
    return bank


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
        "# served_model_version: only if the app shows the version of the model that actually answered",
        "# (e.g. a modelVersion field in a raw-response or model-details panel), and it was the same for",
        "# every answer in this file. A matching version verifies them; another model's version rejects them.",
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
    if CITATION.search(answer):
        issues.append("citations_or_search")
    refusal = bool(REFUSAL.match(answer))
    if refusal:
        issues.append("possible_refusal")
    if re.search(r"^\s*(#{1,6} |[-*•] |\d+\. )|\*\*|__", answer, flags=re.MULTILINE):
        issues.append("markdown")
    if re.search(r"\n\s*\n", answer):
        issues.append("multi_paragraph")
    if ODD_WHITESPACE.search(answer):
        issues.append("unusual_whitespace")
    if requested_words and (m := re.fullmatch(r"(\d+)(?:-(\d+))?", requested_words)):
        # A band "90-130", or a single target ("about 101 words" in pool prompts) read as ±20%.
        lo, hi = (int(m[1]), int(m[2])) if m[2] else (round(int(m[1]) * 0.8), round(int(m[1]) * 1.2))
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
        "tool_use_observed": "citations_or_search" in issues,
        "duplicate_of": None,
        "issues": issues,
    }


def _yes(value: str) -> bool:
    return value.strip().lower() in ("yes", "y", "true", "1")


def served_status(label: str, version: str) -> str:
    """Verification status implied by an app-reported served model version.

    The label itself or a pinned numeric version of it (e.g. "-001", "-2026-09-01") verifies the
    snapshot; another suffix ("-lite", "-preview-x") is ambiguous; anything else is a different model.
    """
    version = version.strip().lower().removeprefix("models/")
    if version == label:
        return "snapshot_verified"
    if version.startswith(label + "-"):
        return "snapshot_verified" if re.fullmatch(r"[0-9][0-9.-]*", version[len(label) + 1:]) else "family_proxy"
    return "rejected_exact_label"


def source_verification(label: str, header: dict) -> dict:
    """Verification block from the answer file's header: a served version beats the picker check."""
    checked = _yes(header.get("model_label_checked_in_app", ""))
    shown = header.get("displayed_model_label") or label
    version = header.get("served_model_version", "").strip()
    if version:
        status = served_status(label, version)
        evidence = [f"operator recorded the app-reported served model version {version!r}"]
    else:
        status = "ui_label_verified" if checked else "family_proxy"
        evidence = [f"operator confirmed the app's model picker showed {shown}"] if checked \
            else ["model label not yet confirmed by the operator"]
    return {
        "status": status,
        "evidence": evidence,
        "routing_uncertainty": "consumer chat: routing and system instructions are not observable"
                               + ("" if version else "; served snapshot unknown"),
        "competition_snapshot_match": "unknown",
        "verification_date": None,
    }


def make_record(entry: dict, attempt: int, parent: str | None, header: dict, prompt: dict,
                withdrawn: bool = False, eligibility: str = "pending_clarification") -> dict:
    """One schema-v1.0 record for one logged answer; header fields come from its answer file.

    `withdrawn` marks an answer that has since been deleted or replaced in its file (a mis-paste).
    `eligibility` is the competition-eligibility status of the whole collection plan (report I1).
    """
    answer = entry["answer"].strip()  # the paste's surrounding blank lines are not part of the answer
    raw_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    label = entry["label"]
    source = source_verification(label, header)
    version = header.get("served_model_version", "").strip() or None
    shown = header.get("displayed_model_label") or None
    setting = {"off": "off", "on": "on"}.get(header.get("memory_and_custom_instructions", "").lower(), "unknown")
    date = header.get("collected_on", "")
    collected = f"{date}T00:00:00Z" if re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) else entry["first_seen_utc"]
    asked = (entry.get("prompt_text") or prompt["prompt_text"]).strip()  # the text in the file when answered
    checks = quality(answer, asked, prompt.get("requested_words"))
    if withdrawn:
        checks["issues"].append("withdrawn")
    fresh = _yes(header.get("fresh_chat_per_prompt", ""))
    quarantine = (bool(QUARANTINE_FLAGS & {i.split(":")[0] for i in checks["issues"]})
                  or source["status"] not in ADMISSIBLE_STATUS or not fresh)
    return {
        "record_id": "x-" + hashlib.sha256(f"{label}|{entry['prompt_id']}|{raw_sha}".encode()).hexdigest()[:16],
        "schema_version": "1.0",
        "target_label": label,
        "served_model_id": version,
        "displayed_model_label": shown,
        "snapshot_id": version if source["status"] == "snapshot_verified" else None,
        "provider": "unspecified",
        "interface": "manual_chat:unspecified",
        "interface_version": None,
        "endpoint_or_product_url": None,
        "region": None,
        "collected_at_utc": collected,
        "prompt_id": prompt["prompt_id"],
        "prompt_family_id": prompt["prompt_family_id"],
        "template_family_id": prompt["template_family_id"],
        "wrapper_id": wrapper_for(prompt, asked),
        "prompt_bank_version": prompt["bank_version"],
        "prompt_text": asked,
        "raw_response": answer,
        "rendered_text_export": None,
        "raw_response_sha256": raw_sha,
        "visible_reasoning": None,
        "final_answer_span": None,
        "source_verification": source,
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
            "model_fallback_observed": True if source["status"] == "rejected_exact_label" else None,
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


def wrapper_for(prompt: dict, asked: str) -> str | None:
    """The bank wrapper id, without the no-tools suffix when the asked text lacked that line."""
    wrapper = prompt.get("wrapper_id")
    if wrapper and NO_TOOLS_LINE not in asked:
        return wrapper.removesuffix(f"+{NO_TOOLS_ID}")
    return wrapper


def same_prompt(asked: str, prompt: dict) -> bool:
    """True when the asked text is the bank prompt, with or without the no-tools line."""
    return asked.strip() in (prompt["prompt_text"], prompt["prompt_text"].removesuffix(f" {NO_TOOLS_LINE}"))


def refresh_answer_file(old: str, label: str, wave: str, prompts: list[dict]) -> str:
    """A new template carrying over the old file's header values and filled answers.

    A filled block keeps the prompt text it was answered under; empty blocks get the current text.
    """
    header, blocks = parse_answer_file(old)
    filled = {pid: (text, answer) for pid, text, answer in blocks if answer.strip()}
    lines = answer_template(label, wave, prompts).split("\n")
    for i, line in enumerate(lines):
        key = line.split(":", 1)[0]
        if key in HEADER_DEFAULTS and key != "model" and not line.startswith("#"):
            lines[i] = f"{key}: {header[key]}".rstrip()
    new = "\n".join(lines)
    for pid, (text, answer) in filled.items():
        current = next(p["prompt_text"] for p in prompts if p["prompt_id"] == pid)
        new = new.replace(f"<<<PROMPT {pid}>>>\n{current}\n<<<ANSWER {pid}>>>\n\n<<<END {pid}>>>",
                          f"<<<PROMPT {pid}>>>\n{text}\n<<<ANSWER {pid}>>>\n{answer}<<<END {pid}>>>")
    return new


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


def parse_generate_content(payload: dict) -> dict:
    """Final answer, visible thought summary, served model, finish reason and usage of a generateContent response."""
    candidates = payload.get("candidates") or []
    parts = ((candidates[0].get("content") or {}).get("parts") or []) if candidates else []
    usage = payload.get("usageMetadata") or {}
    served = payload.get("modelVersion")
    thoughts = usage.get("thoughtsTokenCount")
    return {
        "text": "".join(p.get("text", "") for p in parts if "text" in p and not p.get("thought")),
        "reasoning": "\n\n".join(p.get("text", "") for p in parts if p.get("thought")) or None,
        "served": served.removeprefix("models/") if served else None,
        "response_id": payload.get("responseId"),
        "finish": candidates[0].get("finishReason", "UNKNOWN") if candidates
        else "BLOCKED:" + str((payload.get("promptFeedback") or {}).get("blockReason")),
        "tool_use": any("functionCall" in p or "executableCode" in p for p in parts),
        "input_tokens": usage.get("promptTokenCount"),
        "output_tokens": usage.get("candidatesTokenCount"),
        "reasoning_tokens": thoughts,
        # thinking is billed as output on top of the candidates
        "billed_output_tokens": None if usage.get("candidatesTokenCount") is None
        else usage["candidatesTokenCount"] + (thoughts or 0),
    }


def parse_anthropic_message(payload: dict) -> dict:
    """The same fields from a Claude Messages API response (a Message as a dict)."""
    blocks = payload.get("content") or []
    usage = payload.get("usage") or {}
    return {
        "text": "".join(b.get("text", "") for b in blocks if b.get("type") == "text"),
        "reasoning": "\n\n".join(b["thinking"] for b in blocks if b.get("type") == "thinking" and b.get("thinking"))
        or None,
        "served": payload.get("model"),
        "response_id": payload.get("id"),
        "finish": payload.get("stop_reason") or "unknown",
        "tool_use": any(b.get("type") in ("tool_use", "server_tool_use") for b in blocks),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": None,  # included in output_tokens, not reported separately
        "billed_output_tokens": usage.get("output_tokens"),
    }


def parse_codex_events(payload: dict) -> dict:
    """The same fields from a `codex exec --json` event stream; the last agent message is the answer.

    Codex does not report the served model. Any item other than messages and reasoning means the
    agent ran a tool (shell command, file change, web search, MCP call) while answering.
    """
    events = payload.get("events") or []
    items = [e.get("item") or {} for e in events if e.get("type") == "item.completed"]
    messages = [i.get("text", "") for i in items if i.get("type") == "agent_message"]
    usage = next((e.get("usage") or {} for e in events if e.get("type") == "turn.completed"), {})
    return {
        "text": messages[-1] if messages else "",
        "reasoning": "\n\n".join(i["text"] for i in items if i.get("type") == "reasoning" and i.get("text")) or None,
        "served": None,
        "response_id": next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None),
        "finish": "completed" if any(e.get("type") == "turn.completed" for e in events) else "failed",
        "tool_use": any(i.get("type") not in ("agent_message", "reasoning", "todo_list") for i in items),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reasoning_tokens": usage.get("reasoning_output_tokens"),
        "billed_output_tokens": usage.get("output_tokens"),
    }


def parse_chat_completion(payload: dict) -> dict:
    """The same fields from an OpenAI-compatible chat.completion, as the WorkBuddy gateway returns it.

    The response's model field is only what the gateway reports, which may echo the request, so it is
    kept as "reported" and not treated as a served model. GLM and Kimi return thinking as reasoning_content.
    """
    choices = payload.get("choices") or []
    message = (choices[0].get("message") or {}) if choices else {}
    content = message.get("content") or ""
    if isinstance(content, list):  # content parts
        content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
    usage = payload.get("usage") or {}
    return {
        "text": content,
        "reasoning": message.get("reasoning_content") or message.get("reasoning") or None,
        "served": None,
        "reported": payload.get("model"),
        "response_id": payload.get("id"),
        "finish": (choices[0].get("finish_reason") or "unknown") if choices else "no_choices",
        "tool_use": bool(message.get("tool_calls") or message.get("function_call")),
        "input_tokens": usage.get("prompt_tokens"),
        "output_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": (usage.get("completion_tokens_details") or {}).get("reasoning_tokens"),
        "billed_output_tokens": usage.get("completion_tokens"),
    }


def parse_official_chat_completion(payload: dict) -> dict:
    """parse_chat_completion for a provider's own API, whose model field names the served model."""
    parsed = parse_chat_completion(payload)
    return parsed | {"served": parsed["reported"]}


def parse_antigravity_transcript(payload: dict) -> dict:
    """The same fields from an Antigravity conversation transcript (agentapi new-conversation).

    The answer is the last model reply. Antigravity names no served model. Anything besides the prompt and
    one reply (a tool step, a system message, a second planner turn) means the agent did more than answer.
    """
    steps = payload.get("steps") or []
    replies = [s for s in steps if s.get("source") == "MODEL" and s.get("type") == "PLANNER_RESPONSE"]
    last = replies[-1] if replies else {}
    return {
        "text": last.get("content") or "",
        "reasoning": "\n\n".join(s["thinking"] for s in replies if s.get("thinking")) or None,
        "served": None,
        "reported": None,
        "response_id": None,
        "finish": last.get("status") or "missing",
        "tool_use": len(replies) != 1 or any(s.get("type") not in ("USER_INPUT", "PLANNER_RESPONSE") for s in steps),
        "input_tokens": last.get("input_tokens"),
        "output_tokens": sum(s.get("output_tokens") or 0 for s in replies) or None,
        "reasoning_tokens": None,
        "billed_output_tokens": None,
    }


def parse_qoder_result(payload: dict) -> dict:
    """The same fields from a `qoderclicn -p -o json` result.

    Qoder reports credits rather than tokens (its token fields are zeros) and names the model only by an
    internal key in modelUsage. More than one turn, or any server-side search, means a tool ran.
    """
    usage = payload.get("usage") or {}
    keys = list(payload.get("modelUsage") or {})
    searched = usage.get("server_tool_use") or {}
    return {
        "text": payload.get("result") or "",
        "reasoning": None,
        "served": None,
        "reported": ",".join(keys) or None,
        "response_id": usage.get("request_id") or payload.get("session_id"),
        "finish": "error" if payload.get("is_error") else payload.get("stop_reason") or "unknown",
        "tool_use": (payload.get("num_turns") or 1) > 1
        or any(searched.get(k) for k in ("web_search_requests", "web_fetch_requests")),
        "input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "billed_output_tokens": None,
    }


# source -> provider (or label -> provider), interface, parser, finish values meaning a complete answer,
# finish values meaning cut off
API_SOURCES = {
    "agent_platform": ("google", "api:agent_platform_generate_content", parse_generate_content, {"STOP"},
                       {"MAX_TOKENS"}),
    "anthropic": ("anthropic", "api:anthropic_message_batches", parse_anthropic_message, {"end_turn"},
                  {"max_tokens"}),
    "codex": ("openai", "codex_cli:exec", parse_codex_events, {"completed"}, set()),
    "workbuddy": ({"glm-5.3": "zai", "glm-5.3-flash": "zai", "kimi-k3": "moonshot", "gpt-5.5": "openai"},
                  "api:workbuddy_gateway_chat_completions", parse_chat_completion, {"stop"}, {"length"}),
    "qoder": ("alibaba", "qoder_cli_cn:print", parse_qoder_result, {"end_turn"}, {"max_tokens"}),
    "moonshot": ("moonshot", "api:moonshot_chat_completions", parse_official_chat_completion, {"stop"}, {"length"}),
    "antigravity": ("google", "antigravity:agentapi", parse_antigravity_transcript, {"DONE"}, set()),
}
ANTIGRAVITY_INSTRUCTIONS = ("Antigravity's agent system prompt and tool definitions (~9k tokens, compiled into its "
                            "language_server.exe); the prompt is wrapped in <USER_REQUEST> tags with a local-time line")
# Model names a client reports instead of the selected model id, where they differ.
REPORTED_MODEL_KEYS = {"qwen3.8-max": ("qmodel_38max",)}
# (label, selected model id) pairs found not to be the label's model; their answers are rejected for that label.
RETIRED_SELECTIONS = {
    ("kimi-k3", "kimi-k3-1"): "the owner's CN WorkBuddy account offers kimi-k3-1, not kimi-k3; kimi-k3 answers come from "
                              "the global account, which lists kimi-k3 itself (beside kimi-k3.1-pro)",
}
QODER_INSTRUCTIONS = ("Qoder CLI CN print mode with its system prompt replaced by an empty one, all tools disabled, and "
                      "no user, project or MCP settings; a few hundred tokens of unseen context remain (from billed "
                      "credits)")
CODEX_INSTRUCTIONS = ("Codex CLI built-in agent instructions (~12.7k tokens); user config.toml, rules, AGENTS.md "
                      "and memories not loaded")


def api_record(entry: dict, prompt: dict) -> dict:
    """One schema-v1.0 record for a programmatic answer logged by a collect_*.py script.

    Agent Platform's modelVersion and the Messages API's model field name the served model, which
    verifies the label (snapshot_verified) or contradicts it (rejected_exact_label). Codex reports no
    served model, so its answers rest on the model selected with `codex exec -m` (ui_label_verified).
    Only the bank prompt and an output cap were sent; no system prompt or tools were added.
    """
    source = entry.get("source", "agent_platform")  # Agent Platform entries predate the field
    provider, interface, parse, complete, cut_off = API_SOURCES[source]
    if source == "moonshot" and entry.get("thinking", "default") != "disabled":
        # Thinking-on Kimi writes differently from the official data, so it is its own condition.
        interface = "api:moonshot_thinking_chat_completions"
    parsed = parse(entry["response"])
    answer = parsed["text"].strip()
    label = entry["label"]
    provider = provider[label] if isinstance(provider, dict) else provider
    raw_sha = hashlib.sha256(answer.encode("utf-8")).hexdigest()
    if parsed["served"]:
        status = served_status(label, parsed["served"])
        evidence = [f"{interface} reported served model {parsed['served']!r} for requested model {entry['model_id']!r}"]
    elif source == "codex":
        status = "ui_label_verified"
        evidence = [f"model selected with codex exec -m {entry['model_id']!r}; Codex reports no served model"]
    elif source == "antigravity":
        status = "ui_label_verified"
        evidence = [f"model selected with agentapi new-conversation --model={entry['model_id']} (Antigravity's "
                    + "Gemini 3.8 Flash); Antigravity reports no served model"]
    elif source in ("workbuddy", "qoder"):
        # The gateway's model field may only echo the request, and Qoder reports only an internal model key,
        # so a match verifies nothing beyond the selection; a different name means the model was swapped.
        reported = (parsed["reported"] or "").lower()
        expected = {k.lower() for k in REPORTED_MODEL_KEYS.get(label, (entry["model_id"],))}
        status = "ui_label_verified" if not reported or reported in expected else "rejected_exact_label"
        account = f" ({entry.get('account', 'cn')} account)" if source == "workbuddy" else ""
        evidence = [f"model selected as {entry['model_id']!r} through {interface}{account}, which reported "
                    + repr(parsed["reported"])]
    else:
        status, evidence = "family_proxy", [f"{interface} response carried no served model"]
    if (label, entry["model_id"]) in RETIRED_SELECTIONS:
        status, evidence = "rejected_exact_label", [*evidence, RETIRED_SELECTIONS[label, entry["model_id"]]]
    checks = quality(answer, entry["prompt_text"], prompt.get("requested_words"))
    # Echoed prompts and app clutter are copy-path problems; an API answer has no copy path, so an answer
    # that opens by restating the question is just the model's writing.
    checks["issues"] = [i for i in checks["issues"] if i not in ("prompt_echo", "app_clutter")]
    if parsed["finish"] in cut_off and "possible_truncation" not in checks["issues"]:
        checks["issues"].append("possible_truncation")
    elif parsed["finish"] not in complete | cut_off:
        checks["issues"] += [f"finish:{parsed['finish']}", "possible_refusal"]  # refused, blocked or filtered
    if parsed["tool_use"]:
        checks["issues"].append("tool_use")
    checks["complete"] = bool(answer) and "possible_truncation" not in checks["issues"]
    checks["refusal"] = checks["refusal"] or "possible_refusal" in checks["issues"]
    checks["tool_use_observed"] = checks["tool_use_observed"] or parsed["tool_use"]
    quarantine = bool(QUARANTINE_FLAGS & {i.split(":")[0] for i in checks["issues"]}) or status not in ADMISSIBLE_STATUS
    routing, instructions_status, instructions, tools = {
        "codex": ("the client names no served model", "partially_known", CODEX_INSTRUCTIONS,
                  ["codex read-only sandbox shell"]),
        "workbuddy": ("the gateway's model field may echo the request; WorkBuddy's upstream routing is not observable",
                      "provider_default_unknown", None, []),
        "qoder": ("Qoder names only an internal model key; its upstream routing is not observable", "partially_known",
                  QODER_INSTRUCTIONS, []),
        "antigravity": ("Antigravity names no served model; its server-side additions are not observable",
                        "partially_known", ANTIGRAVITY_INSTRUCTIONS, ["antigravity agent tools"]),
    }.get(source, ("the API names the served model; provider-side serving changes within it are not observable",
                   "known_recorded", "", []))
    if source == "codex":
        tier = entry.get("service_tier")
        settings_source = (f"Codex model defaults, service tier {tier!r} (fast mode)" if tier
                           else "Codex model defaults") + "; only the prompt sent"
    elif source == "workbuddy":
        effort = entry.get("reasoning_effort")
        settings_source = ((f"reasoning effort {effort!r}" if effort else "WorkBuddy default reasoning effort")
                           + "; gateway default temperature 1 (owner's panel setting); only the prompt sent, "
                           "as one user message")
    elif source == "qoder":
        settings_source = "Qoder default thinking and sampling; only the prompt sent, on stdin"
    elif source == "antigravity":
        settings_source = "Antigravity defaults for its Flash model (thinking per the app's model setting); only the prompt sent"
    elif source == "moonshot":
        settings_source = ("API defaults with thinking " + ("disabled" if entry.get("thinking") == "disabled" else "on")
                           + "; an output-token cap sent; no system prompt or tools")
    else:
        settings_source = "API defaults; only an output-token cap sent; no system prompt or tools"
    return {
        "record_id": "x-" + hashlib.sha256(f"{label}|{entry['prompt_id']}|{raw_sha}".encode()).hexdigest()[:16],
        "schema_version": "1.0",
        "target_label": label,
        "served_model_id": parsed["served"],
        "displayed_model_label": entry["model_id"] if source in ("codex", "workbuddy", "qoder", "antigravity") else None,
        "snapshot_id": parsed["served"] if status == "snapshot_verified" else None,
        "provider": provider,
        "interface": interface,
        "interface_version": entry.get("client_version") or "v1",
        "endpoint_or_product_url": entry.get("endpoint"),
        "region": entry.get("location"),
        "collected_at_utc": entry["requested_at_utc"],
        "prompt_id": prompt["prompt_id"],
        "prompt_family_id": prompt["prompt_family_id"],
        "template_family_id": prompt["template_family_id"],
        "wrapper_id": wrapper_for(prompt, entry["prompt_text"]),
        "prompt_bank_version": prompt["bank_version"],
        "prompt_text": entry["prompt_text"],
        "raw_response": answer,
        "rendered_text_export": None,
        "raw_response_sha256": raw_sha,
        "visible_reasoning": parsed["reasoning"],
        "final_answer_span": None,
        "source_verification": {
            "status": status,
            "evidence": evidence,
            "routing_uncertainty": routing,
            "competition_snapshot_match": "unknown",
            "verification_date": entry["requested_at_utc"][:10],
        },
        "session": {
            "fresh_conversation": True,
            "prior_turn_count": 0,
            "session_id_hash": None,
            "memory": "off",
            "personalization": "off",
            "custom_instructions": "empty",
            "system_instructions_status": instructions_status,
            "known_system_instructions": instructions,
            "tools": tools,
            "model_fallback_observed": True if status == "rejected_exact_label" else None,
        },
        "settings": {
            "temperature": None,
            "top_p": None,
            "seed": None,
            "max_output_tokens": entry.get("max_output_tokens")
            or (entry.get("generation_config") or {}).get("maxOutputTokens"),
            "reasoning_effort": entry.get("reasoning_effort")
            or ("none (thinking disabled)" if entry.get("thinking") == "disabled" else None),
            "settings_source": settings_source,
        },
        "provenance": {
            "collection_method": "api",
            "copy_method": f"{interface} response: final answer text, extracted by nllm.collection.{parse.__name__}",
            "collector_id": entry.get("collector") or "owner",
            "original_artifact_reference": f"{entry.get('log', 'api_log.jsonl')}:{entry['request_uid']}",
            "license_or_terms_url": None,
            "source_dataset_url": None,
            "eligibility_status": entry.get("eligibility", "pending_clarification"),
            "eligibility_evidence": None,
            "transforms": ["outer_whitespace_strip"] if answer != parsed["text"] else [],
            "attempt_index": entry.get("attempt", 1),
            "parent_record_id": None,
            "request_id": parsed["response_id"],
        },
        "quality": checks,
        "cost": {
            "input_tokens": parsed["input_tokens"],
            "output_tokens": parsed["billed_output_tokens"],
            "reasoning_tokens": parsed["reasoning_tokens"],
            "usd": entry.get("usd"),
            "operator_minutes": None,
        },
        "split": "quarantine" if quarantine else prompt["split"],
    }


def read_api_logs() -> list[dict]:
    """Every programmatic answer from every api_log*.jsonl, tagged with the log it came from."""
    entries = []
    for path in sorted(EXTERNAL_DIR.glob(API_LOG_GLOB)):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for i, line in enumerate(lines):
            try:
                entries.append({**json.loads(line), "log": path.name})
            except json.JSONDecodeError:
                if i < len(lines) - 1:
                    raise
                # a collector is still writing the last line; the next import will read it whole
    return entries


def api_records(log: list[dict], bank: dict[str, dict], eligibility: str = "pending_clarification") -> list[dict]:
    """Records for every logged programmatic answer, in the order they were collected."""
    return [api_record({**e, "eligibility": eligibility}, bank[e["prompt_id"]])
            for e in sorted(log, key=lambda e: e["requested_at_utc"])]


def load_released() -> set[str]:
    """Record ids of held answers released by a passing scripts/verify_held.py check."""
    if not RELEASE_PATH.exists():
        return set()
    return {rid for entry in json.loads(RELEASE_PATH.read_text(encoding="utf-8")).values() for rid in entry["record_ids"]}


def held_verdict(n_pairs: int, p_value: float, gap_ratio: float, ratio_hi: float) -> str:
    """Pre-registered decision for a held label's matched-pair check (scripts/verify_held.py).

    gap_ratio is the mean held-minus-sibling score gap on the collected pairs divided by the same
    gap on official held-out pairs: about 1 if the app serves the held model, about 0 if it serves
    the sibling under the held label. ratio_hi is the upper end of its 95% bootstrap interval.
    """
    if n_pairs < MIN_PAIRS:
        return "inconclusive"
    if p_value < HELD_ALPHA and gap_ratio >= 0.5:
        return "consistent"
    if ratio_hi < 0.5:
        return "looks_like_sibling"
    return "inconclusive"


def admission(records: list[dict], released: frozenset[str] | set[str] = frozenset()) -> pd.DataFrame:
    """Per-record admission for exact-label training (report E3 step 11).

    Within each (model, prompt, collection method, interface) the first passing attempt is admitted; a
    later attempt is only considered when every earlier one failed on a copy problem or came from another
    model (failed source verification, e.g. a retired selection). Each interface (chat app, Agent Platform,
    Message Batches, Codex, WorkBuddy, Qoder, Moonshot) is a separate condition, so answers to the same
    prompt from two of them may both be admitted; they share the prompt group, so cross-validation keeps
    them together, and scripts/external_data_eval.py --exclude chooses which to train on. Text that occurs more than once
    (exactly, or after lowercasing and whitespace collapsing) is rejected everywhere, since it
    cannot be attributed. A held label's answer needs a verified served version, or its record id
    in `released` (a passing matched-pair check).
    """
    norm = {r["record_id"]: re.sub(r"\s+", " ", r["raw_response"].lower()).strip() for r in records}
    text_count = pd.Series(list(norm.values())).value_counts()
    rows, pair_state = [], {}
    for r in sorted(records, key=lambda r: (r["target_label"], r["prompt_id"], r["provenance"]["collection_method"],
                                            r["provenance"]["attempt_index"])):
        flags = {i.split(":")[0] for i in r["quality"]["issues"]}
        reasons = sorted(QUARANTINE_FLAGS & flags)
        if r["source_verification"]["status"] not in ADMISSIBLE_STATUS:
            reasons.append(f"verification:{r['source_verification']['status']}")
        if not r["session"]["fresh_conversation"]:
            reasons.append("not a fresh chat")
        if r["target_label"] not in TARGETS:
            reasons.append("not a collectable label")
        if (r["target_label"] in HELD and r["source_verification"]["status"] != "snapshot_verified"
                and r["record_id"] not in released):
            reasons.append("held:served model unverified")
        if r["provenance"]["eligibility_status"] == "not_eligible":
            reasons.append("not eligible")
        if norm[r["record_id"]] and text_count[norm[r["record_id"]]] > 1:
            reasons.append("duplicate_text")
        pair = r["target_label"], r["prompt_id"], r["provenance"]["collection_method"], r["interface"]
        state = pair_state.get(pair)
        if state == "admitted":
            reasons.append("later_answer_for_pair")
        elif state == "blocked":
            reasons.append("follows_a_rejected_answer")
        if not reasons:
            pair_state[pair] = "admitted"
        # A copy problem, or an answer from another model than the label's (failed verification), is not
        # the model's own answer, so it does not block a later attempt; anything else (a refusal) does.
        elif state is None and not all(x in COPY_PROBLEMS or x.startswith("verification:") for x in reasons):
            pair_state[pair] = "blocked"
        rows.append({
            "record_id": r["record_id"],
            "target_label": r["target_label"],
            "prompt_id": r["prompt_id"],
            "collection_method": r["provenance"]["collection_method"],
            "interface": r["interface"],
            "attempt_index": r["provenance"]["attempt_index"],
            "split": r["split"],
            "admitted": not reasons,
            "reasons": "; ".join(reasons),
        })
    return pd.DataFrame(rows, columns=["record_id", "target_label", "prompt_id", "collection_method", "interface",
                                       "attempt_index", "split", "admitted", "reasons"])


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
        "collection_method": [r["provenance"]["collection_method"] for r in keep],
        "interface": [r["interface"] for r in keep],
        "eligibility_status": [r["provenance"]["eligibility_status"] for r in keep],
    }, columns=["id", "prompt_id", "model", "text", "split", "wave", "collection_method", "interface",
                "eligibility_status"])
