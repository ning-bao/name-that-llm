import hashlib
import json

import pytest

jsonschema = pytest.importorskip("jsonschema")
from nllm import collection

ANSWER = "Metal pulls heat from your hand faster than wood does — so it feels colder, though both match the room."


@pytest.fixture(scope="module")
def bank():
    return {p["prompt_id"]: p for p in collection.build_bank()}


def test_bank_layout(bank):
    waves = {}
    for p in bank.values():
        waves.setdefault(p["wave"], []).append(p)
    assert len(waves["wave1"]) == 30 and len(waves["diagnostic"]) == 6
    for w in ("wave2", "wave3", "wave4", "wave5"):
        assert len(waves[w]) == 40
        assert sum(p["split"] == "external_holdout" for p in waves[w]) == 8
    new = [p for p in bank.values() if p["bank_version"] == collection.BANK_VERSION]
    assert all(p["requested_words"] in p["prompt_text"] for p in new)
    assert all(p["prompt_family_id"] == p["prompt_id"] for p in new)
    assert "gemini-3.7-flash" in collection.TARGETS and len(collection.TARGETS) == 18
    assert all(p["prompt_text"].endswith(collection.NO_TOOLS_LINE) for p in bank.values())
    assert all(p["wrapper_id"].endswith("+" + collection.NO_TOOLS_ID) for p in bank.values())


def test_prompts_never_ask_for_identity_or_style(bank):
    banned = ("which model", "who are you", "your name", "identify yourself", "in your style", "as an ai")
    assert not [p["prompt_id"] for p in bank.values() if any(b in p["prompt_text"].lower() for b in banned)]


def fill(template: str, answers: dict[str, str]) -> str:
    for pid, text in answers.items():
        template = template.replace(f"<<<ANSWER {pid}>>>\n\n", f"<<<ANSWER {pid}>>>\n{text}\n")
    return template


def test_answer_file_round_trip_with_crlf_and_bom(bank):
    prompts = [bank["N01"], bank["X001"]]
    text = fill(collection.answer_template("claude-opus-5", "wave1", prompts), {"N01": ANSWER})
    text = "\ufeff" + text.replace("\n", "\r\n").replace("model_label_checked_in_app: no", "model_label_checked_in_app: yes")
    header, blocks = collection.parse_answer_file(text)
    assert header["model"] == "claude-opus-5" and header["model_label_checked_in_app"] == "yes"
    assert [(pid, answer.strip()) for pid, _, answer in blocks] == [("N01", ANSWER), ("X001", "")]
    assert blocks[0][1] == bank["N01"]["prompt_text"]


def test_broken_markers_are_reported(bank):
    text = collection.answer_template("gpt-5.5", "wave1", [bank["N01"]]).replace("<<<END N01>>>", "<<END N01>>")
    with pytest.raises(ValueError, match="N01"):
        collection.parse_answer_file(text)


@pytest.mark.parametrize("answer, flag", [
    ("Copy\n" + ANSWER, "app_clutter"),
    ("Thought for 12s\n" + ANSWER, "app_clutter"),
    ("ChatGPT said:\n" + ANSWER, "app_clutter"),
    (ANSWER + "\nGemini 3.8 Flash", "app_clutter"),
    ("Explain why a metal handrail and a wooden handrail can feel different. " + ANSWER, "prompt_echo"),
    (ANSWER[:-1] + " and so", "possible_truncation"),
    ("I’m sorry, but I can't help with that.", "possible_refusal"),
    ("**Heat** flows faster through metal.", "markdown"),
    (ANSWER + "\n\n" + ANSWER, "multi_paragraph"),
    (ANSWER.replace(" ", "\u00a0", 1), "unusual_whitespace"),
    (ANSWER[:-1] + " [1].", "citations_or_search"),
    (ANSWER[:-1] + " [2, 3].", "citations_or_search"),
    (ANSWER[:-1] + " of the underlying model12.", "citations_or_search"),
    (ANSWER[:-1] + " of the model\u00b9\u00b2.", "citations_or_search"),
    (ANSWER + " See https://example.org.", "citations_or_search"),
    (ANSWER + "\nSources:", "citations_or_search"),
])
def test_quality_flags(answer, flag):
    prompt = "Explain why a metal handrail and a wooden handrail can feel different in temperature."
    issues = {i.split(":")[0] for i in collection.quality(answer, prompt, "10-40")["issues"]}
    assert flag in issues


def test_ordinary_prose_is_not_clutter():
    text = "Thinking about heat this way helps. Copy that idea to wood and it still holds."
    assert collection.quality(text, "", None)["issues"] == []


def test_science_notation_is_not_a_citation():
    text = "Plants fix CO2 into sugar, and a 3 m² leaf in 1998 held 2.5 g of H2O [per the model]."
    assert "citations_or_search" not in collection.quality(text, "", None)["issues"]


def test_refresh_keeps_answers_under_the_prompt_they_were_asked(bank):
    old_prompt = bank["N01"]["prompt_text"].removesuffix(" " + collection.NO_TOOLS_LINE)
    old = collection.answer_template("gemini-3.7-flash", "wave1", [bank["N01"], bank["N02"]])
    old = old.replace(bank["N01"]["prompt_text"], old_prompt).replace(bank["N02"]["prompt_text"], "outdated text")
    old = fill(old, {"N01": ANSWER}).replace("collector: ", "collector: me")
    new = collection.refresh_answer_file(old, "gemini-3.7-flash", "wave1", [bank["N01"], bank["N02"]])
    header, blocks = collection.parse_answer_file(new)
    assert header["collector"] == "me"
    assert [(pid, text, answer.strip()) for pid, text, answer in blocks] == [
        ("N01", old_prompt, ANSWER), ("N02", bank["N02"]["prompt_text"], "")]
    assert collection.refresh_answer_file(new, "gemini-3.7-flash", "wave1", [bank["N01"], bank["N02"]]) == new


def test_record_keeps_the_asked_prompt_and_holds_37(bank):
    asked = bank["N01"]["prompt_text"].removesuffix(" " + collection.NO_TOOLS_LINE)
    e = {**entry("gemini-3.7-flash", "N01", ANSWER), "prompt_text": asked}
    (record,) = records_for([e], bank)
    assert record["prompt_text"] == asked and not record["wrapper_id"].endswith(collection.NO_TOOLS_ID)
    assert "held" in collection.admission([record]).reasons.iloc[0]


def entry(label, pid, answer, seen="2026-11-05T10:00:00Z"):
    return {"label": label, "prompt_id": pid, "sha256": hashlib.sha256(answer.strip().encode()).hexdigest(),
            "answer": answer, "first_seen_utc": seen, "file": f"answers/wave1/{label}.txt"}


def checked(label):
    return {**collection.HEADER_DEFAULTS, "model": label, "model_label_checked_in_app": "yes"}


def records_for(entries, bank, unchecked=(), withdrawn=()):
    headers = {e["file"]: checked(e["label"]) if e["label"] not in unchecked else collection.HEADER_DEFAULTS
               for e in entries}
    current = {collection.log_key(e) for e in entries} - set(withdrawn)
    return collection.build_records(entries, headers, bank, current)


def test_records_match_kit_schema(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    good, unchecked = records_for([entry("glm-5.3", "N01", ANSWER), entry("kimi-k3", "N01", "Different " + ANSWER)],
                                  bank, unchecked={"kimi-k3"})
    for record in (good, unchecked):
        validator.validate(record)
    assert good["split"] == "train" and good["source_verification"]["status"] == "ui_label_verified"
    assert good["provenance"]["collection_method"] == "manual_chat" and good["settings"]["temperature"] is None
    assert unchecked["split"] == "quarantine" and unchecked["source_verification"]["status"] == "family_proxy"


def test_admission_rules(bank):
    e = [
        entry("glm-5.3", "N01", ANSWER, "2026-11-05T10:00:00Z"),
        entry("glm-5.3", "N01", "A regenerated " + ANSWER, "2026-11-05T11:00:00Z"),  # later answer: rejected
        entry("gpt-5.5", "N01", "Copy\n" + ANSWER, "2026-11-05T10:00:00Z"),  # copy problem...
        entry("gpt-5.5", "N01", "Re-pasted " + ANSWER, "2026-11-05T11:00:00Z"),  # ...so the re-paste counts
        entry("kimi-k3", "N01", "I'm sorry, I can't.", "2026-11-05T10:00:00Z"),  # refusal...
        entry("kimi-k3", "N01", "Asked again " + ANSWER, "2026-11-05T11:00:00Z"),  # ...blocks a re-ask
        entry("qwen3.8-max", "N02", "Same text twice.", "2026-11-05T10:00:00Z"),
        entry("glm-5.3-flash", "N02", "Same  text twice.", "2026-11-05T10:00:00Z"),  # duplicate after normalising
    ]
    records = records_for(e, bank)
    decisions = collection.admission(records).set_index(["target_label", "prompt_id", "attempt_index"])
    assert decisions.admitted.to_dict() == {
        ("glm-5.3", "N01", 1): True, ("glm-5.3", "N01", 2): False,
        ("glm-5.3-flash", "N02", 1): False,
        ("gpt-5.5", "N01", 1): False, ("gpt-5.5", "N01", 2): True,
        ("kimi-k3", "N01", 1): False, ("kimi-k3", "N01", 2): False,
        ("qwen3.8-max", "N02", 1): False,
    }
    assert "follows_a_rejected_answer" in decisions.loc[("kimi-k3", "N01", 2), "reasons"]
    assert "duplicate_text" in decisions.loc[("qwen3.8-max", "N02", 1), "reasons"]


def test_withdrawn_answer_frees_the_pair(bank):
    first, second = entry("glm-5.3", "N01", ANSWER, "2026-11-05T10:00:00Z"), entry(
        "glm-5.3", "N01", "Corrected " + ANSWER, "2026-11-05T11:00:00Z")
    records = records_for([first, second], bank, withdrawn={collection.log_key(first)})
    decisions = collection.admission(records)
    assert decisions.admitted.tolist() == [False, True]
    frame = collection.admitted_frame(records, decisions, bank)
    assert frame.prompt_id.tolist() == ["xN01"] and frame.model.tolist() == ["glm-5.3"]


@pytest.mark.parametrize("version, status", [
    ("gemini-3.7-flash", "snapshot_verified"),
    ("models/Gemini-3.7-Flash-001", "snapshot_verified"),
    ("gemini-3.7-flash-2026-09-01", "snapshot_verified"),
    ("gemini-3.7-flash-lite", "family_proxy"),
    ("gemini-3.8-flash", "rejected_exact_label"),
])
def test_served_version_status(version, status):
    assert collection.served_status("gemini-3.7-flash", version) == status


def test_served_version_verifies_or_rejects_a_held_label(bank):
    e = entry("gemini-3.7-flash", "N01", ANSWER)
    for version, admitted in (("gemini-3.7-flash-001", True), ("gemini-3.8-flash", False), ("", False)):
        header = {**checked("gemini-3.7-flash"), "served_model_version": version}
        record = collection.build_records([e], {e["file"]: header}, bank, {collection.log_key(e)})[0]
        assert collection.admission([record]).admitted.tolist() == [admitted]
    assert record["source_verification"]["status"] == "ui_label_verified"  # no version: picker check only
    rejected = collection.build_records([e], {e["file"]: {**checked("gemini-3.7-flash"),
                                                          "served_model_version": "gemini-3.8-flash"}},
                                        bank, {collection.log_key(e)})[0]
    assert rejected["split"] == "quarantine" and rejected["served_model_id"] == "gemini-3.8-flash"


def test_release_admits_only_the_listed_held_answers(bank):
    records = records_for([entry("gemini-3.7-flash", "N01", ANSWER),
                           entry("gemini-3.7-flash", "N02", "Another " + ANSWER)], bank)
    decisions = collection.admission(records, released={records[0]["record_id"]})
    assert decisions.admitted.tolist() == [True, False]
    assert "held:served model unverified" in decisions.reasons[1]


@pytest.mark.parametrize("n, p, ratio, hi, verdict", [
    (40, 0.001, 1.0, 1.4, "inconclusive"),        # too few pairs, however strong
    (80, 0.005, 0.8, 1.2, "consistent"),
    (80, 0.03, 0.8, 1.3, "inconclusive"),         # p must clear 0.01, not 0.05
    (80, 0.005, 0.3, 0.6, "inconclusive"),        # real but much smaller than official
    (80, 0.40, 0.05, 0.35, "looks_like_sibling"),
    (80, 0.20, 0.30, 0.80, "inconclusive"),
])
def test_held_verdict(n, p, ratio, hi, verdict):
    assert collection.held_verdict(n, p, ratio, hi) == verdict


def api_entry(label, pid, bank, served, text=ANSWER, finish="STOP"):
    return {"request_uid": f"u-{label}-{pid}-{served}", "label": label, "prompt_id": pid, "model_id": label,
            "prompt_text": bank[pid]["prompt_text"], "location": "global", "requested_at_utc": "2026-11-05T10:00:00Z",
            "endpoint": f"https://aiplatform.googleapis.com/v1/publishers/google/models/{label}:generateContent",
            "attempt": 1, "generation_config": {"maxOutputTokens": 8192}, "usd": 0.001,
            "response": {"modelVersion": served, "responseId": "r1",
                         "candidates": [{"finishReason": finish, "content": {"parts": [
                             {"text": "planning", "thought": True}, {"text": text}]}}],
                         "usageMetadata": {"promptTokenCount": 40, "candidatesTokenCount": 30,
                                           "thoughtsTokenCount": 200}}}


def test_api_records_verify_from_the_served_model(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    genuine, redirected, cut = collection.api_records([
        api_entry("gemini-3.7-flash", "N01", bank, "gemini-3.7-flash"),
        api_entry("gemini-3.7-flash", "N02", bank, "gemini-3.8-flash", text="Other " + ANSWER),
        api_entry("gemini-3.8-flash", "N01", bank, "gemini-3.8-flash", text="Cut off mid", finish="MAX_TOKENS"),
    ], bank)
    for record in (genuine, redirected, cut):
        validator.validate(record)
    assert genuine["source_verification"]["status"] == "snapshot_verified" and genuine["raw_response"] == ANSWER
    assert genuine["visible_reasoning"] == "planning" and genuine["cost"]["output_tokens"] == 230
    assert redirected["source_verification"]["status"] == "rejected_exact_label" and redirected["split"] == "quarantine"
    assert cut["split"] == "quarantine" and not cut["quality"]["complete"]
    decisions = collection.admission([genuine, redirected, cut])
    assert decisions.set_index("record_id").admitted.to_dict() == {
        genuine["record_id"]: True, redirected["record_id"]: False, cut["record_id"]: False}


def test_api_and_chat_answers_to_one_prompt_are_both_admitted(bank):
    chat = records_for([entry("gemini-3.8-flash", "N01", ANSWER)], bank)
    api = collection.api_records([api_entry("gemini-3.8-flash", "N01", bank, "gemini-3.8-flash",
                                            text="Via the API, " + ANSWER)], bank)
    decisions = collection.admission(chat + api)
    assert decisions.admitted.tolist() == [True, True]
    frame = collection.admitted_frame(chat + api, decisions, bank)
    assert sorted(frame.collection_method) == ["api", "manual_chat"] and set(frame.prompt_id) == {"xN01"}


def anthropic_entry(label, pid, bank, served=None, text=ANSWER, stop="end_turn"):
    return {"source": "anthropic", "request_uid": f"b1:{label}__{pid}", "label": label, "prompt_id": pid,
            "model_id": label, "prompt_text": bank[pid]["prompt_text"], "endpoint": "POST /v1/messages/batches",
            "requested_at_utc": "2026-11-05T10:00:00Z", "max_output_tokens": 16000, "usd": 0.002,
            "response": {"id": "msg_1", "model": served or label, "stop_reason": stop,
                         "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": text}],
                         "usage": {"input_tokens": 60, "output_tokens": 900}}}


def codex_entry(label, pid, bank, text=ANSWER, tool=False):
    items = [{"id": "item_0", "type": "reasoning", "text": "plan"}]
    if tool:
        items.append({"id": "item_1", "type": "command_execution", "command": "ls", "status": "completed"})
    items.append({"id": "item_2", "type": "agent_message", "text": text})
    events = [{"type": "thread.started", "thread_id": "t-1"}, {"type": "turn.started"},
              *({"type": "item.completed", "item": i} for i in items),
              {"type": "turn.completed", "usage": {"input_tokens": 12700, "output_tokens": 120,
                                                   "reasoning_output_tokens": 20}}]
    return {"source": "codex", "request_uid": f"c-{label}-{pid}-{tool}", "label": label, "prompt_id": pid,
            "model_id": label, "prompt_text": bank[pid]["prompt_text"], "endpoint": "codex exec (WSL)",
            "client_version": "codex-cli 0.157.1", "requested_at_utc": "2026-11-05T10:00:00Z", "usd": None,
            "response": {"events": events}}


def test_anthropic_and_codex_records(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    claude, refused, swapped, gpt, gpt_tool = collection.api_records([
        anthropic_entry("claude-opus-5", "N01", bank),
        anthropic_entry("claude-fable-5", "N01", bank, text="", stop="refusal"),
        anthropic_entry("claude-opus-4-8", "N01", bank, served="claude-opus-5", text="Swapped " + ANSWER),
        codex_entry("gpt-5.6-luna", "N01", bank, text="GPT says " + ANSWER),
        codex_entry("gpt-5.6-sol", "N01", bank, text="Sol says " + ANSWER, tool=True),
    ], bank)
    for record in (claude, refused, swapped, gpt, gpt_tool):
        validator.validate(record)
    assert claude["source_verification"]["status"] == "snapshot_verified" and claude["provider"] == "anthropic"
    assert claude["cost"]["output_tokens"] == 900 and claude["raw_response"] == ANSWER
    assert refused["quality"]["refusal"] and refused["split"] == "quarantine"
    assert swapped["source_verification"]["status"] == "rejected_exact_label"
    assert gpt["source_verification"]["status"] == "ui_label_verified" and gpt["provider"] == "openai"
    assert gpt["raw_response"] == "GPT says " + ANSWER and gpt["visible_reasoning"] == "plan"
    assert gpt_tool["quality"]["tool_use_observed"] and gpt_tool["split"] == "quarantine"
    decisions = collection.admission([claude, refused, swapped, gpt, gpt_tool]).set_index("record_id").admitted
    assert decisions.to_dict() == {claude["record_id"]: True, refused["record_id"]: False,
                                   swapped["record_id"]: False, gpt["record_id"]: True, gpt_tool["record_id"]: False}


def workbuddy_entry(label, pid, bank, model_id, reported, text=ANSWER, finish="stop", tool=False, effort=None):
    message = {"role": "assistant", "content": text, "reasoning_content": "think first"}
    if tool:
        message["tool_calls"] = [{"id": "t1", "type": "function", "function": {"name": "search", "arguments": "{}"}}]
    return {"source": "workbuddy", "request_uid": f"w-{label}-{pid}-{reported}", "label": label, "prompt_id": pid,
            "model_id": model_id, "prompt_text": bank[pid]["prompt_text"], "endpoint": "workbuddy2api gateway",
            "requested_at_utc": "2026-11-05T10:00:00Z", "attempt": 1, "reasoning_effort": effort, "usd": None,
            "response": {"id": "chatcmpl-1", "model": reported,
                         "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 60, "completion_tokens": 500,
                                   "completion_tokens_details": {"reasoning_tokens": 300}}}}


def test_workbuddy_records(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    glm, kimi, swapped, cut, tool, retired, k31 = collection.api_records([
        workbuddy_entry("glm-5.3", "N01", bank, "glm-5.3", "glm-5.3"),
        workbuddy_entry("kimi-k3", "N01", bank, "kimi-k3", "kimi-k3", text="Kimi " + ANSWER, effort="medium"),
        workbuddy_entry("glm-5.3", "N02", bank, "glm-5.3", "glm-4.6", text="Old " + ANSWER),
        workbuddy_entry("glm-5.3-flash", "N01", bank, "glm-5.3-flash", None, text="Cut off mid", finish="length"),
        workbuddy_entry("glm-5.3-flash", "N02", bank, "glm-5.3-flash", "glm-5.3-flash", text="Tool " + ANSWER, tool=True),
        workbuddy_entry("kimi-k3", "N02", bank, "kimi-k3-1", "kimi-k3-1", text="CN " + ANSWER),
        workbuddy_entry("kimi-k3", "N03", bank, "kimi-k3", "kimi-k3-1", text="Swapped " + ANSWER),
    ], bank)
    for record in (glm, kimi, swapped, cut, tool, retired, k31):
        validator.validate(record)
    assert glm["source_verification"]["status"] == "ui_label_verified" and glm["provider"] == "zai"
    assert glm["visible_reasoning"] == "think first" and glm["cost"]["reasoning_tokens"] == 300
    assert glm["session"]["system_instructions_status"] == "provider_default_unknown"
    assert kimi["source_verification"]["status"] == "ui_label_verified" and kimi["provider"] == "moonshot"
    assert kimi["settings"]["reasoning_effort"] == "medium" and glm["settings"]["reasoning_effort"] is None
    assert swapped["source_verification"]["status"] == "rejected_exact_label" and swapped["split"] == "quarantine"
    # the CN account's kimi-k3-1 is not Kimi K3, and a near-miss reported name is not a match
    assert retired["source_verification"]["status"] == "rejected_exact_label" and retired["split"] == "quarantine"
    assert k31["source_verification"]["status"] == "rejected_exact_label"
    # an answer from another model does not block the right model's later answer to the same prompt
    (global_kimi,) = collection.api_records([workbuddy_entry("kimi-k3", "N02", bank, "kimi-k3", "kimi-k3",
                                                             text="Global " + ANSWER)], bank)
    decisions = collection.admission([retired, global_kimi]).set_index("record_id").admitted
    assert decisions.to_dict() == {retired["record_id"]: False, global_kimi["record_id"]: True}
    assert "possible_truncation" in cut["quality"]["issues"]
    assert tool["quality"]["tool_use_observed"] and tool["split"] == "quarantine"


def test_moonshot_records_and_admission_per_interface(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    official = {**workbuddy_entry("kimi-k3", "N01", bank, "kimi-k3", "kimi-k3", text="Official " + ANSWER),
                "source": "moonshot", "request_uid": "m-1", "max_output_tokens": 16000, "thinking": "disabled",
                "usd": 0.02}
    swapped = {**official, "request_uid": "m-2", "prompt_id": "N02", "prompt_text": bank["N02"]["prompt_text"],
               "response": {**official["response"], "model": "kimi-k3.1-pro"}}
    again = {**official, "request_uid": "m-3", "response": {**official["response"], "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "Second " + ANSWER}, "finish_reason": "stop"}]}}
    gateway = workbuddy_entry("kimi-k3", "N01", bank, "kimi-k3", "kimi-k3", text="Gateway " + ANSWER)
    # entries from before the thinking setting existed ran with the API's default, thinking on
    thinking = {k: v for k, v in again.items() if k != "thinking"} | {
        "request_uid": "m-4", "response": {**official["response"], "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Thought " + ANSWER}, "finish_reason": "stop"}]}}
    moon, wrong, moon_again, wb, moon_thinking = collection.api_records([official, swapped, again, gateway, thinking],
                                                                        bank)
    for record in (moon, wrong, moon_again, wb, moon_thinking):
        validator.validate(record)
    assert moon["source_verification"]["status"] == "snapshot_verified" and moon["served_model_id"] == "kimi-k3"
    assert moon["provider"] == "moonshot" and moon["settings"]["max_output_tokens"] == 16000
    assert moon["interface"] == "api:moonshot_chat_completions"
    assert moon_thinking["interface"] == "api:moonshot_thinking_chat_completions"
    assert moon["settings"]["reasoning_effort"].startswith("none") and moon_thinking["settings"]["reasoning_effort"] is None
    assert wrong["source_verification"]["status"] == "rejected_exact_label"
    # the same prompt through two interfaces (or thinking settings) is two conditions; a second answer through
    # one is not
    decisions = collection.admission([moon, wb, moon_again, moon_thinking]).set_index("record_id")
    assert decisions.admitted.to_dict() == {moon["record_id"]: True, wb["record_id"]: True,
                                            moon_again["record_id"]: False, moon_thinking["record_id"]: True}
    assert decisions.loc[moon_again["record_id"], "reasons"] == "later_answer_for_pair"


def antigravity_entry(pid, bank, text=ANSWER, tool=False):
    steps = [{"step_index": 0, "source": "USER_EXPLICIT", "type": "USER_INPUT", "status": "DONE",
              "content": f"<USER_REQUEST>\n{bank[pid]['prompt_text']}\n</USER_REQUEST>"}]
    if tool:
        steps += [{"step_index": 1, "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE", "content": ""},
                  {"step_index": 2, "source": "MODEL", "type": "GENERIC", "status": "DONE", "content": "ran a command"}]
    steps.append({"step_index": len(steps), "source": "MODEL", "type": "PLANNER_RESPONSE", "status": "DONE",
                  "input_tokens": 9117, "output_tokens": 4079, "content": text, "thinking": "plan the paragraph"})
    return {"source": "antigravity", "request_uid": f"a-{pid}-{tool}", "label": "gemini-3.8-flash", "prompt_id": pid,
            "model_id": "flash", "prompt_text": bank[pid]["prompt_text"], "endpoint": "antigravity agentapi",
            "conversation_id": "c-1", "requested_at_utc": "2026-11-05T10:00:00Z", "attempt": 1, "usd": None,
            "response": {"steps": steps, "metadata": {}}}


def test_antigravity_records(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    clean, tooled = collection.api_records([antigravity_entry("N01", bank),
                                            antigravity_entry("N02", bank, text="Tool " + ANSWER, tool=True)], bank)
    for record in (clean, tooled):
        validator.validate(record)
    assert clean["source_verification"]["status"] == "ui_label_verified" and clean["provider"] == "google"
    assert clean["raw_response"] == ANSWER and clean["visible_reasoning"] == "plan the paragraph"
    assert clean["interface"] == "antigravity:agentapi" and clean["split"] != "quarantine"
    assert tooled["quality"]["tool_use_observed"] and tooled["split"] == "quarantine"


def qoder_entry(label, pid, bank, key, text=ANSWER, stop="end_turn", turns=1):
    return {"source": "qoder", "request_uid": f"q-{label}-{pid}-{key}", "label": label, "prompt_id": pid,
            "model_id": "Qwen3.8-Max", "prompt_text": bank[pid]["prompt_text"], "endpoint": "qoderclicn -p",
            "client_version": "qoderclicn 1.1.81", "requested_at_utc": "2026-11-05T10:00:00Z", "attempt": 1,
            "usd": None, "response": {"type": "result", "subtype": "success", "is_error": False, "num_turns": turns,
                                      "result": text, "stop_reason": stop, "total_credits": 0.13,
                                      "usage": {"input_tokens": 0, "output_tokens": 0, "request_id": "r-1",
                                                "server_tool_use": {"web_search_requests": 0}},
                                      "modelUsage": {key: {"inputTokens": 0, "credits": 0.13}}}}


def test_qoder_records(bank):
    schema = json.loads(collection.SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema, format_checker=jsonschema.FormatChecker())
    qwen, swapped, agent = collection.api_records([
        qoder_entry("qwen3.8-max", "N01", bank, "qmodel_38max"),
        qoder_entry("qwen3.8-max", "N02", bank, "qmodel_38flash", text="Flash " + ANSWER),
        qoder_entry("qwen3.8-max", "N03", bank, "qmodel_38max", text="Looked " + ANSWER, turns=3),
    ], bank)
    for record in (qwen, swapped, agent):
        validator.validate(record)
    assert qwen["source_verification"]["status"] == "ui_label_verified" and qwen["provider"] == "alibaba"
    assert qwen["raw_response"] == ANSWER and qwen["cost"]["output_tokens"] is None
    assert qwen["session"]["system_instructions_status"] == "partially_known"
    assert swapped["source_verification"]["status"] == "rejected_exact_label"
    assert agent["quality"]["tool_use_observed"] and agent["split"] == "quarantine"


def test_read_api_logs_merges_every_source(bank, tmp_path, monkeypatch):
    monkeypatch.setattr(collection, "EXTERNAL_DIR", tmp_path)
    for name, entry in (("api_log.jsonl", api_entry("gemini-3.8-flash", "N01", bank, "gemini-3.8-flash")),
                        ("api_log_anthropic.jsonl", anthropic_entry("claude-opus-5", "N01", bank)),
                        ("api_log_codex.jsonl", codex_entry("gpt-5.6-luna", "N01", bank))):
        (tmp_path / name).write_text(json.dumps(entry) + "\n", encoding="utf-8")
    log = collection.read_api_logs()
    assert sorted(e["log"] for e in log) == ["api_log.jsonl", "api_log_anthropic.jsonl", "api_log_codex.jsonl"]
    providers = sorted(r["provider"] for r in collection.api_records(log, bank))
    assert providers == ["anthropic", "google", "openai"]


@pytest.mark.parametrize("text, cited", [
    ("Mail clients fetch it using IMAP or POP3. Then it shows.", False),
    ("The model is trained on data12. Then it predicts.", True),
    ("Light bends[1] in water.", True),
])
def test_citation_marks_but_not_acronyms(text, cited):
    assert ("citations_or_search" in collection.quality(text, "", None)["issues"]) is cited


def test_api_answer_restating_the_question_is_not_an_echo(bank):
    restated = bank["N04"]["prompt_text"][:60] + " because abstraction keeps only the structure we need."
    (record,) = collection.api_records([api_entry("gemini-3.8-flash", "N04", bank, "gemini-3.8-flash",
                                                  text=restated)], bank)
    assert "prompt_echo" not in record["quality"]["issues"]
