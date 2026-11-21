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
    assert "gemini-3.7-flash" not in collection.TARGETS and len(collection.TARGETS) == 17


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
])
def test_quality_flags(answer, flag):
    prompt = "Explain why a metal handrail and a wooden handrail can feel different in temperature."
    issues = {i.split(":")[0] for i in collection.quality(answer, prompt, "10-40")["issues"]}
    assert flag in issues


def test_ordinary_prose_is_not_clutter():
    text = "Thinking about heat this way helps. Copy that idea to wood and it still holds."
    assert collection.quality(text, "", None)["issues"] == []


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
