"""Collect gemini-3.8-flash answers through Antigravity's own agent (agentapi), on the owner's Antigravity plan.

Gemini through Agent Platform writes em dashes in about 37% of answers; official gemini-3.8-flash
passages have none. A few answers from the Gemini app's work mode and from Antigravity had none either,
and copying Antigravity's identity and style lines into an API system prompt did not remove them, so this
script runs each bank prompt through Antigravity itself. It did not help: the first run (stopped on 3 Nov
2026 after 29 answers) had em dashes in 8 of 29. The script stays for reference.
`language_server.exe agentapi new-conversation
--model=flash` starts one fresh conversation per prompt with Antigravity's full agent context (~9k
tokens), and the answer is read from that conversation's transcript under ~/.gemini/antigravity/brain/.
Any step besides the prompt and a single reply (a tool call, a second planner turn) quarantines the
answer. Antigravity names no served model, so answers are verified only by the selection (ui_label_verified).

agentapi needs ANTIGRAVITY_LS_ADDRESS and ANTIGRAVITY_CSRF_TOKEN, which Antigravity sets only in the
terminals its agent opens. Run this script by asking Antigravity's agent to run it (and approve the
command); it refuses to start elsewhere. Each prompt adds a conversation titled "nllm <prompt id>" to
the Antigravity sidebar and uses the plan's quota. Progress goes to outputs/antigravity_collect.log,
answers to data/external/api_log_antigravity.jsonl (never edited), failures to
api_failures_antigravity.jsonl. Rerunning resumes.

Usage (from Antigravity's agent terminal, in the project folder):
  uv run python scripts/collect_antigravity.py --execute --limit 1
  uv run python scripts/collect_antigravity.py --execute --workers 2
"""

import argparse
import json
import os
import queue
import random
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from nllm import collection
from nllm.paths import OUTPUTS

MODELS = {"gemini-3.8-flash": "flash"}  # label -> agentapi --model
LS_DEFAULT = Path.home() / "AppData" / "Local" / "Programs" / "Antigravity" / "resources" / "bin" / "language_server.exe"
BRAIN = Path.home() / ".gemini" / "antigravity" / "brain"
FAILURES_PATH = collection.EXTERNAL_DIR / "api_failures_antigravity.jsonl"
PROGRESS_PATH = OUTPUTS / "antigravity_collect.log"
SETTLE_SECONDS = 15  # the transcript must stay unchanged this long after the reply before it is read
MAX_WAIT = 900
MAX_TRIES = 3
MAX_START_FAILURES = 3  # consecutive failures to start a conversation stop the run (agentapi itself is broken)


def agentapi(*args: str, timeout: float = 120) -> dict:
    exe = os.environ.get("ANTIGRAVITY_AGENTAPI_EXE") or str(LS_DEFAULT)
    cmd = [exe, *args] if exe.lower().endswith("agentapi.bat") else [exe, "agentapi", *args]
    result = subprocess.run(cmd, capture_output=True, timeout=timeout, check=False)
    text = result.stdout.decode("utf-8", "replace").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": f"exit {result.returncode}: {text[:500]} {result.stderr.decode('utf-8', 'replace')[:300]}"}


def read_steps(conversation_id: str) -> list[dict]:
    path = BRAIN / conversation_id / ".system_generated" / "logs" / "transcript_full.jsonl"
    if not path.exists():
        return []
    steps = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            break  # still being written
    return steps


def wait_for_reply(conversation_id: str) -> list[dict] | None:
    """The transcript once the last step is a finished model reply and nothing has changed for SETTLE_SECONDS."""
    started, last, stable_since = time.time(), None, None
    while time.time() - started < MAX_WAIT:
        steps = read_steps(conversation_id)
        done = bool(steps) and steps[-1].get("source") == "MODEL" and steps[-1].get("type") == "PLANNER_RESPONSE" \
            and steps[-1].get("status") == "DONE" and (steps[-1].get("content") or "").strip()
        if steps != last:
            last, stable_since = steps, time.time()
        elif done and time.time() - stable_since >= SETTLE_SECONDS:
            return steps
        time.sleep(3)
    return None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--waves", nargs="+", default=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"],
                    choices=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"])
    ap.add_argument("--execute", action="store_true", help="start conversations; otherwise a dry run")
    ap.add_argument("--limit", type=int, help="at most this many new prompts per model (probe with 1)")
    ap.add_argument("--workers", type=int, default=2, help="conversations in flight at once")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = collection.load_bank()
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    logged = {(e["label"], e["prompt_id"]) for e in read_jsonl(collection.ANTIGRAVITY_LOG_PATH)}
    tries = defaultdict(int)
    for f in read_jsonl(FAILURES_PATH):
        tries[f["label"], f["prompt_id"]] += 1
    jobs = []
    for m in args.models:
        pending = [p for p in prompts if (m, p["prompt_id"]) not in logged]
        random.Random(f"{args.seed}|{m}").shuffle(pending)
        jobs += [(m, p) for p in (pending[: args.limit] if args.limit else pending)]

    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    lock = threading.Lock()

    def say(message: str) -> None:
        line = f"{datetime.now().strftime('%H:%M:%S')} {message}"
        print(line, flush=True)
        with lock, open(PROGRESS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    in_antigravity = bool(os.environ.get("ANTIGRAVITY_LS_ADDRESS") and os.environ.get("ANTIGRAVITY_CSRF_TOKEN"))
    say(f"Antigravity agentapi: {'available' if in_antigravity else 'NOT available (run from the Antigravity agent)'}; "
        f"{len(prompts)} prompts, {len(logged)} answers logged, {len(jobs)} to run with {args.workers} workers")
    if not args.execute:
        say("dry run: nothing started. Add --execute (with --limit 1 for a probe).")
        return
    if not in_antigravity:
        raise SystemExit("agentapi needs ANTIGRAVITY_LS_ADDRESS and ANTIGRAVITY_CSRF_TOKEN, which only terminals "
                         "opened by Antigravity's agent have: ask the agent to run this command")

    state = {"ok": 0, "failed": 0, "start_failures": 0}
    stop = threading.Event()
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def fail(m, p, error, requested):
        with lock:
            tries[m, p["prompt_id"]] += 1
            state["failed"] += 1
            with open(FAILURES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"label": m, "prompt_id": p["prompt_id"], "error": error[:2000], "at_utc": requested},
                                   ensure_ascii=False) + "\n")
        say(f"FAIL {m} {p['prompt_id']}: {error[:300]}")

    def worker() -> None:
        while not stop.is_set():
            try:
                m, p = pending.get_nowait()
            except queue.Empty:
                return
            for _ in range(MAX_TRIES):
                requested = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                started = agentapi("new-conversation", f"--model={MODELS[m]}", f"--title=nllm {p['prompt_id']}",
                                   p["prompt_text"])
                cid = ((started.get("response") or {}).get("newConversation") or {}).get("conversationId")
                if not cid:
                    fail(m, p, f"no conversation: {json.dumps(started)[:800]}", requested)
                    with lock:
                        state["start_failures"] += 1
                        if state["start_failures"] >= MAX_START_FAILURES:
                            stop.set()
                    if stop.is_set():
                        say("stopping: agentapi failed to start a conversation several times in a row")
                        return
                    time.sleep(30)
                    continue
                with lock:
                    state["start_failures"] = 0
                steps = wait_for_reply(cid)
                if steps is None:
                    fail(m, p, f"no settled reply within {MAX_WAIT}s in conversation {cid}", requested)
                    continue
                metadata = agentapi("get-conversation-metadata", cid)
                entry = {"source": "antigravity", "request_uid": uuid.uuid4().hex, "label": m,
                         "prompt_id": p["prompt_id"], "model_id": MODELS[m], "prompt_text": p["prompt_text"],
                         "endpoint": "antigravity agentapi new-conversation", "conversation_id": cid,
                         "requested_at_utc": requested, "attempt": tries[m, p["prompt_id"]] + 1, "usd": None,
                         "response": {"steps": steps, "metadata": metadata}}
                parsed = collection.parse_antigravity_transcript(entry["response"])
                with lock:
                    with open(collection.ANTIGRAVITY_LOG_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    state["ok"] += 1
                text = parsed["text"]
                say(f"{m} {p['prompt_id']}: {len(text.split())} words, em dashes {text.count(chr(0x2014))}, "
                    f"{parsed['input_tokens']}+{parsed['output_tokens']} tokens{', TOOL USE' if parsed['tool_use'] else ''} "
                    f"[{state['ok']}/{len(jobs)}]")
                break

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    say(f"done: {state['ok']} answers logged, {state['failed']} failures. Next: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
