"""Collect GLM, Kimi and GPT-5.5 answers through the owner's WorkBuddy-to-API gateway (W2A), on the WorkBuddy plan.

W2A is a self-hosted workbuddy2api gateway on the owner's tailnet that exposes the WorkBuddy account
as an OpenAI-compatible Chat Completions API. Each bank prompt is sent once per model as one user
message, with no system prompt or tools. Sampling is left to the gateway (the owner set its default
temperature to 1), and reasoning effort to WorkBuddy's default unless --effort is given. WorkBuddy's
upstream instructions are not visible, and the response's model field may only echo the request, so
these answers are verified only by the selected model id (ui_label_verified); a different reported
model rejects the answer.

Labels and gateway model ids: glm-5.3, glm-5.3-flash, kimi-k3 and gpt-5.5, the same names. The GLM answers came
from the owner's CN WorkBuddy account; kimi-k3 and gpt-5.5 need the global account (the CN account offers only
kimi-k3-1, whose identity is unconfirmed, and no gpt-5.5, which Codex no longer offers either), so the
gateway's account is recorded with --account.
Gateway URL: $WORKBUDDY_BASE_URL, or the first line of ~/.config/nllm/workbuddy_base_url (e.g. http://host:8788/v1).
Key: $WORKBUDDY_API_KEY, or the first line of ~/.config/nllm/workbuddy_api_key (the panel's settings page).
Answers are appended to data/external/api_log_workbuddy_<label>.jsonl (never edited; the first GLM/Kimi run
wrote api_log_workbuddy.jsonl), failures to api_failures_workbuddy_<label>.jsonl. One file per model lets
runs for different models go in parallel; don't run the same model twice at once. Each response's usage
reports the WorkBuddy credits it used (on average 0.11 for glm-5.3 and 0.01 for glm-5.3-flash at the default
effort on the CN account). Rerunning resumes (prompts with a logged answer are skipped).

Usage:
  uv run python scripts/collect_workbuddy.py                         # dry run: plan
  uv run python scripts/collect_workbuddy.py --execute --limit 1     # probe: one prompt per model
  uv run python scripts/collect_workbuddy.py --execute --workers 2
"""

import argparse
import json
import os
import queue
import random
import re
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from nllm import collection

# label -> gateway id. kimi-k3 comes from the global WorkBuddy account; the CN account offers only kimi-k3-1, not
# kimi-k3 (collection.RETIRED_SELECTIONS), so its answers no longer count as logged for kimi-k3.
MODELS = {"glm-5.3": "glm-5.3", "glm-5.3-flash": "glm-5.3-flash", "kimi-k3": "kimi-k3", "gpt-5.5": "gpt-5.5"}
EFFORTS = {"glm-5.3": ("low", "high", "max"), "glm-5.3-flash": ("low", "high", "max"),
           "kimi-k3": ("medium",),  # fixed on the global account
           "gpt-5.5": ("low", "medium", "high", "xhigh")}
CONFIG_DIR = Path.home() / ".config" / "nllm"


def log_path(label: str) -> Path:
    return collection.EXTERNAL_DIR / f"api_log_workbuddy_{label}.jsonl"


def failures_path(label: str) -> Path:
    return collection.EXTERNAL_DIR / f"api_failures_workbuddy_{label}.jsonl"
RETRY = {0, 408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524}
MAX_TRIES = 6
# An exhausted plan allowance is account-wide and stops every model.
QUOTA = re.compile(r"quota|insufficient|额度|余额|用量", re.IGNORECASE)


def setting(env: str, file: str) -> str:
    value = os.environ.get(env, "").strip()
    path = CONFIG_DIR / file
    if not value and path.exists():
        value = (path.read_text(encoding="utf-8-sig").strip().splitlines() or [""])[0].strip()
    return value


def post(url: str, key: str, body: dict, timeout: float) -> tuple[int, dict | str]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:1500]
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
        return 0, f"{type(e).__name__}: {e}"[:500]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--waves", nargs="+", default=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"],
                    choices=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"])
    ap.add_argument("--execute", action="store_true", help="send requests; otherwise a dry run")
    ap.add_argument("--limit", type=int, help="at most this many new prompts per model (probe with 1)")
    ap.add_argument("--workers", type=int, default=2, help="parallel requests per model")
    ap.add_argument("--effort", help="reasoning effort sent to every model (GLM: low/high/max, Kimi: low/high/xhigh); "
                                     "default: WorkBuddy's own default")
    ap.add_argument("--account", default="global", choices=["global", "cn"],
                    help="which WorkBuddy account the gateway is signed in to (recorded with each answer)")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for m in args.models:
        if args.effort and args.effort not in EFFORTS[m]:
            ap.error(f"{m} takes reasoning effort {'/'.join(EFFORTS[m])}, not {args.effort!r}")

    bank = collection.load_bank()
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    logged = {(e["label"], e["prompt_id"]) for path in collection.EXTERNAL_DIR.glob("api_log_workbuddy*.jsonl")
              for e in read_jsonl(path) if e["model_id"] == MODELS[e["label"]]}
    tries = defaultdict(int)
    for f in (f for path in collection.EXTERNAL_DIR.glob("api_failures_workbuddy*.jsonl") for f in read_jsonl(path)):
        tries[f["label"], f["prompt_id"]] += 1
    todo = {}
    for m in args.models:
        pending = [p for p in prompts if (m, p["prompt_id"]) not in logged]
        random.Random(f"{args.seed}|{m}").shuffle(pending)
        todo[m] = pending[: args.limit] if args.limit else pending

    base, key = setting("WORKBUDDY_BASE_URL", "workbuddy_base_url"), setting("WORKBUDDY_API_KEY", "workbuddy_api_key")
    url = base.rstrip("/") + "/chat/completions" if base else None
    print(f"WorkBuddy gateway ({args.account} account): {url or 'NO URL'}; key: {'found' if key else 'MISSING'}; "
          f"reasoning effort: {args.effort or 'default'}")
    print(f"{len(prompts)} prompts ({', '.join(args.waves)}); {len(logged)} answers logged")
    for m in args.models:
        print(f"  {m:14} -> {MODELS[m]:14} {len(todo[m]):4} to request")
    if not args.execute:
        print("dry run: nothing sent. Add --execute (with --limit 1 for a probe).")
        return
    if not url or not key:
        raise SystemExit(f"set WORKBUDDY_BASE_URL / WORKBUDDY_API_KEY, or write {CONFIG_DIR / 'workbuddy_base_url'} "
                         f"and {CONFIG_DIR / 'workbuddy_api_key'}")

    collection.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    state = {"ok": 0, "failed": 0, "credits": 0.0}
    total = sum(map(len, todo.values()))
    queues = {m: queue.Queue() for m in args.models}
    for m in args.models:
        for p in todo[m]:
            queues[m].put(p)
    halted = {m: threading.Event() for m in args.models}

    def log_failure(m, p, status, message):
        with lock:
            tries[m, p["prompt_id"]] += 1
            state["failed"] += 1
            with open(failures_path(m), "a", encoding="utf-8") as f:
                f.write(json.dumps({"label": m, "prompt_id": p["prompt_id"], "status": status, "message": message,
                                    "at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
                                   ensure_ascii=False) + "\n")
        print(f"  FAIL {m} {p['prompt_id']} HTTP {status}: {str(message)[:300]}", flush=True)

    def worker(m: str) -> None:
        body = {"model": MODELS[m], "stream": False} | ({"reasoning_effort": args.effort} if args.effort else {})
        while not stop.is_set() and not halted[m].is_set():
            try:
                p = queues[m].get_nowait()
            except queue.Empty:
                return
            for attempt_try in range(MAX_TRIES):
                if stop.is_set() or halted[m].is_set():
                    return
                requested = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                status, payload = post(url, key, body | {"messages": [{"role": "user", "content": p["prompt_text"]}]},
                                       args.timeout)
                parsed = collection.parse_chat_completion(payload) if isinstance(payload, dict) else None
                if status == 200 and parsed and parsed["text"].strip():
                    entry = {"source": "workbuddy", "request_uid": uuid.uuid4().hex, "label": m,
                             "prompt_id": p["prompt_id"], "model_id": MODELS[m], "prompt_text": p["prompt_text"],
                             "endpoint": "workbuddy2api gateway /v1/chat/completions", "requested_at_utc": requested,
                             "attempt": tries[m, p["prompt_id"]] + 1, "account": args.account,
                             "reasoning_effort": args.effort,
                             "usd": None, "response": payload}
                    with lock:
                        with open(log_path(m), "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        state["ok"] += 1
                        state["credits"] += float((payload.get("usage") or {}).get("credit") or 0)
                    thinking = len((parsed["reasoning"] or "").split())
                    print(f"  {m} {p['prompt_id']}: {len(parsed['text'].split())} words, reported {parsed['reported']}, "
                          f"{parsed['finish']}, {parsed['input_tokens']}+{parsed['output_tokens']} tokens, "
                          f"{thinking} thinking words, {state['credits']:.2f} credits [{state['ok']}/{total}]", flush=True)
                    break
                # a 200 without an answer (gateway error object, empty content) is retried like a server error
                message = payload if status != 200 else f"no answer in response: {json.dumps(payload)[:800]}"
                log_failure(m, p, status, message)
                if QUOTA.search(str(message)):
                    stop.set()
                    print("  stopping: the WorkBuddy allowance looks exhausted; rerun later to resume", flush=True)
                    return
                if status not in RETRY | {200}:
                    halted[m].set()
                    print(f"  stopping {m}: HTTP {status} will not succeed on retry (model id, key or access)", flush=True)
                    return
                time.sleep(min(240.0, 10 * 2**attempt_try) * random.uniform(0.5, 1.5))

    threads = [threading.Thread(target=worker, args=(m,)) for m in args.models if todo[m] for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n{state['ok']} answers logged ({state['credits']:.2f} WorkBuddy credits), {state['failed']} failed requests."
          f"\nNext: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
