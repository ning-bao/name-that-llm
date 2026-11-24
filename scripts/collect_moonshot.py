"""Collect kimi-k3 answers from Moonshot's own API (OpenAI-compatible Chat Completions, api.moonshot.ai).

Each bank prompt is sent once as a single user message with no system prompt or tools, an output-token
cap (MAX_TOKENS) and, by default, thinking turned off ("thinking": {"type": "disabled"}); temperature is
the API's default. With thinking off, 20 test answers matched the official kimi-k3 passages' em-dash and
curly-apostrophe rates, while thinking-on answers (the API default, and WorkBuddy's fixed setting) use em
dashes in ~90% of answers and almost no curly apostrophes. `--thinking default` keeps the API's default;
those answers are their own condition (interface api:moonshot_thinking_chat_completions), as are the
WorkBuddy kimi-k3 answers, so all may be admitted for the same prompt (admission is per interface).
The response's model field is Moonshot's own, so it verifies the label (snapshot_verified) or rejects it.

Key: $MOONSHOT_API_KEY, or the first line of ~/.config/nllm/moonshot_api_key. Cost is computed from the
reported tokens at PRICES (third-party trackers' list price; check platform.kimi.ai) and capped by --max-usd.
Answers are appended to data/external/api_log_moonshot.jsonl (never edited), failures to
api_failures_moonshot.jsonl. Rerunning resumes.

Usage:
  uv run python scripts/collect_moonshot.py                         # dry run: plan
  uv run python scripts/collect_moonshot.py --execute --limit 1     # probe
  uv run python scripts/collect_moonshot.py --execute
"""

import argparse
import json
import os
import queue
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from nllm import collection

BASE_URL = "https://api.moonshot.ai/v1"
MODELS = {"kimi-k3": "kimi-k3"}  # label -> Moonshot model id
PRICES = {"kimi-k3": (3.0, 15.0)}  # USD per million input / output tokens
MAX_TOKENS = 16000
KEY_ENV = "MOONSHOT_API_KEY"
KEY_FILE = Path.home() / ".config" / "nllm" / "moonshot_api_key"
FAILURES_PATH = collection.EXTERNAL_DIR / "api_failures_moonshot.jsonl"
RETRY = {0, 408, 409, 425, 429, 500, 502, 503, 504}
MAX_TRIES = 6


def read_key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key and KEY_FILE.exists():
        key = (KEY_FILE.read_text(encoding="utf-8-sig").strip().splitlines() or [""])[0].strip()
    return key


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
    ap.add_argument("--workers", type=int, default=1, help="parallel requests per model (the owner's account allows 1)")
    ap.add_argument("--thinking", choices=["disabled", "default"], default="disabled",
                    help="turn thinking off (matches the official data), or keep the API's default (on)")
    ap.add_argument("--max-usd", type=float, default=10.0, help="stop once logged spend reaches this")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = collection.load_bank()
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    log = read_jsonl(collection.MOONSHOT_LOG_PATH)
    # entries from before --thinking existed ran with the API's default
    logged = {(e["label"], e["prompt_id"]) for e in log if e.get("thinking", "default") == args.thinking}
    spent = sum(e.get("usd") or 0 for e in log)
    tries = defaultdict(int)
    for f in read_jsonl(FAILURES_PATH):
        tries[f["label"], f["prompt_id"]] += 1
    todo = {}
    for m in args.models:
        pending = [p for p in prompts if (m, p["prompt_id"]) not in logged]
        random.Random(f"{args.seed}|{m}").shuffle(pending)
        todo[m] = pending[: args.limit] if args.limit else pending

    key = read_key()
    print(f"Moonshot API {BASE_URL}; thinking {args.thinking}; key: {'found' if key else 'MISSING'}; "
          f"{len(log)} answers logged, ${spent:.2f} spent")
    for m in args.models:
        print(f"  {m:10} -> {MODELS[m]:10} {len(todo[m]):4} to request")
    if not args.execute:
        print("dry run: nothing sent. Add --execute (with --limit 1 for a probe).")
        return
    if not key:
        raise SystemExit(f"set {KEY_ENV} or write the key to {KEY_FILE}")
    if spent >= args.max_usd:
        raise SystemExit(f"logged spend ${spent:.2f} already reaches --max-usd")

    collection.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    state = {"ok": 0, "failed": 0, "spent": spent}
    total = sum(map(len, todo.values()))
    queues = {m: queue.Queue() for m in args.models}
    for m in args.models:
        for p in todo[m]:
            queues[m].put(p)
    halted = {m: threading.Event() for m in args.models}

    def worker(m: str) -> None:
        url = BASE_URL + "/chat/completions"
        while not stop.is_set() and not halted[m].is_set():
            try:
                p = queues[m].get_nowait()
            except queue.Empty:
                return
            for attempt_try in range(MAX_TRIES):
                if stop.is_set() or halted[m].is_set():
                    return
                body = {"model": MODELS[m], "max_tokens": MAX_TOKENS,
                        "messages": [{"role": "user", "content": p["prompt_text"]}]}
                if args.thinking == "disabled":
                    body["thinking"] = {"type": "disabled"}
                requested = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                status, payload = post(url, key, body, args.timeout)
                parsed = collection.parse_official_chat_completion(payload) if isinstance(payload, dict) else None
                if status == 200 and parsed and parsed["text"].strip():
                    rin, rout = PRICES[m]
                    usd = ((parsed["input_tokens"] or 0) * rin + (parsed["output_tokens"] or 0) * rout) / 1e6
                    entry = {"source": "moonshot", "request_uid": uuid.uuid4().hex, "label": m,
                             "prompt_id": p["prompt_id"], "model_id": MODELS[m], "prompt_text": p["prompt_text"],
                             "endpoint": url, "requested_at_utc": requested, "attempt": tries[m, p["prompt_id"]] + 1,
                             "max_output_tokens": MAX_TOKENS, "thinking": args.thinking, "usd": usd,
                             "response": payload}
                    with lock:
                        with open(collection.MOONSHOT_LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        state["ok"] += 1
                        state["spent"] += usd
                        if state["spent"] >= args.max_usd:
                            stop.set()
                    print(f"  {m} {p['prompt_id']}: {len(parsed['text'].split())} words, served {parsed['served']}, "
                          f"{parsed['finish']}, {parsed['input_tokens']}+{parsed['output_tokens']} tokens "
                          f"({parsed['reasoning_tokens']} reasoning), total ${state['spent']:.2f} [{state['ok']}/{total}]",
                          flush=True)
                    break
                message = payload if status != 200 else f"no answer in response: {json.dumps(payload)[:800]}"
                with lock:
                    tries[m, p["prompt_id"]] += 1
                    state["failed"] += 1
                    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"label": m, "prompt_id": p["prompt_id"], "status": status, "message": message,
                                            "at_utc": requested}, ensure_ascii=False) + "\n")
                print(f"  FAIL {m} {p['prompt_id']} HTTP {status}: {str(message)[:300]}", flush=True)
                if status not in RETRY | {200}:
                    halted[m].set()
                    print(f"  stopping {m}: HTTP {status} will not succeed on retry (model id, key or balance)", flush=True)
                    return
                time.sleep(min(240.0, 10 * 2**attempt_try) * random.uniform(0.5, 1.5))

    threads = [threading.Thread(target=worker, args=(m,)) for m in args.models if todo[m] for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    note = " (stopped at the spending cap)" if stop.is_set() else ""
    print(f"\n{state['ok']} answers logged, {state['failed']} failed requests, ${state['spent']:.2f} logged spend{note}."
          f"\nNext: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
