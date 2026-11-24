"""Collect Gemini answers through Google Cloud's Agent Platform API (formerly Vertex AI).

On Agent Platform, unlike the consumer Gemini API, gemini-3.7-flash is still served (retires
28 Jan 2027; gemini-3.6-flash retires 19 Nov 2026), and every generateContent response names
the served model in modelVersion, which verifies or rejects the label answer by answer. Each
bank prompt goes once to each model as a fresh single-turn request with API defaults: no
system instruction, no tools, only maxOutputTokens. Responses are appended to
data/external/api_log.jsonl (never edited), and scripts/import_answers.py turns them into
kit-schema records alongside the pasted chat answers. Failed requests go to
api_failures.jsonl. Rerunning resumes: prompts that already have a logged response are skipped.

Authentication (the key or token is never printed, logged or written anywhere):
  API key: $AGENT_PLATFORM_API_KEY, or the first line of ~/.config/nllm/agent_platform_api_key
  --auth gcloud: a token from `gcloud auth print-access-token` (Google Cloud SDK, signed in);
                 the project defaults to `gcloud config get-value project`
With --project (or $GOOGLE_CLOUD_PROJECT) requests go to projects/<id>/locations/<location>;
without it they use the express-mode path, which works with an express-mode API key.

Usage:
  uv run python scripts/collect_agent_platform.py                          # dry run: plan and cost
  uv run python scripts/collect_agent_platform.py --execute --limit 1      # probe: one prompt per model
  uv run python scripts/collect_agent_platform.py --execute --max-usd 30   # the rest
  uv run python scripts/import_answers.py
"""

import argparse
import json
import os
import queue
import random
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nllm import collection
from nllm.data import LABELS
from nllm.paths import KIT_DIR

MODELS = [m for m in LABELS if m.startswith("gemini-")]
HOST = "https://aiplatform.googleapis.com/v1"
KEY_ENV = "AGENT_PLATFORM_API_KEY"
KEY_FILE = Path.home() / ".config" / "nllm" / "agent_platform_api_key"
FAILURES_PATH = collection.EXTERNAL_DIR / "api_failures.jsonl"
MAX_OUTPUT_TOKENS = 8192  # thinking counts against it; answers are 40-250 words
RETRY = {0, 408, 429, 500, 502, 503, 504}
MAX_TRIES = 7  # backoff 10, 20, 40, 80, 160, 240 s (with jitter) before a prompt is left for the next run


def rates() -> dict[str, tuple[float, float]]:
    """USD per million input/output tokens from the kit; 3.7 has no listed rate, so 3.8's is used."""
    table = pd.read_csv(KIT_DIR / "model_access_and_prices.csv", encoding="utf-8").set_index("target_label")
    out = {}
    for m in MODELS:
        row = table.loc[m] if not pd.isna(table.loc[m, "usd_per_million_input"]) else table.loc["gemini-3.8-flash"]
        out[m] = (float(row.usd_per_million_input), float(row.usd_per_million_output))
    return out


def find_gcloud() -> str | None:
    """gcloud on PATH, else the Google Cloud SDK's default Windows install (shells started before it lack PATH)."""
    default = Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Cloud SDK/google-cloud-sdk/bin/gcloud.cmd"
    return shutil.which("gcloud") or (str(default) if default.exists() else None)


def gcloud_project() -> str | None:
    gcloud = find_gcloud()
    if not gcloud:
        return None
    result = subprocess.run([gcloud, "config", "get-value", "project"], check=False, capture_output=True, text=True)
    return result.stdout.strip() or None if result.returncode == 0 else None


def endpoint(model: str, project: str | None, location: str) -> str:
    base = f"{HOST}/projects/{project}/locations/{location}" if project else HOST
    return f"{base}/publishers/google/models/{model}:generateContent"


class Credentials:
    """Request headers from an API key or a gcloud access token (refreshed every 30 minutes)."""

    def __init__(self, auth: str):
        self.auth, self.lock, self.token, self.fetched = auth, threading.Lock(), None, 0.0
        if auth == "key":
            self.key = os.environ.get(KEY_ENV, "").strip()
            if not self.key and KEY_FILE.exists():
                self.key = (KEY_FILE.read_text(encoding="utf-8-sig").strip().splitlines() or [""])[0].strip()

    def available(self) -> str | None:
        if self.auth == "gcloud":
            return "gcloud" if find_gcloud() else None
        return ("environment" if os.environ.get(KEY_ENV) else "key file") if self.key else None

    def headers(self) -> dict[str, str]:
        if self.auth == "key":
            return {"x-goog-api-key": self.key}
        with self.lock:
            # gcloud can hand back a cached token close to expiry, so refresh often and on any 401 (expire()).
            if self.token is None or time.time() - self.fetched > 600:
                result = subprocess.run([find_gcloud(), "auth", "print-access-token"], check=False,
                                        capture_output=True, text=True)
                if result.returncode != 0:
                    raise SystemExit("gcloud auth print-access-token failed: sign in with the Google Cloud SDK first")
                self.token, self.fetched = result.stdout.strip(), time.time()
            return {"Authorization": f"Bearer {self.token}"}

    def expire(self) -> None:
        with self.lock:
            self.token = None


def post(url: str, headers: dict, body: dict, timeout: float) -> tuple[int, dict | str]:
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:1500]
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return 0, f"{type(e).__name__}: {e}"[:500]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--waves", nargs="+", default=["wave1", "wave2", "wave3", "wave4", "wave5"],
                    choices=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"])
    ap.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
                    help="Google Cloud project id (omit for an express-mode key)")
    ap.add_argument("--location", default="global")
    ap.add_argument("--auth", choices=["key", "gcloud"], default="key")
    ap.add_argument("--execute", action="store_true", help="send paid requests; otherwise a dry run")
    ap.add_argument("--limit", type=int, help="at most this many new answers per model (probe with 1)")
    ap.add_argument("--max-usd", type=float, default=30.0, help="stop once logged spend reaches this")
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--workers", type=int, default=4, help="parallel requests per model (rate limits back off)")
    ap.add_argument("--seed", type=int, default=0, help="seeds each model's prompt order")
    args = ap.parse_args()
    if args.auth == "gcloud" and not args.project:
        args.project = gcloud_project()  # gcloud tokens need the project path, not the express-mode one

    bank = collection.load_bank()
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    log = read_jsonl(collection.API_LOG_PATH)
    done = {(e["label"], e["prompt_id"]) for e in log}
    tries = defaultdict(int)
    for f in read_jsonl(FAILURES_PATH):
        tries[f["label"], f["prompt_id"]] += 1
    spent = sum(e.get("usd") or 0.0 for e in log)
    price = rates()
    todo = {}
    for m in args.models:
        pending = [p for p in prompts if (m, p["prompt_id"]) not in done]
        random.Random(f"{args.seed}|{m}").shuffle(pending)
        todo[m] = pending[: args.limit] if args.limit else pending

    creds = Credentials(args.auth)
    target = f"project {args.project}, location {args.location}" if args.project else "express mode (no project)"
    print(f"Agent Platform generateContent, {target}; credentials: {creds.available() or 'MISSING'}")
    print(f"{len(prompts)} prompts ({', '.join(args.waves)}); {len(log)} answers already logged, ${spent:.2f} spent")
    total_lo = total_hi = 0.0
    for m in args.models:
        rin, rout = price[m]
        lo = len(todo[m]) * (150 * rin + 300 * rout) / 1e6
        hi = len(todo[m]) * (150 * rin + 2300 * rout) / 1e6
        total_lo, total_hi = total_lo + lo, total_hi + hi
        print(f"  {m:24} {len(todo[m]):4} to request   est. ${lo:.2f} - ${hi:.2f} (answer only / +2,000 thinking tokens)")
    print(f"estimated total ${total_lo:.2f} - ${total_hi:.2f}; spending cap ${args.max_usd:.2f}")
    if not args.execute:
        print("dry run: nothing sent. Add --execute (with --limit 1 for a probe).")
        return
    if not creds.available():
        raise SystemExit(f"no credentials: set {KEY_ENV}, write the key to {KEY_FILE}, or use --auth gcloud")
    if spent >= args.max_usd:
        raise SystemExit(f"logged spend ${spent:.2f} already reaches --max-usd")

    collection.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    state = {"spent": spent, "ok": 0, "failed": 0}

    def log_failure(m, p, status, message):
        with lock:
            state["failed"] += 1
            with open(FAILURES_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps({"label": m, "prompt_id": p["prompt_id"], "status": status, "message": message,
                                    "at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}) + "\n")
        print(f"  FAIL {m} {p['prompt_id']} HTTP {status}: {str(message)[:300]}", flush=True)

    queues = {m: queue.Queue() for m in args.models}
    for m in args.models:
        for p in todo[m]:
            queues[m].put(p)
    halted = {m: threading.Event() for m in args.models}

    def worker(m: str) -> None:
        """Takes this model's prompts off its queue; several workers per model run in parallel."""
        url = endpoint(m, args.project, args.location)
        body_config = {"maxOutputTokens": MAX_OUTPUT_TOKENS}
        while not stop.is_set() and not halted[m].is_set():
            try:
                p = queues[m].get_nowait()
            except queue.Empty:
                return
            for attempt_try in range(MAX_TRIES):
                if stop.is_set() or halted[m].is_set():
                    return
                body = {"contents": [{"role": "user", "parts": [{"text": p["prompt_text"]}]}],
                        "generationConfig": body_config}
                requested = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                status, payload = post(url, creds.headers(), body, args.timeout)
                if status == 200 and isinstance(payload, dict):
                    parsed = collection.parse_generate_content(payload)
                    rin, rout = price[m]
                    out_tokens = (parsed["output_tokens"] or 0) + (parsed["reasoning_tokens"] or 0)
                    usd = ((parsed["input_tokens"] or 0) * rin + out_tokens * rout) / 1e6
                    with lock:
                        attempt = tries[m, p["prompt_id"]] + 1
                    entry = {"source": "agent_platform", "request_uid": uuid.uuid4().hex, "label": m,
                             "prompt_id": p["prompt_id"], "model_id": m,
                             "prompt_text": p["prompt_text"], "endpoint": url, "location": args.location,
                             "requested_at_utc": requested, "attempt": attempt,
                             "generation_config": body_config, "usd": usd, "response": payload}
                    with lock:
                        with open(collection.API_LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        state["ok"] += 1
                        state["spent"] += usd
                        if state["spent"] >= args.max_usd:
                            stop.set()
                    print(f"  {m} {p['prompt_id']}: {len(parsed['text'].split())} words, served {parsed['served']}, "
                          f"{parsed['finish']}, {parsed['input_tokens']}+{out_tokens} tokens, "
                          f"total ${state['spent']:.2f}", flush=True)
                    break
                with lock:
                    tries[m, p["prompt_id"]] += 1
                log_failure(m, p, status, payload)
                if status == 401 and args.auth == "gcloud" and attempt_try < 2:
                    creds.expire()  # an expired access token: fetch a fresh one and retry
                    continue
                if status not in RETRY:
                    halted[m].set()
                    print(f"  stopping {m}: HTTP {status} will not succeed on retry (model id, access or quota setup)")
                    return
                # Rate limits: exponential backoff with jitter, so parallel workers do not retry in lockstep.
                time.sleep(min(240.0, 10 * 2**attempt_try) * random.uniform(0.5, 1.5))

    threads = [threading.Thread(target=worker, args=(m,)) for m in args.models if todo[m]
               for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    note = " (stopped at the spending cap)" if stop.is_set() else ""
    print(f"\n{state['ok']} answers logged, {state['failed']} failed requests, ${state['spent']:.2f} logged spend{note}."
          f"\nNext: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
