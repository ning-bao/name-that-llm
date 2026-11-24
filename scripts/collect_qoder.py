"""Collect qwen3.8-max answers through Qoder CLI CN (qoderclicn), on the owner's Qoder CN plan.

Each bank prompt runs once as a non-interactive `qoderclicn -p -o json` in a fresh empty directory,
with the prompt on stdin, Qoder's system prompt replaced by an empty one, every built-in tool disabled,
no MCP servers, no user/project settings or skills (--setting-sources ""), and no saved session. Billed
credits show that a few hundred tokens of unseen context remain, far less than Codex's agent
instructions. Qoder reports credits instead of tokens and names the model only by an internal key
(qmodel_38max), so these answers are verified only by the selected model (ui_label_verified); another
key rejects the answer. Thinking and sampling are Qoder's defaults.

The CLI is ~/.qoder-cn/bin/qoderclicn/qoderclicn-<version>.exe (the newest is used); sign in once with
`qoderclicn login`. Each answer costs about 0.13 Qoder credits. Answers are appended to
data/external/api_log_qoder.jsonl (never edited), failures to api_failures_qoder.jsonl. Rerunning resumes.

Usage:
  uv run python scripts/collect_qoder.py                         # dry run: plan
  uv run python scripts/collect_qoder.py --execute --limit 1     # probe
  uv run python scripts/collect_qoder.py --execute --workers 3
"""

import argparse
import json
import queue
import random
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from nllm import collection

MODELS = {"qwen3.8-max": "Qwen3.8-Max"}  # label -> Qoder model name
CLI_DIR = Path.home() / ".qoder-cn" / "bin" / "qoderclicn"
FAILURES_PATH = collection.EXTERNAL_DIR / "api_failures_qoder.jsonl"
QUOTA = re.compile(r"credit|quota|insufficient|额度|余额|not logged in", re.IGNORECASE)
RATE_LIMIT = re.compile(r"too many requests|rate limit|\b429\b|限流", re.IGNORECASE)
MAX_TRIES = 4


def newest_cli() -> Path | None:
    builds = list(CLI_DIR.glob("qoderclicn*.exe"))
    key = lambda p: tuple(int(x) for x in re.findall(r"\d+", p.stem))
    return max(builds, key=key) if builds else None


def run_qoder(cli: Path, model: str, prompt: str, timeout: float) -> tuple[dict | None, str]:
    # The CLI can still hold its working directory for a moment after exiting, so cleanup may be left to Windows.
    with tempfile.TemporaryDirectory(prefix="qoder-", ignore_cleanup_errors=True) as d:
        result = subprocess.run([str(cli), "-p", "-m", model, "-o", "json", "--system-prompt", "", "--tools", "",
                                 "--strict-mcp-config", "--setting-sources", "", "--no-session-persistence", "-w", d],
                                input=prompt.encode("utf-8"), capture_output=True, timeout=timeout, check=False)
    lines = result.stdout.decode("utf-8", "replace").splitlines()
    payload = next((json.loads(line) for line in reversed(lines) if line.startswith("{")), None)
    if result.returncode == 0 and payload and not payload.get("is_error") and (payload.get("result") or "").strip():
        return payload, ""
    detail = json.dumps(payload, ensure_ascii=False)[:1500] if payload else "\n".join(lines)[-800:]
    return payload, f"exit {result.returncode}: {detail} {result.stderr.decode('utf-8', 'replace')[-500:]}".strip()


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--waves", nargs="+", default=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"],
                    choices=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"])
    ap.add_argument("--execute", action="store_true", help="run Qoder; otherwise a dry run")
    ap.add_argument("--limit", type=int, help="at most this many new prompts per model (probe with 1)")
    ap.add_argument("--workers", type=int, default=3, help="parallel Qoder runs across all models")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = collection.load_bank()
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    logged = {(e["label"], e["prompt_id"]) for e in read_jsonl(collection.QODER_LOG_PATH)}
    tries = defaultdict(int)
    for f in read_jsonl(FAILURES_PATH):
        tries[f["label"], f["prompt_id"]] += 1
    jobs = []
    for m in args.models:
        pending = [p for p in prompts if (m, p["prompt_id"]) not in logged]
        random.Random(f"{args.seed}|{m}").shuffle(pending)
        jobs += [(m, p) for p in (pending[: args.limit] if args.limit else pending)]

    cli = newest_cli()
    print(f"Qoder CLI CN: {cli.name if cli else 'NOT FOUND'}")
    print(f"{len(prompts)} prompts ({', '.join(args.waves)}); {len(logged)} answers logged; "
          f"{len(jobs)} runs with {args.workers} workers, about {len(jobs) * 0.13:.0f} credits")
    if not args.execute:
        print("dry run: nothing run. Add --execute (with --limit 1 for a probe).")
        return
    if cli is None:
        raise SystemExit(f"no qoderclicn build under {CLI_DIR}")
    version = subprocess.run([str(cli), "--version"], capture_output=True, check=False).stdout.decode().strip()

    collection.EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    lock, stop = threading.Lock(), threading.Event()
    state = {"ok": 0, "failed": 0, "credits": 0.0}
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)

    def worker() -> None:
        while not stop.is_set():
            try:
                m, p = pending.get_nowait()
            except queue.Empty:
                return
            for attempt_try in range(MAX_TRIES):
                if stop.is_set():
                    return
                requested = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                try:
                    payload, error = run_qoder(cli, MODELS[m], p["prompt_text"], args.timeout)
                except subprocess.TimeoutExpired:
                    payload, error = None, f"timeout after {args.timeout:.0f}s"
                if not error:
                    entry = {"source": "qoder", "request_uid": uuid.uuid4().hex, "label": m, "prompt_id": p["prompt_id"],
                             "model_id": MODELS[m], "prompt_text": p["prompt_text"], "endpoint": "qoderclicn -p",
                             "client_version": f"qoderclicn {version}", "requested_at_utc": requested,
                             "attempt": tries[m, p["prompt_id"]] + 1, "usd": None, "response": payload}
                    parsed = collection.parse_qoder_result(payload)
                    with lock:
                        with open(collection.QODER_LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        state["ok"] += 1
                        state["credits"] += float(payload.get("total_credits") or 0)
                    tools = ", tool use" if parsed["tool_use"] else ""
                    print(f"  {m} {p['prompt_id']}: {len(parsed['text'].split())} words, reported {parsed['reported']}, "
                          f"{parsed['finish']}{tools}, {state['credits']:.2f} credits [{state['ok']}/{len(jobs)}]",
                          flush=True)
                    break
                with lock:
                    tries[m, p["prompt_id"]] += 1
                    state["failed"] += 1
                    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"label": m, "prompt_id": p["prompt_id"], "error": error[:2000],
                                            "at_utc": requested}, ensure_ascii=False) + "\n")
                print(f"  FAIL {m} {p['prompt_id']}: {error[:300]}", flush=True)
                if QUOTA.search(error):
                    stop.set()
                    print("  stopping: Qoder credits or sign-in look exhausted; rerun later to resume", flush=True)
                    return
                time.sleep((120 if RATE_LIMIT.search(error) else 15) * 2**attempt_try * random.uniform(0.5, 1.5))

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n{state['ok']} answers logged ({state['credits']:.2f} Qoder credits), {state['failed']} failed runs."
          f"\nNext: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
