"""Collect GPT answers through the Codex CLI, on the user's ChatGPT/Codex plan.

Codex offers gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra and gpt-6-astra (gpt-5.5 is no longer
in Codex). Each bank prompt runs once per model as a non-interactive `codex exec` in a fresh
empty directory with a read-only sandbox, --ephemeral (no session files), --ignore-user-config
(the user's personality, memories and model settings are not applied; login still comes from
~/.codex) and --ignore-rules. The answer is the agent's last message, and any tool use during
the run quarantines it. Codex adds its own agent instructions (~12.7k tokens) and does not report
the served model, so these answers are their own condition, verified only by the selected model.

The Windows codex.exe bundled with the desktop app (0.130 alpha) is too old for these models, so
the newer Linux build the app installs (~/.codex/packages/app-server-daemon/releases/<v>/bin/codex)
runs through WSL, with CODEX_HOME pointing at the Windows ~/.codex.

--fast selects Codex fast mode (the "priority" service tier the model catalogue advertises: the same
model served 1.5x faster, 2x for astra, at higher plan usage). Each entry records its service tier.

Runs count against the plan's Codex usage limits, not dollars. An exhausted usage limit stops the
run; a 429 rate limit only pauses that model with growing backoff while the others continue.
Rerunning resumes (prompts with a logged answer are skipped). Answers are appended to
data/external/api_log_codex.jsonl (never edited), failures to api_failures_codex.jsonl.

Usage:
  uv run python scripts/collect_codex.py                         # dry run: plan
  uv run python scripts/collect_codex.py --execute --limit 1     # probe: one prompt per model
  uv run python scripts/collect_codex.py --execute --workers 2
  uv run python scripts/collect_codex.py --execute --workers 3 --fast
"""

import argparse
import json
import queue
import random
import re
import subprocess
import threading
import time
import uuid
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from nllm import collection

MODELS = ["gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra", "gpt-6-astra"]
NOT_IN_CODEX = {"gpt-5.5": "retired from ChatGPT and Codex on 14 Oct 2026; collected through WorkBuddy instead"}
FAILURES_PATH = collection.EXTERNAL_DIR / "api_failures_codex.jsonl"
CODEX_HOME = Path.home() / ".codex"
RELEASES = CODEX_HOME / "packages" / "app-server-daemon" / "releases"
# An exhausted plan limit is account-wide and stops everything; a 429 is per model and only pauses that
# model (backing off 2, 4, 8 ... 30 min; after MAX_RATE_STRIKES in a row the model is left for a later run).
USAGE_LIMIT = re.compile(r"usage limit|quota", re.IGNORECASE)
RATE_LIMIT = re.compile(r"too many requests|rate limit|\b429\b", re.IGNORECASE)
MAX_RATE_STRIKES = 6
FAST_TIER = "priority"
# Runs one exec in a fresh empty directory inside WSL: $1 codex binary, $2 CODEX_HOME, $3 model, then any extra
# exec options; prompt on stdin.
RUN = ('d=$(mktemp -d) && CODEX_HOME="$2" "$1" exec --ignore-user-config --ignore-rules --ephemeral '
       '--skip-git-repo-check -s read-only -C "$d" -m "$3" "${@:4}" --json -; rc=$?; rm -rf "$d"; exit $rc')


def wsl_path(path: Path) -> str:
    drive, rest = path.drive, path.as_posix()[len(path.drive):]
    return f"/mnt/{drive[0].lower()}{rest}"


def newest_codex() -> Path | None:
    """The newest Linux codex build the desktop app installed (release dirs are named <version>-<target>)."""
    builds = [p for p in RELEASES.glob("*/bin/codex") if p.is_file()]
    key = lambda p: tuple(int(x) for x in re.findall(r"\d+", p.parent.parent.name.split("-")[0]))
    return max(builds, key=key) if builds else None


def run_codex(codex: str, model: str, prompt: str, timeout: float, tier: str | None) -> tuple[list[dict], str]:
    extra = ["-c", f'service_tier="{tier}"'] if tier else []
    result = subprocess.run(["wsl", "-e", "bash", "-c", RUN, "codex-run", codex, wsl_path(CODEX_HOME), model, *extra],
                            input=prompt.encode("utf-8"), capture_output=True, timeout=timeout, check=False)
    events = []
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.strip().startswith("{"):
            events.append(json.loads(line))
    # Codex reports transient "Reconnecting..." errors and retries by itself; only a turn that did not
    # complete is a failure.
    if result.returncode == 0 and any(e.get("type") == "turn.completed" for e in events):
        return events, ""
    errors = [json.dumps(e) for e in events if e.get("type") in ("error", "turn.failed")]
    if not errors:
        errors.append(f"exit {result.returncode}: {result.stderr.decode('utf-8', 'replace')[-500:]}")
    return events, "; ".join(errors)


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--waves", nargs="+", default=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"],
                    choices=sorted({p["wave"] for p in collection.load_bank().values()}),
                    help="pool1 etc. are screened Open Pool prompts (scripts/select_pool_prompts.py)")
    ap.add_argument("--execute", action="store_true", help="run Codex; otherwise a dry run")
    ap.add_argument("--limit", type=int, help="at most this many new prompts per model (probe with 1)")
    ap.add_argument("--workers", type=int, default=2, help="parallel codex runs across all models")
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--fast", action="store_true", help=f"Codex fast mode (service tier {FAST_TIER!r})")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    bank = collection.load_bank()
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    logged = {(e["label"], e["prompt_id"]) for e in read_jsonl(collection.CODEX_LOG_PATH)}
    tries = defaultdict(int)
    for f in read_jsonl(FAILURES_PATH):
        tries[f["label"], f["prompt_id"]] += 1
    per_model = {}
    for m in args.models:
        pending = [p for p in prompts if (m, p["prompt_id"]) not in logged]
        random.Random(f"{args.seed}|{m}").shuffle(pending)
        per_model[m] = pending[: args.limit] if args.limit else pending
    jobs = [(m, per_model[m][i]) for i in range(max(map(len, per_model.values()), default=0))
            for m in args.models if i < len(per_model[m])]  # interleave models

    codex = newest_codex()
    print(f"Codex CLI via WSL: {codex.parent.parent.name if codex else 'NOT FOUND'}; not in Codex: "
          + "; ".join(f"{m} ({why})" for m, why in NOT_IN_CODEX.items()))
    print(f"{len(prompts)} prompts ({', '.join(args.waves)}); {len(logged)} answers logged")
    for m in args.models:
        print(f"  {m:14} {len(per_model[m]):4} to run")
    tier = FAST_TIER if args.fast else None
    print(f"{len(jobs)} runs with {args.workers} workers{', fast mode' if tier else ''}: roughly {len(jobs) * 30 / args.workers / 60:.0f} min at "
          "~30 s per run, against the plan's Codex usage limits")
    if not args.execute:
        print("dry run: nothing run. Add --execute (with --limit 1 for a probe).")
        return
    if codex is None:
        raise SystemExit(f"no Linux codex build under {RELEASES}; open the Codex desktop app once to install it")
    version = subprocess.run(["wsl", "-e", wsl_path(codex), "--version"], capture_output=True, check=False)
    client_version = version.stdout.decode().strip() or codex.parent.parent.name

    lock, stop = threading.Lock(), threading.Event()
    state = {"ok": 0, "failed": 0}
    pending = queue.Queue()
    for job in jobs:
        pending.put(job)
    cooldown, strikes, halted = defaultdict(float), defaultdict(int), set()

    def worker() -> None:
        while not stop.is_set():
            try:
                m, p = pending.get_nowait()
            except queue.Empty:
                return
            if m in halted:
                continue
            wait = cooldown[m] - time.time()
            if wait > 0:  # this model is rate-limited: requeue and let other models run
                pending.put((m, p))
                time.sleep(min(wait, 15))
                continue
            for attempt_try in range(3):
                if stop.is_set():
                    return
                requested = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
                try:
                    events, error = run_codex(wsl_path(codex), m, p["prompt_text"], args.timeout, tier)
                except subprocess.TimeoutExpired:
                    events, error = [], f"timeout after {args.timeout:.0f}s"
                if not error:
                    entry = {"source": "codex", "request_uid": uuid.uuid4().hex, "label": m, "prompt_id": p["prompt_id"],
                             "model_id": m, "prompt_text": p["prompt_text"], "endpoint": "codex exec (WSL)",
                             "client_version": client_version, "service_tier": tier, "requested_at_utc": requested,
                             "attempt": tries[m, p["prompt_id"]] + 1, "usd": None, "response": {"events": events}}
                    parsed = collection.parse_codex_events(entry["response"])
                    with lock:
                        with open(collection.CODEX_LOG_PATH, "a", encoding="utf-8") as f:
                            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        state["ok"] += 1
                        strikes[m] = 0
                    tools = ", tool use" if parsed["tool_use"] else ""
                    print(f"  {m} {p['prompt_id']}: {len(parsed['text'].split())} words, "
                          f"{parsed['input_tokens']}+{parsed['output_tokens']} tokens{tools} [{state['ok']}/{len(jobs)}]",
                          flush=True)
                    break
                with lock:
                    tries[m, p["prompt_id"]] += 1
                    state["failed"] += 1
                    with open(FAILURES_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"label": m, "prompt_id": p["prompt_id"], "error": error[:2000],
                                            "at_utc": requested}) + "\n")
                print(f"  FAIL {m} {p['prompt_id']}: {error[:300]}", flush=True)
                if USAGE_LIMIT.search(error):
                    stop.set()
                    print("  stopping: the plan's Codex usage limit is reached; rerun later to resume", flush=True)
                    return
                if RATE_LIMIT.search(error):
                    with lock:
                        strikes[m] += 1
                        if strikes[m] >= MAX_RATE_STRIKES:
                            halted.add(m)
                            print(f"  leaving {m} for a later run: rate-limited {strikes[m]} times in a row", flush=True)
                            break
                        pause = min(1800, 120 * 2 ** (strikes[m] - 1))
                        cooldown[m] = time.time() + pause
                    print(f"  pausing {m} for {pause // 60:.0f} min (rate limit); other models continue", flush=True)
                    pending.put((m, p))
                    break
                time.sleep(20 * 2**attempt_try)

    threads = [threading.Thread(target=worker) for _ in range(args.workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\n{state['ok']} answers logged, {state['failed']} failed runs."
          f"\nNext: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
