"""Collect Claude answers through the Message Batches API (official Anthropic SDK).

Each bank prompt goes once to each Claude label as a single-turn request with API defaults: no
system prompt, no tools, and no thinking or effort settings, so each model keeps its own default
(Fable 5, Opus 5 and Sonnet 5 think adaptively; Opus 4.8 and 4.6 do not), with max_tokens 16000.
No refusal fallback is configured: a fallback would answer with a different model under the
requested label, so a refusal stays a refusal and is quarantined. The response's `model` field
verifies the served model. Batches cost half the standard price and usually finish within an hour.

Submitted batches are recorded in data/external/anthropic_batches.json. A rerun first collects
finished batches, then submits only prompts that are neither logged nor in a pending batch.
Results are appended to data/external/api_log_anthropic.jsonl (never edited); failed requests to
api_failures_anthropic.jsonl, and they are resubmitted on the next run.

Credentials come from the SDK's own resolution (ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN or an
`ant auth login` profile), or else the first line of ~/.config/nllm/anthropic_api_key; nothing is
printed or stored. A key not scoped to a workspace also needs the workspace id: --workspace,
$ANTHROPIC_WORKSPACE_ID or ~/.config/nllm/anthropic_workspace_id.

Usage:
  uv run python scripts/collect_anthropic.py                         # dry run: plan and cost
  uv run python scripts/collect_anthropic.py --execute --limit 1     # probe batch: one prompt per model
  uv run python scripts/collect_anthropic.py --execute               # everything left
  uv run python scripts/collect_anthropic.py --collect               # wait for and collect pending batches only
"""

import argparse
import json
import os
import random
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import anthropic
import pandas as pd
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from nllm import collection
from nllm.data import LABELS
from nllm.paths import KIT_DIR

MODELS = [m for m in LABELS if m.startswith("claude-")]
MAX_TOKENS = 16000  # non-streaming ceiling; thinking counts against it
BATCH_DISCOUNT = 0.5
STATE_PATH = collection.EXTERNAL_DIR / "anthropic_batches.json"
FAILURES_PATH = collection.EXTERNAL_DIR / "api_failures_anthropic.jsonl"
THINKS_BY_DEFAULT = {"claude-fable-5", "claude-opus-5", "claude-sonnet-5"}
KEY_FILE = Path.home() / ".config" / "nllm" / "anthropic_api_key"
WORKSPACE_FILE = Path.home() / ".config" / "nllm" / "anthropic_workspace_id"


def rates() -> dict[str, tuple[float, float]]:
    table = pd.read_csv(KIT_DIR / "model_access_and_prices.csv", encoding="utf-8").set_index("target_label")
    return {m: (float(table.loc[m, "usd_per_million_input"]), float(table.loc[m, "usd_per_million_output"]))
            for m in MODELS}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def append_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def custom_id(label: str, prompt_id: str) -> str:
    return f"{label}__{prompt_id}"  # batch ids allow only letters, digits, '_' and '-'


def sdk_credentials() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
                or (Path.home() / ".config" / "anthropic").exists())


def workspace_id(given: str | None) -> str | None:
    """--workspace, else $ANTHROPIC_WORKSPACE_ID, else the workspace file; needed by keys not scoped to a workspace."""
    if given:
        return given.strip()
    if os.environ.get("ANTHROPIC_WORKSPACE_ID"):
        return os.environ["ANTHROPIC_WORKSPACE_ID"].strip()
    if WORKSPACE_FILE.exists():
        return (WORKSPACE_FILE.read_text(encoding="utf-8-sig").strip().splitlines() or [""])[0].strip() or None
    return None


def make_client(workspace: str | None = None) -> anthropic.Anthropic:
    """The SDK's own credential resolution, or the key file when none applies (shells started before a new
    environment variable cannot see it). An organization key also needs the workspace to bill."""
    headers = {"anthropic-workspace-id": workspace} if workspace else None
    if not sdk_credentials() and KEY_FILE.exists():
        key = (KEY_FILE.read_text(encoding="utf-8-sig").strip().splitlines() or [""])[0].strip()
        return anthropic.Anthropic(api_key=key, default_headers=headers)
    return anthropic.Anthropic(default_headers=headers)


def collect(client, batch: dict, bank: dict, price: dict, tries: dict) -> tuple[int, int, float]:
    """Log every result of an ended batch; returns (succeeded, failed, usd)."""
    ok, failed, usd_total, entries, failures = 0, 0, 0.0, [], []
    for result in client.messages.batches.results(batch["id"]):
        label, pid = result.custom_id.split("__", 1)
        if result.result.type == "succeeded":
            message = result.result.message
            rin, rout = price[label]
            usd = (message.usage.input_tokens * rin + message.usage.output_tokens * rout) / 1e6 * BATCH_DISCOUNT
            entries.append({
                "source": "anthropic", "request_uid": f"{batch['id']}:{result.custom_id}", "label": label,
                "prompt_id": pid, "model_id": label, "prompt_text": bank[pid]["prompt_text"],
                "endpoint": "POST /v1/messages/batches", "batch_id": batch["id"],
                "requested_at_utc": batch["created_utc"], "attempt": tries[label, pid] + 1,
                "max_output_tokens": MAX_TOKENS, "usd": usd, "client_version": f"anthropic-python {anthropic.__version__}",
                "response": message.to_dict(),
            })
            ok, usd_total = ok + 1, usd_total + usd
        else:
            error = getattr(result.result, "error", None)
            failures.append({"label": label, "prompt_id": pid, "batch_id": batch["id"], "type": result.result.type,
                             "error": str(error)[:1000] if error else None,
                             "at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")})
            failed += 1
    append_jsonl(collection.ANTHROPIC_LOG_PATH, entries)
    append_jsonl(FAILURES_PATH, failures)
    for f in failures[:5]:
        print(f"  FAIL {f['label']} {f['prompt_id']}: {f['type']} {f['error'] or ''}"[:300])
    return ok, failed, usd_total


def wait_and_collect(client, state: dict, bank: dict, price: dict, tries: dict, poll: float) -> None:
    for batch in state["batches"]:
        if batch.get("collected"):
            continue
        while True:
            info = client.messages.batches.retrieve(batch["id"])
            counts = info.request_counts
            print(f"batch {batch['id']}: {info.processing_status}, {counts.processing} processing, "
                  f"{counts.succeeded} succeeded, {counts.errored} errored", flush=True)
            if info.processing_status == "ended":
                break
            time.sleep(poll * random.uniform(0.9, 1.1))
        ok, failed, usd = collect(client, batch, bank, price, tries)
        batch.update(collected=True, succeeded=ok, failed=failed, usd=usd)
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(f"collected {batch['id']}: {ok} answers, {failed} failed, ${usd:.2f}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--waves", nargs="+", default=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"],
                    choices=["wave1", "wave2", "wave3", "wave4", "wave5", "diagnostic"])
    ap.add_argument("--execute", action="store_true", help="submit a paid batch; otherwise a dry run")
    ap.add_argument("--collect", action="store_true", help="only wait for and collect pending batches")
    ap.add_argument("--limit", type=int, help="at most this many new prompts per model (probe with 1)")
    ap.add_argument("--max-usd", type=float, default=40.0, help="refuse a batch whose upper estimate exceeds this")
    ap.add_argument("--poll-seconds", type=float, default=60.0)
    ap.add_argument("--no-wait", action="store_true", help="submit and exit; collect later with --collect")
    ap.add_argument("--workspace", help="Claude Console workspace id (wrkspc_...), for keys not scoped to one")
    args = ap.parse_args()

    bank = collection.load_bank()
    price = rates()
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.exists() else {"batches": []}
    logged = {(e["label"], e["prompt_id"]) for e in read_jsonl(collection.ANTHROPIC_LOG_PATH)}
    pending = {tuple(pair) for b in state["batches"] if not b.get("collected") for pair in b["pairs"]}
    tries = defaultdict(int)
    for f in read_jsonl(FAILURES_PATH):
        tries[f["label"], f["prompt_id"]] += 1
    prompts = sorted((p for p in bank.values() if p["wave"] in args.waves), key=lambda p: p["prompt_id"])
    todo = []
    for m in args.models:
        pairs = [(m, p["prompt_id"]) for p in prompts if (m, p["prompt_id"]) not in logged | pending]
        todo += pairs[: args.limit] if args.limit else pairs

    found = "SDK" if sdk_credentials() else "key file" if KEY_FILE.exists() else "MISSING"
    workspace = workspace_id(args.workspace)
    print(f"Claude Message Batches; credentials: {found}; workspace: {workspace or 'none (key must be workspace-scoped)'}")
    print(f"{len(prompts)} prompts ({', '.join(args.waves)}); {len(logged)} answers logged, {len(pending)} in "
          f"pending batches; spent ${sum(b.get('usd', 0) for b in state['batches']):.2f}")
    est_lo = est_hi = 0.0
    for m in args.models:
        n = sum(1 for mm, _ in todo if mm == m)
        rin, rout = price[m]
        thinking = 1500 if m in THINKS_BY_DEFAULT else 0
        lo = n * (80 * rin + 250 * rout) / 1e6 * BATCH_DISCOUNT
        hi = n * (80 * rin + (300 + thinking + 1000) * rout) / 1e6 * BATCH_DISCOUNT
        est_lo, est_hi = est_lo + lo, est_hi + hi
        default = "thinks by default" if thinking else "no thinking by default"
        print(f"  {m:18} {n:4} to submit   est. ${lo:.2f} - ${hi:.2f} at batch price ({default})")
    print(f"estimated total ${est_lo:.2f} - ${est_hi:.2f}; cap ${args.max_usd:.2f}")
    if not (args.execute or args.collect):
        print("dry run: nothing sent. Add --execute (with --limit 1 for a probe batch).")
        return
    if found == "MISSING":
        raise SystemExit(f"no Anthropic credentials: set ANTHROPIC_API_KEY, run `ant auth login`, or save the key "
                         f"as the only line of {KEY_FILE}")
    client = make_client(workspace_id(args.workspace))

    if args.collect or not args.no_wait:
        wait_and_collect(client, state, bank, price, tries, args.poll_seconds)
    if args.collect or not todo:
        return
    if est_hi > args.max_usd:
        raise SystemExit(f"upper estimate ${est_hi:.2f} exceeds --max-usd {args.max_usd:.2f}; use --limit or raise it")
    requests = [
        Request(custom_id=custom_id(m, pid), params=MessageCreateParamsNonStreaming(
            model=m, max_tokens=MAX_TOKENS, messages=[{"role": "user", "content": bank[pid]["prompt_text"]}]))
        for m, pid in todo
    ]
    batch = client.messages.batches.create(requests=requests)
    state["batches"].append({"id": batch.id, "created_utc": batch.created_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "pairs": [list(t) for t in todo], "collected": False})
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(f"submitted batch {batch.id} with {len(requests)} requests", flush=True)
    if not args.no_wait:
        wait_and_collect(client, state, bank, price, tries, args.poll_seconds)
    print("Next: uv run python scripts/import_answers.py")


if __name__ == "__main__":
    main()
