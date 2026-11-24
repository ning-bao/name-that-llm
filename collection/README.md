# External answers: manual collection guide

The goal is more training passages written by the same 18 models, collected through your own
chat subscriptions. More independent prompts are what helps: the learning curve in
`EXPERIMENTS.md` gains about 3.6 balanced-accuracy points each time the number of training
prompts doubles. These files follow section E of the research report.

All 18 models are collected. `gemini-3.7-flash` answers are **held**: Google's API routes 3.7
requests to 3.8, so an app showing 3.7 may be served by 3.8. They stay out of training until
the served model is verified, or until a check shows they resemble official 3.7 text rather than 3.8.

Every prompt ends with the same line, "Answer from your own knowledge, without searching the
web or using any tools.", because some apps (Gemini in agent studios) cannot switch search off.
It is on every model's prompts so it is not a per-app difference. Keep it, and still turn
search off wherever the app allows. Answers showing citation markers ([1], "model12.", ¹²),
links or a sources footer are quarantined automatically; don't remove the markers by hand.
Answers pasted under an older prompt text keep that text in their file and in their record.

## What is here

| Path | Contents |
| --- | --- |
| `prompts_v2.tsv` | Source for the 160 new prompts (wave2 to wave5), one row each with domain, genre and length band |
| `prompt_bank.json` | All 196 prompts with id, family, split and wave (built from the kit bank plus `prompts_v2.tsv`) |
| `prompts/<wave>.md` | Each wave's prompts, ready to copy |
| `data/external/answers/<wave>/<model>.txt` | Answer files to paste into (local only, never committed) |

| Pack | Prompts | Answers for 18 models |
| --- | --- | --- |
| wave1 | the kit’s 24 core + 6 holdout | 540 |
| wave2 to wave5 | 40 new each (32 train + 8 holdout) | 720 each |
| diagnostic | the kit's 6 diagnostic prompts (optional, kept separate) | 108 |

Work through the waves in order and finish each wave for every model before starting the
next. A finished wave is a balanced set you can stop after. The holdout prompts (about 20%)
measure transfer and are never trained on. Their ids carry no marker, so treat every prompt the same.

The new prompts match the official corpus: polished science and humanities prose, mostly
single paragraphs. Lengths follow the official mix (about 38% 40-70 words, 33% 90-130, 19%
160-210, 10% 220-250). No prompt asks a model to name itself or exaggerate its style, and
none contains competition text.

## Collecting

For each model, open `data/external/answers/<wave>/<model>.txt` and, for every prompt in it:

1. Start a **new chat** with that model. Turn off memory and custom instructions if the app
   lets you, and keep web search, tools and research modes off.
2. Paste the prompt exactly as written.
3. Copy the **first complete answer** and paste it between that prompt's `<<<ANSWER …>>>` and
   `<<<END …>>>` lines. Copy only the answer text: no buttons, model names, "Thought for 12s"
   lines or visible reasoning.
4. Don't edit the answer, and don't regenerate it to get a better one. If the answer is short,
   long, odd or a refusal, keep it as it is.

At the top of each file, set `model_label_checked_in_app: yes` once you have confirmed that
the app's model picker shows exactly this model. Until then, that file's answers are
quarantined. The other header fields are optional.

Pasting matters, because quote marks and dashes are among the strongest signals. Paste into
a plain-text editor such as VS Code or Notepad, never Word or a rich-text app, which can turn
straight quotes into curly ones. If you made a mistake, such as an answer pasted for the
wrong model or app clutter included, delete it and paste the correct one. The importer keeps
the original in its log, marks it withdrawn and uses the new one.

## Importing and checking

```bash
uv run python scripts/import_answers.py
```

Run it whenever you like: it only adds answers it has not seen. It writes the kit-schema
records to `data/external/records.jsonl` and the admitted rows to
`data/external/admitted.parquet`. It also prints:

- answers and admitted rows per wave and model;
- why answers were quarantined: app clutter, an echoed prompt, a truncated or empty answer,
  a refusal, the same text pasted twice, an unconfirmed model label, or a later answer to a
  prompt that already has one;
- flags on admitted answers worth a look: Markdown, several paragraphs, unusual whitespace,
  length outside the request;
- each model's typography against its official passages. A warning such as "curly
  apostrophe in 90% of external answers vs 0% of official" usually means the copy path
  converted characters. Fix that before collecting more.

## Gemini through the Agent Platform API

Google Cloud's Agent Platform (formerly Vertex AI) still serves `gemini-3.7-flash` until
28 January 2027. `gemini-3.6-flash` retires there on **19 November 2026**. Every response names
the model that actually answered, so these answers verify themselves: no picker check or
matched-pair test is needed. A response served by another model is rejected.

1. In the Cloud console, enable the Agent Platform API on your project and create an API key
   restricted to it, or use an express-mode key.
2. Save the key as the only line of `%USERPROFILE%\.config\nllm\agent_platform_api_key`
   (outside the repo). Alternatively, set `AGENT_PLATFORM_API_KEY`, or sign in to the Google Cloud
   SDK and pass `--auth gcloud`. The script never prints or stores the key.
3. Run:

```bash
uv run python scripts/collect_agent_platform.py --project <project-id>                       # dry run, cost estimate
uv run python scripts/collect_agent_platform.py --project <project-id> --execute --limit 1   # one prompt per model
uv run python scripts/collect_agent_platform.py --project <project-id> --execute             # the rest (cap: $30)
uv run python scripts/import_answers.py
```

Leave out `--project` for an express-mode key. Each prompt goes once to each Gemini model with
API defaults: no system instruction, no tools, only an output-token cap. Responses are appended
to `data/external/api_log.jsonl`, and a rerun skips prompts already answered. API answers are
admitted as their own collection method, so they sit alongside any pasted Gemini answers.
`external_data_eval.py --methods api` tests them on their own.

## Claude through the Message Batches API

`scripts/collect_anthropic.py` sends every prompt to the five Claude labels as one batch, at half
the standard price; most batches finish within an hour. Each model keeps its own defaults: Fable 5,
Opus 5 and Sonnet 5 think, while Opus 4.8 and 4.6 don't. No refusal fallback is set, because it
would answer with a different model under the requested label. The response's `model` field
verifies each answer.

Credentials: `ANTHROPIC_API_KEY`, an `ant auth login` profile, or the key as the only line of
`%USERPROFILE%\.config\nllm\anthropic_api_key` (outside the repo). This needs an API key from the
Claude Console with API credits; a Claude chat subscription doesn't cover API calls.

```bash
uv run python scripts/collect_anthropic.py                       # dry run, cost estimate
uv run python scripts/collect_anthropic.py --execute --limit 1   # probe batch, one prompt per model
uv run python scripts/collect_anthropic.py --execute             # the rest (cap: $40)
uv run python scripts/collect_anthropic.py --collect             # pick up a batch submitted earlier
```

## GPT through the Codex CLI

`scripts/collect_codex.py` runs every prompt once per GPT model with `codex exec`, on your
ChatGPT/Codex plan. It covers gpt-5.6-luna, gpt-5.6-sol, gpt-5.6-terra and gpt-6-astra.
**gpt-5.5 isn't in Codex** (retired from ChatGPT and Codex on 14 Oct 2026), so it still needs
pasting or the OpenAI API.

Each run gets a fresh empty folder, a read-only sandbox, no saved session, and none of your Codex
config, rules or memories. Login still comes from `~/.codex`. The Windows CLI bundled with the
desktop app is too old for these models, so the script runs the app's newer Linux build through
WSL. Codex adds its own agent instructions (about 13k–17k tokens) and doesn't report the served
model, so these answers form their own condition. An answer during which the agent ran a tool is
quarantined. Runs count against your plan's Codex usage limits; a limit error stops the run, and
rerunning resumes. `--fast` uses Codex fast mode (the `priority` service tier: the same model,
1.5–2× faster, at higher plan usage), and each answer records its tier.

```bash
uv run python scripts/collect_codex.py --execute --limit 1    # probe, one prompt per model
uv run python scripts/collect_codex.py --execute --workers 3 --fast  # the rest
```

## GLM and Kimi through WorkBuddy (W2A)

`scripts/collect_workbuddy.py` sends every prompt once to glm-5.3, glm-5.3-flash and kimi-k3
through your workbuddy2api gateway, an OpenAI-compatible proxy for your WorkBuddy plan. The GLM answers
came from your CN account. kimi-k3 comes from your global account (`--account global`, the default),
since the CN account offers only `kimi-k3-1`, whose answers are rejected for the kimi-k3 label. Each prompt goes as one user message, with no system prompt or tools. Temperature is
the gateway's default (you set it to 1), and reasoning effort is WorkBuddy's default unless you pass
`--effort` (GLM: low/high/max, gpt-5.5: low/medium/high/xhigh; kimi-k3 is fixed at medium). At the default effort GLM and Kimi think only
a line before answering, and glm-5.3-flash not at all. Kimi requests carry about 80 tokens of hidden input that
GLM and GPT requests do not, and an explicit system message does not remove it (an empty one is refused; a
single space replaces only 5 tokens and leaves the typography shift in place). The response's model field may only echo the
request, so these answers count as verified by the selected model; a different reported model rejects
the answer. Each answer uses WorkBuddy credits: on average 0.11 for glm-5.3, 0.01 for glm-5.3-flash
and 0.71 for the CN account's kimi-k3-1. That first run of 588 answers used 163; kimi-k3 on the
global account costs about 0.38 and gpt-5.5 about 2.5 at its default (high) effort.

Put the gateway URL in `~/.config/nllm/workbuddy_base_url` (for example `http://<host>:8788/v1`)
and the key from the panel's settings page in `~/.config/nllm/workbuddy_api_key`, or set
`WORKBUDDY_BASE_URL` / `WORKBUDDY_API_KEY`.

```bash
uv run python scripts/collect_workbuddy.py --execute --limit 1    # probe, one prompt per model
uv run python scripts/collect_workbuddy.py --execute --workers 2  # the rest
```

## Qwen through Qoder CLI CN

`scripts/collect_qoder.py` runs every prompt once on qwen3.8-max (`Qwen3.8-Max`) with Qoder CLI CN
(`~/.qoder-cn/bin/qoderclicn/qoderclicn-<version>.exe`), on your Qoder CN plan. Sign in once with
`qoderclicn login`, which opens a browser page to approve. Each run is `qoderclicn -p -o json` in a
fresh empty folder. The prompt goes on stdin, and the run disables every tool and MCP server, skips your
user and project settings and skills, and saves no session. Qoder's own system prompt is replaced by an
empty one. Billed credits suggest a few hundred tokens of context you can't see remain, far less than
Codex adds. Qoder reports credits rather than tokens and names the model only by an internal key
(`qmodel_38max`), so these answers count as verified by the selected model; another key rejects the
answer. Each answer costs about 0.13–0.15 credits.

```bash
uv run python scripts/collect_qoder.py --execute --limit 1    # probe
uv run python scripts/collect_qoder.py --execute --workers 3  # the rest
```

Qoder's Qwen3.8-Max writes differently from the competition's qwen3.8-max: em dashes in about 64% of
answers against 3%, and no curly apostrophes against 28%.

## Kimi through Moonshot's API

`scripts/collect_moonshot.py` sends every prompt once to `kimi-k3` on Moonshot's own API
(`https://api.moonshot.ai/v1`), as one user message with no system prompt or tools, a 16,000-token
output cap and thinking turned off (`"thinking": {"type": "disabled"}`). Temperature is the API's
default. Thinking off is what matches the competition data: 20 test answers had em dashes in 50% and
curly apostrophes in 30% (official kimi-k3: 49% and 34%), while thinking-on answers, the API's default
and WorkBuddy's fixed setting, use em dashes in about 90% and almost no curly apostrophes.
`--thinking default` keeps thinking on; those answers form their own condition
(`api:moonshot_thinking_chat_completions`). The response's model field verifies the label. Your account
allows one request at a time, so the collector runs one worker. It prices answers at $3 / $15 per
million input / output tokens (third-party list price; about $0.002 per answer with thinking off, $0.02 with it on) and stops at `--max-usd`
(default $10). Put the key in `~/.config/nllm/moonshot_api_key` or `MOONSHOT_API_KEY`. These answers are
their own condition beside the WorkBuddy Kimi answers.

```bash
uv run python scripts/collect_moonshot.py --execute --limit 1    # probe
uv run python scripts/collect_moonshot.py --execute              # the rest
```

## Verifying gemini-3.7-flash

Held `gemini-3.7-flash` answers can be released into training in two ways. Until one of them
succeeds, keep collecting 3.7 alongside 3.8; released answers need no re-collection.

1. **The app reports the served model.** If the app shows which model actually answered,
   for example a `modelVersion` field in a raw-response or model-details panel, write it in
   the file's `served_model_version:` header line. Do this only if every answer in that file
   came back with the same version. `gemini-3.7-flash` or a pinned version such as
   `gemini-3.7-flash-001` verifies the file's answers. Any 3.8 version rejects them.
2. **A matched-pair check.** Every prompt goes to both 3.7 and 3.8, so the collected answers
   form pairs. The check scores each answer with the reference classifier, trained on official
   passages, and compares the 3.7-minus-3.8 gap on your pairs with the same gap on held-out
   official prompts. A real 3.7 shows a similar gap; 3.8 under a 3.7 label shows none.

   ```bash
   uv run python scripts/import_answers.py
   uv run python scripts/verify_held.py             # report only
   uv run python scripts/verify_held.py --release   # also release on a passing verdict
   ```

   Official 3.7 and 3.8 text is hard to tell apart (the real 3.7 answer wins in only about 60%
   of official pairs), so the check needs many pairs. It refuses to decide before 70 prompts
   answered by both models, and confirming genuine 3.7 typically takes about 120, roughly
   three waves of both. The verdict is "consistent" (release), "looks like 3.8" (keep held;
   the app is likely redirected), or "inconclusive" (keep held and collect more). The rule
   was fixed before any real answers were scored, and its p < 0.01 threshold allows rerunning
   the check after every wave. A release lists exactly the answers scored, and only those
   from files marked `model_label_checked_in_app: yes`. Rerun it with `--release` to cover
   answers collected later.

## Testing whether it helps

```bash
uv run python scripts/external_data_eval.py              # all admitted answers
uv run python scripts/external_data_eval.py --waves wave1
uv run python scripts/external_data_eval.py --exclude "gemini-*@*" "kimi-k3@*workbuddy*" "qwen3.8-max@*"
```

Each interface (chat app, Agent Platform, Message Batches, Codex, WorkBuddy, Qoder, Moonshot) is a
separate condition, so one prompt can have an admitted answer from each. `--exclude` drops
answers whose `model@interface` matches a wildcard pattern, for example the sources whose typography
differs from the official data.

This compares official-only, external-only and combined training on the same official
held-out prompts. It reports paired confidence intervals and Gemini sibling recalls, and
checks transfer on your holdout prompts. Results go in `EXPERIMENTS.md`. External data joins
the main models only if the combined condition beats official-only in both partitions.

The answers' competition eligibility is recorded as `pending_clarification` until the
organizers confirm the plan (report I1). Once they do, run the importer with
`--eligibility organizer_confirmed`. `--eligible-only` restricts the comparison to
confirmed answers.
