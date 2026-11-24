# Data study plan: matched training data for the weak classes (2026-11-08)

Goal: find a generation setup whose answers look like the official passages, then generate enough of them
(crossed 18-model panels) to move our weakest classes. Nothing has been spent or collected. **Phase 0 is done**
(2026-11-08, EXPERIMENTS.md "Data study Phase 0"): the instrument passes its calibration, and its results revise
Phase 1 below.

## Why data, and which classes

- **Data is our biggest measured lever.** Growing the 8B's stage-1 pool from 36k to 212k rows added +0.07
  (Exp 4c → 4d), and 2.5k → 5k rows per label added +0.036 for the 400M (Exp 6g). A forum team measured about
  +0.03 CV per doubling of *matched* data and put the public pool's matched rows near saturation at ~86k
  (thread 752478). Every other lever we know is worth 0.005-0.02.
- **Nothing public explains 0.71-0.75.** The best public notebook is 0.680. The top teams were already at
  0.742-0.747 by 3-4 Nov, before most public ensembling. An earlier forum post credits a 0.670 single model to
  private data.
- **Our weak classes are where the available data matches official worst.** "Pool match" below is the
  official-trained linear model's recognition rate on pool passages divided by its own recall on unseen
  official prompts (1.0 = the pool looks as official as official does; `outputs/closeness_2026-11-03.csv`).

| Group | Labels | 8B recall (Exp 4d) | Pool match | Our current source's match |
| --- | --- | --- | --- | --- |
| Gemini Flash | 3.6 / 3.7 / 3.8 | 0.72 / 0.49 / 0.56 | 0.84 / 0.69 / 0.54 | 0.81 / 0.55 / 0.35 (Agent Platform) |
| Kimi, Qwen | kimi-k3, qwen3.8-max | 0.51, 0.74 | 0.46, 0.50 | 0.14 (Moonshot), 0.08 (Qoder) |
| GPT | 5.5 / luna / sol / terra / astra | 0.63 / 0.65 / 0.51 / 0.60 / 0.84 | 0.64 / 0.88 / 1.33 / 0.62 / 0.92 | 0.52 (WorkBuddy) / 1.12 / 1.33 / 0.92 / 1.18 (Codex) |
| GLM + fable | glm-5.3, glm-5.3-flash, claude-fable-5 | 0.26, 0.45, 0.42 | 1.29, 0.71, 1.00 | 0.79, 0.62 (WorkBuddy), 0.90 (Anthropic) |
| Other Claude | opus-4-6 / 4-8 / 5, sonnet-5 | 0.77 / 0.72 / 0.83 / 0.68 | 0.82 / 0.90 / 1.05 / 0.84 | 0.69 / 0.70 / 1.14 / 0.91 |

The linear recogniser is weak (its own recall is 0.12-0.74), so these ratios are rough, and the GLM/fable
ones near 1.0 mean little. The clearest mismatches are Kimi, Qwen, Gemini 3.8 and 3.7, gpt-5.5 and terra.
Our Codex GPT answers are the one source we know matches. Our other answers ran 10-30% longer than official
(median 111-130 words vs 84-116 per label) because our prompt bands asked for more.

## Rules and limits (host answers in the rules thread, 28-29 Oct)

- Self-generated data is allowed and need not be published. Spending beyond the value of a prize ($5k) is
  "unreasonable". Proposed cap for the whole study: **$800**, released per phase.
- The Kaggle Benchmarks proxy quota must not be used for generation.
- Prompts never contain competition text and never ask a model to identify itself or exaggerate a style
  (`CLAUDE.md`). We do not know the official prompts (train.csv has none), so no official prompt is reused.
- Labels must be verified where the API reports the served model (Gemini `modelVersion`, Moonshot, Anthropic).
  gemini-3.7 is served only by the Agent Platform until 28 Jan 2027, and AI Studio routes it to 3.8.
  **gemini-3.6-flash retires on 19 Nov 2026**, so its collection goes first.
- Crossed design: every prompt is answered by all 18 labels, or by every member of a confusable group plus
  anchors. Every label gets the same wrapper and post-processing, so no setup detail can stand in for a label.

## Phase 0: the closeness instrument (local, about 1 day, no spend)

A setup is judged by how official its answers look to models trained only on official data.
- Recognisers: (a) the linear text+style reference fitted on all 150 official prompts; (b) a Qwen3-4B LoRA24
  trained on official rows only (local GPU; check it fits); (c) the 8B final as a secondary check. It was
  trained mostly on pool rows, so it rewards pool-like text.
- Per (label, setup) cell, four measures:
  - recognition ratio: the recognisers' recall on the cell divided by their nested recall on unseen official
    prompts;
  - style-only AUC, official vs cell;
  - word-count distance from the label's official distribution;
  - within-family transfer BA: a family classifier trained on the cell's family data only, tested on official
    rows of that family. This is the measure closest to what training needs.
- Calibrate on existing sources: Codex GPT (known close) must score well, and Qoder Qwen and Moonshot Kimi
  (known far) must score badly. Otherwise the instrument is not trusted.

**Result (done).** Calibration passes.
- The official-only 4B (out-of-fold BA 0.470) gives Codex GPT ratios of 0.47-0.97, and gives Moonshot Kimi 0.01,
  WorkBuddy Kimi 0.08 and Qoder Qwen 0.02. It puts almost none of the last three in their own family.
- The 4B ratio is the primary measure. Style AUC barely tracks it (Spearman −0.26) and is high for every source.
- The 8B check (c) was not needed.
- Score a new cell with: `uv run --extra nn python -I scripts/closeness_instrument.py --cells pilot.parquet`
  (columns model, text, cell).

## Phase 1: setup search (about $40-80, 2-3 days)

**Start from what Phase 0 found** (4B ratio, ours vs the best pool source):
1. **kimi-k3: 0.01 vs 0.84.** Only arxitekton's rows (Moonshot API, effort "high" per its card) match official on
   em dashes, curly apostrophes, semicolons (61% official, ours 23%) and length. Cells: Moonshot with thinking off,
   with thinking on, and with the effort settings the API exposes.
2. **qwen3.8-max: 0.02 vs 0.82.** Qoder is not the official setup (64% em dashes against 3%; no curly
   apostrophes against 29%). Cells: DashScope qwen3.8-max (arxitekton: snapshot 0902 at T 0.6) at the default and
   at T 0.6.
3. **gpt-5.5: 0.30 vs 0.86.** Use the OpenAI API, not WorkBuddy, which uses no straight apostrophes and too many
   curly double quotes.
4. **glm-5.3: 0.34 vs 0.98** (WorkBuddy vs elliotthankinson). Use Z.ai direct.
5. **gemini-3.8: 0.39 vs 1.28** (chosunghwan). Read its card for the setup first. Ours uses em dashes in 38% of
   answers; official never does.
6. **gemini-3.7: 0.27, and no source above 0.50.** The lowest priority, since nobody has a matching setup.

Claude (Anthropic batches, 0.56-1.18) and the Codex GPT-5.6/6 labels (0.47-0.97) are already near the pool's best.
They only need the new wrapper and length targets.

- 48 new prompts. Word targets are drawn from the official word-count distribution (deciles 49 / 64 / 101 /
  157 / 206), not from fixed bands.
- Per provider, 4-6 cells chosen from these factors (a fractional design, not every combination):
  - wrapper: the current "one paragraph of about N words", "in 40-250 words" (the data page's range), or no
    length line;
  - system prompt: none, or a generic one;
  - thinking or reasoning: the provider's default, off, or low;
  - temperature: the default, or 1.0;
  - interface: a direct provider API instead of a gateway or CLI. This matters for GLM (Z.ai), Qwen
    (DashScope) and gpt-5.5 (OpenAI API), whose current routes match worst.
- Fixed post-processing for every label: whitespace collapsed to one line (every official passage is one line
  with no double spaces). Answers with markdown, lists or citations are dropped. Both rules are logged.
- Output: one chosen setup per provider, or "no matched setup found".
- Gate (revised after Phase 0): scale up a provider only if its best cell reaches all three:
  - a 4B recognition ratio ≥ 0.8;
  - a words KS ≤ 0.15;
  - a style AUC no worse than the label's pool median (0.66-0.86).

  The pool's best style AUC is the minimum over up to 29 noisy sources, which no single setup can be expected to
  beat. 48 prompts give about 48 rows per cell, so a ratio's standard error is about 0.15-0.2. Read a borderline
  cell together with its family share.

## Phase 2: prompt bank (in parallel, about 1 day)

- About 3,000 new prompts in the official genre (one-paragraph explanation or exposition across science and
  humanities). They are written from our own topic lists; LLM help is fine, since no competition text is sent.
- Plus screened Open Pool prompts (`select_pool_prompts.py`), so new answers complete existing pool groups and
  feed the centred-triplet loss.
- Every prompt is screened locally against official passages (TF-IDF cosine < 0.25, as for the pool waves) and
  deduplicated.

## Phase 3: generation, two waves (about $300-700)

- **Wave A:** about 1,000 prompts × 18 labels (18k rows) with each provider's chosen setup. gemini-3.6 runs
  first. Use batch APIs where they exist (Anthropic, OpenAI and Gemini batch modes, about half price), and
  Codex for the GPT-5.6 / 6 labels.
- **Wave B**, only if Wave A passes Phase 4: 2,000-4,000 more prompts, concentrated on the groups that gained:
  - Gemini Flash trio;
  - Kimi and Qwen;
  - the GPT-5.x set;
  - GLM pair with fable;
  - plus full 18-label anchor panels on every 10th prompt.
- Rough answer-only cost per 1,000 answers is below. Reasoning or thinking tokens can multiply the
  GPT-5.6-sol, Gemini and GLM figures. Phase 1 measures the real figures before Wave A is priced.
  - **$1-3:** Gemini Flash, glm-5.3-flash, glm-5.3, Qwen.
  - **$5-10:** Claude 4.x / 5, gpt-5.5, gemini-3.1-pro.
  - **$10:** fable, gpt-6-astra.

## Phase 4: does it help (TC2 + local)

- Screen: `external_data_eval.py`-style nested linear run, official vs official + new rows, both partitions,
  with per-class recalls. Exp 5 showed some sources helping one family and hurting another.
- Decisive: an 8B LoRA24 shared stage 1 on pool + new rows (uniform recipe, so it reads no official rows and
  the nested CV stays clean), then the nested stage 2 on both partitions. Compare with Exp 4d (0.623 / 0.624),
  or with E3 if E3 wins. About 3 h of stage 1 and 2.5 h of stage 2.
- Gate: a paired Δ CI above 0 on both partitions, and no class losing more than 0.05 recall.
- Then: a 150-prompt final with centring, submitted to Kaggle.

## Risks and stop points

- **The hosts' setup cannot be matched** (unknown system prompt or post-processing). Phase 1 shows this for
  about $60, and we stop there.
- **Collection artifacts become label shortcuts.** Mitigated by crossed panels, identical wrapper and
  post-processing, and by checking batch-held-out accuracy. New rows stay a minority of stage 1 unless Phase 4
  says otherwise.
- **Label routing.** 3.7 → 3.8 is known. GLM may route some requests to Claude (forum 755314, unverified).
  Served-model checks catch the first. For GLM, compare the pilot's GLM-vs-fable statistics with official.
- **Rate limits and account set-up time.** No provider keys are configured on this Linux machine yet
  (`~/.config/nllm/` does not exist).

## What the owner decides

1. The budget cap. $800 total is proposed, released per phase: Phase 1 about $80, then Wave A.
2. Which direct API accounts to set up: OpenAI (gpt-5.5, batch), Z.ai (GLM), DashScope (Qwen), Google (Agent
   Platform for 3.7), Anthropic, Moonshot. Keys go in `~/.config/nllm/` and are never printed.
3. ~~Starting Phase 0, which spends nothing.~~ Done 2026-11-08.
