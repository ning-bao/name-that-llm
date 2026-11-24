# Experiment log

Balanced accuracy (BA) is concatenated outer-fold BA over the 120 development prompts
(nested CV from `validation_plan.json`). Intervals are 95% prompt-cluster bootstrap
(2,000 draws). Paired differences are computed on the same resamples. The 30
confirmation prompts have not been scored. Chance is 1/18 = 0.056.

## Exp 0: input and split audit (2026-11-02)

`scripts/audit_inputs.py`: all 17 checks pass. Hashes match the research kit's record,
the 150 × 18 design is complete apart from `p002`/`claude-fable-5`, and both partitions'
outer and inner folds are prompt-disjoint and cover the development set exactly.

## Exp 1: TF-IDF / style linear baseline (2026-11-02)

Run `outputs/baseline_linear/20261102T092238Z`, 45 min on 15 cores, seed 17.

Grid (64 configs): raw char 3-5 and 2-6, case-preserving word 1-2 with punctuation tokens,
and both char+word hstacks; raw or sublinear TF; min_df 2; LR C ∈ {1, 10, 100}, LinearSVC
C ∈ {0.1, 1, 10}. Style: 51 numeric features with LR C ∈ {0.01, 0.1, 1} or a small HGB.
For each outer fold, the config per branch and the text+style blend weight (scaled decision
scores, weight ∈ {0 … 5}) are chosen on inner out-of-fold BA.

| Branch | BA p0 [95% CI] | BA p1 [95% CI] | Δ vs text p0 [CI] | Δ vs text p1 [CI] |
| --- | --- | --- | --- | --- |
| char | 0.295 [0.278, 0.310] | 0.306 [0.285, 0.326] | −0.026 [−0.040, −0.013] | −0.026 [−0.041, −0.011] |
| word | 0.313 [0.296, 0.329] | 0.316 [0.297, 0.334] | −0.007 [−0.021, 0.007] | −0.016 [−0.030, −0.002] |
| style | 0.297 [0.281, 0.315] | 0.307 [0.290, 0.324] | −0.023 [−0.042, −0.004] | −0.024 [−0.047, 0.000] |
| text (= char+word) | 0.320 [0.303, 0.336] | 0.332 [0.311, 0.353] | — | — |
| **text+style** | **0.348 [0.328, 0.369]** | **0.361 [0.342, 0.381]** | **+0.028 [+0.012, +0.044]** | **+0.030 [+0.015, +0.044]** |

Findings:

- **text+style is the baseline to beat.** Its gain over text has a paired lower bound above zero in
  both partitions. Style features alone match char n-grams, and their errors complement text.
- Inner selection always picked a char+word hstack for the text branch, mostly LR with raw TF. LR C=100,
  the top of the grid, was chosen in 6 of 10 folds for text and 8 of 10 for char, so C=1000 is worth adding.
  Style LR C=0.01, the bottom of its grid, was chosen twice.
- The predicted provider family is right on about 69% of rows, and about half of all errors are
  siblings from the same family. Random errors would land in-family only about 18% of the time.
  The largest confusions (counts summed over both partitions) are Gemini 3.7 ↔ 3.8 (74/65),
  glm-5.3 → glm-5.3-flash (57), glm-5.3 → claude-opus-5 (50), gemini-3.6 → 3.1-pro (50),
  claude-fable-5 → glm-5.3 (49) and gpt-5.6-sol → luna (48). Anthropic ↔ ZAI confusion matches the
  report's typography finding: both families use em dashes and no curly apostrophes.
- Weakest classes for text+style: claude-fable-5, glm-5.3, gpt-5.6-sol and claude-opus-4-8, all at or below
  0.22 recall. Strongest: gemini-3.1-pro-preview (≈0.76) and claude-opus-5 (≈0.60).
- Rerun `outputs/baseline_linear/20261102T122807Z` reproduces this run bit for bit (scores, predictions,
  selections) and adds `inner_oof.parquet` for nested blends; use it as the linear reference from now on.

## Prompt-count learning curve, linear text+style (2026-11-02)

`scripts/learning_curve.py`, run `outputs/learning_curve/20261102T133615Z`. Fixed config (the most-selected
in Exp 1: char35+word12 LR C=100 raw TF + 1.0 × style LR C=0.1), fitted on random subsets of each outer
fold's 96 training prompts and scored on its 24 held-out prompts. BA concatenated over the 5 outer folds,
mean over both partitions and 3 subsets per size. Family BA weights the 6 providers equally.

| Training prompts | 24 | 48 | 72 | 96 |
| --- | --- | --- | --- | --- |
| text+style BA | 0.284 | 0.318 | 0.342 | 0.356 |
| text+style family BA | 0.518 | 0.551 | 0.576 | 0.584 |
| text BA | 0.247 | 0.286 | 0.308 | 0.326 |
| style BA | 0.250 | 0.276 | 0.293 | 0.303 |

- About +3.6 BA points per doubling of training prompts for text+style (+3.9 text, +2.7 style), with no
  flattening in log scale (24→48 +3.4, 48→96 +3.8). Family BA rises at the same rate, so extra prompts help
  sibling separation and provider separation about equally.
- Log-linear extrapolation (optimistic at the far end): the final fit on all 150 official prompts is
  worth about +2 points over a 96-prompt fit; adding ~150 independent external prompts per model
  (≈300 total) would be worth about +6 points for this model, if the external data matched the
  competition's serving distribution. Whether it does is what report E6's official-only / external-only /
  combined comparison has to establish.

## Exp 3: fine-tuned encoder screen, partition 0 (2026-11-02/03)

`scripts/encoder_finetune.py`, partition 0 only. Runs: Ettin-400m `outputs/encoder/ettin-400m/20261102T122705Z`
(`jhu-clsp/ettin-encoder-400m`, mean pooling, 110 min) and RoBERTa-large
`outputs/encoder/roberta-large/20261102T141719Z` (`FacebookAI/roberta-large`, about 130 min, the last two outer
folds resumed at half GPU duty). Full fine-tuning, max length 512, bf16 autocast, effective batch 32
(micro-batch 8), AdamW wd 0.01, 6% warmup then linear decay, seed 17. Per outer fold, lr ∈ {1e-5, 3e-5} and the
epoch (1–10) are chosen on inner out-of-fold BA; the outer refit stops at the chosen epoch. Blends:
`scripts/blend_oof.py` against the Exp 1 rerun (`outputs/blend/20261102T1651{51,55,59}Z`), with component
weights chosen per outer fold on inner OOF BA.

| Branch (p0) | BA [95% CI] | Δ vs text+style [CI] |
| --- | --- | --- |
| text+style (Exp 1) | 0.348 [0.328, 0.369] | — |
| RoBERTa-large | 0.331 [0.310, 0.352] | −0.017 [−0.042, +0.007] |
| Ettin-400m | 0.388 [0.367, 0.409] | +0.040 [+0.017, +0.063] |
| text+style + RoBERTa | 0.391 [0.368, 0.414] | +0.043 [+0.023, +0.061] |
| text+style + Ettin | 0.408 [0.387, 0.428] | +0.060 [+0.043, +0.076] |
| **text+style + Ettin + RoBERTa** | **0.425 [0.403, 0.448]** | **+0.077 [+0.057, +0.098]** |

Findings:

- **Ettin-400m beats the linear baseline alone**, and every blend beats its components. The three-way blend
  beats text+style + Ettin by +0.017 [+0.004, +0.031], so RoBERTa's errors complement the others even though
  it is weaker alone.
- The selections sit at the top of the grid. lr 3e-5 was chosen in 9 of 10 fold selections, and the epoch was
  9 or 10 in every fold, so both encoders are probably under-trained. Next screen: lr 5e-5 and 15 epochs.
- Inner BA (models trained on 64 prompts) runs 4–8 points below outer BA (96 prompts), consistent with the
  learning curve.
- Within-family error share (p0): text+style 0.54, Ettin 0.56, RoBERTa 0.59, so the encoders get the provider
  right more often and leave a larger share of their errors among siblings. The blend's biggest recall gains
  over text+style are claude-opus-4-6/4-8, gpt-5.6-luna/sol and gemini-3.6. claude-fable-5 (0.18) and
  glm-5.3 (0.20) stay weakest.
- Partition 1 has not been run, so the encoder results have no partition-stability evidence yet.

## Exp 5: own external data, official vs combined, linear text+style (2026-11-02/03)

`scripts/external_data_eval.py`, both partitions, about 35 min per run. The model is fixed (the Exp 1 reference
text+style config, not re-selected), so only the training data changes. In the combined condition every class
gets equal total weight and its external rows a share s ∈ {0.1, 0.25, 0.5} of it, chosen per outer fold on inner
OOF BA. External rows come from the 152 training prompts of the collection bank (all 18 labels, 152 per label;
gemini-3.8-flash 178, kimi-k3 315). Three runs differ only in which sources are excluded:

- all sources: `outputs/external_eval/20261102T205210Z`, 2,925 rows;
- no Gemini (`--exclude 'gemini-*@*'`): `20261102T213017Z`, 2,291 rows;
- no Gemini, no WorkBuddy or thinking-mode Kimi, no Qwen: `20261102T220347Z`, 1,976 rows.

| External data | BA p0 | Δ vs official p0 [CI] | BA p1 | Δ vs official p1 [CI] |
| --- | --- | --- | --- | --- |
| none (official only) | 0.354 | — | 0.358 | — |
| **all sources** | **0.379** | **+0.025 [+0.010, +0.040]** | **0.375** | **+0.016 [+0.001, +0.032]** |
| no Gemini | 0.368 | +0.014 [−0.001, +0.029] | 0.373 | +0.015 [+0.001, +0.028] |
| no Gemini, no shifted Kimi or Qwen | 0.368 | +0.014 [−0.001, +0.029] | 0.377 | +0.019 [+0.006, +0.032] |

Paired against all sources, dropping Gemini costs −0.011 [−0.021, 0.000] on p0 and −0.002 [−0.013, +0.009] on p1.
Dropping the shifted Kimi and Qwen sources as well changes nothing (−0.000 and +0.004 vs no Gemini, CIs span 0).

Findings:

- **Own external data helps the linear model a little, in both partitions.** About +0.02 BA, less than half
  of the ~+6 points the learning curve predicts for 150 matched prompts, so the collected data is only partly
  matched to the official serving setup. The official-only reference (0.354 / 0.358) is the fixed config,
  slightly above the nested Exp 1 numbers.
- **Gemini rows help Gemini despite their em dashes.** Pooled over partitions, gemini-3.8 recall is 0.371
  official only, 0.467 with all sources and 0.350 without Gemini rows; the four Gemini recalls together gain
  0.21 from the Gemini rows. claude-opus-4-6, opus-4-8 and sonnet-5 lose 0.02–0.04 recall each when Gemini rows
  are added.
- **Those Claude losses are not em-dash passages pulled to Gemini.** Official non-Gemini passages predicted as a
  Gemini label: with an em dash 0.7% (official only), 0.5% (all), 0.1% (no Gemini); without one 6.3%, 8.2%, 4.5%.
  With official rows at half the weight or more, the linear model keeps "em dash means not Gemini". This may
  not hold for a model trained mostly on external rows.
- **qwen3.8-max loses 0.10–0.15 recall in every combined run** (0.479 → 0.346 with Qoder rows, 0.333
  without). This matches the forum's coverage finding: when Qwen's external rows don't match official Qwen,
  new prompts read as "not Qwen". The Qoder rows don't fix it.
- **Kimi gains from the shifted sources.** kimi-k3 recall is 0.275 official only, 0.367 with WorkBuddy and
  thinking-mode rows, 0.292 with only non-thinking Moonshot rows.
- **External-only training is far worse:** external17 0.270 on both partitions vs official17 0.378 / 0.374. On
  the external holdout the gap reverses (0.364 vs 0.27), so the shift runs both ways.
- external17 is meaningless in the two runs with excluded labels (BA 1/17). Labels without external rows get
  −inf scores, `linear.scale` turns every row into NaN and argmax returns claude-fable-5. Combined conditions
  are unaffected because every label has official rows.

## Gemini typography diagnostics (2026-11-03)

Official Gemini passages (all four labels, 600 rows) contain one em dash, and only 7–18% contain a hyphenated
word, against 38–49% for Claude, GLM, Kimi and Qwen. Our Agent Platform Gemini answers have em dashes in 21–35%
and hyphenated words in 47–57% of answers. Common compounds in ours ("long-term", "well-being", "real-world")
never appear in official Gemini, hyphenated, spaced or joined, which points to a dash-avoidance instruction or
regeneration in the hosts' Gemini setup rather than to the models' own habits.

gemini-3.8-flash on Agent Platform, bank prompts only, never imported (`data/external/diagnostics/`):

| System instruction | Answers | Em-dash rate |
| --- | --- | --- |
| none (plain API, same prompts) | 60 | 32–35% |
| leaked Gemini app prompt, verbatim | 20 | 50% |
| same, its own em dashes replaced by commas | 40 | 50% |
| leaked Gemini Workspace prompt | 56 (4 failed with MALFORMED_FUNCTION_CALL) | 14% |

The app prompt's own em dashes do not prime the model, and no public route reproduces official Gemini's 0%.
Dropping em-dash rows keeps 102 of 152 verified 3.7 rows (99–120 for the other Gemini labels), but the hyphen
gap stays (43–50%).

## Source closeness: our answers and the Open Pool vs official (2026-11-03)

`scripts/source_closeness.py`, results in `outputs/closeness_2026-11-03.csv`. For each label, every source of our
admitted answers and the pool (all rows, proxy and non-proxy rows, its four largest source datasets; up to 500
or 300 sampled passages) is compared with the official passages of the 120 dev prompts. "Recognised" is the
share of a source's passages that the text+style reference, fitted on all official dev rows, gives the claimed
label ("family" counts the right provider); the last column is that model's own nested-CV recall on unseen
official prompts, for scale. "AUC" is a style-features-only discriminator, official vs source (0.5 = cannot tell
them apart). Words are medians.

| Label | Our source | Recognised: ours / pool / best pool source | Family: ours / pool | AUC: ours / pool | Words: ours / pool / official | Official CV recall |
| --- | --- | --- | --- | --- | --- | --- |
| claude-fable-5 | Anthropic batches | 0.11 / 0.12 / 0.10 | 0.62 / 0.53 | 0.75 / 0.63 | 125 / 104 / 98 | 0.12 |
| claude-opus-4-6 | Anthropic batches | 0.31 / 0.37 / 0.38 | 0.75 / 0.77 | 0.81 / 0.69 | 125 / 113 / 98 | 0.45 |
| claude-opus-4-8 | Anthropic batches | 0.14 / 0.18 / 0.20 | 0.63 / 0.60 | 0.80 / 0.68 | 122 / 112 / 107 | 0.20 |
| claude-opus-5 | Anthropic batches | 0.72 / 0.66 / 0.70 | 0.82 / 0.80 | 0.76 / 0.60 | 125 / 117 / 106 | 0.63 |
| claude-sonnet-5 | Anthropic batches | 0.39 / 0.36 / 0.47 | 0.68 / 0.68 | 0.81 / 0.69 | 125 / 108 / 108 | 0.43 |
| gemini-3.1-pro-preview | Agent Platform | 0.55 / 0.59 / 0.63 | 0.77 / 0.88 | 0.91 / 0.77 | 124 / 95 / 105 | 0.74 |
| gemini-3.6-flash | Agent Platform | 0.26 / 0.27 / 0.32 | 0.60 / 0.81 | 0.85 / 0.76 | 113 / 90 / 96 | 0.32 |
| gemini-3.7-flash | Agent Platform | 0.23 / 0.29 / 0.30 | 0.63 / 0.80 | 0.88 / 0.80 | 111 / 92 / 100 | 0.42 |
| gemini-3.8-flash | Agent Platform | 0.13 / 0.20 / 0.29 | 0.61 / 0.78 | 0.89 / 0.80 | 113 / 98 / 84 | 0.37 |
| glm-5.3 | WorkBuddy | 0.11 / 0.18 / 0.23 | 0.25 / 0.31 | 0.81 / 0.65 | 129 / 92 / 96 | 0.14 |
| glm-5.3-flash | WorkBuddy | 0.13 / 0.15 / 0.22 | 0.23 / 0.35 | 0.86 / 0.70 | 130 / 102 / 115 | 0.21 |
| gpt-5.5 | WorkBuddy | 0.22 / 0.27 / 0.33 | 0.82 / 0.71 | 0.91 / 0.79 | 114 / 99 / 102 | 0.42 |
| **gpt-5.6-luna** | **Codex CLI** | **0.37** / 0.29 / 0.36 | **0.86** / 0.77 | 0.84 / 0.73 | 112 / 107 / 104 | 0.33 |
| **gpt-5.6-sol** | **Codex CLI** | **0.16** / 0.16 / 0.15 | **0.86** / 0.75 | 0.83 / 0.76 | 116 / 100 / 107 | 0.12 |
| **gpt-5.6-terra** | **Codex CLI** | **0.24** / 0.16 / 0.20 | **0.88** / 0.79 | 0.84 / 0.75 | 111 / 106 / 100 | 0.26 |
| **gpt-6-astra** | **Codex CLI** | **0.59** / 0.46 / 0.50 | **0.88** / 0.82 | 0.88 / 0.77 | 115 / 107 / 101 | 0.50 |
| kimi-k3 | Moonshot (thinking off) | 0.04 / 0.13 / 0.28 | 0.04 / 0.13 | 0.91 / 0.77 | 112 / 90 / 86 | 0.28 |
| qwen3.8-max | Qoder CLI | 0.04 / 0.24 / 0.32 | 0.04 / 0.24 | 0.95 / 0.70 | 118 / 90 / 91 | 0.48 |

Findings:

- **Only our Codex GPT answers are closer to official than the pool.** The official-trained model recognises
  them at or above the best pool source for all four Codex labels, at or above its own CV recall on unseen
  official prompts, and puts 86-88% in the GPT family (pool 75-82%).
- **Everything else we collected is at or below the pool**, clearly so for Gemini, GLM, WorkBuddy gpt-5.5, Kimi
  (Moonshot 0.04, WorkBuddy 0.10) and Qoder Qwen (0.04). Claude is mixed.
- **Every source of ours is easier to tell from official by style alone** (AUC 0.75-0.95 vs 0.60-0.80 for the
  pool), partly because our answers run 10-30% longer than official (median 111-130 words vs 84-115). Our
  prompt bank's length bands ask for more than the official mix delivers.
- Inside the pool, sources differ a lot (e.g. kimi-k3: arxitekton v2 0.28, proxy rows 0.11). Experiment 6 turns
  this measurement into in-fold source multipliers (`pool.source_quality`).

Follow-up: a Codex wave `pool1` on 400 screened Open Pool prompts (`scripts/select_pool_prompts.py`: groups
with ≥12 labels in the pool, one paragraph of about N words, no model or style words, TF-IDF cosine to every
official passage < 0.25, observed max 0.11; sampled to the official length mix). That is 1,600 answers, which
join the pool's prompt groups.

## Exp 6: community pool training data, linear text+style (2026-11-03)

`scripts/pool_eval.py`, run `outputs/pool_eval/20261103T044607Z`, both partitions, 70 min on 8 workers. The model is
the fixed Exp 1 reference text+style config, so only the training data changes. Pool conditions draw 1,000 rows per
label from the Open Pool v13 (190,588 passages; plus 3,229 of our admitted answers where noted), class-balanced,
with the same seed per fold. Everything fitted on official data uses the fold's 96 training prompts only:
match weights (`pool.match_weights`: per-label exponential tilting so 15 marker rates and 4 length bins match
the fold's official rows, ridge 0.02; the pool's own `match_weight` column, fitted on all 150 official prompts,
is never read) and source multipliers (`pool.source_multipliers`: (label, source) cells that the fold's
official-only model recognises more often are drawn more, clipped to 0.25-4). "+official" adds the fold's
official rows at half of each class's weight.

| Condition | BA p0 | Δ vs official p0 [CI] | BA p1 | Δ vs official p1 [CI] |
| --- | --- | --- | --- | --- |
| official only | 0.357 | — | 0.364 | — |
| pool, uniform | 0.396 | +0.040 [+0.016, +0.062] | 0.398 | +0.033 [+0.009, +0.058] |
| pool, match weights | 0.402 | +0.045 [+0.024, +0.068] | 0.416 | +0.052 [+0.030, +0.076] |
| pool, match weights, no proxy rows | 0.404 | +0.047 [+0.026, +0.069] | 0.413 | +0.049 [+0.025, +0.074] |
| pool + own, match weights | 0.405 | +0.049 [+0.026, +0.071] | 0.415 | +0.051 [+0.029, +0.073] |
| pool + own, match weights × source multipliers | 0.403 | +0.046 [+0.022, +0.069] | 0.419 | +0.054 [+0.032, +0.077] |
| **pool (match weights) + official** | **0.418** | **+0.062 [+0.042, +0.080]** | **0.433** | **+0.068 [+0.047, +0.091]** |
| pool + own (× source multipliers) + official | 0.423 | +0.066 [+0.045, +0.087] | 0.430 | +0.066 [+0.045, +0.088] |

Paired against pool with match weights: uniform sampling −0.006 [−0.022, +0.012] (p0) and −0.019 [−0.036, −0.002]
(p1); adding official rows +0.016 [+0.004, +0.029] and +0.016 [+0.002, +0.030]; dropping proxy rows, adding our own
rows and source multipliers all within ±0.004, intervals spanning 0.

Findings:

- **Pool data alone beats the official training rows**, by +0.04 to +0.05 BA in both partitions, for a model whose
  data is 1,000 pool rows per label from prompts it never sees in validation. Pool plus official is best:
  0.418 / 0.433, as high as the three-model blend of Exp 3 (0.425 on p0) from one linear model.
- **Match weights help** (+0.006 / +0.019), consistent with the pool author's +0.026 at full size.
- **Kaggle Benchmarks proxy rows are harmless**; keep them.
- **Our own rows cannot be judged at their natural share.** They are 1-4% of each label's sample (no Qwen, since
  Qoder is excluded). Source multipliers barely move them either: Codex ×1.2-1.4, Moonshot Kimi ×0.33,
  WorkBuddy GLM ×0.64-0.84; pool cells 0.74-1.22 (10th-90th percentile).
- **Per class (mean of partitions)**, pool data lifts Qwen from 0.48 to 0.61, claude-opus-4-8 from 0.20 to 0.31-0.33,
  gemini-3.6 from 0.32 to 0.44-0.48, gemini-3.8 from 0.37 to 0.48-0.54 and gpt-5.6-sol from 0.15 to 0.27-0.31 (with
  official rows). kimi-k3 falls from 0.30 to 0.12-0.24, and gpt-5.5 from 0.44 to 0.36 without official rows. Source
  multipliers raise Kimi (0.18 → 0.24) by drawing fewer of the pool's weak Kimi sources, at a cost to gpt-5.6-sol.

### Exp 6b: does drawing our Codex GPT answers more often help? (2026-11-03)

Run `outputs/pool_eval/20261103T064158Z`, same design as Exp 6, after importing the Codex `pool1` wave (our own rows
4,828; about 596 per Codex GPT label). `w_codex5` keeps only our Codex rows from our own data and gives them 5× the
starting weight, so they make up 15-20% of each Codex label's sample (4% at their natural share in `w_own`).

| Condition | BA p0 | Δ vs pool_w p0 [CI] | BA p1 | Δ vs pool_w p1 [CI] |
| --- | --- | --- | --- | --- |
| pool_w | 0.402 | — | 0.416 | — |
| pool_w_own | 0.404 | +0.002 [−0.007, +0.012] | 0.417 | +0.001 [−0.010, +0.012] |
| pool_w_codex5 | 0.403 | +0.002 [−0.008, +0.011] | 0.412 | −0.005 [−0.015, +0.006] |
| pool_w+official | 0.418 | +0.016 [+0.004, +0.029] | 0.433 | +0.016 [+0.002, +0.030] |
| pool_w_own+official | 0.425 | +0.023 [+0.009, +0.036] | 0.424 | +0.008 [−0.007, +0.023] |
| pool_w_codex5+official | 0.427 | +0.024 [+0.011, +0.038] | 0.427 | +0.011 [−0.003, +0.027] |

- **No.** Boosting Codex moves BA by +0.002 / −0.005 alone and +0.008 / −0.005 next to `pool_w+official`. The sign
  flips between partitions. GPT recalls (mean of partitions, pool_w → codex5) don't rise: luna 0.32 → 0.33, sol
  0.24 → 0.22, terra 0.36 → 0.35, astra 0.56 → 0.51.
- Being recognised as the right label by an official-trained model (the closeness check) did not make the Codex
  answers better training data at a higher share. Codex wraps every prompt in ~12.7k tokens of agent instructions
  and its answers run longer. More of the same serving condition therefore adds sameness, not coverage. Our own
  answers stay at their natural share; no more Codex waves are planned.

### Exp 6c: Ettin-400M with a pool stage 1, partition 0 (2026-11-03)

`scripts/encoder_finetune.py --backbone ettin-400m --partitions 0 --pool-per-label 2500 --pool-recipe w --epochs 6`,
run `outputs/encoder/ettin-400m/20261103T055726Z`, 129 min. For each outer fold: stage 1 fine-tunes the pretrained
encoder for 2 epochs at lr 3e-5 on 45,000 Open Pool rows (2,500 per label, match weights fitted on that fold's 96
official training prompts). Stage 2 is the usual nested fine-tune on official rows (lr ∈ {1e-5, 3e-5}, up to 6
epochs, chosen on inner folds), starting from stage 1. `oof_stage1.parquet` scores stage 1 alone.

| Branch (p0) | BA [95% CI] | Paired Δ |
| --- | --- | --- |
| Ettin-400M, official only (Exp 3) | 0.388 | — |
| text+style + Ettin + RoBERTa blend (Exp 3) | 0.425 | — |
| pool stage 1 alone | 0.523 [0.501, 0.544] | +0.135 [+0.113, +0.157] vs Ettin official-only |
| **pool stage 1 + official stage 2** | **0.530 [0.510, 0.551]** | +0.142 [+0.119, +0.166] vs Ettin official-only; +0.105 [+0.080, +0.129] vs the blend |

- **The pool is the biggest gain so far.** +0.14 BA for the same encoder; one model now beats the old three-model
  blend by +0.10.
- **Stage 2 on official rows adds little**: +0.007 [−0.010, +0.023] over stage 1 alone. It chose the lowest lr
  (1e-5) in every fold and 1-4 epochs, so a gentler stage 2 (lr 5e-6, fewer epochs) or official rows mixed into
  stage 1 may do better.
- Stage 2 trades between siblings: gpt-5.6-luna 0.34 → 0.54 and claude-opus-4-8 0.48 → 0.56, but gemini-3.8
  0.71 → 0.56 and gpt-5.6-sol 0.50 → 0.38.
- **Weakest classes**: claude-fable-5 0.23, glm-5.3 0.22, glm-5.3-flash 0.32, gemini-3.7 0.34; gpt-6-astra
  (0.83) and claude-opus-5 (0.79) are strongest.
- Partition 1 not run yet. Inner selection BA (0.54-0.57) is close to outer BA. Each fold's stage-1 weights saw the
  inner-validation prompts' marker rates, a small leak into the (lr, epoch) choice only.

Follow-up (3 Nov 2026): gemini-3.7-flash on Agent Platform (every response served as gemini-3.7-flash), the first 40
prompts of the same seed-7 order, scored on answers that contain text:

| Frame | Answers | Em dash | Hyphenated | Curly ’ | Straight ' | Colon | Semicolon | Median words |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| official 3.7 (dev) | 120 | 1% | 18% | 27% | 14% | 23% | 23% | 100 |
| plain API, same prompts | 40 | 30% | 60% | 35% | 22% | 18% | 30% | 111 |
| leaked gemini-3.7-flash app prompt, verbatim | 40 | 50% | 57% | 25% | 30% | 28% | 32% | 113 |
| leaked Workspace prompt, verbatim | 15 of 40 | 20% | 47% | 20% | 40% | 13% | 13% | 111 |

With the Workspace prompt, 3.7 tried to call the tools the prompt describes in 25 of 40 requests (23
MALFORMED_FUNCTION_CALL, 2 function calls). Neither leaked frame comes near official 3.7: em dashes stay at 20-50%
and hyphenated words at about half the answers, as with 3.8.

## More gemini-3.7 data: search and verification (2026-11-03)

Searched Hugging Face, Kaggle (datasets updated after Open Pool v13) and GitHub. Hugging Face and GitHub had no 3.7
passages (one empty placeholder). Four Kaggle datasets carry new 3.7 rows; each was downloaded to its own folder under
`data/external/third_party/` (manifest updated). A char+word TF-IDF logistic regression trained to tell 3.7 from 3.8
on the pool's four Vertex-documented sources (prompt-grouped CV AUC only 0.59) scored them. Matched pairs (prompts
answered by both labels): how often the 3.7 answer scores higher.

| Source | 3.7 rows | Route | Mean P(3.7) (3.8 rows) | Paired: 3.7 ahead | Verdict |
| --- | --- | --- | --- | --- | --- |
| official (all 150 prompts) | 150 | — | 0.53 (0.47) | 57%, p = 0.09 | reference |
| ours, Agent Platform, modelVersion-verified | 196 | Vertex | 0.60 (0.50) | 63%, p = 0.0003 | reference |
| zeminglii/ntl-own-exact-route-generations v2 | 319 | OpenRouter pinned to Google Vertex | 0.54 (0.48, via proxy) | 15 pairs only | **usable** |
| kilupy/ntl-18-model-corpus v1 | 300 | OpenRouter, provider not recorded | 0.61 (0.57) | 60%, p = 0.001 | **usable**: behaves like real 3.7 |
| abdullah1afify/ntl-generations v10 | 160 | Google AI Studio | 0.46 | — | reject: AI Studio serves 3.8 since 8 Oct |
| elliotthankinson/ntl-exact-gens-v3 | 1,055 | OpenRouter | — | — | the author's own gate rejected them |

Both usable sets are CC BY 4.0, absent from the pool, single paragraphs of median 98-101 words, and at most 0.20
TF-IDF cosine from any official passage. Their Gemini answers have em dashes in 17-18% of rows, like other API
output, so `typography-source-filter-v1` applies. Together they add about 620 3.7 rows to the pool's 3,278.
kilupy also has 3,338 prompts answered by all five Claude models (16.6k rows, Anthropic API), which bears on the
claude-fable-5 confusion. 25 of its rows are within 0.3 cosine of an official passage and would be dropped.

### Exp 6d: zeminglii and kilupy rows in the pool sample, linear text+style (2026-11-03)

`scripts/prepare_extra_sources.py` cleaned both sets into `data/external/extra_sources.parquet` (21,167 rows: label in
LABELS, one paragraph of 36-275 words, not in the pool, no duplicates, TF-IDF cosine < 0.3 to official passages and
< 0.4 to official prompt centroids, `typography-source-filter-v1`). Recipes `wx` and `wx_own` add them to the
candidates; weighting is unchanged (in-fold match weights). Share of extra rows among each label's candidates: Claude
labels 25-34% (kilupy's five-model block), gemini-3.7-flash 13.5% (510 rows), every other label 2.5-6.6%. Same
protocol as exp 6, 1,000 rows per label. Run `outputs/pool_eval/20261103T090836Z`.

| Condition | BA p0 | BA p1 | vs pool_w (p0 / p1) |
| --- | --- | --- | --- |
| pool_w | 0.402 | 0.416 | — |
| pool_wx | 0.401 | 0.410 | −0.001 / −0.006 |
| pool_wx_own | 0.409 | 0.409 | +0.007 / −0.007 |
| pool_w+official | 0.418 | 0.433 | +0.016 / +0.016 |
| pool_wx+official | 0.427 | 0.426 | +0.025 / +0.010 |
| pool_wx_own+official | 0.418 | 0.419 | +0.016 / +0.003 |

- The extra rows do not move the linear model. Against pool_w+official, pool_wx+official is +0.009 on p0 and −0.007 on
  p1; every paired interval against pool_w on the same footing crosses zero. Not promoted.
- gemini-3.7-flash recall rises on p0 (0.39 → 0.47 without official rows, 0.42 → 0.45 with) but not on p1 (0.36 → 0.38,
  0.39 → 0.38). claude-fable-5 recall is unchanged (0.13-0.22 in every condition), so the extra Claude rows do not
  untangle fable from the GLM pair for a linear model.
- The GPU queue (experiment 7 and Ettin-1B) and the staged TC2 jobs use `wx_own`. By this screen it is equivalent to
  `w`, so comparing them with exp 6c (recipe `w`) changes backbone or method, not data, to within about ±0.01.

## Decoder pool path smoke test, and the experiment-7 memory fix (2026-11-03)

`encoder_finetune.py --backbone qwen3-1.7b-base --pool-per-label 20 --pool-recipe wx_own --pool-epochs 1 --epochs 1
--lrs 1e-4` on partition 0 (`outputs/encoder/qwen3-1.7b-base/smoke_pool`) exercised the decoder stage 1 → LoRA-adapter
reload → nested stage 2 path that the TC2 Qwen3-4B/8B pool jobs use. It ran end to end: 25 fits, peak 6.4-7.8 GB, 5-7.5
min each (2.3 h in all). BA 0.141 (stage 1 alone 0.060) is not a result; 360 pool rows and one epoch only check the
plumbing.

The first experiment-7 launch spilled: the label-conditioned LM held full (8, ≤384, 152k) logits plus an fp32 copy,
reaching 14.8 GB dedicated and 24.8 GB shared memory. It was stopped before finishing a fold. `genlm._token_logprobs`
now runs the decoder body once and passes only passage-token hidden states through the vocabulary head, 1,024 tokens
at a time, checkpointed in training. A test checks that this matches the full-logit computation. Pilot on Qwen3-1.7B
(576 pool rows, 85k tokens, one epoch):

| Micro-batch | Train time | Peak allocated |
| --- | --- | --- |
| 8 | 67 s | 5.6 GB |
| 16 | 45 s | 5.9 GB |
| 32 | 32 s | 7.0 GB |

`genlm_eval.py` now defaults to micro-batch 32 (no accumulation; scoring batch 32): about 25 min per outer fold.

## Exp 4 on TC2: Qwen3-8B pilot, and the first long jobs (2026-11-04)

The project moved to native Linux (the repository root; the NTFS copy is read-only there). TC2 got the code,
data and pool, and a job built the environment (`~/nllm/venv`, 7.3 GB, torch 2.11.0+cu128 on the L40S). HiFT freed
35 GiB of the shared quota first; with the env and the 8B weights the account sits at about 78%.

Pilot `nllm_pilot8b` (job 50893, code a9ef72895d96): Qwen3-8B-Base, LoRA r16 on all projections plus an fp32 head,
p0/o0/k0 (1,151 training and 576 inner-validation rows), lr 2e-4, micro-batch 4 × accumulation 8, no checkpointing.

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inner-val BA | 0.118 | 0.198 | 0.283 | 0.314 | 0.344 | 0.333 | 0.344 | 0.356 |

57 s per epoch (about 20 rows/s), 772 s for the fit, peak 27.9 GB. It passes the System 1 session's gate (best epoch
≥ 0.35, against Ettin-400M's 0.349 on the same split; ≤ 40 GB; ≤ 30 min), narrowly. The fitted adapter and head are in
`outputs/tc2/pilot-qwen3-8b-base-001/model` as the reference for the Kaggle T4 fp16 agreement check.

Submitted at 04:10 (+08), code b49b20ac1ab1, in run order on the single GPU:
- job 50896 `nllm_8bpool_p0_c1`: pool stage 1 (recipe `wx_own`, 2,000 rows per label, 1 epoch, lr 1e-4, about 30 min
  per fold) then a nested stage 2 (lr 1e-4, 3 epochs), partition 0. Expected to finish in one chunk (~4.2 h).
- job 50897 `nllm_8b_p0_c1`: the official-only nested 8B run (lrs 1e-4/2e-4, 8 epochs), partition 0; ~68 min per
  fold, so a short second chunk is needed.
`--stage1-minutes` (new) gives the stage-1 fit its own time reserve; the per-fit reserves come from the pilot timing.

## Exp 7: label-conditioned language model (generative classifier), partition 0 (2026-11-04)

`scripts/genlm_eval.py` with Qwen3-1.7B-Base, LoRA r16, 1 epoch at lr 2e-4 on 19.7k rows per fold (recipe `wx_own`,
1,000 pool rows per label plus the fold's official rows; 40 more pool rows per label set the balancing offsets).
Run `outputs/genlm/qwen3-1.7b-base/20261103T124210Z`, on the local 5070 Ti under Linux: 18 min per fold, peak 7.3 GB.

| | BA p0 | 95% CI | vs Ettin-400M + pool (0.530) |
| --- | --- | --- | --- |
| genlm, with offsets | 0.280 | [0.262, 0.298] | −0.250 [−0.276, −0.226] |
| genlm, no offsets | 0.294 | | |

Every label is below the discriminative baseline; the worst are GPT labels (gpt-5.5 0.08 vs 0.55, gpt-6-astra 0.26 vs
0.82) and opus-4-8 (0.19 vs 0.56). Summed log-likelihood under a label prefix is dominated by how predictable the
passage's content is, and one epoch of LoRA does not move it enough per label. The offsets do not help. Dropped: the
discriminative classifiers (Ettin, Qwen3 with a classification head) are the better use of GPU time.

### Exp 6e: Ettin-1B with a pool stage 1, and a two-encoder blend, partition 0 (2026-11-04)

`scripts/encoder_finetune.py --backbone ettin-1b --optimizer adafactor --micro-batch 4 --gradient-checkpointing
--pool-per-label 1500 --pool-recipe wx_own --pool-epochs 1 --lrs 1e-5 --epochs 3`, run
`outputs/encoder/ettin-1b/20261103T210602Z`, 121 min on the local 5070 Ti under Linux (stage 1: 27,000 rows, about
15 min per fold, peak 10.6 GB). This is not a clean size comparison with Exp 6c: Ettin-1B saw less pool data (1,500
vs 2,500 rows per label, 1 vs 2 stage-1 epochs, recipe `wx_own` vs `w`) and only lr 1e-5 in stage 2.

| Branch (p0) | BA [95% CI] | Paired Δ vs Ettin-400M + pool |
| --- | --- | --- |
| Ettin-400M + pool (Exp 6c) | 0.530 [0.510, 0.551] | — |
| Ettin-1B pool stage 1 alone | 0.498 [0.479, 0.518] | |
| Ettin-1B + pool | 0.517 [0.499, 0.536] | −0.013 [−0.035, +0.009] |
| **nested blend Ettin-400M + Ettin-1B** (`outputs/blend/20261103T230655Z`) | **0.559 [0.538, 0.578]** | **+0.029 [+0.014, +0.043]** |
| same blend + linear text+style (`outputs/blend/20261103T230712Z`) | 0.552 [0.533, 0.571] | +0.022 [+0.008, +0.037] |

- **Ettin-1B alone is no better than Ettin-400M** on this budget. Stage 2 adds +0.019 over its own stage 1 (more than
  the +0.007 it added to the 400M).
- **The two encoders blend well**: +0.029, with the CI clear of zero. Weights chosen on inner folds: 0.5 for the 1B in
  three folds, 0.3 in one, 1.0 in one. The gain is spread across labels: gpt-5.6-terra 0.36 → 0.47, kimi-k3 0.42 →
  0.49, glm-5.3-flash 0.32 → 0.39, claude-fable-5 0.23 → 0.29; nothing drops by more than 0.03.
- The official-only linear model no longer helps once two pool-trained encoders are in the blend.

Partition 1 (same settings; runs `outputs/encoder/ettin-400m/20261103T230815Z`, 104 min, and
`outputs/encoder/ettin-1b/20261104T005202Z`, 120 min; blend `outputs/blend/20261104T025243Z`):

| Branch | BA p0 | BA p1 | Paired Δ vs Ettin-400M + pool, p0 / p1 |
| --- | --- | --- | --- |
| Ettin-400M + pool | 0.530 | 0.534 [0.514, 0.553] | — |
| Ettin-1B + pool | 0.517 | 0.522 [0.499, 0.544] | −0.013 / −0.012 [−0.032, +0.007] |
| nested blend (weights from inner folds) | 0.559 | 0.545 [0.525, 0.565] | +0.029 / +0.011 [−0.001, +0.024] |
| fixed equal blend (weight 1 after inner scaling) | 0.552 | 0.550 | +0.021 [+0.006, +0.038] / +0.016 [+0.003, +0.029] |

- **The pool-trained encoders reproduce on partition 1**: Ettin-400M + pool 0.534, and stage 2 again adds about
  +0.01 over stage 1 alone (0.523). Ettin-1B is again about 0.012 below the 400M.
- **The inner-chosen blend weight is unstable on p1** (0.3 to 2.0 across folds), and its gain shrinks to +0.011 with
  the CI touching zero. A fixed equal weight gains on both partitions with CIs above zero. Equal weight was chosen
  after seeing the p1 nested result, but it is the default prior rather than a tuned value, so the two-encoder
  average with weight 1 is the recipe to carry forward.
- p1 per-class changes from the nested blend: gpt-5.5 0.51 → 0.58, qwen3.8-max 0.59 → 0.65, gemini-3.1-pro 0.76 →
  0.82; gemini-3.7 0.48 → 0.41 and glm-5.3 0.26 → 0.20 drop.

### Exp 4b: Qwen3-8B with a pool stage 1 on TC2, partition 0 (2026-11-04)

TC2 job 50896 (`nllm_8bpool_p0_c1`, 3 h 48 min on one L40S), run `outputs/tc2/qwen3-8b-pool-p0-001`. Frozen bf16
Qwen3-8B-Base + LoRA r16. Stage 1 per outer fold: 1 epoch at lr 1e-4 on 36,000 rows (recipe `wx_own`, 2,000 per
label), about 29.5 min, peak 31.7 GB. Stage 2: nested, lr 1e-4, up to 3 epochs. Every fold chose epoch 3, the most
allowed, so a longer stage 2 may still gain.

| Partition 0 | BA [95% CI] | Paired Δ vs Ettin-400M + pool | Paired Δ vs equal Ettin-400M + Ettin-1B |
| --- | --- | --- | --- |
| Qwen3-8B pool stage 1 alone | 0.489 [0.469, 0.508] | | |
| Qwen3-8B + pool | 0.507 [0.488, 0.526] | −0.023 [−0.046, −0.000] | |
| equal Ettin-400M + Qwen3-8B | 0.562 | +0.031 [+0.012, +0.051] | +0.010 [−0.007, +0.028] |
| equal Ettin-400M + Ettin-1B + Qwen3-8B | 0.566 | +0.036 [+0.019, +0.053] | +0.015 [+0.001, +0.029] |
| nested three-way blend (`outputs/blend/20261104T043243Z`) | 0.569 [0.548, 0.588] | +0.039 [+0.022, +0.055] | |

- **The 8B alone is below Ettin-400M** with the same kind of pool stage 1, and costs about 15× the training time.
- **It adds to the encoders**, +0.015 over the equal two-encoder blend with the CI just above zero, on partition 0
  only. The three-way blend's largest gains over Ettin-400M are gpt-5.6-terra 0.36 → 0.48, kimi-k3 0.42 → 0.51,
  claude-fable-5 0.23 → 0.31; gemini-3.6-flash alone falls to 0.48 under the 8B but the blend keeps 0.68.
- Kaggle cost: bf16 8B weights (about 16 GB) do not fit one T4, so inference needs both T4s or quantised weights,
  and the agreement check against the pilot reference still has to be done.

Partition 1 (job 50995, `nllm_8bpool_p1`, 3 h 47 min, run `outputs/tc2/qwen3-8b-pool-p1-001`; stage 1 about 30 min
per fold, peak 30.7 GB; epoch 3 chosen in four folds). Equal weights after inner scaling throughout:

| Partition 1 | BA | Paired Δ vs Ettin-400M | Paired Δ vs Ettin-400M + Ettin-1B | Paired Δ vs Ettin-400M × 2 + Ettin-1B |
| --- | --- | --- | --- | --- |
| Qwen3-8B pool stage 1 alone | 0.471 | | | |
| Qwen3-8B + pool | 0.508 [0.489, 0.527] | −0.026 [−0.047, −0.005] | | |
| Ettin-400M + Qwen3-8B | 0.554 | +0.020 [+0.002, +0.037] | +0.004 [−0.013, +0.021] | |
| Ettin-400M + Ettin-1B + Qwen3-8B | 0.568 | +0.034 [+0.017, +0.051] | +0.018 [+0.003, +0.032] | +0.006 [−0.008, +0.021] |
| Ettin-400M × 2 seeds + Ettin-1B (Exp 6f) | 0.562 | +0.028 [+0.014, +0.042] | +0.012 [+0.003, +0.022] | — |
| all four | 0.572 | +0.039 [+0.022, +0.054] | +0.023 [+0.010, +0.036] | +0.011 [−0.001, +0.022] |

- **Partition 1 repeats partition 0** to within 0.002: 8B alone 0.508 (0.507), three-way 0.568 (0.566), all four
  0.572 (0.575).
- **The 8B adds about what one more encoder run adds.** In place of a second 400M seed it gains +0.006 (not
  significant). On top of three encoder runs it gains +0.011 on p1 and +0.012 [−0.001, +0.025] on p0, with both CIs
  touching zero. It is the weakest single model on gemini-3.7-flash (0.28) and gpt-5.6-terra (0.34).
- A third Ettin-400M seed (Exp 6f) later added less than the 8B does (+0.007 / +0.004 on top of three encoder
  runs), so seeds alone do not replace it. The 8B belongs in the final blend if two-T4 or quantised inference fits
  the Kaggle time limit.

System 1's official-only 8B run (jobs 50897 + 50987, 5 h 42 min in two chunks, run `outputs/tc2/qwen3-8b-base-p0-001`):
nested, lr 1e-4/2e-4, up to 8 epochs, no pool data. The fold-0 gate passed (inner BA 0.355 ≥ 0.35, peak 27.9 GB ≤ 40,
9.6 min per fit ≤ 30). Selections sit at the top of the grid again: lr 2e-4 in four folds, epoch 7 or 8 of 8.

| Partition 0, official data only | BA [95% CI] | Paired Δ vs Ettin-400M (Exp 3) |
| --- | --- | --- |
| Ettin-400M, lr 1e-5/3e-5, 10 epochs (Exp 3) | 0.388 [0.367, 0.409] | — |
| Ettin-400M, lr 3e-5/5e-5, 15 epochs (`outputs/encoder/ettin-400m/20261102T165538Z`) | 0.396 | +0.009 [−0.010, +0.027] |
| Qwen3-8B-Base LoRA | 0.396 [0.375, 0.415] | +0.008 [−0.015, +0.032] |
| equal Ettin-400M + Qwen3-8B | 0.432 | +0.044 [+0.027, +0.062] |

- **On official data alone the 8B ties Ettin-400M** at about 30× the training time. The two blend well (+0.044), as
  the Exp 3 encoders did, but that is the official-only regime.
- **Pool data matters far more than model size**: the same 8B with a pool stage 1 (above) scores 0.507, +0.111.
- Adding the official-only 8B to the four pool-trained runs lowers BA, 0.575 → 0.565, a paired change of −0.010
  [−0.024, +0.004]. Official-only decoders are dropped; any further 8B work uses the pool stage 1.

### Exp 6f: more Ettin-400M seeds, and how far averaging runs goes (2026-11-04)

The Exp 6c recipe again with `--seed 43` (run `outputs/encoder/ettin-400m/20261104T025354Z`, 109 min locally).
The seed changes initialisation and data order only: the pool sample is seeded by fold, so both seeds saw the same
pool rows. All blends below give each run weight 1 after its inner scale, with no tuning.

| Partition 0 | BA | Paired Δ vs Ettin-400M seed 17 | Paired Δ vs Ettin-400M + Ettin-1B |
| --- | --- | --- | --- |
| Ettin-400M seed 17 (Exp 6c) | 0.530 | — | −0.021 [−0.038, −0.006] |
| Ettin-400M seed 43 | 0.518 | −0.013 [−0.032, +0.007] | −0.034 [−0.052, −0.016] |
| Ettin-400M seeds 17 + 43 | 0.549 | +0.019 [+0.003, +0.033] | −0.003 [−0.019, +0.013] |
| Ettin-400M + Ettin-1B (Exp 6e) | 0.552 | +0.021 [+0.006, +0.038] | — |
| Ettin-400M × 2 seeds + Ettin-1B | 0.563 | +0.033 [+0.016, +0.050] | +0.012 [−0.001, +0.024] |
| Ettin-400M + Ettin-1B + Qwen3-8B (Exp 4b) | 0.566 | +0.036 [+0.019, +0.053] | +0.015 [+0.001, +0.029] |
| all four | 0.575 | +0.044 [+0.027, +0.062] | +0.023 [+0.009, +0.038] |

- **Seed noise is as large as the 400M-vs-1B gap**: the two 400M seeds differ by 0.013, the same as Ettin-1B's
  deficit. Neither size difference is measurable at this budget.
- **The two-encoder gain is mostly from averaging two runs**: two 400M seeds gain +0.019, nearly the +0.021 of
  400M + 1B. Each further run still adds about +0.01, and the 8B adds about as much as one more encoder run.
- Several Ettin-400M seeds are the cheapest way to get this (about 2 h each locally, and fast at Kaggle inference);
  whether the 8B is worth its T4 cost depends on whether it adds more than another encoder seed does.

Partition 1: seed 43 ran on TC2 (job 50988, `nllm_e400pool_s43_p1`, 1 h 18 min on the L40S, stage 1 about 10 min
per fold; run `outputs/tc2/ettin-400m-pool-s43-p1-001`). The partition 1 runs for seed 17 and Ettin-1B are from Exp 6e.

| Partition 1 | BA | Paired Δ vs Ettin-400M seed 17 | Paired Δ vs Ettin-400M + Ettin-1B |
| --- | --- | --- | --- |
| Ettin-400M seed 17 | 0.534 | — | −0.016 [−0.029, −0.003] |
| Ettin-400M seed 43 | 0.529 | −0.005 [−0.025, +0.014] | −0.021 [−0.040, −0.002] |
| Ettin-400M seeds 17 + 43 | 0.546 | +0.012 [−0.001, +0.024] | −0.005 [−0.019, +0.011] |
| Ettin-400M + Ettin-1B | 0.550 | +0.016 [+0.003, +0.029] | — |
| Ettin-400M × 2 seeds + Ettin-1B | 0.562 | +0.028 [+0.014, +0.042] | +0.012 [+0.003, +0.022] |

- **Partition 1 repeats partition 0**: the three-run average is 0.562 (p0 0.563), and the third run adds +0.012 on
  both partitions. On p1 its CI clears zero.
- A second 400M seed gains slightly less than adding the 1B on both partitions (−0.003 / −0.005, not significant), so
  the 1B may add a little diversity of its own. The Kaggle ensemble should mix sizes rather than only stack seeds.
- Per class, the three-run average gains most on gpt-5.5 (0.51 → 0.62), gemini-3.1-pro (0.76 → 0.82), qwen3.8-max
  (0.59 → 0.67), gpt-6-astra (0.72 → 0.78) and kimi-k3 (0.47 → 0.52). It loses on gemini-3.7-flash (0.48 → 0.44),
  glm-5.3 (0.26 → 0.23) and claude-fable-5 (0.24 → 0.21).

Third seed, both partitions: `--seed 44` on TC2 (job 51039, `nllm_e400pool_s44`, 2 h 35 min for both partitions;
run `outputs/tc2/ettin-400m-pool-s44-001`). Equal weights after inner scaling, as above.

| Blend | p0 BA | p0 Δ vs 400M × 2 + 1B | p1 BA | p1 Δ vs 400M × 2 + 1B |
| --- | --- | --- | --- | --- |
| Ettin-400M seed 44 alone | 0.531 | | 0.536 | |
| Ettin-400M × 3 seeds | 0.551 | −0.012 [−0.023, −0.001] | 0.558 | −0.004 [−0.017, +0.009] |
| Ettin-400M × 2 + Ettin-1B | 0.563 | — | 0.562 | — |
| Ettin-400M × 3 + Ettin-1B | 0.570 | +0.007 [−0.001, +0.015] | 0.565 | +0.004 [−0.006, +0.013] |
| Ettin-400M × 2 + Ettin-1B + Qwen3-8B | 0.575 | +0.012 [−0.001, +0.025] | 0.572 | +0.011 [−0.001, +0.022] |
| Ettin-400M × 3 + Ettin-1B + Qwen3-8B | 0.575 | +0.012 [+0.000, +0.024] | 0.579 | +0.017 [+0.006, +0.029] |

- **Seed 44 matches seed 17** (0.531 / 0.536 against 0.530 / 0.534); seed 43 was the low one.
- **More 400M seeds give less each time.** The third seed adds +0.002 (p0) and +0.012 (p1) to the two-seed average,
  and +0.007 / +0.004 to 400M × 2 + 1B, neither significant.
- **The 8B adds more than a third 400M seed does.** Swapped in for seed 44 it is ahead by +0.005 / +0.007 (CIs cross
  zero), and adding seed 44 to the four-run blend changes BA by +0.000 [−0.008, +0.008] / +0.007 [−0.002, +0.015].
  A different model adds more than another seed of the same one, so the next run for the ensemble should be another
  architecture or a second Ettin-1B seed, not a fourth 400M seed.
- The best blend so far is all five runs, 0.575 / 0.579, the only one whose gain over 400M × 2 + 1B clears zero on
  both partitions.
- Single seeds differ a lot per class: seed 44 reaches 0.39 / 0.41 on claude-fable-5 (seed 17: 0.23 / 0.24) but only
  0.09 / 0.11 on glm-5.3 (0.22 / 0.26).

Second Ettin-1B seed: `--seed 43` with the Exp 6e recipe, both partitions on TC2 (job 51148, `nllm_e1bpool_s43`,
3 h 15 min, about 20 min per fold; run `outputs/tc2/ettin-1b-pool-s43-001`). Alone it scores 0.517 / 0.531 (seed 17:
0.517 / 0.522), and its pool stage 1 alone 0.490 / 0.501.

| Blend | p0 BA | p0 Δ | p1 BA | p1 Δ |
| --- | --- | --- | --- | --- |
| Ettin-1B seeds 17 + 43 | 0.530 | | 0.536 | |
| 400M × 3 + 1B | 0.570 | — | 0.565 | — |
| 400M × 3 + 1B × 2 | 0.560 | −0.010 [−0.018, −0.002] | 0.570 | +0.005 [−0.003, +0.013] |
| 400M × 3 + 1B + 8B | 0.575 | — | 0.579 | — |
| 400M × 3 + 1B × 2 + 8B | 0.570 | −0.005 [−0.013, +0.002] | 0.588 | +0.009 [+0.001, +0.017] |

- **A second 1B seed does not help reliably.** It costs 0.010 on partition 0, with a CI clear of zero, and gains
  0.005-0.009 on partition 1, so the two partitions disagree in sign. Exp 6f's rule that a different model adds
  more than another seed holds here too: the gain came from adding the 1B, not from 1B seeds.
- Equal-weight blends of the encoders flatten out at about 0.565-0.570. Only the 8B adds a gain that is positive on
  both partitions (+0.005 / +0.014 on top of 400M × 3 + 1B), though only the p1 CI clears zero.
- The Kaggle ensemble stays 400M × 3 + 1B (seed 17), whose final fits are in TC2 job 51187. The 1B seed 43 gets no
  final fit.

## First Kaggle model: Ettin-400M + pool on all 120 dev prompts (2026-11-04)

For a first Kaggle submission, to test the pipeline and how CV maps to the leaderboard, not to pick a model.
TC2 job 51074 (`nllm_e400_final`, 11 min on the L40S), `scripts/encoder_finetune.py --final`, run
`outputs/tc2/ettin-400m-final-001`. The recipe is the Exp 6c one, with the schedule fixed from the nested runs (lr
1e-5 in all 20 fold selections, median epoch 3): pool stage 1 on 45,000 rows (recipe `w`, 2,500 per label, 2 epochs
at 3e-5, 593 s), then stage 2 on all 2,159 dev rows at lr 1e-5 with a 6-epoch schedule stopped after epoch 3, seed 17.
The 30 confirmation prompts were not used. Its expected BA is about that of one Ettin-400M seed in nested CV,
0.52-0.54 on the dev prompts.

`scripts/kaggle_package.py` builds `outputs/kaggle/ettin-400m-final-001/` (dataset 1.48 GiB, `nllm_infer.ipynb`).
The notebook rescores 72 reference training rows (4 per label, read from the competition's `train.csv`) and refuses
to write a submission if fewer than 95% of predictions agree with the training-time scores. On the CPU, fp32 scores
match TC2's to within 0.0003 under transformers 5.22.1 and 4.57.1.

The checkpoint's `config.json` comes from transformers 5 and stores RoPE settings only as `rope_parameters`.
transformers 4.57.1 ignores that key and runs the sliding-window layers at its default theta of 10,000 instead of
Ettin's 160,000. The model still loads, but only 81% of reference predictions agree (max logit difference 8.3). The
package therefore also writes 4.x's `global_rope_theta` / `local_rope_theta` keys.

Kaggle run (2026-11-04): the package was uploaded with the Kaggle CLI as the private dataset
`rhysbao/nllm-ettin-400m-final-001`. Notebook `rhysbao/nllm-ettin-400m-final` version 1 ran on a T4 with internet off,
under Python 3.13.16, torch 2.11.0+cu130, transformers 5.20.2 and tokenizers 0.23.3. Reference rows: argmax
agreement 1.000, max |diff| 0.0006 against TC2's fp32 scores. Before the reference check, 221 s went mostly on
loading the 1.58 GB weights. The 3 public test rows scored in under a second. `submission.parquet` (`id`, `model`
as `large_string`) is identical to the local CPU dry run.

Submitted as 57990133 (2026-11-04 23:42, scored in about 6 min): **public LB 0.546**. Single Ettin-400M runs score
0.518-0.536 in nested CV on 96 training prompts per fold, and this model trained on all 120. The public score is at or
slightly above that range, so this first look shows no CV-to-LB drop. One public split with an unknown, unequal class
mix is a weak check, and the private split may differ.

## Kaggle ensemble: member scales without inner folds (2026-11-05)

The equal blends divide each run's outer scores by its inner scale for that fold (`linear.scale`: std of
row-centred scores on the inner out-of-fold rows). A final model trained on all 120 prompts has no inner folds, so
the Kaggle notebook divides each member by a constant: the mean of that member's inner scales over the 10 folds of
its nested runs (both partitions). For Ettin-400M seeds 17 / 43 / 44 and Ettin-1B seed 17 these are 6.130, 6.094,
5.910 and 6.384. Fold-to-fold variation is 11-15%, and outer scales are within 7% of inner ones.

| Ettin-400M × 3 + Ettin-1B | p0 BA | Δ vs per-fold scales | p1 BA | Δ vs per-fold scales |
| --- | --- | --- | --- | --- |
| per-fold inner scales (as in Exp 6f) | 0.570 | — | 0.565 | — |
| one constant scale per member | 0.568 | −0.002 [−0.007, +0.003] | 0.562 | −0.004 [−0.009, +0.001] |
| scale computed on the outer rows themselves | 0.570 | −0.000 [−0.003, +0.003] | 0.564 | −0.001 [−0.005, +0.001] |
| no scaling | 0.570 | −0.001 [−0.006, +0.004] | 0.563 | −0.002 [−0.007, +0.002] |

Scaling barely matters among these four, because their scales are within 8% of one another. The notebook uses the
constant per-member scales, which keeps every statistic out of the test rows. The scale step matters again if the
8B joins, since its logits may have a different spread.

## Second Kaggle submission: four-model encoder ensemble (2026-11-05)

Final fits on all 120 dev prompts, one TC2 job (51187, `nllm_finals_ens1`, 39 min): Ettin-1B (Exp 6e recipe, epoch
3) and Ettin-400M seeds 43 and 44 (the first model's recipe). With the seed-17 model they form the package
`outputs/kaggle/ens-400mx3-1b-001` (8.28 GiB, private dataset `rhysbao/nllm-ens-400mx3-1b-001`), whose scales are the
constants above. Notebook `rhysbao/nllm-ensemble-400mx3-1b` version 1 ran on Kaggle's GPU T4 x2 in 422 s. All four
members reached reference agreement 1.000 (max |diff| ≤ 0.0006).

| Submission | CV p0 / p1 | Public LB |
| --- | --- | --- |
| 57990133: Ettin-400M seed 17 alone | 0.530 / 0.534 (seed 17's nested run) | 0.546 |
| 57996685: Ettin-400M × 3 + Ettin-1B, constant scales | 0.568 / 0.562 | 0.566 |

The ensemble gains +0.020 on the public LB, against +0.03 to +0.04 in CV. Both submissions sit at or slightly above
their CV, so CV still ranks them correctly. The two leaderboard scores come from one public split with an unknown
class mix, so their difference is a single noisy observation.

### Qwen3-8B final fit and its Kaggle path (2026-11-05)

TC2 job 51236 (`nllm_8b_final`, 36 min), run `outputs/tc2/qwen3-8b-final-001`: Exp 4b's recipe on all 120 dev
prompts. Stage 1 took 30 min on 36,000 pool rows (`wx_own`, 2,000 per label, 1 epoch at 1e-4). Stage 2 ran on 2,159
rows at 1e-4 and stopped at epoch 3 of 3, taking about 5.5 min. The run saves only the LoRA adapters and head
(175 MB). For the reference rows, the adapters were merged into a fresh copy of the base weights: fp16 and bf16
argmax agreement is 1.000 (max |diff| 0.50), so fp16 inference on a T4 keeps the predictions.

On Kaggle the notebook merges the adapters by hand (`inference.load_decoder`, no peft) into the base weights from the
private dataset `rhysbao/qwen3-8b-base-49e3418f`. That dataset is the unmodified Hugging Face snapshot at revision
49e3418f, the revision TC2 trained on, plus a `REVISION` file. The model runs in fp16 split across the two T4s
(`device_map="auto"`). On a tiny Qwen3, the hand merge matches peft's fp16 scores to within 0.01
(tests/test_final.py). The 8B's ensemble scale is its mean inner scale, 4.741, lower than the encoders' 5.9-6.4.
With constant scales the five-model blend scores 0.575 / 0.575 in CV, against 0.568 / 0.562 for the four
encoders. Per-fold scales give 0.575 / 0.579, so constant scales cost 0.004 on p1.

Five-model notebook (`rhysbao/nllm-ensemble-5`, datasets `rhysbao/nllm-ens5-001` + `rhysbao/qwen3-8b-base-49e3418f`),
version 1 (2026-11-05 04:36-09:10). All five members passed the reference check; the 8B's fp16 scores on two T4s
agree with TC2's fp16 reference at 1.000 (max |diff| 0.047). The run took 4.6 h, almost all of it loading. Each
encoder took 210-230 s against about 100 s in the four-model run. The 8B's `from_pretrained(device_map="auto")`
on the 16 GB base took 4 h 15 min, read in place from the `/kaggle/input` mount. The hand merge, 72 reference rows
and 3 test rows then took under 3 min. Version 2 copies every model directory to local disk (`/tmp/nllm_stage`)
before loading and deletes the copy afterwards.
Version 2, which stages each model to local disk, ran in 305 s. The encoders copied at about 210 MB/s (1.48 GiB in
7 s) and loaded in under 20 s. The 8B base copied in 96 s (15.27 GiB) and loaded in about 100 s, and the 72
reference rows took about 11 s, so the 4,000 hidden rows should take about 10 min. Agreement was 1.000 for all
five members. Submitted as 58004957.

| Submission | CV p0 / p1 (constant scales) | Public LB |
| --- | --- | --- |
| 57990133: Ettin-400M seed 17 | 0.530 / 0.534 | 0.546 |
| 57996685: Ettin-400M × 3 + Ettin-1B | 0.568 / 0.562 | 0.566 |
| 58004957: Ettin-400M × 3 + Ettin-1B + Qwen3-8B (fp16, two T4s) | 0.575 / 0.575 | **0.582** |

The 8B adds +0.016 on the public LB, against +0.007 / +0.013 in CV. Each step up in CV has also gone up on the LB,
and each LB score is within about 0.01 of its CV, so CV still orders the submissions correctly. Scoring the hidden
test took about 32 min. Per Fable's advice (the 8B's gain is near the forum's estimate of LB noise between models),
keep the four-encoder ensemble as the other final selection unless something better replaces it.

### Exp 6g: Ettin-400M with 5,000 pool rows per label (2026-11-05)

TC2 job 51237 (`nllm_e400pool_n5000`, 4 h 04 min, both partitions in one chunk), run
`outputs/tc2/ettin-400m-pool5000-001`. This is the Exp 6c recipe (recipe `w`, 2 stage-1 epochs at 3e-5, stage-2 lr
∈ {1e-5, 3e-5} on a 6-epoch schedule, seed 17) with only one change: 5,000 pool rows per label instead of 2,500.
gemini-3.7-flash (3,278 pool rows) and qwen3.8-max (4,271) run out of rows, so stage 1 has 87,549 rows rather than
90,000. Stage 1 took about 19 min per fold on the L40S, peaking at 11.6 GB.

| Branch | p0 BA [95% CI] | Paired Δ vs 2,500 per label | p1 BA [95% CI] | Paired Δ vs 2,500 per label |
| --- | --- | --- | --- | --- |
| Ettin-400M + pool, 2,500 per label (Exp 6c, seed 17) | 0.530 | — | 0.534 | — |
| Ettin-400M + pool, 5,000 per label | **0.566** [0.543, 0.588] | +0.036 [+0.018, +0.054] | **0.569** [0.546, 0.591] | +0.035 [+0.016, +0.053] |
| its pool stage 1 alone (2,500: 0.523 / 0.523) | 0.557 | | 0.550 | |

- **Doubling the pool data is the largest single gain since the pool itself.** The gain is the same on both
  partitions, and both CIs exclude zero. A single 400M now matches the four-encoder ensemble (0.568 / 0.562). Most
  of it comes from stage 1 (+0.034 / +0.027), and stage 2 still adds +0.010 / +0.018.
- lr 1e-5 was chosen in all 10 folds (the lowest value in the grid), with epochs 1-3 (median 2), earlier than the
  median of 3 at 2,500 per label.
- Per class, gemini-3.7-flash rose from 0.24 to 0.39 (p0), but gemini-3.8-flash fell from 0.76 to 0.59. 3.7 runs out of
  pool rows while 3.8 gets the full 5,000, so the stage-1 prior moved rather than the pair separating. glm-5.3
  (0.15 / 0.28), claude-fable-5 (0.33 / 0.27) and glm-5.3-flash (0.34 / 0.33) are still the weakest classes.
- Fable's read of the confusions: about 60% of errors stay within one family, in four clusters: gemini-3.7 ↔ 3.8
  (3.7 → 3.8 is the largest single cell, 0.37 / 0.43 of 3.7 rows), the gpt-5.6 sol / luna / terra triangle,
  glm-5.3 ↔ glm-5.3-flash, and one large leak across families, claude-fable-5 ↔ both GLMs.

## Third Kaggle ensemble: the 5,000-per-label 400M replaces seed 17 (2026-11-05)

Final fit: TC2 job 51359 (`nllm_e400_final_n5000`, 20 min), run `outputs/tc2/ettin-400m-final-n5000-001`. It uses
the Exp 6g recipe on all 120 dev prompts, with stage 2 at lr 1e-5 stopped at epoch 2 (the median of the nested
selections), seed 17. Stage-1 loss fell from 1.52 to 0.78 over the 2 epochs. On the reference rows, fp32 and bf16
agree on argmax for 0.986 of rows (1 of 72), and the earlier finals agreed on all of them. The Kaggle check compares
against fp32, so this does not affect the notebook. Its constant scale (the mean inner scale over its 10 nested
folds) is 6.203. Recomputed the same way, the other members' scales match the recorded 6.130 / 6.094 / 5.910 /
6.384 / 4.741 exactly.

Whether to replace or add, as equal blends with constant scales, paired against the current five (0.575 / 0.575):

| Blend | p0 BA | Δ | p1 BA | Δ |
| --- | --- | --- | --- | --- |
| current five (Ettin-400M × 3 + Ettin-1B + Qwen3-8B) | 0.575 | — | 0.575 | — |
| **5,000-per-label 400M replaces seed 17** | **0.584** | +0.009 [+0.001, +0.017] | **0.596** | +0.021 [+0.011, +0.032] |
| 5,000-per-label 400M added as a sixth (per-fold scales; Δ vs the five at per-fold scales, 0.575 / 0.579) | 0.580 | +0.005 [−0.002, +0.013] | 0.590 | +0.011 [+0.003, +0.019] |

With per-fold scales, replacing scores 0.584 / 0.596 (Δ +0.009 [+0.001, +0.018] / +0.017 [+0.006, +0.027]).

Replacing beats adding. The new model agrees with seed 17 on 69.6% of rows, against 66.1% for seed 43 vs seed 17:
it is a stronger version of seed 17 rather than a different model. Per class (both partitions pooled), replacing
gains most on gpt-5.6-sol (+0.046), glm-5.3-flash (+0.046), gemini-3.6-flash (+0.042), glm-5.3 (+0.029) and
claude-fable-5 (+0.025). No class drops by more than 0.013 (gemini-3.8-flash).

Package `outputs/kaggle/ens5-n5k-001` (8.5 GiB, private dataset `rhysbao/nllm-ens5-n5k-001`, uploaded with the
`kaggle` CLI 2.2.5 via `uvx --with six kaggle`, `datasets create -r tar`). The notebook source is unchanged from
`rhysbao/nllm-ensemble-5` v2. A local CPU dry run that included the 8B took 28 min, and all five members agreed at
1.000. Notebook `rhysbao/nllm-ensemble-5-n5k` version 1 ran on 2×T4 in 346 s, with all five members agreeing with
TC2 at 1.000 (max |diff| 0.0005 for the encoders, 0.047 for the 8B in fp16). Its 3-row `submission.parquet` is
identical to the dry run's. Submitted as 58026808.

| Submission | CV p0 / p1 (constant scales) | Public LB |
| --- | --- | --- |
| 57990133: Ettin-400M seed 17 | 0.530 / 0.534 | 0.546 |
| 57996685: Ettin-400M × 3 + Ettin-1B | 0.568 / 0.562 | 0.566 |
| 58004957: Ettin-400M × 3 + Ettin-1B + Qwen3-8B | 0.575 / 0.575 | 0.582 |
| 58026808: 400M (5,000 per label) + 400M s43, s44 + Ettin-1B + Qwen3-8B | 0.584 / 0.596 | **0.590** |

CV and the LB still rise together. The +0.008 on the LB is within the 0.009-0.013 LB noise between models that
the forum estimates, so the CV gain is the stronger evidence. Final selections for now: 58026808 and 58004957,
chosen by CV.

Next steps, ranked by Fable from the confusions above:
1. **Scale the 400M's pool further (about 4-6 h on TC2).** At 10,000 per label several more labels run out of rows
   (opus-5 6,288, opus-4-6 6,799, opus-4-8 7,507), so use recipe `wx`, class-balanced sampling with replacement or
   class weights, and add lr 5e-6 to the stage-2 grid. A cheaper arm is 5,000 per label for 3 epochs. Then re-seed
   the winner to replace seeds 43 and 44.
2. **Retrain Ettin-1B with the 5,000-per-label recipe.** It was trained on 1,500 per label for 1 epoch and is the
   weakest member. Leave the 8B as it is; a nested 8B run at 5,000 per label would take about 12 h per partition.
3. **A low-cost specialist for claude-fable-5 / glm-5.3 / glm-5.3-flash**, judged on 18-class BA. Spend no GPU on
   gemini-3.7 ↔ 3.8: Fable measured a within-source AUC of 0.59 for that pair, near its ceiling.

### Exp 4c: Qwen3-8B with the public LoRA24 recipe, partition 0 (2026-11-06)

Why the public notebooks beat us is in `notes/why_public_beats_us_2026-11-05.md`. The main finding: at the same
data, the public "LoRA24" decoder recipe scores about 0.06-0.08 higher than ours. It is now `encoder.PooledDecoder`
plus `TrainConfig` options (`pool_layers`, `keep_layers`, `head_lr_mult`, `centred_weight`, `class_weighted`,
`schedule`), and `encoder_finetune.py --pool-layers ... --shared-stage1`:
- the first 24 layers only, with the final norm off;
- hidden states mean-pooled (first token masked) at layers 2, 4, 6, 8, 12, 16, 20 and 24, each through its own
  LayerNorm, into a linear head;
- batches of same-prompt triplets (18 rows), with a second head trained on triplet-centred features (loss weight 1);
- heads at 10× the LoRA lr, cosine decay.

Local A/B on one inner fold (Qwen3-1.7B, official rows only, 1,151 train / 576 val rows): our last-token recipe
peaked at 0.306 inner-val BA (lr 2e-4, epoch 8). LoRA24 reached 0.391 (lr 1e-4, epoch 6, still rising).

TC2 job 51519 (`nllm_8b_l24_p0`, 3 h 31 min, run `outputs/tc2/qwen3-8b-l24-p0-001`) keeps Exp 4b's per-fold stage 1
(`wx_own`, 2,000 rows per label, 1 epoch at 1e-4, now with the LoRA24 model) and changes stage 2 to lr 5e-5 on a
3-epoch cosine schedule. Stage 1 took 28 min per fold and each stage-2 fit 3-4.5 min, with a peak of 25.6 GB.

| Partition 0 | BA [95% CI] | Paired Δ vs Exp 4b 8B |
| --- | --- | --- |
| Qwen3-8B + pool, Exp 4b (last token) | 0.507 | — |
| Qwen3-8B LoRA24, pool stage 1 alone | 0.519 [0.498, 0.537] | |
| **Qwen3-8B LoRA24 + pool** | **0.550 [0.530, 0.570]** | **+0.043 [+0.025, +0.062]** |

Blends on partition 0. The encoder block is n5000 400M + 400M s43 + s44 + 1B at equal weights, as in the current
submission. Each 8B weight is chosen per outer fold on inner OOF BA from `linear.BLEND_WEIGHTS`. Δ is paired against
block + old 8B.

| Blend | p0 BA | Δ |
| --- | --- | --- |
| encoder block alone | 0.571 | |
| block + old 8B | 0.587 | — |
| block + LoRA24 8B (inner weights 1.5-5) | 0.584 | −0.003 [−0.015, +0.010] |
| block + old 8B + LoRA24 8B | 0.589 | +0.002 [−0.009, +0.013] |
| n5000 400M + LoRA24 8B | 0.595 | +0.009 [−0.006, +0.024] |

- **The recipe works.** Alone, the 8B gains +0.043 with the CI well clear of zero. The largest per-class gains are
  gpt-5.6-terra (+0.19), gemini-3.6 (+0.15), gemini-3.7 (+0.12), claude-opus-4-8 (+0.11) and gpt-5.6-sol (+0.10).
  glm-5.3 (−0.05) and claude-fable-5 (−0.02) did not move. 0.550 is about the public recipe's 0.569 (their folds
  train on 120 prompts, ours on 96).
- **It does not add to our blend yet.** It agrees with the n5000 400M on 60% of rows, against 53% for the old 8B,
  so it repeats more of what the encoders already know. On the inner folds it looks valuable (weights 1.5-5), but
  the outer scores do not move. This matches the forum's diversity finding: a stronger member trained on the same
  data adds little.
- Every fold chose epoch 3, the most allowed, so a longer stage 2 may still gain. Partition 1 was not run.
- The real candidate is the full-pool version (job 51520, `nllm_8b_l24full`): one class-weighted stage 1 over
  ~212k pool + extra rows, shared by all folds, then the nested stage 2 on both partitions. That tests whether
  6× more data lifts the 8B as it lifted the 400M (+0.036 per doubling).

### Exp 4d: Qwen3-8B LoRA24 on the whole pool, one shared stage 1, both partitions (2026-11-06)

TC2 job 51520 (`nllm_8b_l24full`, 5 h 03 min in one chunk, run `outputs/tc2/qwen3-8b-l24-full-001`).
- **Stage 1:** one class-weighted epoch at lr 1e-4 over every Open Pool + extra row (recipe `ux`: uniform, so it
  reads no official rows, and every dev prompt is unseen by it). That is 211,755 rows, 3,788-22,156 per label,
  taking 2 h 42 min at a peak of 29.8 GB. It is shared by all 10 outer folds.
- **Stage 2:** nested as in Exp 4c (LoRA24 model, lr 5e-5, 3-epoch cosine). Epoch 3 was chosen in 9 of 10 folds.

| | p0 BA [95% CI] | p1 BA [95% CI] |
| --- | --- | --- |
| Qwen3-8B, Exp 4b (last token, 36k pool rows per fold) | 0.507 | 0.508 |
| Ettin-400M, 5,000 per label (Exp 6g, previous best single model) | 0.566 | 0.569 |
| current five-model ensemble (submission 58026808, LB 0.590) | 0.584 | 0.596 |
| full-pool LoRA24 stage 1 alone | 0.588 [0.567, 0.607] | 0.588 [0.568, 0.608] |
| **full-pool LoRA24 Qwen3-8B** | **0.623 [0.602, 0.645]** | **0.624 [0.603, 0.645]** |

Paired differences (prompt bootstrap, 2,000 draws):

| Comparison | p0 Δ | p1 Δ |
| --- | --- | --- |
| full-pool 8B vs Exp 4b 8B | +0.117 [+0.096, +0.138] | +0.116 [+0.095, +0.139] |
| full-pool 8B vs n5000 400M | +0.057 [+0.036, +0.078] | +0.055 [+0.034, +0.076] |
| full-pool 8B vs current five-model ensemble | +0.039 [+0.021, +0.058] | +0.028 [+0.008, +0.048] |

Blends, paired against the current five (per-fold inner scales; "inner" weights chosen per outer fold on inner OOF
BA from `linear.BLEND_WEIGHTS`):

| Blend | p0 BA | Δ | p1 BA | Δ |
| --- | --- | --- | --- | --- |
| full 8B alone | 0.623 | +0.039 | 0.624 | +0.028 |
| full 8B + n5000 400M (400M weight on inner: 0.3-1.0) | 0.627 | +0.042 [+0.028, +0.058] | 0.624 | +0.028 [+0.010, +0.046] |
| full 8B + n5000 400M, equal | 0.624 | +0.040 | 0.616 | +0.021 |
| encoder block + full 8B (8B weight on inner: 3-5) | 0.622 | +0.038 | 0.623 | +0.028 |
| current five, full 8B in place of old 8B, equal | 0.599 | +0.015 | 0.602 | +0.006 |
| current five + full 8B, equal | 0.602 | +0.018 | 0.613 | +0.017 |

- **The data volume did what it did for the 400M, on a stronger model.** On partition 0 the same recipe gains +0.073
  going from 36k pool rows per fold (Exp 4c, 0.550) to the whole pool. Exp 4c did not run partition 1. Stage 1 alone (0.588) already beats every
  earlier model. Both partitions agree to within 0.001.
- **One model now beats our ensemble.** Blends add at most +0.004 on top of the 8B. Equal-weight blends that give
  the weaker encoders a full vote dilute it, and inner-fold selection gives the 8B weight 3-5 against the encoder
  block.
- Per class (both partitions pooled) against the n5000 400M: gpt-5.6-terra +0.14, gemini-3.7 +0.13, claude-opus-4-8
  +0.13, claude-fable-5 +0.12, glm-5.3-flash +0.11. gemini-3.8 is −0.05, and glm-5.3 (0.26) and claude-fable-5
  (0.42) are still the weakest. It agrees with the n5000 400M on 62.7% of rows.
- Mean inner scale over the 10 folds is 5.986 (for a constant-scale blend).
- **Not yet on Kaggle.** `encoder_finetune.py --final` refuses pooled decoders until `nllm.inference` can load one
  (hand-merged adapters on the truncated base, LayerNorm heads), and the final fit must retrain its stage 1. The
  nested run deleted its shared stage 1, since `--keep-stage1` was not set.

## Fourth Kaggle submission: the full-pool LoRA24 8B alone (2026-11-08)

TC2 job 52142 (`nllm_8b_l24full_final`, 2 h 48 min) trained Exp 4d's recipe on all 120 dev prompts: the same shared
uniform stage 1 (211,755 rows, 2 h 42 min, kept under `stage1/`), then stage 2 stopped at epoch 3. Reference rows
through the peft-free loader: fp16 vs bf16 argmax agreement 1.000, max |diff| 0.125. Package `q8-l24-full-001`
(0.12 GiB plus the 8B base dataset) ran on Kaggle's T4s in 248 s with reference agreement 1.000.

Submission 58101792: **public LB 0.599**, against nested CV 0.623 / 0.624 and LB 0.590 for the five-model ensemble
(CV 0.584 / 0.596). The CV-to-LB drop (−0.024) is larger than the earlier submissions' (−0.007 to +0.007), but the
public split is 40% of a test with its own class mixture, and repeat submissions vary by up to 0.009.

## Exp 8: test-time soft centring inside content clusters (plan E6, 2026-11-08)

Passages answering the same prompt share a topic offset in a model's scores. `nllm.centring` clusters passages by
content (stop-word-free sublinear TF-IDF, cosine distance, average linkage, cut c, fitted on the passages being
scored) and subtracts w × the mean score of each row's cluster mates. The "cavity" mean leaves the row itself out,
because with two or three answers per prompt the plain mean is mostly the row's own author signal (forum 750213).
The hidden test's answers per prompt are unknown and probably few, so `scripts/test_time_centring.py` cuts every
prompt to m random answers (5 draws per m) and recovers the clusters on that cut. Per outer fold, (w, cavity, c) is
chosen on the inner OOF scores of the 96 training prompts (selected lr and epoch), averaging m = 2, 3 and 6. The
outer clustering runs over each partition's whole cut (120 prompts). Paired prompt bootstrap vs no centring, on
Exp 4d's 8B (`outputs/centring/q8-l24-full-001`):

| answers per prompt | p0 plain → centred | Δ [95% CI] | p1 plain → centred | Δ [95% CI] |
| --- | --- | --- | --- | --- |
| 2 | 0.611 → 0.621 | +0.010 [−0.009, +0.025] | 0.622 → 0.622 | +0.001 [−0.015, +0.014] |
| 3 | 0.620 → 0.628 | +0.008 [−0.005, +0.019] | 0.611 → 0.636 | +0.025 [+0.011, +0.036] |
| 6 | 0.624 → 0.640 | +0.017 [+0.006, +0.028] | 0.628 → 0.648 | +0.020 [+0.009, +0.031] |
| 18 | 0.623 → 0.646 | +0.022 [+0.011, +0.034] | 0.624 → 0.647 | +0.023 [+0.013, +0.033] |

- Inner selection chose w 0.25-0.4 and cut 0.85-0.9 in every fold, cavity in 2 of 10. No m loses.
- Fixed settings, both partitions averaged (Δ at m = 2 / 3 / 6 / 18): w 0.25, cavity, cut 0.85 gives +0.008 /
  +0.013 / +0.015 / +0.020; w 0.4, plain, cut 0.9 gives +0.006 / +0.015 / +0.020 / +0.026. The Kaggle package
  uses the first (`kaggle_package.py --centring 0.25 0.85`). It is gentler where groups are small, and at m = 2
  its clusters are 0.975 pair-precise against 0.852 at cut 0.9. The test's ~4,000 passages hold far more prompts than
  our 120, so more chance merges are expected there. The forum's version (LoRA member, w 0.25, cut 0.85) moved
  its public LB from 0.595 to 0.602 (thread 752708).
- Earlier, per-class logit offsets fitted nestedly on inner OOF hurt (−0.008 / −0.014), so the gain is not a prior
  correction.

### Plan E5: are some pool sources' GLM labels inconsistent? (2026-11-08)

Stage 2 lowers glm-5.3's recall (0.283 → 0.258), so Fable suspected pool sources that label the GLM pair
differently from official serving. Audit without official rows: for each pool source, a char 1-4-gram LinearSVC
over glm-5.3 / glm-5.3-flash / claude-fable-5, trained on every *other* source, scores that source's rows
(`outputs/centring/glm_loso.csv`). Every source's rows are recognised at 0.26-0.57 (chance 0.33), and no source
shows a swap (its glm-5.3 rows called glm-5.3-flash more often than glm-5.3, or the reverse). No source stands out,
so no recipe drops sources: the GLM/fable triangle is hard in every source alike.

### 14B LoRA24 nested, and an 8B + 14B blend (2026-11-08)

TC2 jobs 52203 + 52207 (`nllm_14b_l24full`, 5 h 27 min + 42 min, run `outputs/tc2/qwen3-14b-l24-full-001`): Qwen3-14B
with the LoRA24 recipe (24 of 40 layers, micro-batch 3), a shared uniform stage 1 capped at 10,000 per label
(166,945 rows, 3 h 05 min, peak 29.8 GB; kept under `stage1/shared` for a final), then the nested stage 2.

| | p0 BA [95% CI] | p1 BA [95% CI] |
| --- | --- | --- |
| 14B stage 1 alone | 0.588 [0.566, 0.611] | 0.588 [0.566, 0.610] |
| **14B LoRA24** | **0.609 [0.590, 0.628]** | **0.621 [0.601, 0.641]** |
| 8B LoRA24, full pool (Exp 4d) | 0.623 | 0.624 |
| 8B + 14B, nested weights (`outputs/blend/q8-l24full+q14-l24full-20261108`) | 0.623 (Δ −0.000 [−0.016, +0.015]) | 0.628 (Δ +0.005 [−0.010, +0.020]) |
| 8B + 14B, equal (each / its scale) | 0.628 (Δ +0.005 [−0.008, +0.019]) | 0.631 (Δ +0.008 [−0.007, +0.023]) |

- The 14B alone is 0.015 / 0.003 below the 8B. Its stage 1 saw 21% fewer pool rows (the 10k cap that kept it inside
  one job), which may be all of it: stage 1 alone ties the 8B's at 0.588. Epoch 2 was chosen in 3 of 10 folds.
- They agree on 73.5% of rows. An equal blend gains on both partitions, but neither CI excludes 0. A 14B Kaggle member
  also needs its 30 GB base uploaded (or a 24-layer cut of it).

## Exp 9: the single use of the confirmation set (2026-11-08)

With the user's approval, the 30 locked confirmation prompts (540 rows) were scored once with the frozen recipe:
the Kaggle model `qwen3-8b-l24-full-final-001` (trained on the 120 dev prompts only), loaded as the notebook loads it
(hand-merged fp16) on the local GPU, then centred as packaged (w 0.25, cut 0.85, cavity). `scripts/confirm_locked.py`,
`outputs/confirmation/qwen3-8b-l24-full-final-001`; `outputs/confirmation/USED.json` stops a second run.

| | BA [95% CI] | Δ from centring [95% CI] |
| --- | --- | --- |
| **8B LoRA24 final, plain** | **0.633 [0.600, 0.665]** | |
| centred, all 18 answers per prompt | 0.637 [0.602, 0.670] | +0.004 [−0.013, +0.022] |
| centred, prompts cut to 2 / 3 / 6 answers (5 draws) | 0.605 / 0.633 / 0.644 | −0.005 / −0.005 / −0.002 (each CI spans 0) |

- **The model holds up on unseen prompts:** 0.633, against 0.623 / 0.624 nested (the final also trains on 120
  prompts, not 96). The public LB of the same model is 0.599, so the LB's class mixture or prompts are harder than
  either split of train.
- **Centring is not confirmed:** about zero here, against +0.008 to +0.023 on the dev OOF. Only 30 prompts, so the
  intervals are ±0.02 and still admit a gain near +0.01. The fixed setting was also picked on the dev outer OOF.
  The Kaggle LB of the centred package (`q8-l24-full-c25-001`) is the remaining check.
- **Kaggle settles it in centring's favour:** submission 58114807, the same model and weights as 58101792 with
  centring added (w 0.25, cut 0.85, cavity), scored **0.619** on the public LB against 0.599. Identical kernels
  rescore identically (forum 756088), so the +0.020 is centring's effect on the public 40%. It is larger than the
  forum's +0.007, which suggests the hidden test has more answers per prompt than the 2-3 the forum assumed.
- From here on, the 30 prompts are training data for final fits (`--final --all-prompts`). They are never used for
  selection.

## Data study Phase 0: the closeness instrument (2026-11-08)

Phase 0 of `notes/data_study_plan_2026-11-08.md`, which spends nothing. It asks how official a batch of generated
answers (a "cell": one source or setup) looks to models trained on official rows only. `nllm.closeness`,
`scripts/closeness_recogniser.py`, `scripts/closeness_instrument.py`, outputs under `outputs/closeness/`.

- **Two recognisers, both trained on official rows only**, over 5 prompt-grouped folds of all 150 prompts:
  - the linear text+style reference (CV BA 0.386);
  - a Qwen3-4B LoRA24 (Exp 4c recipe, no pool stage 1, lr 1e-4, 4 epochs, micro-batch 3; 2-3 min and 11.3 GB per
    fold locally). Its out-of-fold BA is 0.470 (epochs 1-4: 0.307 / 0.390 / 0.448 / 0.470). Saved under
    `recogniser-4b/fold<k>/`.
- **Measures per (cell, label):**
  - recognition ratio: the rate at which the recogniser gives the cell's passages their claimed label, divided by
    its own out-of-fold recall on that label. 1.0 means the cell looks as official as unseen official prompts do.
    For the 4B, the rate is the mean over its 5 fold models.
  - family share: how often the 4B puts the passages in the right provider family;
  - style-only AUC, official vs cell;
  - median words and KS distance from the label's official word counts;
  - within-family transfer (linear only): a family model trained on 100 cell rows per label is tested on official
    rows, against one trained on 100 official rows per label.
- **Calibration set:** our admitted answers by interface, and every Open Pool source with ≥ 100 rows per label (up
  to 200 sampled). That makes 52 cells and 324 (cell, label) rows. Runs: `instrument-linear` (with transfer) and
  `instrument-4b`.

**Calibration passes.** Both recognisers put the sources we knew were close high, and the sources we knew were far
at the bottom:

| Our source | Labels | Ratio, linear | Ratio, 4B | Right family (4B) | Style AUC | Median words (official) |
| --- | --- | --- | --- | --- | --- | --- |
| Codex CLI | luna / sol / terra / astra | 0.75 / 0.69 / 0.62 / 1.10 | 0.97 / 0.47 / 0.77 / 0.94 | 0.85-0.91 | 0.77-0.82 | 99-106 (100-112) |
| Anthropic batches | fable / 4-6 / 4-8 / opus-5 / sonnet-5 | 1.12 / 0.76 / 0.49 / 1.08 / 0.93 | 1.10 / 0.85 / 0.56 / 1.18 / 0.61 | 0.50-0.88 | 0.75-0.84 | 125 (99-117) |
| Agent Platform | 3.1-pro / 3.6 / 3.7 / 3.8 | 0.76 / 0.65 / 0.60 / 0.53 | 0.82 / 0.83 / 0.27 / 0.39 | 0.75-0.89 | 0.88-0.93 | 111-124 (84-104) |
| WorkBuddy | glm-5.3 / glm-flash / gpt-5.5 | 0.46 / 0.74 / 0.55 | 0.34 / 1.13 / 0.30 | 0.36-0.91 | 0.81-0.93 | 115-130 (96-112) |
| Moonshot, thinking off | kimi-k3 | 0.12 | **0.01** | 0.005 | 0.93 | 112 (89) |
| WorkBuddy | kimi-k3 | 0.30 | **0.08** | 0.04 | 0.89 | 126 (89) |
| Qoder CLI | qwen3.8-max | 0.06 | **0.02** | 0.005 | 0.96 | 119 (99) |

- **The 4B is the sharper instrument.** It separates the two groups further than the linear model does. It almost
  never puts our Kimi or Qwen answers in their own family (0.5-4%), so those answers do not even read as Kimi or
  Qwen. The two recognisers agree only moderately across all 324 rows (Spearman 0.59). From here the 4B ratio is
  the primary measure.
- **Style AUC is a weak gate.** Every source is fairly easy to tell from official by style (pool medians 0.66-0.86
  per label), and it barely tracks the 4B ratio (Spearman −0.26).
- **One check of predictive power already exists.** Exp 6 found that pool data *lowered* Kimi's recall
  (0.30 → 0.12-0.24), and that source multipliers raised it again by drawing less from the weak Kimi sources. The
  pool's Kimi sources have the lowest median 4B ratio of any label (0.16). Phase 4 is the real test.

**Best pool sources for our weak labels (4B ratio).** Each best is the maximum over 4-29 noisy cells, so it reads
high. One ratio has a standard error of about 0.1-0.15, more for labels with low recall (fable 0.20, GLM 0.25-0.29,
sol 0.29).

| Label | Pool median (sources) | Best pool source | Ours |
| --- | --- | --- | --- |
| kimi-k3 | 0.16 (9) | arxitekton v1 / v2 0.84 / 0.84; next drjcfmd 0.34 | 0.01 / 0.08 |
| qwen3.8-max | 0.56 (7) | drjcfmd 0.82, lxdlxdlxd 0.79 | 0.02 |
| gemini-3.8-flash | 0.62 (29) | chosunghwan/ntl-gemini-new-prompts 1.28 | 0.39 (Antigravity 0.70, 29 rows) |
| gemini-3.7-flash | 0.45 (4) | davidpiln 0.50 | 0.27 |
| gpt-5.5 | 0.71 (24) | ziyuzhou1207 0.86 | 0.30 |
| gpt-5.6-sol | 0.62 (21) | dophuchung 0.77 | 0.47 |
| glm-5.3 | 0.65 (12) | elliotthankinson v3 0.98 | 0.34 |
| claude-opus-4-8 | 0.67 (16) | yutoab 0.84 | 0.56 |
| claude-sonnet-5 | 0.69 (20) | drjcfmd 1.10 | 0.61 |

**What separates the close sources from ours.** Typography and length, official vs our route vs the best pool
sources:

| Label | Official | Ours | Best pool |
| --- | --- | --- | --- |
| kimi-k3 | 52% em dash, 32% curly apostrophe, 61% semicolon, 89 words | Moonshot thinking off: 48% / 35% / 23%, 113 words | arxitekton: 50% / 43% / 54%, 107 words |
| qwen3.8-max | 3% em dash, 29% curly apostrophe | Qoder: 64% / 0% | drjcfmd, lxdlxdlxd, arxitekton: 3-7% / 44-57% |
| gemini-3.8-flash | 0% em dash | Agent Platform: 38% | chosunghwan, drjcfmd: 0% |

- **Kimi:** our Moonshot answers match the official dash and apostrophe rates, yet score 0.01. The thinking-on
  routes are further still: WorkBuddy is 78% / 11% and drjcfmd (Moonshot via OpenRouter) 85% / 8%. Only
  arxitekton's rows match official on all three marks and on length. Its card lists the Moonshot API with effort
  "high".
- **Qwen:** Qoder does not serve the official Qwen setup.
- **gpt-5.5 via WorkBuddy:** 0% straight apostrophes (official 11%) and 32% curly double quotes (official 9%).
- **Length:** every API source of ours runs 7-41% longer than official (KS 0.17-0.29). Codex and most pool sources
  are within about 10% (KS 0.05-0.12).

**Family transfer (linear, 100 rows per label):**

| Family | Pool median lift | Ours |
| --- | --- | --- |
| Claude | 0.87 | 0.96 (3rd of 18) |
| Gemini | 0.68 | 0.77 |
| GPT | 0.65 | Codex 0.66 |

GLM cannot be measured this way: 100 official rows per label give a linear BA of only 0.51 for the pair. Kimi and
Qwen are single-label families.

**Limits.** A cell's rate also depends on its prompts, so a source whose prompts resemble the official genre gains
for that reason too. Phase 1 holds the prompts fixed, so its cells compare setups only.

**Consequences for Phase 1** (plan updated):
- Kimi and Qwen first. Our routes are the furthest sources measured, while close setups exist: arxitekton's
  Moonshot Kimi, and drjcfmd's and arxitekton's Qwen (arxitekton documents qwen3.8-max-0902 at T 0.6).
- Then gpt-5.5 through the OpenAI API, glm-5.3 through Z.ai, and Gemini 3.8 with a setup like chosunghwan's.
- Word targets come from the official distribution, not our bands.
- The gate becomes a 4B ratio ≥ 0.8, words KS ≤ 0.15, and style AUC no worse than the label's pool median.

## Architecture review of the 8B, and the leaderboard on 12 Nov (2026-11-12)

Read-only analysis of Exp 4d's outer-fold scores (`outputs/tc2/qwen3-8b-l24-full-001/oof.parquet`, both partitions),
and the public leaderboard pulled with the Kaggle CLI.

| Leaderboard, 12 Nov (5,582 teams) | Score |
| --- | --- |
| rank 1 / 10 / 20 / 50 / 100 | 0.768 / 0.747 / 0.733 / 0.707 / 0.687 |
| public notebook that anyone can fork (278 teams at exactly this score; 749 teams in 0.68-0.69) | 0.684 |
| ours (58114807: full-pool 8B LoRA24 + centring), rank 2,538 | 0.619 |

Where the 8B's errors are (p0 / p1):

- **The family is right far more often than the sibling.** Predicted family right on 0.847 / 0.846 of rows; given
  that, the exact label is right on 0.736 / 0.738. With the true family known, BA would be 0.728 / 0.733; with a
  perfect choice inside the predicted family, 0.847 / 0.846. Top-2 accuracy 0.80, top-3 0.90.
- **Three groups hold 47% of all errors** (762 of 1,625 over both partitions): gemini-3.6 / 3.7 / 3.8 (253),
  gpt-5.5 / luna / sol / terra (257) and claude-fable-5 / glm-5.3 / glm-5.3-flash (252). Removing one group's
  errors would add about 0.06 accuracy. Largest cells: gemini-3.7 ↔ 3.8 (73 / 71), glm-5.3 → glm-5.3-flash (64),
  glm-5.3 → claude-fable-5 (50), gpt-5.5 → kimi-k3 (39).
- **Short passages are much harder**: accuracy by word-count fifth is 0.45 (36-59 words), 0.60, 0.63, 0.69 and
  0.76 (170-274 words).
- Predicted label counts are even (98-138 per label and partition), so there is no prior to correct.

Checked and dropped:

- **Stage-1 + stage-2 logit blends**: adding 0.25 / 0.5 / 1.0 × the stage-1 scores to the stage-2 scores changes BA
  by −0.002 / −0.001 / −0.008 (p0) and −0.003 / −0.004 / −0.009 (p1).
- **Distinct-label assignment inside true prompt groups** (each label at most once per prompt, prompts cut to m
  answers, 5 draws): plain → assigned BA 0.623 → 0.623 (m = 2), 0.622 → 0.636 (3), 0.618 → 0.641 (6), 0.630 → 0.684
  (10) and 0.623 → 0.777 (18) on p0; p1 the same to within 0.008. Public teams lost 0.009-0.017 on the board with it,
  which fits a test with few answers per prompt. Not pursued.
- **Confident pseudo-labels**: the most confident 30% of rows are 92% right but cover only 16 labels, so
  self-training on test rows would mostly re-teach the easy labels.
- **The nested inner loop** took 1.39 of 4.71 training hours in Exp 4d (30 of 41 fits) to choose an epoch that was
  3 in 18 of 20 folds from a one-value lr grid. A plain 5-fold stage 2 from the kept stage-1 adapters
  (`outputs/tc2/qwen3-8b-l24-full-final-001/stage1`, `qwen3-14b-l24-full-001/stage1/shared`) would cost minutes.

What the public 0.684 pipeline is (alexchilton's "Name That LLM 0.684: All Public + What Failed", version 25, read
but never run locally): six weighted members (multi-depth LoRA readouts with twin heads on Qwen3-14B × 2, Qwen3-8B,
Llama-3.1-8B and Qwen2.5-14B, and a DeBERTa-v3 encoder), a stylometric stack mixed in at 0.15, a five-fold
Qwen3-Embedding-8B expert at a 0.40 share, a Qwen3.8-27B expert at 0.03 and Sinkhorn balancing at 0.30. Its authors'
own board tests: more LoRA members of the same kind move the score by at most 0.004, Hungarian assignment loses,
intra-family gates add nothing, and the twin rule took one of their blends from 0.560 to 0.588. Two things follow
for us. Its `l24` members have exactly our pooled decoder's architecture (truncated base, the same eight depths,
first token masked, one LayerNorm per depth, plain and centred heads), so our 8B can join as a native member. And
our own inference dropped the centred head, which that pipeline uses for every member.

## Public-stack base: forks of the 0.684 notebook (2026-11-12)

Decision (owner, 12 Nov): build on the public 0.684 stack and test our additions against it.

`scripts/build_public_fork.py` (replacing the earlier, never-submitted fork of the 0.672 notebook) builds two
private notebooks from the pulled source, whose 20 code cells must hash to version 25's:

- `base` (`outputs/kaggle/public-fork-684/base`, kernel `rhysbao/nllm-public684-base`): every code cell unchanged.
  The 17 datasets are pinned to the versions the scored session (368561683) ran with, read from its run record;
  each was also the latest version on 12 Nov. Expected public LB 0.684.
- `plus` (`outputs/kaggle/public-fork-684/plus-w2-t0.35`, kernel `rhysbao/nllm-public684-plus-ours`): our full-pool
  8B LoRA24 final (dataset `rhysbao/nllm-q8-l24-c25-001`, already on Kaggle) as a seventh member at weight 2.0 and
  twin 0.35, the settings of the pipeline's own Qwen3-8B member. `kaggle/public_fork/ours_l24.py` only renames our
  saved tensors into the layout `infer.py` reads (`head` → `plain`, `centred_head` → `centered`) and copies the
  adapter; the pipeline scores it with its own code and applies its own grouping and twin rule. The run cell
  changes by three lines.
- **Commit-time check.** The scoring rerun's log is never visible, so the `plus` commit (3-row public test) runs
  `infer.py` itself on our 72 reference training rows, with our member alone and its plain head only, and stops
  unless the log-probabilities agree with the fp16 scores saved at training time (argmax agreement ≥ 0.95, every
  value within 1.0; fp16 and bf16 of our own loader differ by at most 0.20). A version that commits has therefore
  been read by the pipeline the way training read it.
- The earlier design (our ensemble in its own process, added to `tot`) would have had no effect on this version:
  `infer.py` now decides from `blend(LP)`, and `tot` is only saved.

Pushed and submitted on 12 Nov with the owner's go-ahead (`kaggle kernels push`, both private, version 1 each). The
push accepted the pinned dataset versions and every source.

| Commit run (3-row public test, T4 x2) | `base` | `plus` |
| --- | --- | --- |
| sources / models mounted | 25 of 25 / 8 of 8 | 25 of 25 / 8 of 8, plus our dataset |
| wall time | 164 s | 276 s |
| our member through `infer.py`, 72 reference rows | | argmax agreement 1.000, max abs log-prob diff 0.0255 |

In the `plus` commit our member took 46 s in all to load, merge and score the 72 rows (no bf16 redo batches). Its
time on the 4,000 hidden rows was not measured; our own notebook's fp16 runs put it at roughly 5-10 minutes.

| Submission | Notebook | Public LB | Submit to score |
| --- | --- | --- | --- |
| 58265513 | `rhysbao/nllm-public684-base` v1: the public pipeline, code cells unchanged | 0.684 | 5 h 12 to 5 h 22 min |
| 58265516 | `rhysbao/nllm-public684-plus-ours` v1: plus our 8B LoRA24 at weight 2.0, twin 0.35 | 0.680 | 5 h 22 to 5 h 32 min |

Both were submitted at 07:40 UTC and scored by 13:12 UTC; the times come from a ten-minute poll and include any
queueing.

- `base` reproduces the public score exactly, so the pinned fork is a faithful base: 0.684 against our own best of
  0.619 (8 Nov). On the board at 13:12 UTC (5,622 teams) 117 teams are above 0.684 and 335 sit exactly on it; first
  is 0.768, 20th 0.733, 50th 0.708.
- `plus` is 0.004 below `base`. That is inside the ±0.009 repeat-submission noise and the size the pipeline's
  authors report for one more LoRA member of the same kind, so this is no evidence that our 8B helps or hurts. It
  was the expected outcome: the pipeline already has a Qwen3-8B member trained much as ours was. Adding our current
  8B unchanged is not a gain, and `base` is the one to keep.
- The member slot works on the hidden rows: `plus` scored without error and cost about ten more minutes (0 to 20 at
  the poll's resolution), leaving over two hours of the pipeline's 8 h budget. Use the slot for members of ours that
  differ (a retrain with sibling-stratified same-prompt units, a different data view, the 150-prompt final).

## How the leaders reach 0.75: evidence, and the open-weight candidates (2026-11-13)

Question from the owner: something bolder than member tweaks; how do others get to 0.75? Read on 13 Nov: the
forum (sorted by votes, four pages), the public notebooks' notes, the Kaggle dataset listings and the Hugging Face
Hub. Nothing here is from our own experiments yet.

**The hidden test has about two answers per prompt.** GoPain (comment on discussion 753177, quoted in the public
0.684 notebook): scoring out-of-fold on 2 answers per prompt tracked the board at correlation 0.91, against 0.13 with
all 18. The 0.684 notebook's authors lost 0.009-0.017 with Hungarian assignment on every base they tried, and their
own blend fell from 0.6865 in CV (18-answer groups) to 0.56 on the board. So roughly 2,000 prompts for 4,000
passages: within-prompt assignment has no room, and grouped CV on 18 answers overstates any centring trick.

**Single models at the top are 0.65-0.68; ours is 0.60.** Discussion 754860: Horikita Saku (9th) about 0.670 with
a Qwen3-14B and some own data, about 0.65 on public data only; Xiao (22nd) 0.67-0.68 with post-processing;
somerandomguy (3rd) 0.65 a week earlier; Nishant Dahal (17th) about 0.65; Nguyên Nguyên 0.657 (14B, pretrained on
external data, full fine-tune on host data); Carrot Production (the Open Pool's author) 0.637 to 0.644 as the pool
improved; GoPain 0.628 with a Qwen3-8B LoRA on pool v10 (OOF 0.6224, the same as ours). Several say nearly all
of their gain came from data. Our pool copy is v13 (3 Nov); the current one is v16.

**More data is not the whole answer.** Discussion 749447 (the most voted): Keanan (6th when writing) reached the
top ten without paid generation; Horikita Saku (9th) reports that some models did not improve when the data grew
a lot. Free generation exists: Kaggle Benchmarks credits ($100 a month, or $10 a day by another account).

**Three candidates have public weights, and the top teams very likely score with them.** This is the lead that
fits the size of the gap.

- `zai-org/GLM-5.3-Flash` (MIT, architecture `glm5_next`, about 320 B parameters) and `zai-org/GLM-5.3`
  (`glm_moe_dsa`, 78 layers) are on the Hub; transformers 5.22 implements both architectures.
- Kaggle holds them cut into layer segments for streaming: `jinghengwu/ntl-flash-iq3xxs-segs` (6 files, 117 GB,
  367 downloads), `jinghengwu/ntl-glm53-iq3xxs-segs-a` and `-b` (19 files of about 14.6 GB), a mirror by
  jamesliebel, `ggayoayogg/glm53-ud-iq2m-b` (uploaded 11 Nov by AyoGG, 42nd), and Kimi K3 as
  `lianmatsuo/kimik3-iq2xxs-slices-1` to `-4` (about 750 GB, 900 to 1,600 downloads each).
- Discussion 757199 (vishu phalswal, about 0.68): out-of-fold accuracy 0.32 on glm-5.3, 0.44 on glm-5.3-flash and
  0.48 on claude-fable-5; likelihoods from GLM-4-9B and Qwen3-8B-Base added nothing (−0.002), "so the real
  model's likelihood looks like the missing signal". Replies: read one layer, run every passage through it, free
  it, next layer; 128-256 tokens are enough. AyoGG: look at "the publicly available models from the
  top-performing group".
- No public notebook uses these weights. The 0.684 stack's Qwen3.8-27B expert is a relative of qwen3.8-max at a
  3% share, not a candidate's own likelihood.

**Why this is worth several points.** Our 8B's recall is 0.24 / 0.28 on glm-5.3, 0.46 / 0.43 on glm-5.3-flash,
0.39 / 0.45 on claude-fable-5 (its errors are mostly GLM) and 0.52 / 0.51 on kimi-k3. If the four reached 0.85,
balanced accuracy would rise by about 0.10. The gap from the public stack to the top ten is 0.06-0.08.

## Exp 10: likelihood under the real GLM-5.3-Flash (2026-11-13)

Two of the candidates' own weights, read from the public Kaggle layer segments and run one layer at a time on the
local 16 GB GPU (`nllm.candidate_lm`, `scripts/candidate_likelihood.py`). This entry is GLM-5.3-Flash
(`jinghengwu/ntl-flash-iq3xxs-segs`: 117 GB, experts at IQ2_S / IQ3_S, 45 layers, 288 experts); GLM-5.3 follows.

**Is the model read correctly?** The control flow matches transformers' forward pass on small random models to
6e-5, and the GPU expert decoders equal gguf-py's bit for bit (`tests/test_candidate_lm.py`). On the real weights:

| Text | CPU float32: perplexity, top-1 | GPU bf16: perplexity, top-1 |
| --- | --- | --- |
| fluent English, paragraph 1 | 5.2, 0.60 | 5.3, 0.62 |
| the same words shuffled | 666.5, 0.09 | 597.7, 0.11 |
| fluent English, paragraph 2 | 6.8, 0.57 | 6.7, 0.55 |
| the same words shuffled | 331.4, 0.11 | 351.6, 0.09 |

**Run.** All 2,699 official passages (444k tokens with an 18-token chat prefix), each as a finished assistant
answer to the generic prompt "Write a single paragraph.": 2,011 s, 7.8 GB of GPU memory. The library's reference
linear attention is the slow part and needs about 0.6 GB per row and 64-token chunk, so it runs four rows at a
time. Output: `outputs/candidates/flash-official-001.npz` (16 per-passage features, per-token statistics).

**What the likelihood looks like** (120 development prompts, mean per label):

| Passages by | mean token log-prob under Flash | top-1 rate |
| --- | --- | --- |
| glm-5.3-flash (its own) | −1.91 | 0.567 |
| claude-fable-5, claude-opus-4-6 | −2.05 | 0.556 |
| claude-opus-4-8 / sonnet-5 / opus-5, glm-5.3 | −2.12 to −2.30 | 0.52-0.54 |
| GPT, Kimi, Qwen | −2.58 to −2.73 | 0.45-0.47 |
| Gemini | −2.73 to −3.04 | 0.44-0.46 |

One-vs-rest AUC for glm-5.3-flash: 0.858 (curvature: log-prob plus entropy over the root of the variance, as in
Fast-DetectGPT), 0.815 (mean log-prob). Pairwise, curvature separates Flash's own text from glm-5.3 at 0.805, from
claude-fable-5 at 0.743 and from GPT or Kimi at 0.87-0.88; it does not separate glm-5.3 from claude-fable-5 (0.43).
So the model prefers its own text, but Claude text is nearly as likely to it. The signal is real and moderate, not
decisive.

**What it adds to the 8B** (`scripts/candidate_eval.py` on Exp 4d's run: a correction `a × base log-probabilities
+ bias + features @ W` fitted on inner out-of-fold rows only, L2 penalty by grouped CV inside them; `recal` is the
same without features):

| Features | p0 | p1 | plus − recal, 95% prompt-bootstrap CI |
| --- | --- | --- | --- |
| none: the 8B (base) / recal | 0.6233 / 0.6205 | 0.6238 / 0.6224 | |
| random features (null check) | 0.6182 | 0.6220 | −0.002 / −0.001, both CIs span 0 |
| neutral reference only (Qwen3-1.7B-Base on the raw text) | 0.6210 | 0.6192 | +0.001 / −0.003, both CIs span 0 |
| Flash | 0.6312 | 0.6275 | +0.011 [+0.002, +0.020] / +0.005 [−0.006, +0.015] |
| Flash + neutral reference | **0.6368** | **0.6396** | +0.016 [+0.005, +0.027] / +0.017 [+0.006, +0.029] |

Recall with Flash + reference (p0 / p1): glm-5.3 0.24 → 0.40 / 0.28 → 0.41, glm-5.3-flash 0.46 → 0.63 /
0.43 → 0.58, claude-fable-5 0.39 → 0.40 / 0.45 → 0.49. Other labels move by at most ±0.05.

Reading: the real model's likelihood is worth about +0.015 on its own corner of the problem, and only once a
neutral model takes out how predictable the content is (the reference alone adds nothing). This is one candidate
model, at about 3 bits per weight, without the prompt. It is not yet the 0.06-0.08 that separates the public stack
from the top ten.

## Exp 11: sibling-drawn same-prompt units at stage 2 — no gain (2026-11-13)

Hypothesis from the architecture review: three sibling groups hold 47% of the 8B's errors, so cut each prompt's
units of three so that labels the model confuses share a unit, and let the centred head's loss push the shared
trunk towards what separates them. `TrainConfig.sibling_units` (`--sibling-units S`): each unit takes an anchor
row, and each partner is drawn with probability S in proportion to the starting model's confusion with the
anchor's label (mean predicted probability, symmetrised, measured on the fit's own training rows), otherwise
uniformly. Nothing is hand-coded and no held-out row is read.

Screen (`--fixed-epoch 3`, no inner fits; epoch 3 was chosen in 16 of Exp 4d's 20 folds), stage 2 only, both arms
from the kept uniform stage 1 of the 120-prompt final, on the local RTX 5070 Ti (micro-batch 3, gradient
checkpointing, `NLLM_EVAL_BATCH=4`: 12 GB, 11 min per fold; scoring at 16 rows a batch ran out of memory beside
the desktop). Partition 0, five folds each; partition 1 was not run once the answer was clear.

| Partition 0 | random units (baseline) | sibling units, S = 1.0 | difference, 95% prompt-bootstrap CI |
| --- | --- | --- | --- |
| same-family pairs inside units | 0.17 | 0.48 | |
| plain head, 18 answers per prompt | 0.6169 | 0.6090 | −0.008 [−0.018, +0.002] |
| family right / exact given family | 0.843 / 0.732 | 0.839 / 0.726 | |
| twin rule (0.35), prompts cut to 2, true groups | 0.6297 | 0.6262 | −0.005 [−0.018, +0.008] |
| twin rule, cut to 2, content clusters | 0.6286 | 0.6288 | −0.002 [−0.018, +0.013] |
| twin rule, cut to 3, true groups / clusters | 0.6369 / 0.6356 | 0.6339 / 0.6299 | −0.003 / −0.006 |
| twin rule, cut to 6, true groups / clusters | 0.6427 / 0.6449 | 0.6356 / 0.6350 | −0.007 / −0.010 |

- **The units did what was intended** (Gemini 3.7 and 3.8 in one unit 57% of the time against 12%), and it did
  not help: every comparison is zero to slightly negative, on the plain head and on the twin rule. Dropped.
- The kept stage 1 scores 0.575 on the development prompts here against 0.588 for the instance Exp 4d's nested run
  trained (same recipe and seed, 86% of predictions equal), so this baseline (0.617) sits a little under Exp 4d's
  0.623. Both arms share it.
- **A by-product worth keeping:** fits now save the centred head's outputs (`oof.parquet`, `centred:<label>`), and
  `scripts/twin_eval.py` scores the public pipelines' twin rule from them. On the baseline arm the twin rule adds
  +0.013 at two answers per prompt (0.616 → 0.630, true groups; 0.629 with our content clusters) and +0.02 at
  three. Our own Kaggle inference drops that head; the public pipeline uses it for every member.

## GLM-5.3 stopped at layer 54 of 78; weights removed; no model runs on this machine (2026-11-14)

**Owner's decision, 14 Nov:** no language models are run on the local machine any more, and the GLM weights go.
Kimi K3's slices (about 750 GB) were never downloaded.

- **State found.** The GLM-5.3 scoring run ended with the earlier session at layer 54 of 78 (2,283 s in; segments
  13-15 half downloaded). Nothing was running. It is not resumed.
- **Kept from it.** The checkpoint already held each passage's mean hidden state after layers 8, 16, ..., 48
  (`--pool-layers` default), and the states after layer 54 were averaged the same way before deletion:
  `outputs/candidates/glm53-partial-pooled-l54.npz` (2,703 × 7 depths × 6,144, float16, 233 MB). No token
  likelihoods exist for GLM-5.3: those need all 78 layers.
- **Deleted** (143 GB; free disk 42 GB → 185 GB): both segment directories under `data/external/third_party/`
  (Flash 110 GB, GLM-5.3 28 GB), the 6.1 GB checkpoint, the tokenizer/config copies in `outputs/candidates/hf/`,
  an empty Hub cache entry, and the launch scripts. The Flash result files (Exp 10) are kept.

**Do GLM-5.3's hidden states help?** CPU only, the same nested correction as Exp 10 on Exp 4d's run; the mean
state at one depth is standardised and reduced to 64 principal components fitted on each correction's training
rows. Depths 32 and 54 were fixed before looking.

| Features added to the 8B | p0 | p1 | difference to recal, 95% prompt-bootstrap CI |
| --- | --- | --- | --- |
| none (recal) | 0.6205 | 0.6224 | |
| GLM-5.3 state, layer 32 | 0.6215 | 0.6173 | +0.001 [−0.004, +0.006] / −0.005 [−0.010, −0.001] |
| GLM-5.3 state, layer 54 | 0.6201 | 0.6192 | −0.000 [−0.007, +0.005] / −0.003 [−0.008, +0.002] |
| Flash likelihood + neutral reference (Exp 10, rerun) | 0.6368 | 0.6396 | +0.016 [+0.005, +0.027] / +0.017 [+0.006, +0.029] |
| the same + GLM-5.3 state, layer 54 | 0.6326 | 0.6340 | +0.012 / +0.012; against the row above −0.004 / −0.006, CIs span 0 |

A logistic regression on the full layer-54 state alone scored 0.38 and 0.42 on the two folds that were run
(layers 16-48: 0.38-0.44): above the linear text+style baseline (0.35-0.36), far below the 8B. So the
candidate's mean hidden state carries nothing the 8B lacks; the value of a candidate's own weights is in the token likelihoods, and Exp 10's +0.016 from Flash
is the whole result of this line. GLM-5.3's likelihood stays unmeasured.

**Where this leaves the 0.75 question.** Nothing tested here is worth more than about +0.015, measured on our own
8B (0.62) and not on the public stack (0.684). Using even that needs Flash to score the hidden test inside the
Kaggle notebook (117 GB of segments, 45 layers; 2,011 s for 2,699 passages on the local card, unmeasured on T4s).
The training-side features already exist for all 2,699 official passages, so no further local run would be needed.

## The board on 14 Nov, and what 0.75 would take (2026-11-14)

Read-only: the leaderboard and the new public notebooks' source (Kaggle CLI, 12:42 UTC), the forum, and CPU
analysis of saved predictions. The full account, with sources and the plan, is
`notes/path_to_075_2026-11-14.md`.

| Leaderboard, 14 Nov (5,840 teams, two months to go) | Score |
| --- | --- |
| rank 1 / 10 / 20 / 50 / 100 / 150 / 500 | 0.771 / 0.750 / 0.737 / 0.716 / 0.701 / 0.694 / 0.684 |
| best public notebooks: the 0.684 stack plus real GLM-5.3 and GLM-5.3-Flash likelihood | 0.694, 0.693 |
| ours (the unchanged 0.684 fork), rank 610 | 0.684 |

- **The public now has Exp 10's idea, finished** (`estopadilla/0-689-real-glm-5-3-likelihood-on-2xt4`): a patched
  llama.cpp runs the layer segments on two T4s over the texts with the most GLM mass. Their out-of-fold figures:
  GLM-5.3 alone +0.004, the Flash − GLM-5.3 contrast +0.035 (recall glm-5.3 0.27 → 0.63, glm-5.3-flash
  0.45 → 0.65). On the board +0.007 to +0.010. Our Flash-only +0.016 (Exp 10) sits between the two.
- **Oracles on Exp 4d's 8B** (p0 / p1): as it is 0.623 / 0.624; the GLM pair identified perfectly 0.714 / 0.712;
  with kimi-k3 too 0.752 / 0.752; right sibling whenever the family is right: Gemini 0.697 / 0.697, GPT
  0.699 / 0.700, Claude 0.672 / 0.672. The three open-weight labels hold more of the error than any closed
  family, and the public stage captures about a third of the GLM part.
- **Learning curve from saved fits** (official prompts in stage 2: 0 / 64 / 96; the 64-prompt figure is the inner
  out-of-fold score at the selected epoch): 8B 0.588 / 0.613 / 0.623 on both partitions, about +0.018 per
  doubling; 14B 0.588 / 0.615-0.618 / 0.609-0.621, flat within its noise.
- **Pool structure:** 31,355 prompts; 78% of rows answer prompts that six or more labels answer. 37% of rows came
  through the Kaggle Benchmarks proxy, and their closeness cells look as official as other routes and no more
  (median 4B ratio 0.73 against 0.73).
- **Reading.** No change to the classifier's architecture is supported: more or different members move the public
  stack by ±0.004, and ours scored 0.680 against 0.684. The gap to the top ten (0.056 above the public best) is
  best explained by stronger single models from official-like data and by real-model stages beyond the public's
  first version (Kimi K3 slices have been public since 29 Oct). Realistic without new data: 0.70-0.72.
