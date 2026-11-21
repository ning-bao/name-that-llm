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
