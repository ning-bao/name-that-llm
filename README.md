# Name That LLM

Kaggle competition: attribute a single passage to one of 18 exact source-model labels
(GPT, Gemini, Claude, Qwen, GLM and Kimi versions). Metric: balanced accuracy on ~4,000
hidden passages from prompts not seen in training.

## Status and findings

The project is finished and archived. The full write-up is in [`report/report.pdf`](report/report.pdf)
(LaTeX source in `report/`), and the run-by-run log is in `EXPERIMENTS.md`.

- Best single model: a truncated, multi-depth-pooled Qwen3-8B LoRA classifier, at 0.623 balanced accuracy
  in nested prompt-grouped CV and 0.633 on the locked confirmation prompts. On the public leaderboard it
  scores 0.619 with test-time soft centring. The stylometric linear baseline scores 0.355.
- Data mattered most. Pre-training on the 212k-passage community pool added +0.14 BA to the same encoder.
  Architecture changes, ensembling, sibling-focused training and within-provider specialists each added
  0.01 or less.
- The provider family is right on 85% of passages, so most remaining errors confuse sibling models. Word
  choice separates providers and model generations, but not close siblings.
- The final public score was 0.684, from the best public notebook at the time. Adding our 8B model to that
  ensemble did not raise it (0.680).

Trained weights are not in this repository.

## Setup

Requires [uv](https://docs.astral.sh/uv/). Place the competition files in `data/raw/`
(`train.csv`, `test.csv`, `sample_submission.parquet`), then:

```bash
uv sync
uv run python scripts/audit_inputs.py
uv run pytest
```

For encoder/neural experiments, `uv sync --extra nn` installs torch (CUDA 12.8 wheels),
transformers, accelerate, peft and the SentencePiece tokenizer dependencies.

## Layout

| Path | Contents |
| --- | --- |
| `src/nllm/` | Shared library: loaders and label order, frozen splits, balanced accuracy and prompt-cluster bootstrap, submission writer, linear/style features, encoder fine-tuning, nested blending, likelihoods under the open-weight candidate models |
| `scripts/` | Runnable entry points: input audit, tokenizer audit, linear baseline (exp 1), encoder screen (exp 3), nested blend of finished runs, learning curve, external-answer collection and evaluation (exp 5) |
| `collection/` | External-data prompt bank and copy-ready prompt files; `collection/README.md` is the manual collection guide |
| `EXPERIMENTS.md` | Results log |
| `report/` | Final report (PDF and LaTeX source, figures, analysis scripts) |
| `tests/` | pytest suite |
| `Name_That_LLM_Research_Kit/` | Research report, validation plan, forensic statistics and external-data collection kit |
| `data/raw/` | Competition files (not committed) |
| `outputs/` | Run artifacts (not committed) |

## Validation

Splits come from `Name_That_LLM_Research_Kit/validation_plan.json`: 30 prompt groups are
locked for a single final confirmation; the other 120 carry nested cross-validation
(2 partitions × 5 outer folds × 3 inner folds). All rows from a prompt stay together.
See section F of the research report for the full protocol.
