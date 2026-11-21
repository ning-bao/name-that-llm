# Name That LLM

Kaggle competition: attribute a single passage to one of 18 exact source-model labels
(GPT, Gemini, Claude, Qwen, GLM and Kimi versions). Metric: balanced accuracy on ~4,000
hidden passages from prompts not seen in training.

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
| `src/nllm/` | Shared library: loaders and label order, frozen splits, balanced accuracy and prompt-cluster bootstrap, submission writer, linear/style features, encoder fine-tuning, nested blending |
| `scripts/` | Runnable entry points: input audit, tokenizer audit, linear baseline (exp 1), encoder screen (exp 3), nested blend of finished runs, learning curve, external-answer collection and evaluation (exp 5) |
| `collection/` | External-data prompt bank and copy-ready prompt files; `collection/README.md` is the manual collection guide |
| `EXPERIMENTS.md` | Results log |
| `tests/` | pytest suite |
| `Name_That_LLM_Research_Kit/` | Research report, validation plan, forensic statistics and external-data collection kit |
| `data/raw/` | Competition files (not committed) |
| `outputs/` | Run artifacts (not committed) |

## Validation

Splits come from `Name_That_LLM_Research_Kit/validation_plan.json`: 30 prompt groups are
locked for a single final confirmation; the other 120 carry nested cross-validation
(2 partitions × 5 outer folds × 3 inner folds). All rows from a prompt stay together.
See section F of the research report for the full protocol.

## Data

Competition data is not included: download it from the Kaggle competition page into `data/raw/`
(the competition rules restrict redistribution). Corpus statistics derived from the training set
(`forensic_summary.json`, `model_statistics.csv`, `family_statistics.csv`,
`near_duplicate_candidates.csv`) are omitted from the research kit for the same reason;
`Name_That_LLM_Research_Kit/forensic_analysis.py` regenerates them from your own copy.
`scripts/audit_inputs.py` checks your files against `Name_That_LLM_Research_Kit/data_hashes.json`.

## License

MIT, see `LICENSE`.
