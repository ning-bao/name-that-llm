# External answers: manual collection guide

The goal is more training passages written by the same 18 models, collected through your own
chat subscriptions. More independent prompts are what helps: the learning curve in
`EXPERIMENTS.md` gains about 3.6 balanced-accuracy points each time the number of training
prompts doubles. These files follow section E of the research report.

`gemini-3.7-flash` is not collected. Google now routes 3.7 requests to 3.8, and 3.8 output
must never be labelled 3.7. That leaves 17 models.

## What is here

| Path | Contents |
| --- | --- |
| `prompts_v2.tsv` | Source for the 160 new prompts (wave2 to wave5), one row each with domain, genre and length band |
| `prompt_bank.json` | All 196 prompts with id, family, split and wave (built from the kit bank plus `prompts_v2.tsv`) |
| `prompts/<wave>.md` | Each wave's prompts, ready to copy |
| `data/external/answers/<wave>/<model>.txt` | Answer files to paste into (local only, never committed) |

| Pack | Prompts | Answers for 17 models |
| --- | --- | --- |
| wave1 | the kit's 24 core + 6 holdout | 510 |
| wave2 to wave5 | 40 new each (32 train + 8 holdout) | 680 each |
| diagnostic | the kit's 6 diagnostic prompts (optional, kept separate) | 102 |

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

## Testing whether it helps

```bash
uv run python scripts/external_data_eval.py              # all admitted answers
uv run python scripts/external_data_eval.py --waves wave1
```

This compares official-only, external-only and combined training on the same official
held-out prompts. It reports paired confidence intervals and Gemini sibling recalls, and
checks transfer on your holdout prompts. Results go in `EXPERIMENTS.md`. External data joins
the main models only if the combined condition beats official-only in both partitions.

The answers' competition eligibility is recorded as `pending_clarification` until the
organizers confirm the plan (report I1). Once they do, run the importer with
`--eligibility organizer_confirmed`. `--eligible-only` restricts the comparison to
confirmed answers.
