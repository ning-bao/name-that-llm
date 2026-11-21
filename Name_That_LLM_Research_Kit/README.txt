NAME THAT LLM RESEARCH KIT - 2 NOVEMBER 2026

Start with the PDF or complete text research report. The prompt file contains
24 core training prompts, six external holdout sentinels and six diagnostics.
Use one prompt per fresh conversation, and verify source identity first.
The JSON schema and template define a collection record; the template is not
training data. No source-model calls, classifier training or predictions have
been made. Raw competition passages are deliberately not included.

Reproduce read-only numeric forensics with:
  python forensic_analysis.py --input PATH_TO_SUPPLIED_FILES --output audit
Dependencies: pandas, numpy, scipy, scikit-learn, pyarrow, tiktoken.
Input directory must contain train.csv, test.csv and sample_submission.parquet.
The first tokenizer use may download its public o200k_base vocabulary.
The analysis never edits those inputs and does not fit an attribution model.

The feature statistics use whitespace word counts, ASCII lexical regexes,
heuristic sentence/LaTeX markers and one common tokenizer probe. Controlled
R-squared values are descriptive in-sample effects, not validation accuracy.
topic_audit.json contains manual inferred annotations, not recovered prompts.
The nearest-cross-prompt list is a selected nearest-neighbor diagnostic, not
an exhaustive ranking. Full threshold and maximum-similarity checks are given.

validation_plan.json freezes a POST-EDA 30-prompt confirmation set and nested
five-outer/three-inner development folds on the other 120 prompts. It also
contains a second development partition. No model errors have been used.
All broad EDA already used the full corpus; do not call this a pristine test.

Prices/access are documentation-based at the cutoff, not authenticated access
tests. Paid/private external-data eligibility and exact historical snapshots
remain conditional. gemini-3.7-flash currently redirects to 3.8 in Google's API.

To regenerate the report, place these files in the layout expected by the
build scripts: forensics/ for statistic files, inputs/ for original supplied
files, topic_audit.json and report_source.txt at root; run build_collection_kit.py
then build_research_package.py. PDF dependencies: reportlab, matplotlib and
DejaVu fonts. Build scripts perform no model training or collection.
