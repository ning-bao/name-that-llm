# Kaggle forum summary, 3 Nov 2026

All 93 topics and 241 comments of the Name That LLM forum, as of 3 Nov 2026 (forum opened 14 Oct). The raw
export is in `outputs/kaggle_discussions/` (gitignored). Topic ids in brackets are
`https://www.kaggle.com/competitions/name-that-llm/discussion/<id>`. Unless marked as a host statement,
everything is a participant's claim, mostly unverified. The public leaderboard is scored on about 40% of the test.

## Where the leaderboard is, and what gets there

| Approach | Score | Source |
| --- | --- | --- |
| TF-IDF char+word + LR/SVC, train only | LB 0.31–0.36 (group CV 0.34–0.37) | [753526], [749593], [750390], [751802] |
| Frozen Qwen3-4B early layers + TF-IDF | CV 0.439 / LB 0.420 | [750462] |
| Qwen3-14B-Base LoRA, train only | fold-0 CV 0.443 / LB 0.466 | [750390] |
| Qwen3.5-9B (19 layers) LoRA, train only | CV 0.503 / LB 0.483 | [751871] |
| n-gram SVC trained with a community pool | CV 0.467 / LB 0.423 | [750390] |
| Two-stage Qwen3-8B LoRA (stage 1 on pool, stage 2 on train) | CV 0.576 / LB 0.538–0.549 | [750390] |
| Best single model (one Qwen adapter, Open Pool v10) | LB 0.637–0.65 | [754860] |
| Public blend notebooks (Qwen 8B/14B, Llama-8B LoRA members, DeBERTa, SVC, centred twins, ATLAS, Sinkhorn) | LB 0.60–0.663 | [750447], [753177], [754131] |
| Gold zone / top of the LB | about 0.653 / 0.700 | [752357] |

Our numbers for comparison (nested CV on 96 training prompts, partition 0): text+style 0.348, Ettin-400m 0.388,
text+style + Ettin + RoBERTa 0.425. Train-only results in the forum line up with ours. The gap to the top is
external data plus 8–14B decoder LoRAs.

## Host statements

- **Data spending and generation ([752084], MartynaPlomecka, 28–29 Oct).**
  - "not a data generation competition".
  - Buying data with token spend beyond the value of a prize (about $5k) "would be considered unreasonable".
  - Self-generated data "within reasonable cost" need not be published.
  - The Kaggle Benchmarks model-proxy quota is for benchmark runs, not for generating training data. Earlier
    use is not penalised.
  - Public community datasets, including proxy-generated ones, and public model weights on Kaggle are fine to
    use.
  - For the winners' write-up, an OSI-approved licence is acceptable.
- **Dataset description (quoted in [750408]).** Only light formatting cleanup was applied, for example removing a
  bold wrapper around a whole passage. [749455] relays that each generation had its own random target length.
- **Infrastructure ([750924], Dustin).**
  - Reading datasets mounted under /kaggle/input was slow.
  - Load weights attached as a Kaggle Model, or copy them to /tmp first.
- **Not answered by hosts:**
  - How passages were generated: API or app, settings, system prompts ([749664], [750059], [752280]).
  - Answers per test prompt, and whether the public/private split is by prompt or by passage ([750768], [751802]).
  - Whether test-time use of relationships between test passages (clustering, pseudo-labels) is allowed ([752084]).
  - Whether the 3.7 labels predate Google's 8 Oct routing of 3.7 to 3.8 ([751954]).

## External data in circulation

Licences are those shown on Kaggle. Sizes are as reported in the forum.

| Dataset | Content | Reported effect |
| --- | --- | --- |
| crtproduction/name-that-llm-open-pool (CC BY-SA 4.0) | 190,588 passages merged from about 59 community datasets. Cleaned: API dumps, refusals, cut-offs, answers to train prompts and typography a label never uses in train are removed. 592 prompts have all 18 labels. Also has `prompts.parquet` (13k), per-row `match_weight`, a `flag_kb_proxy` flag (about 70k proxy rows) and a separate file of unverified 3.7 rows. | Cleaning +0.027; `match_weight` sampling +0.026; Vertex-only 3.7 +0.009 (Qwen3-1.7B LoRA, scored on train) [752766], [753779] |
| nuridudak/ntl-130k-clean-passages (CC BY-SA 4.0) | All public sets merged and deduplicated, 124k rows. Contains 96 `train_regen` rows that answer train prompts. | Pool 36k → 124k: CV 0.618 → 0.639, LB 0.586 → 0.617 [752810] |
| prvsiyan extended (CC BY 4.0) | 631 prompts × 18 models, API-logged. Has a `train_regen` split that answers the host's train prompts, which leaks into CV. | −0.009 in the Open Pool test [752766] |
| olegzholobov response corpus (mixed licences) | 135k responses from HF datasets and eval logs, mostly multi-paragraph and Markdown. | Hurt CV: 0.326 → 0.279 when flattened [749629] |
| enaimaan OpenAI responses | 9,719 GPT-5.6 / GPT-6 rows with exact prompts and settings. | Not reported |
| seyitkaangunes extra API passages (CC BY 4.0) | About 5,000 Claude paragraphs on 1,000 new prompts. | Best used for a within-Claude head; in the 18-class training it hurt other classes [750224] |
| danielandresmiranda OpenRouter GLM/Kimi (CC BY 4.0) | 1,516 passages (glm-5.3 / glm-5.3-flash / kimi-k3). | Filling the uncovered labels: 0.447 → 0.508 [751871] |
| darragh1walsh ntl-kaggle-ai-gen (CC0) | Daily Kaggle-allowance generations, few details. | Not reported |
| pjmathematician reconstructed train prompts | 150 train prompts guessed by an LLM from the passages; no passages. | Not applicable |

Lessons the forum reports about external data:

- **Coverage of all 18 labels matters.** With partial coverage, the uncovered labels lost about 6 recall points
  because the model learned "new prompt means not GLM" [751871].
- **Size helps.** About +0.03 CV per doubling of matched data [752478]; 35k → 84k was the biggest single lever,
  +0.028 [752357].
- **Format must match.** Passages should be single paragraphs, with no line breaks or bold [749447], [749629].
- **Pool-trained CV runs 0.02–0.04 above the LB** [751941], [750390]. Compare pool-trained models with each
  other, not with train-only models.

## Typography and Gemini

- **Gemini em dashes.** No one has reproduced official Gemini's near-zero em-dash rate.
  - Proxy or API Gemini text has em dashes in 13–43% of passages, depending on reasoning settings ([750664],
    citing dataset notes).
  - Community pools therefore drop Gemini passages that contain an em dash, and Claude passages that contain a
    curly apostrophe ([750390], [753779]).
  - The pool's Gemini passages also use word hyphens in 44% of passages, against 12% in official [752766].
- **Lab-level rules.**
  - Claude and GLM: em dashes, never curly apostrophes.
  - GPT and Gemini: curly apostrophes.
  - A spaced em dash (" — ") appears only in claude-opus-4-6, claude-opus-5 and both GLMs [751871].
- **3.7 vs 3.8.** Two separate 3.7-vs-3.8 classifiers scored AUC 0.498 inside one source, for 3.7 data from
  sources other than Vertex. That is chance level, so those 3.7 labels are likely 3.8 output [752766]. Vertex
  still serves 3.7 and is the only source people trust for it.
- **OpenRouter reasoning.** Only kimi-k3 accepts reasoning off, and without reasoning its em-dash rate matches
  train better [751871]. This matches our Moonshot finding.

## Test structure and test-time tricks

- **The test probably has few passages per prompt.** Participants infer 2–18 per content cluster rather than 18
  ([750706], [752708]). The hosts haven't confirmed it.
- **Hungarian or within-cluster assignment:**
  - Gains about 0.10 in train CV, where every prompt has all 18 labels.
  - Flat or negative on the LB ([751802], [750706], [750390], [751871]).
- **Light centring inside content clusters** helps a little: LB 0.595 → 0.603 at weight 0.25. A "centred twin
  head" reached 0.620 [752708], [753053]. Heavier centring hurts.
- **Sinkhorn class balancing:**
  - 1–3 iterations give small gains (+0.002 to +0.011 LB).
  - Full balancing can lose a lot when the class mix is uneven ([752365]).
  - Under balanced accuracy, the optimal decision does not depend on the test prior.
- **Use a test-like validation proxy.** Subsample each validation prompt to 2–3 answers. That proxy tracked the
  LB better than full-prompt OOF (Pearson 0.79 vs 0.13 over 15 submissions, outlier-driven) [753177].
- **LB noise:**
  - The noise of a gap between two submissions is about 0.009–0.013 for independent models and 0.003 for
    near-identical blends [754084].
  - Identical notebooks have scored up to 0.009 apart [754427].

## Kaggle runtime

- **Hardware.**
  - T4 ×2, fp16 (no bf16). A 14B model fits across both GPUs, merged LoRA, at about 17–20 min per 4,000 rows
    [750390].
  - The free TPU v5e-8 trains a Qwen3-8B LoRA about 14× faster than 2×T4, but its queue can take about a day
    [754657].
- **Loading weights.** Attach weights as a Kaggle Model, or copy them to /tmp, because dataset mounts can be
  very slow ([750924], [751268]).
- **Hidden reruns.**
  - They fail opaquely. Wrap each stage and always write `submission.parquet` [750553].
  - `KAGGLE_IS_COMPETITION_RERUN` detects the real rerun [754426].

## Implications for this project (my reading, for review)

1. **Our CV is in line with the forum's train-only numbers.** The forum's train-only numbers hold up; the large
   gains come from external data at 50k–190k passages and from 8–14B decoder LoRAs. Our own collection (about
   3,500 rows) is small next to the community pools.
2. **Gemini em dashes are an unsolved serving difference for everyone.** The community answer is to filter
   Gemini passages that contain em dashes, not to match the serving setup. Our leaked-prompt tests point the same
   way:
   - the app prompt gives 50% (n = 20);
   - the Workspace prompt gives 14% (n = 56);
   - plain API calls give 35% on the same 60 prompts;
   - official is 0 of 120.
3. **Test-time assignment tricks don't transfer**, consistent with the rule in CLAUDE.md. Light centring and
   1–3 Sinkhorn steps are the only test-time steps with LB support.
4. **Leakage risk.** `train_regen` rows and the reconstructed train prompts answer the host's train prompts.
   They must be removed before any prompt-grouped CV, or CV is inflated.
