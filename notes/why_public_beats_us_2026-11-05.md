# Why the public notebooks score 0.67 and we score 0.59 (2026-11-05)

Sources: public notebooks pulled with `kaggle kernels pull` (source only, not run), forum threads 754860, 753177 and
756088, competition rules section 6. Our numbers come from `EXPERIMENTS.md`. Public numbers are the authors' own
claims. Their CV is 5-fold over all 150 prompts (120 training prompts per fold); ours is nested over the 120 dev
prompts (96 training prompts per outer fold).

## The headline: same data, better recipe

| Qwen3-8B LoRA | Official data only | + ~35k pool stage 1, stage 1 alone | + stage 2 on official |
| --- | --- | --- | --- |
| Public "LoRA24" recipe (lucasboesen, nishantdahal) | 0.480 | 0.483 | **0.569** |
| Ours (Exp 4, Exp 4b) | 0.396 | 0.489 / 0.471 | **0.507 / 0.508** |

- With pool data, both stage 1s land in the same place. The gap opens in **stage 2 on official rows**: theirs adds
  about +0.09, ours +0.02 to +0.04. Our Ettin runs show the same pattern (stage 2 adds +0.007 to +0.018).
- On official data alone, the recipe is worth about +0.08 at the same backbone.
- Their folds train on 25% more prompts. By our learning curve (+3.6 BA per doubling of prompts), that explains
  only about 1 point.

## The LoRA24 recipe (what differs from ours)

1. **Truncated backbone:** the first 24 of 36 Qwen3-8B layers (27 of 40 for 14B, 32 of 48 for Qwen2.5-14B), with
   the final norm off. The authors' rationale: style lives in early and middle layers, while late layers carry
   topic, which does not transfer to unseen prompts.
2. **Multi-layer mean pooling:** mean over tokens (first token masked) at layers 2, 4, 6, 8, 12, 16, 20 and 24,
   each with its own LayerNorm, concatenated (8 × 4096) into a linear head. Ours pools the last token of the final
   layer.
3. **Within-prompt contrast in training:** batches are triplets of answers to the same prompt (6 per step). The loss
   is the CE of the plain head plus the CE of a second, "centred" head on triplet-mean-subtracted features. The
   official data has all 18 labels per prompt, which may be why their stage 2 is where the big gain comes.
4. **Learning rates and schedule:** LoRA r16 / α32 / dropout 0.05 on all projections. Heads at 10× the LoRA lr.
   Stage 1: 1 epoch at 1e-4 / 1e-3, class-weighted CE over the whole pool, 10% warmup then cosine. Stage 2: 2
   epochs at 5e-5 / 5e-4. More stage-2 epochs over-fit, and mixing pool rows back into stage 2 hurt.
5. **max_len 384**, bf16 training. An 8B fit takes about 30 min and a 14B about 90 min on an RTX 5090 (35k pool).

## Then data volume and ensemble breadth

- **More pool data, per member:** retraining a 14B member on the 120k pool raised its OOF from 0.569 to 0.605. The
  nlm-v27 notebook retrained three members on that pool, taking the LB from 0.619 to 0.651. GoPain's Qwen3-8B on
  train + Open Pool v10 scores 0.622 OOF / 0.628 LB. Single models reach 0.65-0.67 on the LB, and 3rd place reports
  0.670 from one Qwen3-14B with private data.
- **Ensemble:** the 0.672 notebook blends six LoRA24-style members (Qwen3-14B ×2, Qwen3-8B, Llama-3.1-8B,
  Qwen2.5-14B, DeBERTa) with a 4-view n-gram SVC on the 190k pool and smaller parts. Removing the Qwen3-14B member
  costs 0.010-0.020 on the LB, the most of any part. Members trained on a different data mix add more than their
  OOF suggests.
- **Test-time prompt structure:** soft centring within inferred prompt clusters is worth +0.006 to +0.017 on the
  LB. Hungarian assignment loses 0.009-0.017, because the hidden test has few answers per prompt.
- **Each other part (SVC, ATLAS, verifier, typography likelihood, Sinkhorn 0.30)** is worth 0.001-0.011 depending
  on the base, often within the 0.009 repeat-submission noise.

## Things that are not the gap

- **Data quality:** Open Pool v13 already respects the hard typography invariants (Gemini em dash 0.000;
  Claude/GLM curly quotes 0.000).
- **Encoders:** our Ettin-400M with 5,000 pool rows per label (0.566 nested) is close to their ModernBERT-large
  member (0.588 OOF).
- **Final training data:** we hold out the 30 confirmation prompts, so our final fits use 120 prompts instead of
  150. That costs roughly 1 point.

## Rough decomposition of the 0.08 LB gap (estimates)

| Source | Estimated LB effect |
| --- | --- |
| Decoder recipe (truncation + multi-layer pooling + within-prompt contrast + head lr) | +0.03 to +0.05 via much stronger decoder members |
| Pool size for decoders (36k → 120-200k) | +0.02 to +0.035 per member |
| More and larger decoder members (14B, Llama, Qwen2.5) | +0.01 to +0.02 |
| Soft prompt centring at test time | +0.006 to +0.017 |
| 150 instead of 120 training prompts in final fits | about +0.01 |
| SVC, ATLAS, verifier, typography, Sinkhorn | 0 to +0.01 combined, near noise |

The parts overlap, so they don't add up exactly.
