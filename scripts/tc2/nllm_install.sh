#!/bin/bash
# Experiment 4 on TC2, login-node side: ONLY file operations + sbatch/squeue/sacct (no python, conda or downloads).
# Everything stays under $HOME/nllm. Idempotent: verifies uploads by sha256, unpacks the code once into
# nllm/repo_<code id>, installs train.csv and the baseline predictions once, then submits the named job unless it is
# already queued, running or completed. The account is shared (QoS: 2 submitted jobs per user), so a refused
# submission stops here. Never scancel.
#   bash nllm_install.sh unpack                # verify + install uploads only
#   bash nllm_install.sh submit NAME [CHUNK [AFTER]]   # one job from the table below; nested runs take CHUNK 1, 2,
#                                              # ... (each chunk resumes the same run dir; job name NAME_c<CHUNK>).
#                                              # AFTER: a job id this one waits for (--dependency=afterany, so a
#                                              # failed predecessor never leaves it pending in a slot forever)
set -uo pipefail
cd "$HOME" || exit 1
[ "$(pwd)" = "$HOME" ] || { echo "unexpected home $(pwd)"; exit 1; }
B=$HOME/nllm; U=$B/upload; mkdir -p "$B/logs" "$B/runs" "$B/data"
STAGE=${1:?stage: unpack | submit NAME [CHUNK]}
if [ -f "$U/nllm_code.tar" ]; then
  (cd "$U" && sha256sum -c nllm_code.sha256) || { echo "code archive sha256 mismatch"; exit 1; }
  C=$(cat "$U/CODE_ID"); R=$B/repo_$C
  if [ -d "$R" ]; then echo "code already unpacked at $R"; else
    mkdir "$R" && tar -xf "$U/nllm_code.tar" -C "$R" || exit 1
    (cd "$R" && sha256sum -c --quiet "$U/nllm_payload.sha256") || { echo "payload sha256 mismatch"; exit 1; }
    cp "$U/CODE_ID" "$R/CODE_ID"
    echo "unpacked $R"
  fi
  rm -f "$U/nllm_code.tar"
fi
if [ -f "$U/train.csv" ]; then
  (cd "$U" && sha256sum -c nllm_data.sha256) || { echo "data sha256 mismatch"; exit 1; }
  mkdir -p "$B/data/baseline"
  mv "$U/train.csv" "$B/data/train.csv" && mv "$U/baseline_oof.parquet" "$B/data/baseline/oof.parquet" || exit 1
  cp "$U/nllm_data.sha256" "$U/BASELINE_RUN" "$B/data/" && touch "$B/data/.ready"
  echo "data installed (baseline $(cat "$B/data/BASELINE_RUN"))"
fi
if [ -f "$U/nllm_pool.tar" ]; then
  (cd "$U" && sha256sum -c nllm_pool.sha256) || { echo "pool sha256 mismatch"; exit 1; }
  rm -rf "$B/data/pool" && mkdir -p "$B/data/pool" && tar -xf "$U/nllm_pool.tar" -C "$B/data/pool" || exit 1
  cp "$U/nllm_pool.sha256" "$B/data/pool/" && rm -f "$U/nllm_pool.tar" && touch "$B/data/pool/.ready"
  echo "pool data installed"
fi
C=$(cat "$U/CODE_ID"); R=$B/repo_$C
[ -d "$R" ] || { echo "no unpacked repo for code $C"; exit 1; }
[ "$STAGE" = unpack ] && exit 0
[ "$STAGE" = submit ] || { echo "unknown stage $STAGE"; exit 1; }

seen() { squeue -h -u "$USER" -o "%j" | grep -qx "$1" || sacct -n -X -u "$USER" -S 2026-11-02 -o JobName%30,State | grep -qE "^ *$1 +(COMPLETED|RUNNING|PENDING)"; }
submit() {  # job-name sbatch-file [extra sbatch args...]
  local name=$1 file=$2; shift 2
  if seen "$name"; then echo "$name already queued/running/completed"; return 0; fi
  local out; out=$(sbatch --parsable --nodelist=TC2N03 -J "$name" "$@" "$R/scripts/tc2/$file" 2>&1) \
    && { echo "SUBMITTED $name=$out (code $C)"; return 0; } || { echo "NOT SUBMITTED $name: $(echo "$out" | head -c 160)"; return 1; }
}
echo "queue before:"; squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
NAME=${2:?job name}; CHUNK=${3:-1}; AFTER=${4:-}
case $CHUNK in [1-9]) ;; *) echo "NOT SUBMITTED $NAME: bad chunk $CHUNK"; exit 1 ;; esac
DEP=()
if [ -n "$AFTER" ]; then
  case $AFTER in *[!0-9]*) echo "NOT SUBMITTED $NAME: bad job id $AFTER"; exit 1 ;; esac
  squeue -h -u "$USER" -j "$AFTER" -o "%i" 2>/dev/null | grep -qx "$AFTER" || { echo "NOT SUBMITTED $NAME: job $AFTER is not queued or running"; exit 1; }
  DEP=(--dependency="afterany:$AFTER")
fi
if [ "$NAME" = nllm_env ]; then submit nllm_env nllm_env.sbatch --export=ALL,REPO="$R"; exit $?; fi
[ -f "$B/venv/.ready" ] && [ -f "$B/data/.ready" ] || { echo "NOT SUBMITTED $NAME: env/data not ready"; exit 1; }
case $NAME in *pool*) [ -f "$B/data/pool/.ready" ] || { echo "NOT SUBMITTED $NAME: pool data not ready"; exit 1; } ;; esac
# --fit-minutes reserves time for the longest single fit. Measured on the L40S (nllm_pilot8b, 2026-11-04): one 8B fit
# of 8 epochs on 1,151 rows takes 12.9 min (about 20 rows/s), peak 27.9 GB.
NESTED="--time-budget-minutes 340 --baseline $B/data/baseline"
# Final fits of the other Kaggle ensemble members, run in turn by one job (nllm_multi.sbatch): Ettin-400M seeds 43
# and 44 with nllm_e400_final's recipe (about 11 min each), and Ettin-1B with the Exp 6e recipe (its nested runs chose
# lr 1e-5 and epoch 3 in 6 of 10 folds, epoch 2 otherwise).
if [ "$NAME" = nllm_finals_ens1 ]; then
  F400="--final --micro-batch 8 --pool-per-label 2500 --pool-recipe w --pool-epochs 2 --lrs 1e-5 --epochs 6 --final-epoch 3"
  F1B="--final --optimizer adafactor --micro-batch 4 --gradient-checkpointing --pool-per-label 1500 --pool-recipe wx_own --pool-epochs 1 --lrs 1e-5 --epochs 3 --final-epoch 3"
  # 1B first: its stage-1 and final weights (about 4 GB each, fp32) peak at 8 GB before any other archive exists.
  RUNS="ettin-1b-final-001|ettin-1b|$F1B;ettin-400m-final-s43-001|ettin-400m|$F400 --seed 43;ettin-400m-final-s44-001|ettin-400m|$F400 --seed 44"
  submit "${NAME}_c$CHUNK" nllm_multi.sbatch --export=ALL,REPO="$R",RUNS="$RUNS"
  STATUS=$?; squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"; exit $STATUS
fi
POOL="--pool-recipe wx_own --pool-epochs 1 --lrs 1e-4 --epochs 3"
L24="--pool-layers 2 4 6 8 12 16 20 24 --head-lr-mult 10 --centred-weight 1 --schedule cosine --batch-size 18 --micro-batch 6 --lrs 5e-5 --epochs 3"
case $NAME in  # job name | run id | backbone | script args (8B: micro-batch 4 x accumulation 8, no checkpointing)
  nllm_pilot8b) RUN=pilot-qwen3-8b-base-001; BB=qwen3-8b-base; ARGS="--pilot --save-model --micro-batch 4" ;;
  nllm_pilot4b) RUN=pilot-qwen3-4b-base-001; BB=qwen3-4b-base; ARGS="--pilot --save-model --micro-batch 8" ;;
  nllm_8b_p0)   RUN=qwen3-8b-base-p0-001;    BB=qwen3-8b-base; ARGS="--partitions 0 --micro-batch 4 --fit-minutes 15 $NESTED" ;;
  nllm_8b_p1)   RUN=qwen3-8b-base-p1-001;    BB=qwen3-8b-base; ARGS="--partitions 1 --micro-batch 4 --fit-minutes 15 $NESTED" ;;
  nllm_4b_p0)   RUN=qwen3-4b-base-p0-001;    BB=qwen3-4b-base; ARGS="--partitions 0 --micro-batch 8 --fit-minutes 15 $NESTED" ;;
  nllm_4b_p1)   RUN=qwen3-4b-base-p1-001;    BB=qwen3-4b-base; ARGS="--partitions 1 --micro-batch 8 --fit-minutes 15 $NESTED" ;;
  # Pool stage 1 (2,000 rows per label, 1 epoch: about 30 min for the 8B) then a short official stage 2 (3 epochs,
  # about 5 min per fit); the stage-1 fit gets its own, longer reserve. 8B partition 0 took 3 h 48 min in one chunk.
  nllm_4bpool_p0) RUN=qwen3-4b-pool-p0-001; BB=qwen3-4b-base
                  ARGS="--partitions 0 --micro-batch 8 --pool-per-label 2000 $POOL --fit-minutes 10 --stage1-minutes 30 $NESTED" ;;
  nllm_8bpool_p0) RUN=qwen3-8b-pool-p0-001; BB=qwen3-8b-base
                  ARGS="--partitions 0 --micro-batch 4 --pool-per-label 2000 $POOL --fit-minutes 10 --stage1-minutes 40 $NESTED" ;;
  nllm_8bpool_p1) RUN=qwen3-8b-pool-p1-001; BB=qwen3-8b-base
                  ARGS="--partitions 1 --micro-batch 4 --pool-per-label 2000 $POOL --fit-minutes 10 --stage1-minutes 40 $NESTED" ;;
  # Ettin-400M with a pool stage 1 (Exp 6c recipe; about 13 min stage 1 and 2 min per fit on a 5070 Ti), moved off
  # the local GPU. SEED picks the initialisation; the pool sample is seeded by fold.
  nllm_e400pool_s43_p1) RUN=ettin-400m-pool-s43-p1-001; BB=ettin-400m
                  ARGS="--partitions 1 --micro-batch 8 --pool-per-label 2500 --pool-recipe w --pool-epochs 2 --lrs 1e-5 3e-5 --epochs 6 --seed 43 --fit-minutes 10 --stage1-minutes 25 $NESTED" ;;
  # Third seed, both partitions in one job (seed 43 on p1 took 78 min on the L40S).
  nllm_e400pool_s44) RUN=ettin-400m-pool-s44-001; BB=ettin-400m
                  ARGS="--partitions 0 1 --micro-batch 8 --pool-per-label 2500 --pool-recipe w --pool-epochs 2 --lrs 1e-5 3e-5 --epochs 6 --seed 44 --fit-minutes 10 --stage1-minutes 25 $NESTED" ;;
  # Pool stage-1 size: Ettin-400M seed 17 with 5,000 rows per label instead of 2,500 (87,549 rows; labels with fewer
  # pool rows use all of them), otherwise the Exp 6c recipe, so the comparison with seed 17 isolates data size.
  # About 4.1 h: stage 1 ~18 min per fold (6.2 ms per row-epoch on the L40S), stage 2 ~6 min per fold.
  nllm_e400pool_n5000) RUN=ettin-400m-pool5000-001; BB=ettin-400m
                  ARGS="--partitions 0 1 --micro-batch 8 --pool-per-label 5000 --pool-recipe w --pool-epochs 2 --lrs 1e-5 3e-5 --epochs 6 --seed 17 --fit-minutes 10 --stage1-minutes 45 $NESTED" ;;
  # Second Ettin-1B seed, both partitions, Exp 6e recipe unchanged (adafactor, checkpointing, 1,500 rows per label
  # from wx_own, 1 epoch). Locally about 2 h per partition: stage 1 15 min and 2-3 min per fit.
  nllm_e1bpool_s43) RUN=ettin-1b-pool-s43-001; BB=ettin-1b
                  ARGS="--partitions 0 1 --optimizer adafactor --micro-batch 4 --gradient-checkpointing --pool-per-label 1500 --pool-recipe wx_own --pool-epochs 1 --lrs 1e-5 --epochs 3 --seed 43 --fit-minutes 10 --stage1-minutes 30 $NESTED" ;;
  # The public LoRA24 recipe (notes/why_public_beats_us_2026-11-05.md): first 24 layers, mean-pooled at 8 depths
  # with a LayerNorm each, same-prompt triplet batches (18 rows) with a centred head, heads at 10x lr, cosine.
  # nllm_8b_l24_p0 keeps Exp 4b's per-fold stage 1 (wx_own, 2,000 per label, 1 epoch at 1e-4) and changes only the
  # model and stage 2, so it compares with Exp 4b's 0.507 on partition 0. About 3 h.
  # nllm_8b_l24full trains one class-weighted stage 1 on every pool + extra row (recipe ux, uniform, so no dev rows
  # are read; ~212k rows, ~2-2.5 h), shared by all folds of both partitions, then the nested stage 2. ~4.5 h.
  nllm_8b_l24_p0) RUN=qwen3-8b-l24-p0-001; BB=qwen3-8b-base
                  ARGS="--partitions 0 $L24 --pool-per-label 2000 --pool-recipe wx_own --pool-epochs 1 --pool-lr 1e-4 --fit-minutes 10 --stage1-minutes 35 $NESTED" ;;
  nllm_8b_l24full) RUN=qwen3-8b-l24-full-001; BB=qwen3-8b-base
                  ARGS="--partitions 0 1 $L24 --pool-per-label 30000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --shared-stage1 --fit-minutes 10 --stage1-minutes 200 $NESTED" ;;
  # Qwen3-14B with the LoRA24 recipe, both partitions. 24 layers kept (the deepest pooled depth: layers above it
  # cannot change the features), micro-batch 3 (one triplet) for memory (~17 GB of bf16 weights). Stage 1 is the
  # shared uniform pool capped at 10,000 per label (~167k rows), so it fits one job (~3.9 h at ~1.8x the 8B's cost).
  # Stage 2 (~4.3 h) continues in chunk 2 from the saved shared stage 1. Downloads Qwen3-14B-Base (~30 GB) once.
  # --keep-stage1 (added before chunk 2) keeps the shared stage 1 for nllm_14b_l24full_final.
  nllm_14b_l24full) RUN=qwen3-14b-l24-full-001; BB=qwen3-14b-base
                  ARGS="--partitions 0 1 ${L24/--micro-batch 6/--micro-batch 3} --pool-per-label 10000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --shared-stage1 --keep-stage1 --fit-minutes 15 --stage1-minutes 300 $NESTED" ;;
  # The 14B on all 120 dev prompts, stage 2 only, from the nested run's shared stage 1 (--stage1-from: the same
  # uniform pool model, read no dev rows, never deleted). Stage 2 stops at the nested runs' modal epoch. ~20 min.
  nllm_14b_l24full_final) RUN=qwen3-14b-l24-full-final-001; BB=qwen3-14b-base
                  ARGS="--final ${L24/--micro-batch 6/--micro-batch 3} --final-epoch 3 --pool-per-label 10000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --stage1-from $B/runs/qwen3-14b-l24-full-001/stage1/shared" ;;
  # Plan E3 (2026-11-08): a second stage-1 epoch for the 8B. Continues the final run's saved stage 1 (uniform ux,
  # 212k rows, 1 epoch at 1e-4, which read no dev rows) for one more epoch at 5e-5 on a reshuffled sample
  # (--pool-seed 1), then the nested stage 2 on both partitions; compare with Exp 4d's 0.623 / 0.624. ~2.7 h + 2.5 h.
  nllm_8b_l24_s1x2) RUN=qwen3-8b-l24-s1x2-001; BB=qwen3-8b-base
                  ARGS="--partitions 0 1 $L24 --pool-per-label 30000 --pool-recipe ux --pool-epochs 1 --pool-lr 5e-5 --pool-class-weighted --pool-seed 1 --shared-stage1 --stage1-init $B/runs/qwen3-8b-l24-full-final-001/stage1 --keep-stage1 --fit-minutes 10 --stage1-minutes 180 $NESTED" ;;
  # Plan E4: a second decoder family for ensemble diversity, Qwen2.5-14B (48 layers; the same 24 kept and pooled
  # depths as the Qwen3 members), with nllm_14b_l24full's 10k-cap shared stage 1. ~3.5 h stage 1 + ~4 h stage 2.
  nllm_q25_14b_l24full) RUN=qwen2.5-14b-l24-full-001; BB=qwen2.5-14b
                  ARGS="--partitions 0 1 ${L24/--micro-batch 6/--micro-batch 3} --pool-per-label 10000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --shared-stage1 --keep-stage1 --fit-minutes 15 --stage1-minutes 300 $NESTED" ;;
  # Plan E2: 8B stage-2 sweep on partition 0, from the final run's saved stage 1 used as-is (the same model as
  # Exp 4d's shared stage 1: same uniform sample and seed). lr 5e-5 and 1e-4 on a 5-epoch cosine. ~3.6 h.
  nllm_8b_l24_s2sweep) RUN=qwen3-8b-l24-s2sweep-001; BB=qwen3-8b-base
                  ARGS="--partitions 0 ${L24/--lrs 5e-5 --epochs 3/--lrs 5e-5 1e-4 --epochs 5} --pool-per-label 30000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --shared-stage1 --stage1-from $B/runs/qwen3-8b-l24-full-final-001/stage1 --fit-minutes 10 $NESTED" ;;
  # The full-pool LoRA24 8B (Exp 4d, 0.623 / 0.624 nested) on all 120 dev prompts for Kaggle: the same uniform,
  # class-weighted stage 1 over every pool + extra row, then stage 2 stopped at epoch 3 (chosen in 9 of 10 folds).
  # Reference rows are scored through nllm.inference's peft-free loader in bf16 and fp16. About 3 h; stage 1 kept.
  nllm_8b_l24full_final) RUN=qwen3-8b-l24-full-final-001; BB=qwen3-8b-base
                  ARGS="--final $L24 --final-epoch 3 --pool-per-label 30000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --keep-stage1" ;;
  # The same 8B on all 150 prompts (--all-prompts: the 30 confirmation prompts were used once on 2026-11-08, Exp 9,
  # and are training data from then on), stage 2 only, from the 120-prompt final's saved stage 1 (uniform, so it
  # read no official rows; --stage1-from never deletes it). Stage 2 on 2,699 rows, stopped at epoch 3. ~20 min.
  nllm_8b_l24full_final150) RUN=qwen3-8b-l24-full-final150-001; BB=qwen3-8b-base
                  ARGS="--final --all-prompts $L24 --final-epoch 3 --pool-per-label 30000 --pool-recipe ux --pool-epochs 1 --pool-lr 1e-4 --pool-class-weighted --stage1-from $B/runs/qwen3-8b-l24-full-final-001/stage1" ;;
  # Qwen3-8B on all 120 dev prompts for the Kaggle ensemble: Exp 4b's pool recipe, stage 2 stopped at epoch 3 (the
  # epoch every p0 fold and four p1 folds chose). Saves the LoRA adapters and head, and scores the reference rows in
  # bf16 and fp16 (a T4 has no bf16). About 50 min: stage 1 ~35, stage 2 ~6, two 16 GB reloads.
  nllm_8b_final) RUN=qwen3-8b-final-001; BB=qwen3-8b-base
                  ARGS="--final --micro-batch 4 --pool-per-label 2000 $POOL --final-epoch 3" ;;
  # One model on all 120 dev prompts for the first Kaggle submission (--final, no CV). Schedule from the nested
  # runs: lr 1e-5 in all 20 fold selections, median epoch 3 of 6. About 15 min; the run dir holds the 1.6 GB model.
  nllm_e400_final) RUN=ettin-400m-final-001; BB=ettin-400m
                  ARGS="--final --micro-batch 8 --pool-per-label 2500 --pool-recipe w --pool-epochs 2 --lrs 1e-5 --epochs 6 --final-epoch 3" ;;
  # The same with 5,000 pool rows per label (nllm_e400pool_n5000: +0.036 / +0.035 over 2,500, paired CIs above 0).
  # Its nested runs chose lr 1e-5 in all 10 folds, median epoch 2. About 30 min: stage 1 ~20 min on ~88k rows.
  nllm_e400_final_n5000) RUN=ettin-400m-final-n5000-001; BB=ettin-400m
                  ARGS="--final --micro-batch 8 --pool-per-label 5000 --pool-recipe w --pool-epochs 2 --lrs 1e-5 --epochs 6 --final-epoch 2" ;;
  *) echo "NOT SUBMITTED $NAME: unknown job"; exit 1 ;;
esac
case $NAME in
  nllm_pilot*) [ -e "$B/runs/$RUN" ] && { echo "NOT SUBMITTED $NAME: pilot dir $RUN exists (never reused)"; exit 1; }
               JOB=$NAME ;;
  *) [ -f "$B/runs/$RUN/results.csv" ] || [ -f "$B/runs/$RUN/manifest.json" ] &&
       { echo "NOT SUBMITTED $NAME: $RUN already finished"; exit 0; }
     JOB=${NAME}_c$CHUNK ;;
esac
submit "$JOB" nllm_train.sbatch ${DEP[@]+"${DEP[@]}"} --export=ALL,REPO="$R",RUN_ID="$RUN",BACKBONE="$BB",ARGS="$ARGS"
STATUS=$?
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
exit $STATUS
