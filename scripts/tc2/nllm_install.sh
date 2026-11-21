#!/bin/bash
# Experiment 4 on TC2, login-node side: ONLY file operations + sbatch/squeue/sacct (no python, conda or downloads).
# Everything stays under $HOME/nllm. Idempotent: verifies uploads by sha256, unpacks the code once into
# nllm/repo_<code id>, installs train.csv and the baseline predictions once, then submits the named job unless it is
# already queued, running or completed. The account is shared (QoS: 2 submitted jobs per user), so a refused
# submission stops here. Never scancel.
#   bash nllm_install.sh unpack                # verify + install uploads only
#   bash nllm_install.sh submit NAME [CHUNK]   # one job from the table below; nested runs take CHUNK 1, 2, ...
#                                              # (each chunk resumes the same run dir; job name NAME_c<CHUNK>)
set -uo pipefail
cd "$HOME" || exit 1
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
C=$(cat "$U/CODE_ID"); R=$B/repo_$C
[ -d "$R" ] || { echo "no unpacked repo for code $C"; exit 1; }
[ "$STAGE" = unpack ] && exit 0
[ "$STAGE" = submit ] || { echo "unknown stage $STAGE"; exit 1; }

seen() { squeue -h -u "$USER" -o "%j" | grep -qx "$1" || sacct -n -X -u "$USER" -S 2026-11-02 -o JobName%30,State | grep -qE "^ *$1 +(COMPLETED|RUNNING|PENDING)"; }
submit() {  # job-name sbatch-file [extra sbatch args...]
  local name=$1 file=$2; shift 2
  if seen "$name"; then echo "$name already queued/running/completed"; return 0; fi
  local out; out=$(sbatch --parsable ${TC2_NODELIST:+--nodelist=$TC2_NODELIST} -J "$name" "$@" "$R/scripts/tc2/$file" 2>&1) \
    && { echo "SUBMITTED $name=$out (code $C)"; return 0; } || { echo "NOT SUBMITTED $name: $(echo "$out" | head -c 160)"; return 1; }
}
echo "queue before:"; squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
NAME=${2:?job name}; CHUNK=${3:-1}
case $CHUNK in [1-9]) ;; *) echo "NOT SUBMITTED $NAME: bad chunk $CHUNK"; exit 1 ;; esac
if [ "$NAME" = nllm_env ]; then submit nllm_env nllm_env.sbatch --export=ALL,REPO="$R"; exit $?; fi
[ -f "$B/venv/.ready" ] && [ -f "$B/data/.ready" ] || { echo "NOT SUBMITTED $NAME: env/data not ready"; exit 1; }
NESTED="--time-budget-minutes 340 --fit-minutes 35 --baseline $B/data/baseline"
case $NAME in  # job name | run id | backbone | script args (8B: micro-batch 4 x accumulation 8, no checkpointing)
  nllm_pilot8b) RUN=pilot-qwen3-8b-base-001; BB=qwen3-8b-base; ARGS="--pilot --save-model --micro-batch 4" ;;
  nllm_pilot4b) RUN=pilot-qwen3-4b-base-001; BB=qwen3-4b-base; ARGS="--pilot --save-model --micro-batch 8" ;;
  nllm_8b_p0)   RUN=qwen3-8b-base-p0-001;    BB=qwen3-8b-base; ARGS="--partitions 0 --micro-batch 4 $NESTED" ;;
  nllm_8b_p1)   RUN=qwen3-8b-base-p1-001;    BB=qwen3-8b-base; ARGS="--partitions 1 --micro-batch 4 $NESTED" ;;
  nllm_4b_p0)   RUN=qwen3-4b-base-p0-001;    BB=qwen3-4b-base; ARGS="--partitions 0 --micro-batch 8 $NESTED" ;;
  nllm_4b_p1)   RUN=qwen3-4b-base-p1-001;    BB=qwen3-4b-base; ARGS="--partitions 1 --micro-batch 8 $NESTED" ;;
  *) echo "NOT SUBMITTED $NAME: unknown job"; exit 1 ;;
esac
case $NAME in
  nllm_pilot*) [ -e "$B/runs/$RUN" ] && { echo "NOT SUBMITTED $NAME: pilot dir $RUN exists (never reused)"; exit 1; }
               JOB=$NAME ;;
  *) [ -f "$B/runs/$RUN/results.csv" ] && { echo "NOT SUBMITTED $NAME: $RUN already finished"; exit 0; }
     JOB=${NAME}_c$CHUNK ;;
esac
submit "$JOB" nllm_train.sbatch --export=ALL,REPO="$R",RUN_ID="$RUN",BACKBONE="$BB",ARGS="$ARGS"
STATUS=$?
squeue -u "$USER" -o "%.10i %.20j %.8T %.10M %R"
exit $STATUS
