#!/bin/bash
# Several scripts/encoder_finetune.py --final fits in one job, so they take one slot of the shared account.
# REPO and RUNS come from nllm_install.sh; RUNS is "RUN_ID|BACKBONE|ARGS;RUN_ID|BACKBONE|ARGS;...". Each finished run
# is packed into runs/RUN_ID.tar + .sha256 (scripts/tc2/tc2.sh get) and its directory removed, since the saved models
# are large and the home quota is shared. A run whose tar already exists is skipped, so a resubmission continues.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=30G
#SBATCH --time=05:50:00
#SBATCH --output=$HOME/nllm/logs/%x_%j.out
#SBATCH --error=$HOME/nllm/logs/%x_%j.out
set -o pipefail   # no -u: activate scripts may reference unset variables
B=$HOME/nllm; R=${REPO:?REPO}; RUNS=${RUNS:?RUNS}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$B/venv/.ready" ] || { echo "env not ready"; exit 9; }
[ -f "$B/data/.ready" ] || { echo "data not ready"; exit 10; }
source "$B/venv/bin/activate"
export PYTHONPATH=$R/src PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NLLM_DATA_DIR=$B/data NLLM_POOL_DATA=$B/data/pool HF_HOME=$B/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd "$R"
STATUS=0
IFS=';' read -ra ITEMS <<< "$RUNS"
for ITEM in "${ITEMS[@]}"; do
  IFS='|' read -r RUN_ID BACKBONE ARGS <<< "$ITEM"
  if [ -f "$B/runs/$RUN_ID.tar.sha256" ]; then echo "=== $RUN_ID already packed"; continue; fi
  echo "=== $RUN_ID: $BACKBONE (code $(cat CODE_ID)) args: $ARGS  $(date +%H:%M:%S)"
  # shellcheck disable=SC2086
  python -u scripts/encoder_finetune.py --backbone "$BACKBONE" --out "$B/runs/$RUN_ID" $ARGS || { STATUS=$?; echo "=== $RUN_ID failed ($STATUS)"; continue; }
  (cd "$B/runs" && tar -cf "$RUN_ID.tar" "$RUN_ID" && sha256sum "$RUN_ID.tar" > "$RUN_ID.tar.sha256" && rm -rf "${RUN_ID:?}") \
    || { STATUS=22; echo "=== $RUN_ID: packing failed"; }
  echo "=== $RUN_ID packed  $(date +%H:%M:%S)"
done
echo "=== exit $STATUS $(date --iso-8601=seconds)"
exit $STATUS
