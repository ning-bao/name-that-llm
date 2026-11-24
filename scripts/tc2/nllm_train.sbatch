#!/bin/bash
# One experiment-4 job on TC2N03 (L40S): scripts/encoder_finetune.py for BACKBONE into ~/nllm/runs/RUN_ID.
# REPO, RUN_ID, BACKBONE and ARGS come from nllm_install.sh (--export). A nested run resumes its run dir (every
# fit's logits are cached), and --time-budget-minutes makes the script stop before a fit that might not finish
# (exit 75). That is reported as success with runs/RUN_ID.incomplete, so the next chunk job continues it.
# On exit the run dir is packed into runs/RUN_ID.tar + .sha256 for retrieval (scripts/tc2/tc2.sh get).
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
B=$HOME/nllm; R=${REPO:?REPO}; RUN_ID=${RUN_ID:?RUN_ID}; BACKBONE=${BACKBONE:?BACKBONE}; ARGS=${ARGS:-}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
[ -f "$B/venv/.ready" ] || { echo "env not ready"; exit 9; }
[ -f "$B/data/.ready" ] || { echo "data not ready"; exit 10; }
source "$B/venv/bin/activate"
export PYTHONPATH=$R/src PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NLLM_DATA_DIR=$B/data NLLM_POOL_DATA=$B/data/pool HF_HOME=$B/hf HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
cd "$R"
echo "=== $RUN_ID: $BACKBONE (code $(cat CODE_ID)) args: $ARGS"
# shellcheck disable=SC2086
python -u scripts/encoder_finetune.py --backbone "$BACKBONE" --out "$B/runs/$RUN_ID" $ARGS
STATUS=$?
rm -f "$B/runs/$RUN_ID.incomplete"
if [ $STATUS -eq 75 ]; then touch "$B/runs/$RUN_ID.incomplete"; STATUS=0; echo "stopped for time: submit the next chunk"; fi
if [ -d "$B/runs/$RUN_ID" ]; then
  (cd "$B/runs" && tar -cf "$RUN_ID.tar" "$RUN_ID" && sha256sum "$RUN_ID.tar" > "$RUN_ID.tar.sha256") || STATUS=22
fi
echo "=== exit $STATUS $(date --iso-8601=seconds)"
exit $STATUS
