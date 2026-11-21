#!/bin/bash
# Experiment-4 environment at ~/nllm/venv, built INSIDE a job (no code runs on the login node): conda-forge
# Python 3.12 at ~/nllm/py as the base interpreter, then `uv sync --frozen --extra nn` from the uploaded repo's
# uv.lock (torch cu128 wheels; driver 570 supports CUDA 12.8). The nllm package itself is not installed: jobs put
# their repo's src/ on PYTHONPATH, so one env serves every uploaded code version. Package caches stay under
# ~/nllm and are removed afterwards. Writes ~/nllm/venv/.ready only after a GPU + import check passes.
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=nllm/logs/%x_%j.out   # relative to $HOME, where nllm_install.sh submits
#SBATCH --error=nllm/logs/%x_%j.out
set -eo pipefail   # no -u: the cluster's conda activate.d scripts reference unset variables
B=$HOME/nllm; R=${REPO:?REPO}
date --iso-8601=seconds; hostname; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
if [ -f "$B/venv/.ready" ]; then echo "env already ready"; exit 0; fi
if [ -e "$B/venv" ] || [ -e "$B/py" ]; then echo "partial env under $B exists; refusing to overwrite (inspect by hand)"; exit 3; fi
export CONDA_PKGS_DIRS=$B/.condapkgs PIP_CACHE_DIR=$B/.pipcache UV_CACHE_DIR=$B/.uvcache UV_PYTHON_DOWNLOADS=never
set +e   # cluster conda hooks may return non-zero internally; checked explicitly below
module purge
module load anaconda/25.5.1 || module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
set -e
conda create -y -p "$B/py" -c conda-forge --override-channels python=3.12 pip
"$B/py/bin/python" -m pip install --no-input uv==0.7.3
cd "$R"
UV_PROJECT_ENVIRONMENT=$B/venv "$B/py/bin/uv" sync --frozen --extra nn --no-dev --no-install-project --python "$B/py/bin/python"
PYTHONPATH=$R/src "$B/venv/bin/python" - <<'EOF'
import peft, torch, transformers
import nllm.encoder
assert torch.cuda.is_available(), "CUDA not available"
x = torch.randn(256, 256, device="cuda", dtype=torch.bfloat16)
assert torch.isfinite(x @ x).all()
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "| transformers", transformers.__version__, "| peft", peft.__version__)
EOF
"$B/py/bin/uv" pip freeze --python "$B/venv/bin/python" > "$B/venv/FREEZE.txt"
conda clean -y --all >/dev/null 2>&1 || true
rm -rf "$B/.condapkgs" "$B/.pipcache" "$B/.uvcache"
touch "$B/venv/.ready"
du -sh "$B/py" "$B/venv"
echo "ENV READY $(date --iso-8601=seconds)"
