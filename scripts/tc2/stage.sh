#!/bin/bash
# Stage experiment 4 for TC2 in outputs/tc2_stage/: a code archive of the working tree (src, scripts, tests, the
# uv lock and the kit's validation_plan.json) with a per-file sha256 payload list whose hash is the CODE_ID,
# train.csv plus the baseline's oof.parquet (paired differences), their sha256s, and the login-node installer.
# Upload with `scripts/tc2/tc2.sh upload`. Runs in Git Bash or WSL from anywhere in the repo.
# With --pool it also packs the pool data for pool stage 1 (nllm.pool, NLLM_POOL_DATA on TC2): the Open Pool main
# file, extra_sources.parquet and admitted.parquet, as nllm_pool.tar with nllm_pool.sha256.
#   bash scripts/tc2/stage.sh [--code-only | --pool]
set -euo pipefail
cd "$(dirname "$0")/../.."
S=outputs/tc2_stage
BASELINE=${BASELINE:-outputs/baseline_linear/20261102T122807Z}
rm -rf "$S"; mkdir -p "$S"
mapfile -t FILES < <( { printf '%s\n' pyproject.toml uv.lock .python-version Name_That_LLM_Research_Kit/validation_plan.json
                        git ls-files -co --exclude-standard src scripts tests; } | sort -u )
if grep -lI $'\r' "${FILES[@]}" | grep -E '\.(sh|sbatch)$'; then echo "CRLF line endings in shell scripts above"; exit 1; fi
sha256sum "${FILES[@]}" > "$S/nllm_payload.sha256"
CODE_ID=$(sha256sum < "$S/nllm_payload.sha256" | cut -c1-12)
echo "$CODE_ID" > "$S/CODE_ID"
tar -cf "$S/nllm_code.tar" "${FILES[@]}"
(cd "$S" && sha256sum nllm_code.tar > nllm_code.sha256)
cp scripts/tc2/nllm_install.sh "$S/"
if [ "${1:-}" != --code-only ]; then
  cp data/raw/train.csv "$S/train.csv"
  cp "$BASELINE/oof.parquet" "$S/baseline_oof.parquet"
  echo "$BASELINE" > "$S/BASELINE_RUN"
  (cd "$S" && sha256sum train.csv baseline_oof.parquet > nllm_data.sha256)
fi
if [ "${1:-}" = --pool ]; then
  tar -cf "$S/nllm_pool.tar" -C data/external third_party/crtproduction__name-that-llm-open-pool/ntl_open_pool.parquet     extra_sources.parquet admitted.parquet
  (cd "$S" && sha256sum nllm_pool.tar > nllm_pool.sha256)
fi
echo "staged code $CODE_ID (${#FILES[@]} files) in $S:"
ls -la "$S"
