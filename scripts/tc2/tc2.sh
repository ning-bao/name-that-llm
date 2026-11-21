#!/bin/bash
# Local controller for experiment 4 on TC2. Runs in WSL and only reuses the SSH control socket the user opened:
#   ssh -M -S ~/.ssh/tc2.sock -o ControlPersist=yes "$TC2_HOST"
# Set TC2_HOST to your user@host for the cluster login node (and TC2_SOCK if the socket lives elsewhere).
# It never opens a connection or handles a password, checks the socket before every call and stops if it is down.
# Cluster etiquette: shared account, 2 submitted jobs, never scancel.
#   bash scripts/tc2/tc2.sh COMMAND
#     check                  socket guard only
#     upload [--code-only]   scp outputs/tc2_stage/* to ~/nllm/upload (run stage.sh first)
#     install ARGS...        bash ~/nllm/upload/nllm_install.sh ARGS (unpack | submit NAME [CHUNK])
#     queue                  our jobs in squeue and today's sacct
#     log JOB_NAME JOB_ID    last 60 lines of a job log
#     get RUN_ID             fetch runs/RUN_ID.tar (+ sha256), verify, unpack under outputs/tc2/
set -euo pipefail
H=${TC2_HOST:?set TC2_HOST=user@login-node}
SOCK=${TC2_SOCK:-$HOME/.ssh/tc2.sock}
O=(-o ControlPath=$SOCK -o ControlMaster=no -o BatchMode=yes)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
S=$ROOT/outputs/tc2_stage

guard() { ssh -S "$SOCK" -O check "$H" 2>/dev/null || { echo "TC2 socket is down: reconnect (VPN + control socket), then rerun"; exit 2; }; }
remote() { guard; ssh "${O[@]}" "$H" "$@" | { grep -v '^Loading ' || true; }; }  # module load chatter

CMD=${1:?command: check | upload | install | queue | log | get}; shift
case $CMD in
  check) guard; echo "socket ok" ;;
  upload)
    FILES=(nllm_code.tar nllm_code.sha256 nllm_payload.sha256 CODE_ID nllm_install.sh)
    [ "${1:-}" = --code-only ] || FILES+=(train.csv baseline_oof.parquet nllm_data.sha256 BASELINE_RUN)
    for f in "${FILES[@]}"; do [ -f "$S/$f" ] || { echo "missing $S/$f: run stage.sh"; exit 1; }; done
    remote 'mkdir -p ~/nllm/upload'
    guard; (cd "$S" && scp "${O[@]}" "${FILES[@]}" "$H:nllm/upload/")
    remote "cd ~/nllm/upload && sha256sum -c nllm_code.sha256 && { [ ! -f nllm_data.sha256 ] || [ ! -f train.csv ] || sha256sum -c nllm_data.sha256; }"
    echo "uploaded code $(cat "$S/CODE_ID")" ;;
  install) remote "bash ~/nllm/upload/nllm_install.sh $*" ;;
  queue)
    remote 'squeue -u $USER -o "%.10i %.22j %.8T %.10M %.10l %R"; echo; sacct -n -X -u $USER -S $(date +%F) -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "nllm_" || true' ;;
  log) remote "tail -n 60 ~/nllm/logs/${1:?job name}_${2:?job id}.out" ;;
  get)
    RUN=${1:?run id}; D=$ROOT/outputs/tc2; mkdir -p "$D"
    remote "cd ~/nllm/runs && ls -l $RUN.tar $RUN.tar.sha256 && { [ ! -f $RUN.incomplete ] || echo 'note: run is incomplete (more chunks to go)'; }"
    guard; scp "${O[@]}" "$H:nllm/runs/$RUN.tar" "$H:nllm/runs/$RUN.tar.sha256" "$D/"
    (cd "$D" && sha256sum -c "$RUN.tar.sha256")
    rm -rf "${D:?}/$RUN" && tar -xf "$D/$RUN.tar" -C "$D" && rm "$D/$RUN.tar"
    echo "fetched $D/$RUN" ;;
  *) echo "unknown command $CMD"; exit 1 ;;
esac
