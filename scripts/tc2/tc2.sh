#!/bin/bash
# Local controller for experiment 4 on TC2. Runs on Linux (or WSL) and only reuses the SSH control socket the user opened:
#   ssh -M -S $HOME/.ssh/tc2.sock -o ControlPersist=yes USER@TC2_LOGIN_HOST
# It never opens a connection or handles a password, checks the socket before every call and stops if it is down.
# Rules: the cluster owner's private usage guide (shared account, 2 submitted jobs, never scancel).
#   bash scripts/tc2/tc2.sh COMMAND      (Windows: MSYS_NO_PATHCONV=1 wsl bash /mnt/f/.../scripts/tc2/tc2.sh COMMAND)
#     check                  socket guard only
#     upload [--code-only]   scp outputs/tc2_stage/* to ~/nllm/upload (run stage.sh first; includes nllm_pool.tar
#                            when stage.sh --pool made one)
#     install ARGS...        bash ~/nllm/upload/nllm_install.sh ARGS (unpack | submit NAME [CHUNK])
#     queue                  our jobs in squeue and today's sacct
#     log JOB_NAME JOB_ID    last 60 lines of a job log
#     wait JOB_ID [MINUTES]  block until the job has left the queue: first check after MINUTES (default 15), then
#                            every 5 min; stops if the socket goes down (run it in the background)
#     get RUN_ID             fetch runs/RUN_ID.tar (+ sha256), verify, unpack under outputs/tc2/
set -euo pipefail
H=USER@TC2_LOGIN_HOST
SOCK=$HOME/.ssh/tc2.sock
O=(-o ControlPath=$SOCK -o ControlMaster=no -o BatchMode=yes)
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
S=$ROOT/outputs/tc2_stage

# Every call is wrapped in timeout: a stalled master can hang ssh while `-O check` still passes.
guard() { timeout 20 ssh -S "$SOCK" -O check "$H" 2>/dev/null || { echo "TC2 socket is down: reconnect (VPN + control socket), then rerun"; exit 2; }; }
remote() { guard; timeout 600 ssh "${O[@]}" "$H" "$@" | { grep -v '^Loading ' || true; }; }  # module load chatter

CMD=${1:?command: check | upload | install | queue | log | get}; shift
case $CMD in
  check) guard; echo "socket ok" ;;
  upload)
    FILES=(nllm_code.tar nllm_code.sha256 nllm_payload.sha256 CODE_ID nllm_install.sh)
    [ "${1:-}" = --code-only ] || FILES+=(train.csv baseline_oof.parquet nllm_data.sha256 BASELINE_RUN)
    [ -f "$S/nllm_pool.tar" ] && FILES+=(nllm_pool.tar nllm_pool.sha256)
    for f in "${FILES[@]}"; do [ -f "$S/$f" ] || { echo "missing $S/$f: run stage.sh"; exit 1; }; done
    remote 'mkdir -p ~/nllm/upload'
    guard; (cd "$S" && timeout 5400 scp "${O[@]}" "${FILES[@]}" "$H:nllm/upload/")
    remote "cd ~/nllm/upload && sha256sum -c nllm_code.sha256 && { [ ! -f nllm_data.sha256 ] || [ ! -f train.csv ] || sha256sum -c nllm_data.sha256; } && { [ ! -f nllm_pool.tar ] || sha256sum -c nllm_pool.sha256; }"
    echo "uploaded code $(cat "$S/CODE_ID")" ;;
  install) remote "bash ~/nllm/upload/nllm_install.sh $*" ;;
  queue)
    remote 'squeue -u $USER -o "%.10i %.22j %.8T %.10M %.10l %R"; echo; sacct -n -X -u $USER -S $(date +%F) -o JobID,JobName%22,State,Elapsed,ExitCode | grep -E "nllm_" || true' ;;
  log) remote "tail -n 60 ~/nllm/logs/${1:?job name}_${2:?job id}.out" ;;
  wait)
    JOB=${1:?job id}; sleep $((60 * ${2:-15}))
    while :; do
      guard
      ST=$(remote "sacct -n -X -j $JOB -o State%20" 2>/dev/null | tr -d ' ' | head -n 1)
      case $ST in PENDING|RUNNING|REQUEUED|CONFIGURING|COMPLETING|"") sleep 300 ;; *) echo "job $JOB: $ST"; break ;; esac
    done ;;
  get)
    RUN=${1:?run id}; D=$ROOT/outputs/tc2; mkdir -p "$D"
    remote "cd ~/nllm/runs && ls -l $RUN.tar $RUN.tar.sha256 && { [ ! -f $RUN.incomplete ] || echo 'note: run is incomplete (more chunks to go)'; }"
    guard; timeout 3600 scp "${O[@]}" "$H:nllm/runs/$RUN.tar" "$H:nllm/runs/$RUN.tar.sha256" "$D/"
    (cd "$D" && sha256sum -c "$RUN.tar.sha256")
    rm -rf "${D:?}/$RUN" && tar -xf "$D/$RUN.tar" -C "$D" && rm "$D/$RUN.tar"
    echo "fetched $D/$RUN" ;;
  *) echo "unknown command $CMD"; exit 1 ;;
esac
