#!/usr/bin/env bash
# Per-model vLLM serving benchmark loop. Runs ON the remote GPU host.
#
#   ./run_perf.sh [--run-id ID] [--only SUBSTR] [--models FILE] [--params FILE]
#
# For every model in models.txt: cold-start a vLLM server via start_vllm, wait
# for /health, run benchmark_serving against it, then kill the server and wait
# for GPU memory to drain before moving on. One model failing never aborts the
# rest of the run. Re-running with the same --run-id skips finished models.
#
# Prerequisite: the detached perf container is already up, with results
# bind-mounted (see tests/perf/README.md, or --help below).

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
MODELS_FILE="$SCRIPT_DIR/models.txt"
PARAMS_FILE="$SCRIPT_DIR/params.env"
RUN_ID=""
ONLY=""

CONTAINER_RESULTS=/var/aikit/perf_results

usage() {
    # Print the header comment block (everything after the shebang up to the
    # first non-comment line), stripped of its leading '#'.
    awk 'NR==1 {next} /^#/ {sub(/^# ?/, ""); print; next} {exit}' "${BASH_SOURCE[0]}"
    cat <<'EOF'

Start the container first (from the repo root on the remote host):

  mkdir -p tests/perf/results && chmod 777 tests/perf/results
  docker run -d --restart=no --entrypoint sleep \
    --gpus all --ipc host --network host \
    --mount type=bind,source=$HOME/.cache/huggingface,target=/var/aikit/.cache/huggingface \
    --mount type=bind,source=$PWD/tests/perf/results,target=/var/aikit/perf_results \
    --env HUGGING_FACE_HUB_TOKEN \
    --name aikit_perf_run aikit_perftest infinity
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)  RUN_ID="${2:?--run-id needs a value}"; shift 2 ;;
        --only)    ONLY="${2:?--only needs a value}"; shift 2 ;;
        --models)  MODELS_FILE="${2:?--models needs a value}"; shift 2 ;;
        --params)  PARAMS_FILE="${2:?--params needs a value}"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

# shellcheck source=params.env
source "$PARAMS_FILE"
: "${RUN_ID:=$(date +%Y%m%d-%H%M%S)}"
: "${GIT_SHA:=$(git -C "$SCRIPT_DIR/../.." rev-parse --short HEAD 2>/dev/null || echo unknown)}"

RESULTS_ROOT="$SCRIPT_DIR/results/$RUN_ID"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

# --- helpers that run inside the container -------------------------------
# procps (pgrep/pkill) is not guaranteed in the image, so process handling is
# done by walking /proc with python3, which definitely is present.
PY_PROC='
import os, signal, sys, time

def pids():
    out = []
    me = os.getpid()
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open("/proc/%s/cmdline" % d, "rb") as fh:
                cl = fh.read().decode("utf-8", "replace")
        except OSError:
            continue
        if "vllm" in cl:
            out.append(int(d))
    return out

if sys.argv[1] == "list":
    print(" ".join(str(p) for p in pids()))
    sys.exit(0)

for sig in (signal.SIGTERM, signal.SIGKILL):
    if not pids():
        break
    for p in pids():
        try:
            os.kill(p, sig)
        except OSError:
            pass
    for _ in range(30):
        if not pids():
            break
        time.sleep(1)
'

PY_HEALTH='
import sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
'

cexec()   { docker exec "$CONTAINER" "$@"; }
server_up()   { [[ -n "$(cexec python3 -c "$PY_PROC" list 2>/dev/null)" ]]; }
kill_server() { cexec python3 -c "$PY_PROC" kill >/dev/null 2>&1; }
health_ok()   { cexec python3 -c "$PY_HEALTH" "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; }

gpu_used_mib() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
        | awk '{s+=$1} END {print s+0}'
}

# Wait until GPU memory returns to within GPU_DRAIN_SLACK_MIB of the baseline.
# Without this, the next model OOMs and the failure is misattributed to it.
drain_gpu() {
    local limit=$((GPU_BASELINE_MIB + GPU_DRAIN_SLACK_MIB)) waited=0 used
    while (( waited < GPU_DRAIN_TIMEOUT_S )); do
        used=$(gpu_used_mib)
        (( used <= limit )) && { log "  GPU drained (${used} MiB)"; return 0; }
        sleep 5; waited=$((waited + 5))
    done
    log "  WARNING: GPU still at $(gpu_used_mib) MiB after ${GPU_DRAIN_TIMEOUT_S}s (limit ${limit})"
    return 1
}

# --- preflight -----------------------------------------------------------
command -v docker     >/dev/null || die "docker not found"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found"
[[ -r "$MODELS_FILE" ]] || die "cannot read $MODELS_FILE"

docker inspect -f '{{.State.Running}}' "$CONTAINER" 2>/dev/null | grep -qx true \
    || die "container '$CONTAINER' is not running. See --help for the docker run line."
cexec test -w "$CONTAINER_RESULTS" 2>/dev/null \
    || die "$CONTAINER_RESULTS is not writable in '$CONTAINER'. Recreate it with the results bind mount (see --help)."

mkdir -p "$RESULTS_ROOT"
cp "$PARAMS_FILE" "$RESULTS_ROOT/params.env.snapshot"

GPU_BASELINE_MIB=$(gpu_used_mib)
log "run id      : $RUN_ID"
log "image/sha   : $IMAGE_TAG / $GIT_SHA"
log "results     : $RESULTS_ROOT"
log "gpu baseline: ${GPU_BASELINE_MIB} MiB"

# Nothing should be holding the GPU before we start.
if server_up; then
    log "stray vLLM process in '$CONTAINER' -- killing it"
    kill_server; drain_gpu; GPU_BASELINE_MIB=$(gpu_used_mib)
fi

# --- main loop -----------------------------------------------------------
declare -i n_done=0 n_skip=0 n_fail=0

while IFS= read -r model || [[ -n "$model" ]]; do
    model="${model%%#*}"; model="$(echo "$model" | xargs)"
    [[ -z "$model" ]] && continue
    [[ -n "$ONLY" && "$model" != *"$ONLY"* ]] && continue

    slug="${model//\//__}"
    case_host="$RESULTS_ROOT/$slug"
    case_ctr="$CONTAINER_RESULTS/$RUN_ID/$slug"

    if [[ -f "$case_host/result.json" ]]; then
        log "SKIP $model (result.json already present)"
        n_skip+=1; continue
    fi

    log "=== $model ==="
    mkdir -p "$case_host"; chmod 777 "$case_host" 2>/dev/null
    status=UNKNOWN; ready_s=""

    # 1. Pre-download weights so download time stays out of the cold-start metric.
    log "  downloading weights"
    dl_start=$SECONDS
    if ! cexec python3 -c 'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1])' \
            "$model" >"$case_host/download.log" 2>&1; then
        log "  FAILED_DOWNLOAD (see download.log)"
        { echo "model=$model"; echo "status=FAILED_DOWNLOAD"; } >"$case_host/status.txt"
        n_fail+=1; continue
    fi
    dl_s=$((SECONDS - dl_start))
    log "  downloaded in ${dl_s}s"

    # 2. Cold-start the server. VLLM/NCCL log levels must be passed as CLI flags:
    #    start_vllm sources .vllmenv *after* reading the env, clobbering -e values.
    log "  cold-starting vLLM"
    start_ts=$SECONDS
    # A login shell (-l) is used here on purpose: /etc/profile.d sets up the CUDA
    # toolkit paths the server expects. It also kicks off a background
    # gpu-discovery and may chown the HF cache, but both finish well before
    # /health goes green, so they cannot perturb the measurement below.
    # shellcheck disable=SC2086  # SERVER_EXTRA_ARGS is intentionally word-split
    docker exec -d "$CONTAINER" bash -lc "mkdir -p '$case_ctr' && exec start_vllm \
        --model-path '$model' \
        --served-model-name '$model' \
        --port '$PORT' \
        --max-model-len '$MAX_MODEL_LEN' \
        --max-num-seqs '$MAX_NUM_SEQS' \
        --gpu-memory-utilization '$GPU_MEM_UTIL' \
        --tensor-parallel-size '$TENSOR_PARALLEL_SIZE' \
        --vllm-logging-level '$VLLM_LOG_LEVEL' \
        --nccl-debug '$NCCL_LOG_LEVEL' \
        --log-file '$case_ctr/server.log' \
        $SERVER_EXTRA_ARGS"

    # 3. Wait for /health, bailing early if the server process dies.
    while :; do
        if health_ok; then
            ready_s=$((SECONDS - start_ts)); status=READY
            log "  ready in ${ready_s}s"; break
        fi
        if ! server_up; then
            status=FAILED_STARTUP
            log "  FAILED_STARTUP: server exited after $((SECONDS - start_ts))s"; break
        fi
        if (( SECONDS - start_ts >= READY_TIMEOUT_S )); then
            status=FAILED_READY_TIMEOUT
            log "  FAILED_READY_TIMEOUT after ${READY_TIMEOUT_S}s"; break
        fi
        sleep 5
    done

    # 4. Benchmark.
    if [[ "$status" == READY ]]; then
        log "  benchmarking (${NUM_PROMPTS} prompts, rate=${REQUEST_RATE})"
        # NOT a login shell: /etc/profile.d would spawn a background gpu-discovery
        # (and possibly a `sudo chown -R` of the HF cache) right as measurement
        # starts. The client needs no CUDA env, so it is invoked directly rather
        # than relying on profile.d putting ~/benchmarks on PATH.
        # shellcheck disable=SC2086  # CLIENT_CMD/CLIENT_EXTRA_ARGS are intentionally word-split
        if docker exec "$CONTAINER" bash -c "$CLIENT_CMD \
                --backend '$BACKEND' \
                --model '$model' \
                --host 127.0.0.1 --port '$PORT' \
                --dataset-name '$DATASET_NAME' \
                --dataset-path '$DATASET_PATH' \
                --num-prompts '$NUM_PROMPTS' \
                --request-rate '$REQUEST_RATE' \
                --percentile-metrics '$PERCENTILE_METRICS' \
                --metric-percentiles '$METRIC_PERCENTILES' \
                --save-result \
                --result-dir '$case_ctr' \
                --result-filename result.json \
                $CLIENT_EXTRA_ARGS \
                --metadata image='$IMAGE_TAG' git_sha='$GIT_SHA' run_id='$RUN_ID' \
                          max_num_seqs='$MAX_NUM_SEQS' max_model_len='$MAX_MODEL_LEN' \
                          time_to_ready_s='${ready_s:-}'" >"$case_host/client.log" 2>&1; then
            status=DONE; log "  DONE"
        else
            status=FAILED_BENCHMARK; log "  FAILED_BENCHMARK (see client.log)"
        fi
    fi

    # 5. Tear the server down and wait for the GPU before the next model.
    log "  stopping server"
    kill_server
    drain_gpu || status="${status}+GPU_NOT_DRAINED"

    {
        echo "model=$model"
        echo "status=$status"
        echo "run_id=$RUN_ID"
        echo "image=$IMAGE_TAG"
        echo "git_sha=$GIT_SHA"
        echo "download_s=${dl_s:-}"
        echo "time_to_ready_s=${ready_s:-}"
    } >"$case_host/status.txt"

    [[ "$status" == DONE* ]] && n_done+=1 || n_fail+=1
done <"$MODELS_FILE"

log "finished: $n_done ok, $n_fail failed, $n_skip skipped -> $RESULTS_ROOT"
(( n_fail == 0 ))
