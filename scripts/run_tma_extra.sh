#!/usr/bin/env bash
# run_tma_extra.sh — supplemental TMA passes at tok192 and tok512 (medium bucket).
#
# The main run_benchmark.sh only invokes run_tma() once at tok64. This script
# runs the equivalent TMA measurement at higher token tiers so the report can
# show how the CPU pipeline profile (BE-bound / FE-bound / retiring) shifts
# as max_tokens grows.
#
# Outputs go into the existing run's tma/ directory with tier-suffixed names:
#   tma_toplev_<mode>_tok<TOK>.txt
#   tma_slots_<mode>_tok<TOK>.txt
#
# Usage:
#   scripts/run_tma_extra.sh                    # auto-pick latest run dir, both tiers
#   scripts/run_tma_extra.sh --run RUN_DIR
#   scripts/run_tma_extra.sh --tiers 192        # only one tier
#   scripts/run_tma_extra.sh --tiers 192,512
#
# Safe to run while the main benchmark is still going? NO — kubectl perf -p 1
# would race the active perf process. Wait until run_benchmark.sh exits.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUERIES_DIR="${KERNEL_ROOT}/benchmark_queries"
RESULTS_DIR="${KERNEL_ROOT}/benchmark_results"
NAMESPACE="${BENCHMARK_NAMESPACE:-llm-service}"
BENCHMARK_URL="${BENCHMARK_URL:-http://localhost:8080}"

# Args
RUN_DIR=""
TIERS="192,512"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)   RUN_DIR="$2"; shift 2 ;;
        --tiers) TIERS="$2"; shift 2 ;;
        -h|--help)
            grep -m1 -A 20 "^# run_tma_extra.sh" "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${RUN_DIR}" ]]; then
    RUN_DIR="$(ls -dt "${RESULTS_DIR}"/run_* 2>/dev/null | head -1)"
fi
if [[ -z "${RUN_DIR}" || ! -d "${RUN_DIR}" ]]; then
    echo "ERROR: no run dir found (try --run <path>)" >&2
    exit 1
fi
TMA_DIR="${RUN_DIR}/tma"
mkdir -p "${TMA_DIR}"

# Bail out if main benchmark is still running
if pgrep -f "run_benchmark.sh" > /dev/null; then
    echo "ERROR: run_benchmark.sh is still running. Wait for it to exit first." >&2
    echo "       (running this in parallel would race the active perf process)" >&2
    exit 1
fi
if pgrep -f "kubectl exec.*perf stat" > /dev/null; then
    echo "ERROR: an active kubectl perf is in flight. Wait for it to finish first." >&2
    exit 1
fi

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

get_pod() {
    kubectl get pod -n "${NAMESPACE}" -l app=llm-service-kernel \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null
}

POD="$(get_pod)"
if [[ -z "${POD}" ]]; then
    echo "ERROR: cannot find llm-service-kernel pod in namespace ${NAMESPACE}" >&2
    exit 1
fi
PID=$(kubectl exec -n "${NAMESPACE}" "${POD}" -- \
    sh -c "grep -m1 Pid /proc/1/status | awk '{print \$2}'" 2>/dev/null || echo "1")

log "Run dir : ${RUN_DIR}"
log "TMA dir : ${TMA_DIR}"
log "Pod     : ${POD} (PID ${PID})"
log "Tiers   : ${TIERS}"

run_tma_one() {
    local tok="$1"
    local mode="$2"
    local query_file warmup_file="" sc_arg="" qr_mode="${mode}"
    local bucket="medium"

    case "${mode}" in
        rag)        query_file="${QUERIES_DIR}/rag/${bucket}.txt" ;;
        llm_direct) query_file="${QUERIES_DIR}/llm_direct/${bucket}.txt" ;;
        sc_a)
            query_file="${QUERIES_DIR}/cache/${bucket}_measure.txt"
            warmup_file="${QUERIES_DIR}/cache/${bucket}_warm.txt"
            sc_arg="--sc-scenario a"
            qr_mode="cache_a"
            ;;
        *) echo "Unknown mode ${mode}" >&2; return 1 ;;
    esac
    [[ -f "${query_file}" ]] || { echo "  query file missing: ${query_file}" >&2; return 0; }

    local count=50
    # Size the perf window so it COVERS the full query_runner duration (50 warmup + 50 measure).
    # Per-request seconds are tier+mode dependent — measured from prior runs at this run's data.
    # Multiply by 100 (warmup+measure) and add a 60s tail buffer.
    local secs_per_req
    case "${tok}_${mode}" in
        64_rag)         secs_per_req=2 ;;
        64_llm_direct)  secs_per_req=2 ;;
        64_sc_a)        secs_per_req=1 ;;
        192_rag)        secs_per_req=5 ;;   # observed ~4.5s
        192_llm_direct) secs_per_req=3 ;;   # observed ~2.5s
        192_sc_a)       secs_per_req=1 ;;
        512_rag)        secs_per_req=6 ;;   # observed ~5s
        512_llm_direct) secs_per_req=3 ;;
        512_sc_a)       secs_per_req=1 ;;
        *)              secs_per_req=6 ;;
    esac
    local est_dur=$(( count * 2 * secs_per_req + 60 ))   # 2× count for warmup+measure, +60s buffer
    # Floor at 360s so we always get a meaningful steady-state window even for fast modes.
    if [[ ${est_dur} -lt 360 ]]; then est_dur=360; fi
    log "  est_dur=${est_dur}s (covers ~${secs_per_req}s/req × 100 req + 60s tail)"

    local toplev_out="${TMA_DIR}/tma_toplev_${mode}_tok${tok}.txt"
    local slots_out="${TMA_DIR}/tma_slots_${mode}_tok${tok}.txt"

    log "  TMA toplev -l1: ${mode}/${bucket} [tok=${tok}] → $(basename "${toplev_out}")"
    kubectl exec -n "${NAMESPACE}" "${POD}" -- bash -c \
        "cd /opt/pmu-tools && python3 toplev.py -l1 --no-desc \
         --nodes Retiring,Frontend_Bound,Bad_Speculation,Backend_Bound,MUX \
         -p ${PID} -- sleep ${est_dur} 2>&1" \
        > "${toplev_out}" 2>&1 &
    local toplev_pid=$!

    log "  TMA perf slots: ${mode}/${bucket} [tok=${tok}] → $(basename "${slots_out}")"
    kubectl exec -n "${NAMESPACE}" "${POD}" -- \
        perf stat -p "${PID}" \
        -e slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound \
        -- sleep "${est_dur}" \
        > "${slots_out}" 2>&1 &
    local slots_pid=$!

    BENCHMARK_URL="${BENCHMARK_URL}" \
    python3 "${SCRIPT_DIR}/query_runner.py" \
        --mode "${qr_mode}" \
        --queries "${query_file}" \
        ${warmup_file:+--warmup-queries "${warmup_file}"} \
        --size-bucket "${bucket}" \
        --count "${count}" --warmup 50 \
        --max-tokens "${tok}" \
        --out-dir "${TMA_DIR}" \
        ${sc_arg} \
        2>&1 | tail -3

    wait "${toplev_pid}" 2>/dev/null || true
    wait "${slots_pid}"  2>/dev/null || true
    log "  ✓ ${mode} tok${tok} done"
}

IFS=',' read -ra TIER_LIST <<< "${TIERS}"
for tok in "${TIER_LIST[@]}"; do
    log "=== TMA EXTRA: tok${tok} ==="
    for mode in rag llm_direct sc_a; do
        run_tma_one "${tok}" "${mode}"
    done
done

log "All TMA extras complete. Files in ${TMA_DIR}/"
ls -la "${TMA_DIR}"/tma_*_tok{192,512}.* 2>/dev/null || true
