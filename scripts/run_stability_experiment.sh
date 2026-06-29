#!/usr/bin/env bash
# Stability experiment: find smallest n where perf metrics converge.
# Runs each cell × n with hw passes (via run_benchmark.sh) + a TMA pass (inline).
# Results go to benchmark_results/stability/<cell>_n<n>/
#
# Usage:
#   bash scripts/run_stability_experiment.sh
set -euo pipefail

KERNEL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${KERNEL_ROOT}/scripts"
set -a; source "${KERNEL_ROOT}/deploy/config.env"; set +a

NAMESPACE="${NAMESPACE_SERVICE:-llm-service}"
RESULTS_BASE="${KERNEL_ROOT}/benchmark_results/stability"
QUERIES_DIR="${KERNEL_ROOT}/benchmark_queries"
TOPLEV="/tmp/pmu-tools-benchmark/toplev.py"
BENCHMARK_URL="${BENCHMARK_URL:-http://localhost:18081}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-qwen2.5-0.5b}"
WARMUP="${BENCHMARK_WARMUP:-5}"
STREAM_FLAG="--stream"

log()  { printf '\n\033[1;34m[stability]\033[0m %s\n' "$*"; }
ok()   { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m  ⚠\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[stability] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ── Step 0: Patch ConfigMap and restart FastAPI with SeaweedFS forced ─────────
enable_seaweed() {
    log "Patching ConfigMap to enable RAG_FORCE_SEAWEED_FETCH=1..."
    kubectl patch configmap llm-service-kernel-config-fullstack \
        -n "${NAMESPACE}" \
        --type merge \
        -p '{"data": {"RAG_FORCE_SEAWEED_FETCH": "1"}}' 2>/dev/null || \
        warn "ConfigMap patch failed — may not exist, continuing"

    log "Restarting FastAPI pod..."
    kubectl rollout restart deployment/llm-service-kernel -n "${NAMESPACE}"
    kubectl rollout status deployment/llm-service-kernel -n "${NAMESPACE}" --timeout=120s
    ok "FastAPI restarted"

    # Re-establish port-forward (it dies when the pod restarts)
    log "Re-establishing port-forward on :18081..."
    kill "$(lsof -ti:18081 2>/dev/null)" 2>/dev/null || true
    sleep 2
    kubectl port-forward svc/llm-service-kernel 18081:8080 -n "${NAMESPACE}" \
        >/dev/null 2>&1 &
    sleep 3
    ok "Port-forward re-established"

    local pod
    pod=$(kubectl get pod -n "${NAMESPACE}" -l app=llm-service-kernel \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
    local flag
    flag=$(kubectl exec -n "${NAMESPACE}" "${pod}" -- \
        sh -c 'echo ${RAG_FORCE_SEAWEED_FETCH:-0}' 2>/dev/null || echo "unknown")
    if [[ "${flag}" == "1" ]]; then
        ok "RAG_FORCE_SEAWEED_FETCH=1 confirmed inside pod"
    else
        warn "RAG_FORCE_SEAWEED_FETCH=${flag} inside pod — SeaweedFS may not be on hot path"
    fi
}

# ── PID discovery for TMA ─────────────────────────────────────────────────────
get_fastapi_host_pid() {
    ps -eo pid,cmd 2>/dev/null \
        | grep -E "uvicorn src.service.main" \
        | grep -v grep \
        | awk '{print $1}' | head -1 || true
}

# ── TMA pass: start toplev, run queries, SIGINT when done ────────────────────
run_tma_pass() {
    local run_dir="$1" n="$2" cell="$3"

    local fastapi_pid
    fastapi_pid=$(get_fastapi_host_pid)
    if [[ -z "${fastapi_pid}" ]] || ! [[ "${fastapi_pid}" =~ ^[0-9]+$ ]]; then
        warn "FastAPI host PID not found — skipping TMA pass for ${cell} n=${n}"
        return
    fi

    # Resolve query file, mode, and SC args from cell name
    local query_file qr_mode sc_arg="" warmup_file=""
    case "${cell}" in
        rag_short)     query_file="${QUERIES_DIR}/rag/short.txt";      qr_mode="rag" ;;
        rag_medium)    query_file="${QUERIES_DIR}/rag/medium.txt";      qr_mode="rag" ;;
        rag_long)      query_file="${QUERIES_DIR}/rag/long.txt";        qr_mode="rag" ;;
        rag_very_long) query_file="${QUERIES_DIR}/rag/very_long.txt";   qr_mode="rag" ;;
        llm_medium)    query_file="${QUERIES_DIR}/llm_direct/medium.txt"; qr_mode="llm_direct" ;;
        llm_long)      query_file="${QUERIES_DIR}/llm_direct/long.txt";   qr_mode="llm_direct" ;;
        sc_a_medium)
            query_file="${QUERIES_DIR}/cache/medium_measure.txt"
            warmup_file="${QUERIES_DIR}/cache/medium_warm.txt"
            qr_mode="cache_a"; sc_arg="--sc-scenario a" ;;
        sc_b_medium)
            query_file="${QUERIES_DIR}/cache/medium_measure.txt"
            warmup_file="${QUERIES_DIR}/cache/medium_warm.txt"
            qr_mode="cache_b"; sc_arg="--sc-scenario b" ;;
        *) warn "Unknown cell ${cell} — skipping TMA"; return ;;
    esac

    log "  TMA pass: cell=${cell} n=${n} fastapi_pid=${fastapi_pid}"

    local tma_out="${run_dir}/tma_toplev.txt"
    local tma_csv="${run_dir}/tma_queries.csv"

    # SC warmup before TMA measurement window (keep cache hot, outside window)
    if [[ -n "${warmup_file}" ]]; then
        log "  SC pre-warmup (outside TMA window)..."
        BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
        python3 "${SCRIPT_DIR}/query_runner.py" \
            --mode "${qr_mode}" \
            --queries "${query_file}" \
            --warmup-queries "${warmup_file}" \
            --size-bucket medium \
            --count 1 \
            --warmup "${WARMUP}" \
            --max-tokens 64 \
            --out-dir "${run_dir}" \
            ${sc_arg} \
            --warmup-only 2>&1 | tail -2 || true
    fi

    # Estimate query window so toplev's sleep expires naturally after queries finish.
    # toplev writes output when its sleep subprocess exits — SIGINT does NOT flush reliably.
    # Use n × latency_estimate + 30s buffer so sleep expires ~30s after queries finish.
    local lat_estimate
    case "${cell}" in
        sc_*)        lat_estimate=1 ;;   # SC hits are ~75ms — must be before *_medium
        *_short)     lat_estimate=3 ;;
        *_medium)    lat_estimate=5 ;;
        *_long)      lat_estimate=8 ;;
        *_very_long) lat_estimate=12 ;;
        *)           lat_estimate=5 ;;
    esac
    local sleep_window=$(( n * lat_estimate + 30 ))

    # Start toplev — will print output when sleep expires naturally
    sudo python3 "${TOPLEV}" -l2 --no-desc \
        --nodes "Retiring,Frontend_Bound,Bad_Speculation,Backend_Bound,\
Backend_Bound.Memory_Bound,Backend_Bound.Core_Bound,\
Frontend_Bound.Fetch_Latency,Frontend_Bound.Fetch_Bandwidth" \
        -p "${fastapi_pid}" \
        -- sleep "${sleep_window}" \
        > "${tma_out}" 2>&1 &
    local tma_pid=$!

    sleep 1  # let toplev attach and start collecting

    # Run n queries (measurement window)
    local no_warmup_flag=""
    [[ -n "${warmup_file}" ]] && no_warmup_flag="--no-warmup"

    local bucket="medium"
    case "${cell}" in
        *_short)     bucket="short" ;;
        *_medium)    bucket="medium" ;;
        *_long)      bucket="long" ;;
        *_very_long) bucket="very_long" ;;
    esac

    BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
    python3 "${SCRIPT_DIR}/query_runner.py" \
        --mode "${qr_mode}" \
        --queries "${query_file}" \
        ${warmup_file:+--warmup-queries "${warmup_file}"} \
        --size-bucket "${bucket}" \
        --count "${n}" \
        --warmup "${WARMUP}" \
        --max-tokens 64 \
        --out-dir "${run_dir}" \
        ${sc_arg} \
        ${no_warmup_flag} \
        ${STREAM_FLAG} 2>&1 | tail -4 || true

    # Wait for toplev's sleep to expire and output to be written
    log "  Waiting for toplev window (${sleep_window}s) to expire..."
    wait "${tma_pid}" 2>/dev/null || true

    # Verify toplev produced output
    if grep -qE "Retiring|Frontend_Bound|Backend_Bound" "${tma_out}" 2>/dev/null; then
        ok "  TMA output collected → ${tma_out}"
    else
        warn "  TMA output empty or failed — check ${tma_out}"
    fi
}

# ── Run one cell × n combination ─────────────────────────────────────────────
run_cell_n() {
    local cell="$1" n="$2"
    local run_dir="${RESULTS_BASE}/${cell}_n${n}"
    mkdir -p "${run_dir}"

    log "=== CELL ${cell} n=${n} ==="

    # hw passes (7 passes via run_benchmark.sh)
    RUN_DIR_OVERRIDE="${run_dir}" \
    BENCHMARK_COUNT_64="${n}" \
    BENCHMARK_URL="${BENCHMARK_URL}" \
    bash "${SCRIPT_DIR}/run_benchmark.sh" "${cell}" --tokens 64 2>&1

    # TMA pass (inline)
    run_tma_pass "${run_dir}" "${n}" "${cell}"

    ok "=== DONE: ${cell} n=${n} → ${run_dir} ==="
}

# ── Main ──────────────────────────────────────────────────────────────────────
main() {
    mkdir -p "${RESULTS_BASE}"

    log "Starting stability experiment"
    log "Results: ${RESULTS_BASE}"

    enable_seaweed

    # Disable nmi_watchdog to reduce PMU multiplexing for TMA
    sudo sysctl kernel.nmi_watchdog=0 2>/dev/null || warn "Could not disable nmi_watchdog"

    log "Waiting for service to be ready..."
    local attempts=0
    until curl -sf "${BENCHMARK_URL}/health" \
            | python3 -c "import sys,json; h=json.load(sys.stdin); sys.exit(0 if h.get('ready') else 1)" \
            2>/dev/null; do
        sleep 5
        attempts=$((attempts + 1))
        [[ ${attempts} -gt 60 ]] && die "Service not ready after 5 min"
    done
    ok "Service ready"

    # RAG / LLM cells
    local rag_llm_cells=(rag_short rag_medium rag_long rag_very_long llm_medium llm_long)
    local rag_llm_n=(5 20 50)

    # SC cells
    local sc_cells=(sc_a_medium sc_b_medium)
    local sc_n=(50 150 300)

    for cell in "${rag_llm_cells[@]}"; do
        for n in "${rag_llm_n[@]}"; do
            [[ -d "${RESULTS_BASE}/${cell}_n${n}" ]] && { ok "Skipping ${cell} n=${n} (already done)"; continue; }
            run_cell_n "${cell}" "${n}"
        done
    done

    for cell in "${sc_cells[@]}"; do
        for n in "${sc_n[@]}"; do
            [[ -d "${RESULTS_BASE}/${cell}_n${n}" ]] && { ok "Skipping ${cell} n=${n} (already done)"; continue; }
            run_cell_n "${cell}" "${n}"
        done
    done

    log "All cells complete. Run analyze_stability.py to see results."
    ok "Results in: ${RESULTS_BASE}"
}

main "$@"
