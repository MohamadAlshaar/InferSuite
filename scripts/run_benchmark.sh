#!/usr/bin/env bash
# =============================================================================
# run_benchmark.sh — GenAI workload characterization benchmark orchestrator
#
# Measures hardware-counter cells across ALL pods simultaneously under the
# same query traffic.  Passes 1, 2a, 2b, 3, 4 attach perf to every pod at once;
# pass 3 (uncore IMC) runs once system-wide per node.  TMA runs per-pod
# with toplev -l2 (all pods parallel per path — queries run once, all pods measured simultaneously).
#
# Usage:
#   ./scripts/run_benchmark.sh [cell...] [--tokens N]
#
#   cell: rag_short | rag_medium | rag_long | rag_very_long
#         rag_pure_fetch
#         sc_a_short | sc_a_medium          (SC isolated)
#         sc_b_short | sc_b_medium          (SC full pipeline)
#         llm_short  | llm_medium | llm_long | llm_very_long
#         bge_short  | bge_medium | bge_long | bge_very_long
#         hnsw_short | hnsw_medium| hnsw_long| hnsw_very_long
#         calibration | tma | all (default)
#
#   --tokens N  : override token count (default runs 64+192+320)
#   --stream    : use SSE streaming for all query passes (records real TTFT and TPOT)
#
# Pod label overrides (env vars):
#   FASTAPI_LABEL      default: app=llm-service-kernel
#   MILVUS_LABEL       default: app=milvus
#   MILVUS_ETCD_LABEL  default: app=milvus-etcd
#   MILVUS_MINIO_LABEL default: app=milvus-minio
#   MONGODB_LABEL      default: app=mongodb
#   SEAWEED_MASTER_LABEL  default: app=seaweed-master
#   SEAWEED_VOLUME_LABEL  default: app=seaweed-volume
#   SEAWEED_FILER_LABEL   default: app=seaweed-filer
#   DEPLOY_ENV         minikube (default) | eks  — sets VLLM_NAMESPACE automatically
#   VLLM_LABEL         default: llm-d.ai/role=decode
#   VLLM_NAMESPACE     default: llm-d-local (minikube) / llm-d (EKS via DEPLOY_ENV=eks)
#
# Output: benchmark_results/run_TIMESTAMP/
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUERIES_DIR="${KERNEL_ROOT}/benchmark_queries"
RESULTS_DIR="${KERNEL_ROOT}/benchmark_results"
NAMESPACE="${BENCHMARK_NAMESPACE:-llm-service}"
BENCHMARK_URL="${BENCHMARK_URL:-http://localhost:8080}"
BENCHMARK_MODEL="${BENCHMARK_MODEL:-qwen2.5-0.5b}"

# Default counts per token tier
COUNT_64="${BENCHMARK_COUNT_64:-20}"
COUNT_192="${BENCHMARK_COUNT_192:-20}"
COUNT_320="${BENCHMARK_COUNT_320:-20}"
WARMUP="${BENCHMARK_WARMUP:-0}"
# Tier warmup: queries run ONCE per path per tier before any perf cell (outside all perf windows).
# RAG/LLM-direct: medium bucket queries. SC: cache population (short + medium buckets) + hw prime.
# Replaces the old per-cell PRERUN_WARMUP. Override: BENCHMARK_TIER_WARMUP=N.
TIER_WARMUP_COUNT="${BENCHMARK_TIER_WARMUP:-50}"
# Concurrency: in-flight queries per query_runner invocation (warmup + measurement).
# 1 = strict serial (legacy). Higher values keep CPU busy during the perf window and
# shorten wall-clock; vLLM continuous batching absorbs them on the GPU side.
CONCURRENCY="${BENCHMARK_CONCURRENCY:-${CONCURRENCY:-1}}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${RUN_DIR_OVERRIDE:-${RESULTS_DIR}/run_${TIMESTAMP}}"
mkdir -p "${RUN_DIR}"

# Parse args
CELLS=""
TOKEN_OVERRIDE=""
STREAM_FLAG=""   # set to "--stream" to enable SSE streaming (measures real TTFT/TPOT)
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tokens) TOKEN_OVERRIDE="$2"; shift 2 ;;
        --stream) STREAM_FLAG="--stream"; shift ;;
        *)        CELLS="${CELLS} $1"; shift ;;
    esac
done
CELLS="${CELLS:-all}"
CELLS="${CELLS## }"

# Hard-fail if --stream is not set: generation_ms/TTFT/TPOT will be empty without it,
# making the latency charts meaningless. Always pass --stream for real benchmark runs.
# Override with ALLOW_NO_STREAM=1 for local/dev runs only.
if [[ -z "${STREAM_FLAG}" && "${ALLOW_NO_STREAM:-0}" != "1" ]]; then
    printf '[ERROR] --stream not passed. generation_ms/TTFT/TPOT will be empty in all CSVs.\n' >&2
    printf '        Re-run with: bash scripts/run_benchmark.sh --stream %s\n' "${TOKEN_OVERRIDE:+--tokens ${TOKEN_OVERRIDE} }${CELLS}" >&2
    printf '        (dev/local only: set ALLOW_NO_STREAM=1 to skip this check)\n' >&2
    exit 1
fi

log()  { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
ok()   { printf '  ✓ %s\n' "$*"; }
warn() { printf '  ⚠ %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

# ── Pod label config (overridable via env) ────────────────────────────────────
FASTAPI_LABEL="${FASTAPI_LABEL:-app=llm-service-kernel}"
MILVUS_LABEL="${MILVUS_LABEL:-app=milvus}"
MILVUS_ETCD_LABEL="${MILVUS_ETCD_LABEL:-app=milvus-etcd}"
MILVUS_MINIO_LABEL="${MILVUS_MINIO_LABEL:-app=milvus-minio}"
MONGODB_LABEL="${MONGODB_LABEL:-app=mongodb}"
SEAWEED_MASTER_LABEL="${SEAWEED_MASTER_LABEL:-app=seaweed-master}"
SEAWEED_VOLUME_LABEL="${SEAWEED_VOLUME_LABEL:-app=seaweed-volume}"
SEAWEED_FILER_LABEL="${SEAWEED_FILER_LABEL:-app=seaweed-filer}"
VLLM_LABEL="${VLLM_LABEL:-llm-d.ai/role=decode}"
LLMD_GATEWAY_LABEL="${LLMD_GATEWAY_LABEL:-app.kubernetes.io/component=inference-gateway}"
# minikube → llm-d-local   EKS → llm-d   (set DEPLOY_ENV=eks or VLLM_NAMESPACE explicitly to override)
_default_vllm_ns="llm-d-local"
[[ "${DEPLOY_ENV:-}" == "eks" ]] && _default_vllm_ns="llm-d"
VLLM_NAMESPACE="${VLLM_NAMESPACE:-${_default_vllm_ns}}"
# llmd gateway lives in the same namespace as vLLM (llm-d on EKS, llm-d-local on minikube)
LLMD_GATEWAY_NAMESPACE="${LLMD_GATEWAY_NAMESPACE:-${VLLM_NAMESPACE}}"

# Host-level perf binary
PERF_HOST_BIN="${PERF_HOST_BIN:-$(find /usr/lib/linux-tools* -maxdepth 2 -name perf -type f -executable 2>/dev/null | sort -V | tail -1 || true)}"
PERF_HOST_BIN="${PERF_HOST_BIN:-perf}"

# Local runs: script runs as a regular user → prefix perf with sudo and lower paranoid.
# EKS pod runs as root (runAsUser: 0) → no sudo needed or available.
if [[ $(id -u) -ne 0 ]]; then
    PERF_SUDO="sudo"
    sudo sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null \
        || warn "Could not set perf_event_paranoid — perf counters may be empty"
else
    PERF_SUDO=""
    sysctl -w kernel.perf_event_paranoid=-1 2>/dev/null || true
fi

# perf-agent pod: runs on the GPU node to profile vLLM remotely (EKS multi-node setup)
# On minikube (single node) this is unused — all PIDs are local.
PERF_AGENT_POD="${PERF_AGENT_POD:-perf-agent}"
PERF_AGENT_NS="${PERF_AGENT_NS:-${NAMESPACE}}"

# ── Global associative arrays for pod/pid discovery ───────────────────────────
declare -gA POD_MAP            # pod_key → pod_name
declare -gA PID_MAP            # pod_key → PID inside pod (container namespace, for kubectl exec)
declare -gA HOST_PID_MAP       # pod_key → host-visible PID local to this node
declare -gA REMOTE_HOST_PID_MAP # pod_key → host-visible PID on perf-agent node (e.g. vLLM on GPU node)
declare -gA PERF_BG_PIDS       # pod_key → background perf PID (plus metadata keys prefixed _)
declare -gA POD_NAMESPACE_MAP  # pod_key → kubernetes namespace

# Pod label selectors, container pgrep patterns, and HOST ps patterns per key
declare -gA POD_LABELS
declare -gA POD_PGREP
declare -gA HOST_PS_PATTERN   # ps -eo cmd pattern to find host PID
# NOTE: milvus_etcd, milvus_minio and seaweed_master are intentionally NOT listed
# here. They are idle control-plane / metadata services (their PMU data is noise),
# so they are excluded from deep per-pod measurement. The work pods below are the
# only ones that do meaningful CPU work under query load.
POD_LABELS=(
    [fastapi]="${FASTAPI_LABEL}"
    [milvus]="${MILVUS_LABEL}"
    [mongodb]="${MONGODB_LABEL}"
    [seaweed_volume]="${SEAWEED_VOLUME_LABEL}"
    [seaweed_filer]="${SEAWEED_FILER_LABEL}"
    [vllm]="${VLLM_LABEL}"
    [llmd_gateway]="${LLMD_GATEWAY_LABEL}"
)
POD_PGREP=(
    [fastapi]="uvicorn"
    [milvus]="milvus"
    [mongodb]="mongod"
    [seaweed_volume]="weed.*volume"
    [seaweed_filer]="weed.*filer"
    [vllm]="python.*vllm"
    [llmd_gateway]="envoy"
)
# Patterns match the unique part of each process's cmdline as seen from the host.
# Uses ps -eo pid,cmd output; must not match shell/grep/awk artifacts.
HOST_PS_PATTERN=(
    [fastapi]="uvicorn src.service.main"
    [vllm]="python.*vllm.*serve"
    [milvus]="milvus run standalone"
    [mongodb]="[m]ongod"
    [seaweed_volume]="weed -logtostderr=true volume"
    [seaweed_filer]="weed -logtostderr=true filer"
    [llmd_gateway]="envoy.*istio/proxy"
)

# ── Pod helpers ───────────────────────────────────────────────────────────────

get_pod_by_label() {
    local label="$1" ns="${2:-$NAMESPACE}"
    kubectl get pod -n "${ns}" -l "${label}" \
        -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || true
}

get_pid_in_pod() {
    local pod="$1" pattern="$2" ns="${3:-$NAMESPACE}"
    # Try sh first; fall back to direct pgrep for distroless containers
    kubectl exec -n "${ns}" "${pod}" -- \
        sh -c "pgrep -f '${pattern}' | head -1 2>/dev/null" 2>/dev/null \
    || kubectl exec -n "${ns}" "${pod}" -- \
        pgrep -f "${pattern}" 2>/dev/null | head -1 \
    || true
}

_pod_ns() {
    # Return the namespace for a given pod key
    echo "${POD_NAMESPACE_MAP[$1]:-$NAMESPACE}"
}

_get_host_pid() {
    # Find the host-visible PID for a pod key using ps pattern matching.
    local key="$1"
    local ps_pat="${HOST_PS_PATTERN[$key]:-}"
    [[ -z "${ps_pat}" ]] && return
    ps -eo pid,cmd 2>/dev/null \
        | grep -E "${ps_pat}" \
        | grep -v grep | grep -v ' bash ' | grep -v ' sh ' \
        | awk '{print $1}' | head -1 || true
}

_get_remote_host_pid() {
    # Get host-visible PID for a pod running on the perf-agent node via kubectl exec.
    local key="$1"
    local ps_pat="${HOST_PS_PATTERN[$key]:-}"
    [[ -z "${ps_pat}" ]] && return
    kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- \
        bash -c "ps -eo pid,cmd | grep -E '${ps_pat}' | grep -v grep | awk '{print \$1}' | head -1 || true" \
        2>/dev/null || true
}

discover_all_pods() {
    POD_MAP=()
    PID_MAP=()
    HOST_PID_MAP=()
    REMOTE_HOST_PID_MAP=()
    POD_NAMESPACE_MAP=()
    log "Discovering pods and PIDs..."

    # Check if perf-agent is available (GPU node peer for remote perf)
    local perf_agent_ready=0
    if kubectl get pod -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" \
           --no-headers 2>/dev/null | grep -q "Running"; then
        perf_agent_ready=1
        local agent_perf_ver agent_paranoid
        agent_perf_ver=$(kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- \
            perf --version 2>/dev/null || echo "unknown")
        agent_paranoid=$(kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- \
            cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "unknown")
        ok "perf-agent pod found — perf=${agent_perf_ver} paranoid=${agent_paranoid}"
    else
        warn "perf-agent pod not found/running — GPU-node pods (vLLM) will not have perf measurements"
    fi

    # Build per-key namespace map (all default to NAMESPACE except vllm + llmd_gateway)
    for key in "${!POD_LABELS[@]}"; do
        case "${key}" in
            vllm)         POD_NAMESPACE_MAP[$key]="${VLLM_NAMESPACE}" ;;
            llmd_gateway) POD_NAMESPACE_MAP[$key]="${LLMD_GATEWAY_NAMESPACE}" ;;
            *)            POD_NAMESPACE_MAP[$key]="${NAMESPACE}" ;;
        esac
    done

    for key in "${!POD_LABELS[@]}"; do
        local label="${POD_LABELS[$key]}"
        local pattern="${POD_PGREP[$key]}"
        local ns="${POD_NAMESPACE_MAP[$key]:-$NAMESPACE}"
        local pod pid host_pid
        pod=$(get_pod_by_label "${label}" "${ns}")
        if [[ -z "${pod}" ]]; then
            warn "Pod not found for key=${key} label=${label} ns=${ns} — skipping"
            continue
        fi
        # Container-internal PID (for kubectl exec operations)
        pid=$(get_pid_in_pod "${pod}" "${pattern}" "${ns}")
        if [[ -z "${pid}" ]] || ! [[ "${pid}" =~ ^[0-9]+$ ]]; then
            pid="unknown"
        fi
        # Host-visible PID: try local first, then remote via perf-agent
        local host_pid
        host_pid=$(_get_host_pid "${key}")
        if [[ -n "${host_pid}" ]] && [[ "${host_pid}" =~ ^[0-9]+$ ]]; then
            HOST_PID_MAP[${key}]="${host_pid}"
            ok "  ${key}: pod=${pod} host_pid=${host_pid} ns=${ns} (local)"
        elif [[ "${perf_agent_ready}" == "1" ]]; then
            local remote_pid
            remote_pid=$(_get_remote_host_pid "${key}")
            if [[ -n "${remote_pid}" ]] && [[ "${remote_pid}" =~ ^[0-9]+$ ]]; then
                REMOTE_HOST_PID_MAP[${key}]="${remote_pid}"
                ok "  ${key}: pod=${pod} host_pid=${remote_pid} ns=${ns} (remote via perf-agent)"
            else
                warn "  ${key}: host PID not found locally or on perf-agent — skipping"
            fi
        else
            warn "  ${key}: host PID not visible and no perf-agent available — skipping"
        fi
        POD_MAP[${key}]="${pod}"
        PID_MAP[${key}]="${pid}"
    done
    # Require at least fastapi to be measurable
    if [[ -z "${HOST_PID_MAP[fastapi]:-}" ]]; then
        die "FastAPI host PID not found — benchmark-runner must run on the same node as FastAPI."
    fi
    if [[ ${#HOST_PID_MAP[@]} -eq 0 ]] && [[ ${#REMOTE_HOST_PID_MAP[@]} -eq 0 ]]; then
        die "No host PIDs discovered — cannot run benchmark without perf measurements."
    fi
    ok "perf binary: ${PERF_HOST_BIN}"
}

get_fastapi_pod() {
    echo "${POD_MAP[fastapi]:-}"
}

get_fastapi_node() {
    local pod="${POD_MAP[fastapi]:-}"
    [[ -z "${pod}" ]] && echo "unknown" && return
    kubectl get pod -n "${NAMESPACE}" "${pod}" \
        -o jsonpath='{.spec.nodeName}' 2>/dev/null || echo "unknown"
}

# ── Health check ─────────────────────────────────────────────────────────────

check_health() {
    log "Checking service health..."
    local ready
    ready=$(curl -sf "${BENCHMARK_URL}/health" \
        | python3 -c "import sys,json;h=json.load(sys.stdin);print('ok' if h.get('ready') else 'not_ready')" \
        2>/dev/null || echo "unreachable")
    [ "${ready}" = "ok" ] || die "Service not ready at ${BENCHMARK_URL}"
    ok "Service ready"

    local node="${BENCHMARK_NODE:-$(get_fastapi_node)}"
    if [[ "${node}" != "unknown" && -n "${node}" ]]; then
        local instance
        instance=$(kubectl get node "${node}" \
            -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null || echo "unknown")
        ok "Node: ${node} (${instance})"
    fi

    local fastapi_pod="${POD_MAP[fastapi]:-}"
    if [[ -n "${fastapi_pod}" ]]; then
        local seaweed_flag
        seaweed_flag=$(kubectl exec -n "${NAMESPACE}" "${fastapi_pod}" -- \
            sh -c 'echo ${RAG_FORCE_SEAWEED_FETCH:-0}' 2>/dev/null || echo "0")
        if [ "${seaweed_flag}" = "1" ]; then
            ok "SeaweedFS hot path: ENABLED"
        else
            die "RAG_FORCE_SEAWEED_FETCH not set — SeaweedFS bypassed. Deploy with kustomize EKS overlay or set RAG_FORCE_SEAWEED_FETCH=1 in the configmap."
        fi
    fi
}

# ── Calibration ───────────────────────────────────────────────────────────────

run_calibration() {
    log "=== CALIBRATION ==="
    local fastapi_pod="${POD_MAP[fastapi]:-}"
    [[ -z "${fastapi_pod}" ]] && { warn "No fastapi pod — skipping calibration"; return; }

    local cal_dir="${RUN_DIR}/calibration"
    mkdir -p "${cal_dir}"

    log "Running STREAM benchmark..."
    kubectl exec -n "${NAMESPACE}" "${fastapi_pod}" -- /usr/local/bin/stream \
        > "${cal_dir}/stream.txt" 2>&1 || warn "STREAM failed"
    grep -E "Triad|Copy|Scale|Add" "${cal_dir}/stream.txt" 2>/dev/null | while read -r line; do
        ok "STREAM: ${line}"
    done || true

    log "Verifying perf events (host-level)..."
    {
      ${PERF_SUDO} "${PERF_HOST_BIN}" stat -e cycles,instructions,task-clock,branch-misses,uops_issued.any,uops_retired.slots,uops_executed.core -- sleep 0.1 2>&1
      ${PERF_SUDO} "${PERF_HOST_BIN}" stat -e L1-dcache-load-misses,l2_rqsts.miss,cache-misses,cache-references,cycle_activity.stalls_l3_miss,cycle_activity.stalls_total -- sleep 0.1 2>&1
      ${PERF_SUDO} "${PERF_HOST_BIN}" stat -e fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.scalar_single -- sleep 0.1 2>&1
      ${PERF_SUDO} "${PERF_HOST_BIN}" stat -e mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles -- sleep 0.1 2>&1
      ${PERF_SUDO} "${PERF_HOST_BIN}" stat -e slots,topdown-retiring,topdown-fe-bound,topdown-be-bound,topdown-bad-spec -- sleep 0.1 2>&1
    } > "${cal_dir}/perf_verify.txt" 2>&1 || true

    if grep -Eiq "not supported|No permission|event syntax error|failed to open|Bad event|unable to find" \
            "${cal_dir}/perf_verify.txt" 2>/dev/null; then
        warn "Some perf events unsupported on this kernel — check ${cal_dir}/perf_verify.txt"
    else
        echo "all_events_ok" >> "${cal_dir}/perf_verify.txt"
        ok "All perf events verified"
    fi

    ok "Calibration complete → ${cal_dir}/"
}

# ── Per-pod perf pass infrastructure ─────────────────────────────────────────

_get_events_for_pass() {
    local pass_name="$1"
    case "${pass_name}" in
        pass1)
            # branch-instructions added for misprediction rate (branch-misses / branch-instructions)
            echo "cycles,instructions,task-clock,branch-misses,branch-instructions,uops_issued.any,uops_retired.slots,uops_executed.core,context-switches,cpu-migrations"
            ;;
        pass2a)
            # Per-level load HIT/MISS hierarchy for AMAT + load-based MPKI.
            # mem_load_retired.l1_hit/.l2_hit/.l3_hit/.l3_miss are mutually-exclusive
            # load outcomes; total_loads = their sum, and per-level MPKI is derivable
            # (L1 miss = l2_hit+l3_hit+l3_miss, etc.).
            # AMAT = (l1h*4 + l2h*12 + l3h*40 + l3miss*200) / (l1h+l2h+l3h+l3miss)  [SPR latencies].
            # NOTE (verified on c7i.metal SPR 2026-06): these PEBS events do NOT co-schedule
            # with L1-dcache-load-misses/l2_rqsts.miss/cache-misses — adding those caused
            # multiplexing that dropped l3_hit/l3_miss to <not counted>. Keeping ONLY the
            # 4 PEBS + cycles runs at 100% (no multiplexing).
            echo "cycles,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
            ;;
        pass2b)
            # Stalls + TLB MPKI (Kanev et al. WSC) — 5 GP counters, verify time_running/time_enabled in output
            echo "cycles,cycle_activity.stalls_l3_miss,cycle_activity.stalls_total,iTLB-load-misses,L1-icache-load-misses,dTLB-load-misses"
            ;;
        pass4)
            # load-bound + MLP (pending / pending_cycles) + FP/FLOPs packed in to
            # avoid a separate 6th drive (pass4 had spare GP counters). FP gives the
            # arithmetic-intensity NUMERATOR (FLOPs); the DRAM-bytes denominator is
            # pass3's node-wide uncore IMC. Widths: FP32 scalar/256b/512b cover BGE's
            # FP32-dominant matmul, + scalar FP64 to catch stragglers. 7 GP counters
            # (≤8) → no multiplexing; verify time_running==time_enabled in output.
            # NOTE: BF16/INT8/FP64-vector widths omitted to fit one pass — add a
            # dedicated FP pass if full SIMD-width coverage is needed.
            echo "cycles,exe_activity.bound_on_loads,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.scalar_double"
            ;;
        passfp)
            # Dedicated FULL-WIDTH FP pass. FIX (2026-06-20): pass4 omitted packed-DOUBLE
            # (128/256/512b_packed_double) -> BLIND to numpy/BLAS/Milvus double-precision SIMD
            # -> falsely reported ~0 vectorized FP. This pass adds all packed-double + 128b_single.
            # 8 fp GP events + cycles/instructions (fixed) ; verify time_running==time_enabled
            # (if it multiplexes, perf scales -> FLOP totals still valid). FLOP lane weights:
            # scalar=1; packed_single 128/256/512 = 4/8/16; packed_double 128/256/512 = 2/4/8; FMA x2.
            echo "cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
            ;;
    esac
}

# Software-only event set for remote pods running on virtualized nodes (e.g. g6.12xlarge GPU node).
# Nitro hypervisor does not expose hardware PMU to guests, so only software events are available.
_get_remote_events_for_pass() {
    # Same set for all passes — only software counters available on the GPU node.
    echo "task-clock,context-switches,cpu-migrations,page-faults"
}

# Start perf on ALL discovered pods simultaneously from the host (Option B).
# Uses host-visible PIDs — no perf installation required inside containers.
# Writes: ${cell_dir}/perf_${pass_name}_${pod_key}.txt for each pod.
# Also copies fastapi result to perf_${pass_name}.txt for backward compat.
start_perf_all_pods() {
    local pass_name="$1" cell_dir="$2"
    local events
    events=$(_get_events_for_pass "${pass_name}")
    PERF_BG_PIDS=()
    PERF_BG_PIDS[_current_pass]="${pass_name}"

    # Local pods — perf runs directly on this node
    for key in "${!HOST_PID_MAP[@]}"; do
        local host_pid="${HOST_PID_MAP[$key]}"
        [[ -z "${host_pid}" ]] && continue

        local out_file="${cell_dir}/perf_${pass_name}_${key}.txt"
        ${PERF_SUDO} "${PERF_HOST_BIN}" stat -p "${host_pid}" -e "${events}" \
            > "${out_file}" 2>&1 &
        PERF_BG_PIDS[${key}]=$!

        if [[ "${key}" == "fastapi" ]]; then
            PERF_BG_PIDS[_fastapi_out]="${out_file}"
            PERF_BG_PIDS[_fastapi_compat]="${cell_dir}/perf_${pass_name}.txt"
        fi
    done

    # Remote pods — perf runs on perf-agent pod (GPU node), result fetched later.
    # GPU node is a virtualized Nitro instance with no hardware PMU; use software-only events.
    local remote_events
    remote_events=$(_get_remote_events_for_pass "${pass_name}")
    for key in "${!REMOTE_HOST_PID_MAP[@]}"; do
        local remote_pid="${REMOTE_HOST_PID_MAP[$key]}"
        [[ -z "${remote_pid}" ]] && continue

        local remote_out="/tmp/perf_${pass_name}_${key}.txt"
        local remote_pid_file="/tmp/perf_${pass_name}_${key}.pid"

        # Start perf in background inside perf-agent pod.
        # 'disown' is critical: it detaches the background job from bash so bash exits
        # immediately, allowing kubectl exec to return without blocking the benchmark script.
        kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- bash -c \
            "perf stat -p ${remote_pid} -e ${remote_events} > ${remote_out} 2>&1 & echo \$! > ${remote_pid_file}; disown" \
            </dev/null 2>/dev/null || true

        PERF_BG_PIDS[_remote_${key}_out]="${remote_out}"
        PERF_BG_PIDS[_remote_${key}_local]="${cell_dir}/perf_${pass_name}_${key}.txt"
    done

    # Brief pause so perf attaches before the first query byte hits the process
    sleep 0.3
}

# Send SIGINT to all running host-level perf processes so they write their summary.
kill_perf_all_pods() {
    # Step 1: SIGINT to local background perf processes
    for key in "${!PERF_BG_PIDS[@]}"; do
        [[ "${key}" == _* ]] && continue
        local bg_pid="${PERF_BG_PIDS[$key]:-}"
        [[ -z "${bg_pid}" ]] && continue
        kill -INT "${bg_pid}" 2>/dev/null || true
    done

    # Step 2: SIGINT to remote perf processes on perf-agent
    for key in "${!REMOTE_HOST_PID_MAP[@]}"; do
        local remote_pid_file="/tmp/perf_${PERF_BG_PIDS[_current_pass]:-}_${key}.pid"
        kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- bash -c \
            "[[ -f '${remote_pid_file}' ]] && kill -INT \$(cat '${remote_pid_file}') 2>/dev/null || true" \
            2>/dev/null || true
    done

    # Step 3: Wait for perf to flush
    sleep 3

    # Step 4: Kill local perf processes
    for key in "${!PERF_BG_PIDS[@]}"; do
        [[ "${key}" == _* ]] && continue
        local bg_pid="${PERF_BG_PIDS[$key]:-}"
        [[ -z "${bg_pid}" ]] && continue
        kill -KILL "${bg_pid}" 2>/dev/null || true
        wait "${bg_pid}" 2>/dev/null || true
    done

    # Step 5: Collect remote perf results via kubectl cp
    for key in "${!PERF_BG_PIDS[@]}"; do
        [[ "${key}" != _remote_*_out ]] && continue
        local suffix="${key#_remote_}"
        suffix="${suffix%_out}"
        local remote_out="${PERF_BG_PIDS[$key]}"
        local local_out="${PERF_BG_PIDS[_remote_${suffix}_local]:-}"
        [[ -z "${local_out}" ]] && continue
        kubectl cp "${PERF_AGENT_NS}/${PERF_AGENT_POD}:${remote_out}" "${local_out}" \
            2>/dev/null || true
    done

    # Step 6: Backward compat copy for fastapi
    local fastapi_out="${PERF_BG_PIDS[_fastapi_out]:-}"
    local fastapi_compat="${PERF_BG_PIDS[_fastapi_compat]:-}"
    if [[ -n "${fastapi_out}" && -f "${fastapi_out}" && -n "${fastapi_compat}" ]]; then
        cp "${fastapi_out}" "${fastapi_compat}" 2>/dev/null || true
    fi

    PERF_BG_PIDS=()
}

# Legacy single-pod interface — used for pass3 (node-wide uncore) only
PERF_BG_PID=""

start_perf_pass3_node() {
    local _pod="$1" out_file="$2"
    # Node-wide DRAM bandwidth via CHA (Caching Home Agent) events — correct for
    # Sapphire Rapids (c7i). Each count = 1 cache line (64 bytes).
    # uncore_imc/cas_count_read/ does NOT exist on SPR; use uncore_cha instead.
    ${PERF_SUDO} "${PERF_HOST_BIN}" stat -a \
        -e "uncore_cha/unc_cha_imc_reads_count.normal/,uncore_cha/unc_cha_imc_writes_count.full/" \
        > "${out_file}" 2>&1 &
    PERF_BG_PID=$!
}

kill_perf() {
    # Used for node-wide pass3
    if [ -n "${PERF_BG_PID}" ]; then
        kill -INT "${PERF_BG_PID}" 2>/dev/null || true
        sleep 3
        kill -KILL "${PERF_BG_PID}" 2>/dev/null || true
        wait "${PERF_BG_PID}" 2>/dev/null || true
        PERF_BG_PID=""
    fi
}

# ── GPU monitoring (nvidia-smi dmon + pmon) ───────────────────────────────────
# Runs alongside pass1 queries only. ~1% overhead — does not affect measurements.
# dmon: device-level SM%, HMMA tensor%, DRAM% (overall GPU)
# pmon: per-process SM%, FB memory (identifies vLLM vs display processes)

GPU_DMON_PID=""
GPU_PMON_PID=""

start_gpu_monitor() {
    local cell_dir="$1"
    command -v nvidia-smi &>/dev/null || return 0
    # Device-level: basic util + GPM metrics (SM Activity, SM Occupancy, HMMA Tensor, DRAM Activity)
    nvidia-smi dmon -s u --gpm-metrics 2,3,7,10 -d 1 \
        > "${cell_dir}/gpu_dmon.txt" 2>/dev/null &
    GPU_DMON_PID=$!
    # Per-process: SM util%, memory util%, FB memory
    nvidia-smi pmon -s um -d 1 \
        > "${cell_dir}/gpu_pmon.txt" 2>/dev/null &
    GPU_PMON_PID=$!
}

stop_gpu_monitor() {
    if [[ -n "${GPU_DMON_PID}" ]]; then
        kill "${GPU_DMON_PID}" 2>/dev/null || true
        wait "${GPU_DMON_PID}" 2>/dev/null || true
        GPU_DMON_PID=""
    fi
    if [[ -n "${GPU_PMON_PID}" ]]; then
        kill "${GPU_PMON_PID}" 2>/dev/null || true
        wait "${GPU_PMON_PID}" 2>/dev/null || true
        GPU_PMON_PID=""
    fi
}

# ── vLLM engine metrics (Prometheus /metrics) ─────────────────────────────────
# Polls the vLLM /metrics endpoint during the Pass-1 window. Captures engine
# internals perf/nsys/ncu cannot see: queue depth, KV-cache %, preemptions,
# token throughput. Gauges → min/max/median (low-n convention); counters → delta.
# URL is derived from the vLLM pod IP; override with VLLM_METRICS_URL.
VLLM_METRICS_PID=""

start_vllm_metrics() {
    local cell_dir="$1"
    local url="${VLLM_METRICS_URL:-}"
    if [[ -z "${url}" ]]; then
        local vpod="${POD_MAP[vllm]:-}"
        local vns="${POD_NAMESPACE_MAP[vllm]:-${VLLM_NAMESPACE}}"
        [[ -z "${vpod}" ]] && { warn "no vllm pod — skipping vLLM metrics scrape"; return 0; }
        local vip
        vip=$(kubectl get pod -n "${vns}" "${vpod}" -o jsonpath='{.status.podIP}' 2>/dev/null || true)
        [[ -z "${vip}" ]] && { warn "no vllm pod IP — skipping vLLM metrics scrape"; return 0; }
        url="http://${vip}:${VLLM_METRICS_PORT:-8200}/metrics"
    fi
    python3 "${SCRIPT_DIR}/vllm_metrics_scraper.py" \
        --url "${url}" \
        --interval "${VLLM_METRICS_INTERVAL:-1}" \
        --out-csv "${cell_dir}/vllm_metrics.csv" \
        --out-summary "${cell_dir}/vllm_metrics_summary.json" \
        > "${cell_dir}/vllm_metrics.log" 2>&1 &
    VLLM_METRICS_PID=$!
}

stop_vllm_metrics() {
    [[ -z "${VLLM_METRICS_PID}" ]] && return 0
    kill -TERM "${VLLM_METRICS_PID}" 2>/dev/null || true
    wait "${VLLM_METRICS_PID}" 2>/dev/null || true
    VLLM_METRICS_PID=""
}

# ── vLLM CPU interval perf (pass1 only) ──────────────────────────────────────
# Runs perf stat -I 200 on the vLLM process during pass1 to capture CPU activity
# pattern over time while the GPU is running. Correlate with gpu_dmon.txt (which
# is also time-sampled) to see what the CPU is doing between GPU kernel launches.
# On EKS the vLLM PID is remote (GPU node) — runs via perf-agent.
VLLM_INTERVAL_PID=""
VLLM_INTERVAL_REMOTE=0

start_vllm_interval_perf() {
    local cell_dir="$1"
    local out_file="${cell_dir}/perf_pass1_vllm_interval.txt"

    # Local vLLM (minikube)
    if [[ -n "${HOST_PID_MAP[vllm]:-}" ]]; then
        ${PERF_SUDO} "${PERF_HOST_BIN}" stat \
            -p "${HOST_PID_MAP[vllm]}" \
            -I 200 \
            -e "cycles,instructions,task-clock,context-switches" \
            > "${out_file}" 2>&1 &
        VLLM_INTERVAL_PID=$!
        VLLM_INTERVAL_REMOTE=0
        return
    fi

    # Remote vLLM (EKS perf-agent)
    if [[ -n "${REMOTE_HOST_PID_MAP[vllm]:-}" ]]; then
        if kubectl get pod -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" \
               --no-headers 2>/dev/null | grep -q "Running"; then
            kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- bash -c \
                "perf stat -p ${REMOTE_HOST_PID_MAP[vllm]} -I 200 \
                 -e cycles,instructions,task-clock,context-switches \
                 > /tmp/perf_vllm_interval.txt 2>&1 & echo \$! > /tmp/perf_vllm_interval.pid; disown" \
                </dev/null 2>/dev/null || true
            VLLM_INTERVAL_REMOTE=1
        fi
    fi
}

stop_vllm_interval_perf() {
    local cell_dir="$1"
    local out_file="${cell_dir}/perf_pass1_vllm_interval.txt"

    if [[ "${VLLM_INTERVAL_REMOTE}" == "0" && -n "${VLLM_INTERVAL_PID}" ]]; then
        kill -INT "${VLLM_INTERVAL_PID}" 2>/dev/null || true
        sleep 1
        kill -KILL "${VLLM_INTERVAL_PID}" 2>/dev/null || true
        wait "${VLLM_INTERVAL_PID}" 2>/dev/null || true
        VLLM_INTERVAL_PID=""
    elif [[ "${VLLM_INTERVAL_REMOTE}" == "1" ]]; then
        kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- bash -c \
            "[[ -f /tmp/perf_vllm_interval.pid ]] && \
             kill -INT \$(cat /tmp/perf_vllm_interval.pid) 2>/dev/null; \
             sleep 1; \
             kill -KILL \$(cat /tmp/perf_vllm_interval.pid) 2>/dev/null; true" \
            2>/dev/null || true
        sleep 1
        kubectl cp "${PERF_AGENT_NS}/${PERF_AGENT_POD}:/tmp/perf_vllm_interval.txt" \
            "${out_file}" 2>/dev/null || true
        VLLM_INTERVAL_REMOTE=0
    fi
    VLLM_INTERVAL_PID=""
}

# ── Query runner wrapper ──────────────────────────────────────────────────────

run_queries() {
    local mode="$1" qfile="$2" bucket="$3" out_dir="$4" count="$5" max_tok="$6"
    local extra_args="${7:-}"
    local warmup_file="${8:-}"
    shift 8
    local runner_flags=("$@")   # e.g. --warmup-only, --no-warmup

    local qr_mode
    case "${mode}" in
        sc_a)           qr_mode="cache_a" ;;
        sc_b)           qr_mode="cache_b" ;;
        # rag_pure_fetch maps to "rag" mode at the API level — the bypass flags and query
        # flags are identical to standard RAG.  The "pure fetch" isolation comes entirely
        # from the query file (benchmark_queries/rag_pure_fetch/queries.txt), which contains
        # bare factual questions with no document-specific context hints, forcing the
        # retriever to do a cold full-corpus search.  If cache=bypass and RAG are enabled,
        # every request will hit the retrieve → SeaweedFS → LLM path.
        rag_pure_fetch) qr_mode="rag" ;;
        *)              qr_mode="${mode}" ;;
    esac

    # Warmup-only calls only need to populate cache; streaming adds no value there
    # and may change cache-write behaviour (cache writes happen after full response).
    local _stream_arg="${STREAM_FLAG}"
    [[ " ${runner_flags[*]:-} " == *" --warmup-only "* ]] && _stream_arg=""

    BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
    python3 "${SCRIPT_DIR}/query_runner.py" \
        --mode "${qr_mode}" \
        --queries "${qfile}" \
        ${warmup_file:+--warmup-queries "${warmup_file}"} \
        --size-bucket "${bucket}" \
        --count "${count}" \
        --warmup "${WARMUP}" \
        --max-tokens "${max_tok}" \
        --out-dir "${out_dir}" \
        ${extra_args} \
        ${_stream_arg} \
        "${runner_flags[@]}" \
        2>&1
}

# ── Single cell with 4 perf passes — ALL pods measured simultaneously ─────────

run_cell_hw() {
    local mode="$1" bucket="$2" max_tok="${3:-64}" count_override="${4:-}"
    local cell_name="${mode}_${bucket}"
    log "=== CELL (hw): ${cell_name} [tok=${max_tok}] ==="

    local cell_dir="${RUN_DIR}/tok${max_tok}/cell_${cell_name}"
    mkdir -p "${cell_dir}"

    local query_file warmup_file="" sc_mode_arg=""
    case "${mode}" in
        rag)
            query_file="${QUERIES_DIR}/rag/${bucket}.txt"
            ;;
        rag_pure_fetch)
            query_file="${QUERIES_DIR}/rag_pure_fetch/queries.txt"
            ;;
        sc_a)
            query_file="${QUERIES_DIR}/cache/${bucket}_measure.txt"
            warmup_file="${QUERIES_DIR}/cache/${bucket}_warm.txt"
            sc_mode_arg="--sc-scenario a"
            ;;
        sc_b)
            query_file="${QUERIES_DIR}/cache/${bucket}_measure.txt"
            warmup_file="${QUERIES_DIR}/cache/${bucket}_warm.txt"
            sc_mode_arg="--sc-scenario b"
            ;;
        llm_direct)
            query_file="${QUERIES_DIR}/llm_direct/${bucket}.txt"
            ;;
        bge_isolated|hnsw_isolated)
            query_file="${QUERIES_DIR}/rag/${bucket}.txt"
            ;;
    esac

    [ -f "${query_file}" ] || { warn "Query file missing: ${query_file}"; return; }

    local count="${count_override:-${COUNT_64}}"
    # NOTE: perf measures a batch window (start perf → run N queries → stop perf), not
    # per-query PMU attribution.  PMU counters are totals over the entire batch.
    # Per-cell warmup removed — warmup_tier_path() primes hardware once per path per tier.
    # --no-warmup flag: prevents SC cells from re-populating the cache on every pass.
    # The cache is already populated by warmup_tier_path("sc", tok).
    local no_warmup_flag=""
    [[ -n "${warmup_file}" ]] && no_warmup_flag="--no-warmup"

    # ── Pass 1/4: IPC + ILP + pipeline (ALL pods simultaneously) ─────────────
    log "  Pass 1/4: IPC + ILP + pipeline (all pods)..."
    start_perf_all_pods "pass1" "${cell_dir}"
    start_gpu_monitor "${cell_dir}"
    start_vllm_metrics "${cell_dir}"
    start_vllm_interval_perf "${cell_dir}"
    run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}" "${count}" "${max_tok}" \
        "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tee "${cell_dir}/stdout.txt"
    kill_perf_all_pods
    stop_gpu_monitor
    stop_vllm_metrics
    stop_vllm_interval_perf "${cell_dir}"

    # ── Pass 2a: L1/L2/LLC cache hierarchy (ALL pods simultaneously) ─────────
    log "  Pass 2a/4: L1/L2/LLC cache hierarchy (all pods)..."
    start_perf_all_pods "pass2a" "${cell_dir}"
    mkdir -p "${cell_dir}/pass2a"
    run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/pass2a" "${count}" "${max_tok}" \
        "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tail -8
    kill_perf_all_pods

    # ── Pass 2b: Stalls + TLB MPKI (ALL pods simultaneously) ─────────────────
    log "  Pass 2b/4: Stalls + TLB MPKI (all pods)..."
    start_perf_all_pods "pass2b" "${cell_dir}"
    mkdir -p "${cell_dir}/pass2b"
    run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/pass2b" "${count}" "${max_tok}" \
        "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tail -8
    kill_perf_all_pods

    # ── Pass 4: DRAM bandwidth — ONCE PER NODE (uncore system-wide) ──────────
    log "  Pass 3/4: DRAM bandwidth (node-wide IMC, one run)..."
    local fastapi_pod="${POD_MAP[fastapi]:-}"
    if [[ -n "${fastapi_pod}" ]]; then
        start_perf_pass3_node "${fastapi_pod}" "${cell_dir}/perf_pass3_node.txt"
        mkdir -p "${cell_dir}/pass3"
        run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/pass3" "${count}" "${max_tok}" \
            "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tail -8
        kill_perf
        cp "${cell_dir}/perf_pass3_node.txt" "${cell_dir}/perf_pass3.txt" 2>/dev/null || true
    else
        warn "  No fastapi pod — skipping pass3 IMC"
    fi

    # ── Pass 4/4: load-bound + MLP (ALL pods simultaneously) ─────────────────
    # Merged former 5a+5b. Store-bound, ports, and the L1/L2/L3-hit pyramid were
    # dropped as redundant with TMA's Memory_Bound/Core_Bound decomposition.
    log "  Pass 4/4: load-bound + MLP (all pods)..."
    start_perf_all_pods "pass4" "${cell_dir}"
    mkdir -p "${cell_dir}/pass4"
    run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/pass4" "${count}" "${max_tok}" \
        "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tail -8
    kill_perf_all_pods

    # Pass FP: full-width FP/FLOPs incl packed-DOUBLE (fixes the pass4 single-only blindness)
    log "  Pass FP: full-width FP (incl packed-double) (all pods)..."
    start_perf_all_pods "passfp" "${cell_dir}"
    mkdir -p "${cell_dir}/passfp"
    run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/passfp" "${count}" "${max_tok}" \
        "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tail -8
    kill_perf_all_pods

    ok "  Cell ${cell_name} complete → ${cell_dir}/"
}

# ── Pass-4-only sweep ─────────────────────────────────────────────────────────

run_cell_pass3_only() {
    local mode="$1" bucket="$2" max_tok="${3:-64}" count_override="${4:-}"
    local cell_name="${mode}_${bucket}"
    local cell_dir="${RUN_DIR}/tok${max_tok}/cell_${cell_name}"
    [ -d "${cell_dir}" ] || { warn "Cell dir missing, skipping pass3: ${cell_dir}"; return; }

    log "  Pass3 (IMC BW): ${cell_name} [tok=${max_tok}]"
    local fastapi_pod="${POD_MAP[fastapi]:-}"
    [[ -z "${fastapi_pod}" ]] && { warn "No fastapi pod for pass3"; return; }

    local query_file warmup_file="" sc_mode_arg=""
    case "${mode}" in
        rag)        query_file="${QUERIES_DIR}/rag/${bucket}.txt" ;;
        rag_pure_fetch) query_file="${QUERIES_DIR}/rag_pure_fetch/queries.txt" ;;
        sc_a)       query_file="${QUERIES_DIR}/cache/${bucket}_measure.txt"
                    warmup_file="${QUERIES_DIR}/cache/${bucket}_warm.txt"
                    sc_mode_arg="--sc-scenario a" ;;
        sc_b)       query_file="${QUERIES_DIR}/cache/${bucket}_measure.txt"
                    warmup_file="${QUERIES_DIR}/cache/${bucket}_warm.txt"
                    sc_mode_arg="--sc-scenario b" ;;
        llm_direct) query_file="${QUERIES_DIR}/llm_direct/${bucket}.txt" ;;
        bge_isolated|hnsw_isolated) query_file="${QUERIES_DIR}/rag/${bucket}.txt" ;;
    esac
    [ -f "${query_file}" ] || { warn "Query file missing: ${query_file}"; return; }

    local count="${count_override:-${COUNT_64}}"
    local no_warmup_flag=""
    [[ -n "${warmup_file}" ]] && no_warmup_flag="--no-warmup"

    # SC cache population before pass3 sweep (safety net when pass3_sweep runs standalone).
    # WARMUP=0 globally, so we override it here to run a real cache-population batch.
    if [[ -n "${warmup_file}" ]]; then
        log "  SC cache population for pass3 (outside perf window)..."
        mkdir -p "${cell_dir}/pass3"
        local _old_warmup="${WARMUP}"
        WARMUP=20
        run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/pass3" "${count}" "${max_tok}" \
            "${sc_mode_arg}" "${warmup_file}" --warmup-only 2>&1 | tail -3
        WARMUP="${_old_warmup}"
    fi

    start_perf_pass3_node "${fastapi_pod}" "${cell_dir}/perf_pass3_node.txt"
    mkdir -p "${cell_dir}/pass3"
    run_queries "${mode}" "${query_file}" "${bucket}" "${cell_dir}/pass3" "${count}" "${max_tok}" \
        "${sc_mode_arg}" "${warmup_file}" ${no_warmup_flag} 2>&1 | tail -4
    kill_perf
    cp "${cell_dir}/perf_pass3_node.txt" "${cell_dir}/perf_pass3.txt" 2>/dev/null || true
    ok "    pass3 done → ${cell_dir}/"
}

# ── Per-tier path warmup ──────────────────────────────────────────────────────
# Called ONCE per path per tier before any hw cells for that path.
# Primes LLC, HNSW graph, MongoDB buffer pool, vLLM KV cache, Python heap for
# the given path so all subsequent cells start in steady state with no per-cell warmup.
# SC also populates the semantic cache for all measured buckets (short + medium).

warmup_tier_path() {
    local path_mode="$1" tok="$2"
    local wcount="${TIER_WARMUP_COUNT:-50}"
    local wdir="${RUN_DIR}/warmup"
    mkdir -p "${wdir}"

    case "${path_mode}" in
        rag)
            log "  Tier warmup: rag medium (${wcount} queries, tok=${tok})..."
            run_queries "rag" "${QUERIES_DIR}/rag/medium.txt" "medium" "${wdir}" \
                "${wcount}" "${tok}" "" "" 2>&1 | tail -2
            ;;
        llm_direct)
            log "  Tier warmup: llm_direct medium (${wcount} queries, tok=${tok})..."
            run_queries "llm_direct" "${QUERIES_DIR}/llm_direct/medium.txt" "medium" "${wdir}" \
                "${wcount}" "${tok}" "" "" 2>&1 | tail -2
            ;;
        sc)
            # Populate semantic cache for every measured bucket before any SC cell runs.
            # WARMUP=0 globally — temporarily override so load_queries gets a real count.
            local _old_warmup="${WARMUP}"
            WARMUP="${wcount}"
            for sc_bucket in short medium; do
                local wf="${QUERIES_DIR}/cache/${sc_bucket}_warm.txt"
                local qf="${QUERIES_DIR}/cache/${sc_bucket}_measure.txt"
                [ -f "${wf}" ] || continue
                log "  Tier warmup: SC cache population (${sc_bucket}, ${wcount} q, tok=${tok})..."
                run_queries "sc_a" "${qf}" "${sc_bucket}" "${wdir}" \
                    "${wcount}" "${tok}" "--sc-scenario a" "${wf}" \
                    --warmup-only 2>&1 | tail -3
            done
            WARMUP="${_old_warmup}"
            # Hardware prime: run measurement queries through the SC hot path
            log "  Tier warmup: SC hardware prime (20 queries, tok=${tok})..."
            run_queries "sc_a" "${QUERIES_DIR}/cache/medium_measure.txt" "medium" "${wdir}" \
                20 "${tok}" "--sc-scenario a" "${QUERIES_DIR}/cache/medium_warm.txt" \
                --no-warmup 2>&1 | tail -2
            ;;
    esac
}

# ── TMA — toplev -l2, all pods PARALLEL per path ─────────────────────────────
# Step A: start toplev on ALL local pods simultaneously, run queries once, wait.
# Step B: start perf slots on ALL pods simultaneously, run queries once, SIGINT all.
# PMU multiplexing across PIDs is corrected by toplev via time_running/time_enabled.

run_tma() {
    # Args: $1 = output token budget (default 64), $2 = tier label (default tok64).
    # Report-read files are suffixed with the tier (e.g. _tok192) for every tier
    # except the tok64 baseline, which keeps the unsuffixed names for back-compat.
    local tok="${1:-64}"
    local tier_label="${2:-tok64}"
    local sfx=""
    [[ "${tier_label}" != "tok64" ]] && sfx="_${tier_label}"
    log "=== TMA (toplev -l2, all pods parallel) [${tier_label}, tok=${tok}] ==="
    local tma_dir="${RUN_DIR}/tma"
    mkdir -p "${tma_dir}"

    local fastapi_pod="${POD_MAP[fastapi]:-}"
    local TOPLEV_HOST="/tmp/pmu-tools-benchmark"
    if [[ -n "${fastapi_pod}" && ! -f "${TOPLEV_HOST}/toplev.py" ]]; then
        log "Copying pmu-tools from fastapi container to host..."
        mkdir -p "${TOPLEV_HOST}"
        kubectl cp -n "${NAMESPACE}" "${fastapi_pod}:/opt/pmu-tools" "${TOPLEV_HOST}" 2>/dev/null || true
        if [[ ! -f "${TOPLEV_HOST}/toplev.py" ]]; then
            warn "pmu-tools copy failed — TMA will be skipped"
            return
        fi
        ok "pmu-tools ready at ${TOPLEV_HOST}"
    fi
    [[ -f "${TOPLEV_HOST}/toplev.py" ]] || { warn "pmu-tools not available — skipping TMA"; return; }

    # TMA = RAG path ONLY (2026-06-16). RAG is the full pipeline → every pod does real
    # work, so it's the only path where the per-pod CPU-microarch characterization (the
    # micro_summary three-regime figure) is meaningful.
    #   - sc_a DROPPED: measured queries are cache HITS → milvus/seaweed/mongodb idle.
    #   - llm_direct DROPPED: no retrieval → milvus/seaweed/mongodb idle too, and fastapi
    #     just forwards then blocks on the GPU (low signal). The fastapi embed-vs-forward
    #     contrast is already captured in the per-cell IPC data (rag ~0.7 vs llm ~1.3).
    # Re-add via TMA_PATHS (e.g. TMA_PATHS="rag llm_direct sc_a") if ever needed.
    for path_mode in ${TMA_PATHS:-rag}; do   # was: rag llm_direct sc_a
        local bucket="medium" query_file warmup_file="" sc_arg=""
        case "${path_mode}" in
            rag)        query_file="${QUERIES_DIR}/rag/medium.txt" ;;
            llm_direct) query_file="${QUERIES_DIR}/llm_direct/medium.txt" ;;
            sc_a)
                query_file="${QUERIES_DIR}/cache/medium_measure.txt"
                warmup_file="${QUERIES_DIR}/cache/medium_warm.txt"
                sc_arg="--sc-scenario a"
                ;;
        esac
        [ -f "${query_file}" ] || continue

        # Measured queries per TMA drive. n=20 (was 5) → ~3-4x longer steady-state
        # window so the topdown PERF_METRICS group programs reliably even with
        # fastapi's churning thread pool (count=5 gave racy ~9s tok64 windows that
        # lost fastapi's slots capture). Override with TMA_COUNT.
        local tma_count="${TMA_COUNT:-20}"
        local tma_qr_mode
        case "${path_mode}" in
            sc_a) tma_qr_mode="cache_a" ;;
            *)    tma_qr_mode="${path_mode}" ;;
        esac

        # Pre-warmup BEFORE the toplev window (outside perf) so the measured window
        # only ever sees steady state — same philosophy as run_cell_hw. Defaults to
        # the hw-cell PRERUN_WARMUP; override with TMA_WARMUP.
        # NB: non-SC main() ignores --warmup, so RAG/LLM-direct previously got NO
        # warmup at all — this adds an explicit throwaway warmup pass for them.
        local tma_no_warmup_flag=""
        local tma_warmup="${TMA_WARMUP:-20}"
        if [[ -n "${warmup_file}" ]]; then
            log "  SC pre-warmup for TMA (${tma_warmup} queries, cache population, outside perf window)..."
            BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
            python3 "${SCRIPT_DIR}/query_runner.py" \
                --mode "${tma_qr_mode}" \
                --queries "${query_file}" \
                --warmup-queries "${warmup_file}" \
                --size-bucket "${bucket}" \
                --count "${tma_count}" --warmup "${tma_warmup}" \
                --max-tokens "${tok}" \
                        --out-dir "${tma_dir}" \
                ${sc_arg} \
                --warmup-only \
                2>&1 | tail -3
            tma_no_warmup_flag="--no-warmup"
        else
            # RAG / LLM-direct: warm CPU caches, Python allocator, HTTP pool, Milvus
            # HNSW by running throwaway queries outside the toplev window. The CSV
            # lands in tma_dir and is never read (the report only reads tma_*.txt).
            log "  TMA pre-warmup (${tma_warmup} queries, outside perf window)..."
            BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
            python3 "${SCRIPT_DIR}/query_runner.py" \
                --mode "${tma_qr_mode}" \
                --queries "${query_file}" \
                --size-bucket "${bucket}" \
                --count "${tma_warmup}" \
                --max-tokens "${tok}" \
                        --out-dir "${tma_dir}" \
                2>&1 | tail -2
        fi

        # Build combined pid map: local PIDs and remote PIDs (vLLM on GPU node)
        declare -A _tma_pid_map
        for key in "${!HOST_PID_MAP[@]}"; do
            _tma_pid_map[$key]="${HOST_PID_MAP[$key]}"
        done
        for key in "${!REMOTE_HOST_PID_MAP[@]}"; do
            if kubectl get pod -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" \
                   --no-headers 2>/dev/null | grep -q "Running"; then
                _tma_pid_map[$key]="remote:${REMOTE_HOST_PID_MAP[$key]}"
            fi
        done

        log "  TMA parallel: path=${path_mode} pods=${#_tma_pid_map[@]}"

        # sleep window: all toplevs share this window — they start and expire together.
        # Must COVER the full warmup+measure query duration, which grows with the token
        # budget (more decode steps). Scaled by tier; override with TMA_SEC_PER_QUERY.
        # Tuned 2026-06-16 against measured per-query e2e (rag_medium: tok64 1.6s /
        # tok192 4.6s / tok320 7.5s) at n=20 → 25-query drive ≈ 40/115/188s. These
        # sec/query give windows 160/200/260s = 4.0x/1.7x/1.4x headroom (was 8/12/16
        # → 2x everywhere; trims ~14 min of idle tail off the TMA rerun). tok64 is
        # base-dominated (+60), so 5 is a sensible floor.
        local tma_sec_per_q="${TMA_SEC_PER_QUERY:-}"
        if [[ -z "${tma_sec_per_q}" ]]; then
            case "${tok}" in
                64)  tma_sec_per_q=5 ;;
                192) tma_sec_per_q=7 ;;
                320) tma_sec_per_q=10 ;;
                *)   tma_sec_per_q=10 ;;
            esac
        fi
        local tma_sleep=$(( tma_count * tma_sec_per_q + 60 ))

        # ── Step A: toplev -l2, start all local pods simultaneously ─────────
        declare -A _toplev_bg_pids
        for key in "${!_tma_pid_map[@]}"; do
            local raw_pid="${_tma_pid_map[$key]}"
            if [[ "${raw_pid}" == remote:* ]]; then
                warn "  TMA toplev skipped for remote pod ${key} (pmu-tools not on perf-agent)"
                echo "SKIPPED: remote pod, pmu-tools not available on perf-agent" \
                    > "${tma_dir}/tma_toplev_${path_mode}_${key}${sfx}.txt"
                continue
            fi
            local host_pid="${raw_pid}"
            [[ -z "${host_pid}" ]] && continue
            log "    toplev start: pod=${key} pid=${host_pid} sleep=${tma_sleep}s"
            ${PERF_SUDO} python3 "${TOPLEV_HOST}/toplev.py" -l2 --no-desc \
                --nodes "Retiring,Frontend_Bound,Bad_Speculation,Backend_Bound,Memory_Bound,Core_Bound,L1_Bound,L2_Bound,L3_Bound,DRAM_Bound,Store_Bound,Ports_Utilization,Divider,MUX" \
                -p "${host_pid}" \
                -- sleep "${tma_sleep}" \
                > "${tma_dir}/tma_toplev_${path_mode}_${key}${sfx}.txt" 2>&1 &
            _toplev_bg_pids[$key]=$!
        done

        sleep 1  # let all toplevs attach before queries start

        # Run queries ONCE — all pods measured simultaneously during this window
        BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
        python3 "${SCRIPT_DIR}/query_runner.py" \
            --mode "${tma_qr_mode}" \
            --queries "${query_file}" \
            ${warmup_file:+--warmup-queries "${warmup_file}"} \
            --size-bucket "${bucket}" \
            --count "${tma_count}" --warmup 5 \
            --max-tokens "${tok}" \
                --out-dir "${tma_dir}" \
            ${sc_arg} \
            ${tma_no_warmup_flag} \
            ${STREAM_FLAG} \
            2>&1 | tail -3

        log "  Waiting for all toplev sleep windows (${tma_sleep}s from start)..."
        for key in "${!_toplev_bg_pids[@]}"; do
            wait "${_toplev_bg_pids[$key]}" 2>/dev/null || true
            ok "    toplev done: pod=${key}"
        done
        unset _toplev_bg_pids

        # Copy fastapi result as the primary TMA output (tier-suffixed for the report)
        cp "${tma_dir}/tma_toplev_${path_mode}_fastapi${sfx}.txt" \
           "${tma_dir}/tma_toplev_${path_mode}${sfx}.txt" 2>/dev/null || true

        # ── Step B: perf stat slots, start all pods simultaneously ──────────
        declare -A _local_slots_pids   # key → bg PID of local perf stat
        declare -A _remote_slots_pids  # key → bg PID of kubectl exec (remote)

        for key in "${!_tma_pid_map[@]}"; do
            local raw_pid="${_tma_pid_map[$key]}"
            if [[ "${raw_pid}" == remote:* ]]; then
                local host_pid="${raw_pid#remote:}"
                local remote_slots_out="/tmp/tma_slots_${path_mode}_${key}.txt"
                local remote_pid_file="/tmp/tma_slots_${path_mode}_${key}.pid"
                # Run perf stat on perf-agent; store PID to file for clean SIGINT later
                kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- bash -c \
                    "perf stat -p ${host_pid} -e slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound \
                     > ${remote_slots_out} 2>&1 & echo \$! > ${remote_pid_file}; wait" &
                _remote_slots_pids[$key]=$!
            else
                local host_pid="${raw_pid}"
                [[ -z "${host_pid}" ]] && continue
                ${PERF_SUDO} "${PERF_HOST_BIN}" stat -p "${host_pid}" \
                    -e "slots,topdown-retiring,topdown-fe-bound,topdown-bad-spec,topdown-be-bound" \
                    > "${tma_dir}/tma_slots_${path_mode}_${key}${sfx}.txt" 2>&1 &
                _local_slots_pids[$key]=$!
                log "    slots start: pod=${key} pid=${host_pid}"
            fi
        done

        # Let all perf stat collectors finish attaching before queries fire.
        # FastAPI has ~200+ threads (uvicorn + asyncio + BGE thread pool); without
        # this pause, perf races queries and can lose entire pods' data with
        # "Ignored open failure for pid NNN" — matches the sleep 1 in Step A.
        sleep 1

        # Run queries ONCE — all slots collectors active simultaneously
        BENCHMARK_URL="${BENCHMARK_URL}" BENCHMARK_MODEL="${BENCHMARK_MODEL}" \
        python3 "${SCRIPT_DIR}/query_runner.py" \
            --mode "${tma_qr_mode}" \
            --queries "${query_file}" \
            ${warmup_file:+--warmup-queries "${warmup_file}"} \
            --size-bucket "${bucket}" \
            --count "${tma_count}" --warmup 5 \
            --max-tokens "${tok}" \
                --out-dir "${tma_dir}" \
            ${sc_arg} \
            ${tma_no_warmup_flag} \
            ${STREAM_FLAG} \
            2>&1 | tail -3

        # SIGINT all local perf stat collectors simultaneously
        for key in "${!_local_slots_pids[@]}"; do
            kill -INT "${_local_slots_pids[$key]}" 2>/dev/null || true
        done

        # Stop all remote perf stat collectors via perf-agent
        for key in "${!_remote_slots_pids[@]}"; do
            local remote_pid_file="/tmp/tma_slots_${path_mode}_${key}.pid"
            kubectl exec -n "${PERF_AGENT_NS}" "${PERF_AGENT_POD}" -- bash -c \
                "kill -INT \$(cat ${remote_pid_file}) 2>/dev/null; sleep 2; \
                 kill -KILL \$(cat ${remote_pid_file}) 2>/dev/null; true" 2>/dev/null || true
        done

        sleep 3  # allow perf stat to flush output after SIGINT

        # Wait for all slots collectors and clean up
        for key in "${!_local_slots_pids[@]}"; do
            kill -KILL "${_local_slots_pids[$key]}" 2>/dev/null || true
            wait "${_local_slots_pids[$key]}" 2>/dev/null || true
            ok "    slots done: pod=${key}"
        done
        for key in "${!_remote_slots_pids[@]}"; do
            wait "${_remote_slots_pids[$key]}" 2>/dev/null || true
            kubectl cp "${PERF_AGENT_NS}/${PERF_AGENT_POD}:/tmp/tma_slots_${path_mode}_${key}.txt" \
                "${tma_dir}/tma_slots_${path_mode}_${key}${sfx}.txt" 2>/dev/null || true
            ok "    slots done (remote): pod=${key}"
        done
        unset _local_slots_pids _remote_slots_pids

        # Copy fastapi result as the primary slots output (tier-suffixed for the report)
        cp "${tma_dir}/tma_slots_${path_mode}_fastapi${sfx}.txt" \
           "${tma_dir}/tma_slots_${path_mode}${sfx}.txt" 2>/dev/null || true

        ok "  TMA ${path_mode} complete → ${tma_dir}/"
        unset _tma_pid_map
    done
}

# ── Main ─────────────────────────────────────────────────────────────────────

main() {
    log "Benchmark run: ${TIMESTAMP}"
    log "Output: ${RUN_DIR}"
    log "URL: ${BENCHMARK_URL}  Model: ${BENCHMARK_MODEL}"
    log "Counts: tok64=${COUNT_64}  tok192=${COUNT_192}  tok320=${COUNT_320}  tier_warmup=${TIER_WARMUP_COUNT}"

    # Allow full PMU access — required on EKS (Amazon Linux default paranoia=1 blocks some events).
    # Works silently on minikube too. Requires privileged pod (hostPID: true, runAsUser: 0).
    if echo -1 > /proc/sys/kernel/perf_event_paranoid 2>/dev/null; then
        echo 0 > /proc/sys/kernel/nmi_watchdog 2>/dev/null || true
        ok "perf_event_paranoid=-1, nmi_watchdog=0"
    else
        warn "Could not set perf_event_paranoid — some PMU events may be blocked"
    fi

    discover_all_pods
    check_health

    local node
    node=$(get_fastapi_node)
    local fastapi_pod="${POD_MAP[fastapi]:-}"
    local instance="unknown"
    local seaweed_force="0"
    if [[ "${node}" != "unknown" && -n "${node}" ]]; then
        instance=$(kubectl get node "${node}" \
            -o jsonpath='{.metadata.labels.node\.kubernetes\.io/instance-type}' 2>/dev/null || echo unknown)
    fi
    if [[ -n "${fastapi_pod}" ]]; then
        seaweed_force=$(kubectl exec -n "${NAMESPACE}" "${fastapi_pod}" -- \
            sh -c 'echo ${RAG_FORCE_SEAWEED_FETCH:-0}' 2>/dev/null || echo 0)
    fi

    local pods_json
    pods_json=$(python3 -c "import json,sys; keys=sys.stdin.read().split(); print(json.dumps(keys))" \
        <<< "${!POD_MAP[*]}")

    local _stream_bool
    _stream_bool=$([[ -n "${STREAM_FLAG}" ]] && echo true || echo false)

    cat > "${RUN_DIR}/run_info.json" <<EOF
{
  "timestamp": "${TIMESTAMP}",
  "url": "${BENCHMARK_URL}",
  "model": "${BENCHMARK_MODEL}",
  "count_64": ${COUNT_64},
  "count_192": ${COUNT_192},
  "count_320": ${COUNT_320},
  "warmup": ${WARMUP},
  "tier_warmup_count": ${TIER_WARMUP_COUNT},
  "concurrency": ${CONCURRENCY},
  "node": "${node}",
  "instance": "${instance}",
  "seaweed_force": "${seaweed_force}",
  "stream": ${_stream_bool},
  "pods_discovered": ${pods_json}
}
EOF

    # ── System metadata snapshot ──────────────────────────────────────────────
    local meta_dir="${RUN_DIR}/metadata"
    mkdir -p "${meta_dir}"
    lscpu                              > "${meta_dir}/lscpu.txt"           2>/dev/null || true
    uname -a                           > "${meta_dir}/uname.txt"           2>/dev/null || true
    "${PERF_HOST_BIN}" --version       > "${meta_dir}/perf_version.txt"   2>/dev/null || true
    "${PERF_HOST_BIN}" list           >> "${meta_dir}/perf_version.txt"   2>/dev/null || true
    command -v nvidia-smi &>/dev/null && \
        nvidia-smi -q                  > "${meta_dir}/nvidia_smi.txt"     2>/dev/null || true
    kubectl get nodes -o wide          > "${meta_dir}/nodes.txt"          2>/dev/null || true
    kubectl get pods -n "${NAMESPACE}" -o wide  > "${meta_dir}/pods_llm_service.txt"  2>/dev/null || true
    kubectl get pods -n "${VLLM_NAMESPACE}" -o wide > "${meta_dir}/pods_vllm.txt"    2>/dev/null || true
    # Per-pod container images
    kubectl get pods -n "${NAMESPACE}" \
        -o jsonpath='{range .items[*]}{.metadata.name}{": "}{range .spec.containers[*]}{.image}{" "}{end}{"\n"}{end}' \
        > "${meta_dir}/pod_images.txt" 2>/dev/null || true
    # Save host PID map for reproducibility
    python3 -c "
import json
host_pids = {}
$(for k in "${!HOST_PID_MAP[@]}"; do
    echo "host_pids['${k}'] = '${HOST_PID_MAP[$k]}'";
done)
print(json.dumps(host_pids, indent=2))
" > "${meta_dir}/host_pids.json" 2>/dev/null || true
    ok "System metadata saved → ${meta_dir}/"

    local run_all=0
    [[ "${CELLS}" == "all" ]] && run_all=1
    wants() { [[ "${run_all}" == "1" ]] || [[ " ${CELLS} " == *" $1 "* ]]; }

    local tok_tiers="64 192 320"
    [ -n "${TOKEN_OVERRIDE}" ] && tok_tiers="${TOKEN_OVERRIDE}"

    wants calibration && run_calibration

    # ── 64-token tier ──────────────────────────────────────────────────────────
    if [[ " ${tok_tiers} " == *" 64 "* ]]; then
        mkdir -p "${RUN_DIR}/tok64"

        { wants rag_short || wants rag_medium || wants rag_long || wants rag_very_long || wants rag_pure_fetch; } \
            && warmup_tier_path "rag" 64
        for bucket in short medium long very_long; do
            wants "rag_${bucket}" && run_cell_hw "rag" "${bucket}" 64
        done
        # rag_pure_fetch disabled for this re-run (trim to cut runtime)
        # wants "rag_pure_fetch" && run_cell_hw "rag_pure_fetch" "short" 64

        { wants sc_a_short || wants sc_a_medium || wants sc_b_short || wants sc_b_medium; } \
            && warmup_tier_path "sc" 64
        for bucket in short medium; do
            wants "sc_a_${bucket}" && run_cell_hw "sc_a" "${bucket}" 64
            wants "sc_b_${bucket}" && run_cell_hw "sc_b" "${bucket}" 64
        done

        { wants llm_short || wants llm_medium || wants llm_long || wants llm_very_long; } \
            && warmup_tier_path "llm_direct" 64
        for bucket in short medium long very_long; do
            wants "llm_${bucket}" && run_cell_hw "llm_direct" "${bucket}" 64
        done

        # for bucket in short medium long very_long; do
        #     wants "bge_${bucket}" && run_cell_hw "bge_isolated" "${bucket}" 64
        # done

        # for bucket in short medium long very_long; do
        #     wants "hnsw_${bucket}" && run_cell_hw "hnsw_isolated" "${bucket}" 64
        # done

        wants tma && { run_tma 64 tok64 || warn "TMA tok64 failed — continuing to tok192"; }

        if wants pass3_sweep; then
            log "=== PASS-4 IMC BANDWIDTH SWEEP (tok64) ==="
            for bucket in short medium long very_long; do
                run_cell_pass3_only "rag" "${bucket}" 64
            done
            # rag_pure_fetch disabled for this re-run (trim to cut runtime)
            # run_cell_pass3_only "rag_pure_fetch" "short" 64
            for bucket in short medium; do
                run_cell_pass3_only "sc_a" "${bucket}" 64
                run_cell_pass3_only "sc_b" "${bucket}" 64
            done
            for bucket in short medium long very_long; do
                run_cell_pass3_only "llm_direct" "${bucket}" 64
            done
            ok "Pass-4 sweep complete"
        fi
    fi

    # ── 192-token tier ─────────────────────────────────────────────────────────
    if [[ " ${tok_tiers} " == *" 192 "* ]]; then
        mkdir -p "${RUN_DIR}/tok192"
        log "=== TOKEN TIER: 192 ==="

        { wants rag_short || wants rag_medium || wants rag_long || wants rag_very_long || wants rag_pure_fetch; } \
            && warmup_tier_path "rag" 192
        for bucket in short medium long very_long; do
            wants "rag_${bucket}" && run_cell_hw "rag" "${bucket}" 192 "${COUNT_192}"
        done
        # rag_pure_fetch disabled for this re-run (trim to cut runtime)
        # wants "rag_pure_fetch" && run_cell_hw "rag_pure_fetch" "short" 192 "${COUNT_192}"

        { wants sc_a_short || wants sc_a_medium || wants sc_b_short || wants sc_b_medium; } \
            && warmup_tier_path "sc" 192
        for bucket in short medium; do
            wants "sc_a_${bucket}" && run_cell_hw "sc_a" "${bucket}" 192 "${COUNT_192}"
            wants "sc_b_${bucket}" && run_cell_hw "sc_b" "${bucket}" 192 "${COUNT_192}"
        done

        { wants llm_short || wants llm_medium || wants llm_long || wants llm_very_long; } \
            && warmup_tier_path "llm_direct" 192
        for bucket in short medium long very_long; do
            wants "llm_${bucket}" && run_cell_hw "llm_direct" "${bucket}" 192 "${COUNT_192}"
        done

        wants tma && { run_tma 192 tok192 || warn "TMA tok192 failed — continuing to tok320"; }
    fi

    # ── 320-token tier ─────────────────────────────────────────────────────────
    if [[ " ${tok_tiers} " == *" 320 "* ]]; then
        mkdir -p "${RUN_DIR}/tok320"
        log "=== TOKEN TIER: 320 ==="

        { wants rag_short || wants rag_medium || wants rag_long || wants rag_very_long || wants rag_pure_fetch; } \
            && warmup_tier_path "rag" 320
        for bucket in short medium long very_long; do
            wants "rag_${bucket}" && run_cell_hw "rag" "${bucket}" 320 "${COUNT_320}"
        done
        # rag_pure_fetch disabled for this re-run (trim to cut runtime)
        # wants "rag_pure_fetch" && run_cell_hw "rag_pure_fetch" "short" 320 "${COUNT_320}"

        { wants sc_a_short || wants sc_a_medium || wants sc_b_short || wants sc_b_medium; } \
            && warmup_tier_path "sc" 320
        for bucket in short medium; do
            wants "sc_a_${bucket}" && run_cell_hw "sc_a" "${bucket}" 320 "${COUNT_320}"
            wants "sc_b_${bucket}" && run_cell_hw "sc_b" "${bucket}" 320 "${COUNT_320}"
        done

        { wants llm_short || wants llm_medium || wants llm_long || wants llm_very_long; } \
            && warmup_tier_path "llm_direct" 320
        for bucket in short medium long very_long; do
            wants "llm_${bucket}" && run_cell_hw "llm_direct" "${bucket}" 320 "${COUNT_320}"
        done

        wants tma && { run_tma 320 tok320 || warn "TMA tok320 failed"; }
    fi

    log "=== ALL COMPLETE ==="
    log "Results: ${RUN_DIR}"
    echo ""
    echo "  To view results:"
    echo "  python3 scripts/results_cli.py --results-dir ${RUN_DIR}"
    echo ""
}

main
