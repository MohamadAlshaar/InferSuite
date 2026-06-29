#!/usr/bin/env bash
# =============================================================================
# run_nsys.sh — Nsight Systems profiling for vLLM (minikube local mode)
#
# nsys cannot attach to running processes — it must be the process launcher.
# This script patches the vLLM Deployment to start vLLM under nsys, using
# hostPath volumes so the nsys binary and trace output survive pod restarts.
#
# Usage:
#   ./scripts/run_nsys.sh [--workload W] [--size S] [--count N]
#   ./scripts/run_nsys.sh --restore   # restore original deployment
#
# Options:
#   --workload  rag | llm_direct | cache (default: llm_direct)
#   --size      short | medium | long | very_long (default: medium)
#   --count     number of queries (default: 20)
#   --url       service URL (default: $BENCHMARK_URL or http://localhost:8080)
#   --restore   undo the deployment patch and restart the pod clean
#
# Workflow:
#   setup   → docker cp nsys to minikube node → patch Deployment (one-time)
#   profile → fire queries → SIGINT to nsys PID 1 → trace written to hostPath
#   extract → docker cp trace from minikube → nsys stats
#   restore → re-apply saved original Deployment
#
# Output: nsys_traces/<timestamp>_<workload>_<size>/
#   trace.nsys-rep         raw trace (open with: nsys-ui trace.nsys-rep)
#   kernel_summary.txt     top kernels by total GPU time
#   cuda_api_summary.txt   CUDA API overhead
#   queries.csv            per-request latency
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
QUERIES_DIR="${KERNEL_ROOT}/benchmark_queries"
TRACES_DIR="${KERNEL_ROOT}/nsys_traces"

# ── Kubernetes config ─────────────────────────────────────────────────────────
VLLM_DEPLOY="ms-local-llm-d-modelservice-decode"
VLLM_NS="llm-d-local"
VLLM_CONTAINER="vllm"

# ── Paths ─────────────────────────────────────────────────────────────────────
NSYS_HOST_DIR="/opt/nvidia/nsight-systems/2025.3.2/target-linux-x64"
# /var/ is used instead of /tmp/ — minikube mounts /tmp with noexec so binaries can't run there.
# MINIKUBE_NSYS_PARENT is the parent dir; the binary lives in target-linux-x64/ subdir inside it.
# nsys requires "target-linux-x64" to appear in the path it executes from.
MINIKUBE_NSYS_PARENT="/var/nsys_install"
MINIKUBE_NSYS_DIR="${MINIKUBE_NSYS_PARENT}/target-linux-x64"
MINIKUBE_TRACES_DIR="/var/nsys_traces"
SAVED_DEPLOY="${KERNEL_ROOT}/.nsys_original_deploy.yaml"

# ── Defaults ──────────────────────────────────────────────────────────────────
WORKLOAD="llm_direct"
SIZE="medium"
COUNT=20
URL="${BENCHMARK_URL:-http://localhost:8080}"
MODEL="${BENCHMARK_MODEL:-qwen2.5-0.5b}"
DO_RESTORE=false

log()  { printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"; }
ok()   { printf '  ✓ %s\n' "$*"; }
warn() { printf '  ⚠ %s\n' "$*" >&2; }
die()  { printf '[ERROR] %s\n' "$*" >&2; exit 1; }

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --workload) WORKLOAD="$2"; shift 2 ;;
        --size)     SIZE="$2";     shift 2 ;;
        --count)    COUNT="$2";    shift 2 ;;
        --url)      URL="$2";      shift 2 ;;
        --restore)  DO_RESTORE=true; shift ;;
        *) die "Unknown argument: $1" ;;
    esac
done

# ── Restore function ──────────────────────────────────────────────────────────
restore_deployment() {
    [[ -f "${SAVED_DEPLOY}" ]] || die "No saved deployment found at ${SAVED_DEPLOY}"
    log "Restoring original vLLM deployment..."
    # Strip stale metadata fields that cause resourceVersion conflicts with kubectl apply
    python3 - "${SAVED_DEPLOY}" <<'PYEOF'
import yaml, subprocess, sys
with open(sys.argv[1]) as f:
    doc = yaml.safe_load(f)
for field in ('resourceVersion','generation','managedFields','uid','creationTimestamp'):
    doc.get('metadata',{}).pop(field, None)
doc.pop('status', None)
r = subprocess.run(['kubectl','replace','-f','-'], input=yaml.dump(doc), text=True, capture_output=True)
print(r.stdout or r.stderr)
if r.returncode != 0:
    sys.exit(r.returncode)
PYEOF
    kubectl rollout status deployment/"${VLLM_DEPLOY}" -n "${VLLM_NS}" --timeout=300s
    rm -f "${SAVED_DEPLOY}"
    ok "Deployment restored. nsys patch removed."
}

if [[ "${DO_RESTORE}" == "true" ]]; then
    restore_deployment
    exit 0
fi

# ── Validate ──────────────────────────────────────────────────────────────────
command -v nsys   &>/dev/null || die "nsys not found on host"
command -v docker &>/dev/null || die "docker not found (needed for minikube hostPath setup)"
command -v kubectl &>/dev/null || die "kubectl not found"

QUERY_FILE="${QUERIES_DIR}/${WORKLOAD}/${SIZE}.txt"
[[ -f "${QUERY_FILE}" ]] || die "Query file not found: ${QUERY_FILE}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOCAL_OUT="${TRACES_DIR}/${TIMESTAMP}_${WORKLOAD}_${SIZE}"
mkdir -p "${LOCAL_OUT}"

# ── Map workload → query runner mode ─────────────────────────────────────────
case "${WORKLOAD}" in
    llm_direct) QR_MODE="llm_direct" ;;
    rag)        QR_MODE="rag"        ;;
    cache)      QR_MODE="sc_b"       ;;
    *) die "Unknown workload: ${WORKLOAD}. Use: llm_direct | rag | cache" ;;
esac

case "${SIZE}" in
    short)     MAX_TOKENS=64  ;;
    medium)    MAX_TOKENS=192 ;;
    long)      MAX_TOKENS=512 ;;
    very_long) MAX_TOKENS=512 ;;
    *)         MAX_TOKENS=192 ;;
esac

log "nsys profiling session"
ok "workload : ${WORKLOAD} / ${SIZE} (${COUNT} queries)"
ok "service  : ${URL}"
ok "output   : ${LOCAL_OUT}"

# ── Step 1: Copy nsys to minikube node (idempotent) ──────────────────────────
log "Step 1 — copying nsys to minikube node..."

NSYS_ALREADY_THERE="$(docker exec minikube test -f "${MINIKUBE_NSYS_DIR}/nsys" && echo yes || echo no)"
if [[ "${NSYS_ALREADY_THERE}" == "no" ]]; then
    docker exec minikube mkdir -p "${MINIKUBE_NSYS_DIR}" "${MINIKUBE_TRACES_DIR}"
    # docker cp copies into the minikube container; source is host target-linux-x64 contents
    docker cp "${NSYS_HOST_DIR}/." "minikube:${MINIKUBE_NSYS_DIR}/"
    ok "nsys copied to minikube:${MINIKUBE_NSYS_DIR}"
else
    ok "nsys already on minikube node (skipping copy)"
fi
docker exec minikube mkdir -p "${MINIKUBE_TRACES_DIR}"
docker exec minikube chmod +x "${MINIKUBE_NSYS_DIR}/nsys"
# Verify the binary exists and is executable
docker exec minikube test -x "${MINIKUBE_NSYS_DIR}/nsys" \
    || die "nsys binary not executable at ${MINIKUBE_NSYS_DIR}/nsys — check copy step"

# ── Step 2: Patch the deployment (idempotent) ─────────────────────────────────
log "Step 2 — patching vLLM deployment to run under nsys..."

ALREADY_PATCHED="$(kubectl get deployment "${VLLM_DEPLOY}" -n "${VLLM_NS}" \
    -o jsonpath='{.spec.template.spec.containers[0].args[0]}' 2>/dev/null \
    | grep -c 'nsys_install/target-linux-x64/nsys' || true)"

if [[ "${ALREADY_PATCHED}" -ge 1 ]]; then
    ok "Deployment already patched (skipping)"
else
    # Save original before patching
    kubectl get deployment "${VLLM_DEPLOY}" -n "${VLLM_NS}" -o yaml > "${SAVED_DEPLOY}"
    ok "Original deployment saved to ${SAVED_DEPLOY}"

    # Write patch script to temp file — avoids heredoc/pipe stdin conflict
    _PATCH_SCRIPT="$(mktemp /tmp/nsys_patch_XXXXX.py)"
    cat > "${_PATCH_SCRIPT}" << 'PYEOF'
import json, sys

data = json.load(sys.stdin)
nsys_dir   = sys.argv[1]
traces_dir = sys.argv[2]

spec = data["spec"]["template"]["spec"]

spec.setdefault("volumes", [])
spec["volumes"] += [
    # Parent dir mounted so pod path is /nsys_install/target-linux-x64/nsys
    # nsys requires "target-linux-x64" to appear in its own execution path
    {"name": "nsys-binary", "hostPath": {"path": nsys_dir,   "type": "Directory"}},
    {"name": "nsys-traces", "hostPath": {"path": traces_dir, "type": "DirectoryOrCreate"}},
]

for c in spec["containers"]:
    if c["name"] == "vllm":
        c.setdefault("volumeMounts", [])
        c["volumeMounts"] += [
            {"name": "nsys-binary", "mountPath": "/nsys_install", "readOnly": True},
            {"name": "nsys-traces", "mountPath": "/traces"},
        ]
        # CUDA profiling via CUPTI requires root + privileged — RmProfilingAdminOnly=1
        # blocks non-root access to GPU performance counters on this machine
        c["securityContext"] = {"privileged": True, "runAsUser": 0}
        old_args = c["args"][0]
        new_args = old_args.replace(
            "exec vllm serve",
            "exec /nsys_install/target-linux-x64/nsys profile --trace=cuda,nvtx,osrt,python-gil --sample=cpu --cpuctxsw=process-tree --delay=120 --duration=60 --kill=none --force-overwrite=true -o /traces/trace vllm serve"
        )
        if new_args == old_args:
            sys.stderr.write("[ERROR] Could not find 'exec vllm serve' in startup args\n")
            sys.exit(1)
        c["args"][0] = new_args
        break

data.get("metadata", {}).pop("managedFields", None)
data.get("metadata", {}).pop("resourceVersion", None)
data.get("metadata", {}).pop("generation", None)

print(json.dumps(data))
PYEOF

    kubectl get deployment "${VLLM_DEPLOY}" -n "${VLLM_NS}" -o json \
        | python3 "${_PATCH_SCRIPT}" "${MINIKUBE_NSYS_PARENT}" "${MINIKUBE_TRACES_DIR}" \
        | kubectl apply -f -
    rm -f "${_PATCH_SCRIPT}"

    ok "Deployment patched — pod will restart under nsys"

    log "Waiting for pod to be ready (vLLM startup takes ~60s)..."
    kubectl rollout status deployment/"${VLLM_DEPLOY}" -n "${VLLM_NS}" --timeout=300s
    ok "Pod ready"
fi

# ── Step 3: Verify vLLM is fully ready (model loaded, not just HTTP up) ──────
# Poll FastAPI's health endpoint until generation_backend_usable=true.
# This is more reliable than checking /health directly — it confirms vLLM has
# finished loading the model and is actually serving, not just that the HTTP
# server is accepting connections.
log "Step 3 — waiting for vLLM model to finish loading (up to 3 min)..."
for i in $(seq 1 60); do
    ready="$(curl -sf "${URL}/health" 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('generation_backend_usable',False))" \
        2>/dev/null || echo False)"
    if [[ "${ready}" == "True" ]]; then
        ok "vLLM model loaded and serving at ${URL}"
        break
    fi
    [[ $i -eq 60 ]] && die "vLLM not ready after 5 minutes"
    sleep 5
done

# Wait until nsys capture window opens (--delay=120 from pod start).
# vLLM typically loads in ~75s; nsys starts capturing at 120s from pod start.
# We wait until T=115s so queries fire right as the capture window opens.
POD_START_ISO="$(kubectl get pods -n "${VLLM_NS}" \
    -o jsonpath='{.items[?(@.status.phase=="Running")].status.startTime}' \
    | tr ' ' '\n' | grep -v '^$' | tail -1)"
if [[ -n "${POD_START_ISO}" ]]; then
    POD_START_UNIX="$(date -d "${POD_START_ISO}" +%s 2>/dev/null || echo 0)"
    ELAPSED=$(( $(date +%s) - POD_START_UNIX ))
    WAIT_FOR_CAPTURE=$(( 115 - ELAPSED ))
    if [[ ${WAIT_FOR_CAPTURE} -gt 0 ]]; then
        log "Waiting ${WAIT_FOR_CAPTURE}s for nsys capture window to open (delay=120s from pod start)..."
        sleep "${WAIT_FOR_CAPTURE}"
    fi
fi

# ── Step 4: Fire queries ──────────────────────────────────────────────────────
log "Step 4 — firing ${COUNT} queries (${QR_MODE} / ${SIZE})..."

BENCHMARK_MODEL="${MODEL}" python3 "${SCRIPT_DIR}/query_runner.py" \
    --mode        "${QR_MODE}" \
    --queries     "${QUERY_FILE}" \
    --size-bucket "${SIZE}" \
    --count       "${COUNT}" \
    --warmup      0 \
    --max-tokens  "${MAX_TOKENS}" \
    --url         "${URL}" \
    --out-dir     "${LOCAL_OUT}" \
    2>&1 | tee "${LOCAL_OUT}/queries.log"

# ── Step 5: Wait for nsys to auto-stop ───────────────────────────────────────
# nsys uses --duration=60 --kill=none: it stops capturing after 60s and writes
# the trace automatically. vLLM keeps running. No SIGINT needed.
log "Step 5 — waiting for nsys to finish capture window and write trace (~60s)..."
sleep 70

# ── Step 6: Wait for nsys to write trace, then extract ───────────────────────
log "Step 6 — waiting for nsys to write trace (up to 2 min)..."

TRACE_REP="${MINIKUBE_TRACES_DIR}/trace.nsys-rep"
TRACE_QDSTRM="${MINIKUBE_TRACES_DIR}/trace.qdstrm"
TRACE_FOUND=""

for i in $(seq 1 24); do
    if docker exec minikube test -f "${TRACE_REP}" 2>/dev/null; then
        TRACE_FOUND="${TRACE_REP}"
        break
    fi
    if docker exec minikube test -f "${TRACE_QDSTRM}" 2>/dev/null; then
        # qdstrm exists — check it was modified recently (within last 5 min)
        QDSTRM_AGE="$(docker exec minikube find "${MINIKUBE_TRACES_DIR}" -name "trace.qdstrm" -newer /tmp -maxdepth 1 2>/dev/null | wc -l || echo 0)"
        if [[ "${QDSTRM_AGE}" -ge 1 ]]; then
            TRACE_FOUND="${TRACE_QDSTRM}"
            break
        fi
    fi
    [[ $i -eq 24 ]] && die "Trace not written after 2 minutes — nsys may have crashed"
    sleep 5
done
ok "Trace found: minikube:${TRACE_FOUND}"

log "Copying trace from minikube node..."
if [[ "${TRACE_FOUND}" == "${TRACE_REP}" ]]; then
    docker cp "minikube:${TRACE_REP}" "${LOCAL_OUT}/trace.nsys-rep"
else
    # qdstrm needs conversion via QdstrmImporter
    docker cp "minikube:${TRACE_QDSTRM}" "${LOCAL_OUT}/trace.qdstrm"
    QDSIMPORTER="/opt/nvidia/nsight-systems/2025.3.2/host-linux-x64/QdstrmImporter"
    if [[ -x "${QDSIMPORTER}" ]]; then
        log "Converting qdstrm → nsys-rep..."
        "${QDSIMPORTER}" "${LOCAL_OUT}/trace.qdstrm" -o "${LOCAL_OUT}/trace.nsys-rep"
        rm -f "${LOCAL_OUT}/trace.qdstrm"
    else
        warn "QdstrmImporter not found — keeping trace.qdstrm (convert manually)"
        cp "${LOCAL_OUT}/trace.qdstrm" "${LOCAL_OUT}/trace.nsys-rep" 2>/dev/null || true
    fi
fi
ok "Trace copied: ${LOCAL_OUT}/trace.nsys-rep"

# ── Step 7: Generate stats ────────────────────────────────────────────────────
log "Step 7 — generating kernel summary..."

TRACE="${LOCAL_OUT}/trace.nsys-rep"

nsys stats --report gputrace  "${TRACE}" 2>/dev/null \
    | tee "${LOCAL_OUT}/kernel_summary.txt" | head -40 || warn "gputrace report failed"

nsys stats --report cuda_api_sum "${TRACE}" 2>/dev/null \
    | tee "${LOCAL_OUT}/cuda_api_summary.txt" | head -20 || warn "cuda_api_sum report failed"

printf '\n'
ok "Done."
ok "Raw trace : ${TRACE}"
ok "Open with : nsys-ui ${TRACE}"
ok "To restore: ./scripts/run_nsys.sh --restore"
