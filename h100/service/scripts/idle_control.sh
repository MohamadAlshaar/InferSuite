#!/usr/bin/env bash
# idle_control.sh — vLLM idle-baseline control: same core-group stat + record as the loaded capture,
# but with loadgen scaled to 0 (engine serving nothing). Answers: is the ~2-CPU spin load-induced?
set -o pipefail
export HOME=/home/ubuntu
export KUBECONFIG="$HOME/.kube/config"
PERF=/usr/bin/perf
OUT=$HOME/service_perf/idle
mkdir -p "$OUT"
log(){ printf '[idle] %s\n' "$*"; }

# confirm no load
kubectl get deploy loadgen -n llm-service -o jsonpath='{.spec.replicas}' > "$OUT/loadgen_replicas.txt" 2>&1
POD=$(kubectl get pod -n llm-d-local -l llm-d.ai/role=decode -o jsonpath='{.items[0].metadata.name}')
kubectl logs $POD -n llm-d-local -c vllm --tail=5 2>/dev/null | grep -oE "Running: [0-9]+ reqs.*" | tail -2 > "$OUT/vllm_state.txt"
CID=$(kubectl get pod "$POD" -n llm-d-local -o jsonpath='{.status.containerStatuses[0].containerID}'); CID=${CID##*://}
PID=$(sudo k3s crictl inspect --output go-template --template '{{.info.pid}}' "$CID")
CG=$(sudo cat /proc/$PID/cgroup | sed 's/^0:://'); CG=${CG#/}
echo "$CG" > "$OUT/cgroup.txt"
sudo pkill -9 -x perf 2>/dev/null; sleep 1
log "core stat 20s (idle)"
sudo "$PERF" stat -a -e task-clock,cycles,instructions,branches,branch-misses --for-each-cgroup="$CG" -- sleep 20 2> "$OUT/group_vllm_idle_core.txt"
log "record 25s (idle)"
sudo "$PERF" record -e task-clock -a --cgroup="$CG" -g -F 199 -o "$OUT/rec_vllm_idle.data" -- sleep 25 >/dev/null 2>&1
sudo "$PERF" report -i "$OUT/rec_vllm_idle.data" --stdio -g none --symfs="/proc/$PID/root" 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/vllm_idle_flat.txt"
sudo "$PERF" report -i "$OUT/rec_vllm_idle.data" --stdio --sort=dso 2>/dev/null | grep -E "^\s+[0-9]" > "$OUT/vllm_idle_dso.txt"
# also grab gpu-util + deploy args evidence for PROVENANCE
kubectl get deploy ms-local-llm-d-modelservice-decode -n llm-d-local -o yaml 2>/dev/null | grep -E "gpu-memory-utilization|max-model-len|served-model-name" | head > "$OUT/vllm_args.txt"
log DONE
