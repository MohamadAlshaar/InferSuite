#!/usr/bin/env bash
# run_timing_pass.sh — GPU-vs-CPU time measurement for the LOCAL service run: the benchmark
# protocol (n=20 per cell, concurrency 1, exact tokens) on the RAG path, per OUTPUT TIER x INPUT
# BUCKET (12 cells, like the EKS benchmark). Persists the query_runner CSVs:
#   model_backend_http_ms = GPU generation | frontend_overhead_ms = CPU side | n_output_tokens =
#   exact-token verification. Run AFTER capture_tiers.sh (no perf running, GPU otherwise idle).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export KUBECONFIG="$HOME/.kube/k3s-local.yaml"
OUT_ROOT="$REPO/local_service/data/timing"
COUNT="${COUNT:-20}"
TIERS=(${TIERS:-64 192 320})
BUCKETS=(${BUCKETS:-short medium long very_long})
log(){ printf '[timing] %s\n' "$*"; }

for T in "${TIERS[@]}"; do
  for B in "${BUCKETS[@]}"; do
    OUT="$OUT_ROOT/tok$T"; mkdir -p "$OUT"
    CSV="$OUT/rag_${B}_tok$T.csv"
    [ -s "$CSV" ] && { log "skip tok$T/$B (exists)"; continue; }
    log "cell tok$T x $B: $COUNT requests, concurrency 1, exact tokens"
    POD="timing-$T-${B//_/-}"
    kubectl delete pod "$POD" -n llm-service --ignore-not-found=true --wait=true >/dev/null 2>&1
    kubectl run "$POD" --image=llm-service-kernel:fastapi-selfcontained --image-pull-policy=Never \
      --restart=Never --attach -n llm-service --pod-running-timeout=3m \
      --env BENCHMARK_URL=http://llm-service-kernel.llm-service.svc.cluster.local:8080 \
      --env BENCHMARK_MODEL=qwen2.5-7b-instruct-awq \
      -- /bin/sh -c "python3 /app/scripts/query_runner.py --mode rag \
          --queries /app/benchmark_queries/rag/$B.txt --size-bucket $B \
          --count $COUNT --warmup 2 --max-tokens $T --concurrency 1 --out-dir /tmp/t \
          && echo CSV_BEGIN && cat /tmp/t/*.csv && echo CSV_END" > "$OUT/raw_${B}.log" 2>&1 || true
    kubectl delete pod "$POD" -n llm-service --ignore-not-found=true >/dev/null 2>&1
    awk '/^CSV_BEGIN$/{f=1;next} /^CSV_END$/{f=0} f' "$OUT/raw_${B}.log" > "$CSV"
    rows=$(($(wc -l < "$CSV") - 1)); [ "$rows" -lt 0 ] && rows=0
    log "  saved $rows rows -> $CSV"
    python3 - "$CSV" "$T" <<'PY'
import csv, sys, statistics as st
try: rows = list(csv.DictReader(open(sys.argv[1])))
except Exception: rows = []
tgt = int(sys.argv[2])
if not rows: print("  [timing] EMPTY CSV — FAIL"); sys.exit(0)
toks = [int(float(r["n_output_tokens"])) for r in rows if r.get("n_output_tokens")]
exact = sum(1 for t in toks if t == tgt)
print(f"  [timing] n_output_tokens: {exact}/{len(toks)} exactly {tgt} (min {min(toks)}, max {max(toks)})")
try:
    gpu = [float(r["model_backend_http_ms"]) for r in rows]
    cpu = [float(r["frontend_overhead_ms"]) for r in rows]
    print(f"  [timing] median GPU-gen {st.median(gpu):.0f} ms | median CPU-side {st.median(cpu):.0f} ms "
          f"| GPU share {sum(gpu)/(sum(gpu)+sum(cpu))*100:.1f}%")
except KeyError as e:
    print(f"  [timing] column missing: {e} — headers: {list(rows[0].keys())[:14]}")
PY
  done
done
log "DONE -> $OUT_ROOT (12 cells: 3 tiers x 4 buckets)"
