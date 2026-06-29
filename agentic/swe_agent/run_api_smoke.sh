#!/usr/bin/env bash
# SMOKE TEST: one SWE-bench Verified instance via a hosted Claude model (API).
# Validates end-to-end: Anthropic auth (litellm) + stock function-calling config + agent reasons,
# edits, runs, submits a NON-empty patch. NO perf here. Key comes from .env (gitignored).
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; . ./.env; set +a   # load ANTHROPIC_API_KEY (not echoed)
[ -n "${ANTHROPIC_API_KEY:-}" ] || { echo "FATAL: ANTHROPIC_API_KEY not set"; exit 1; }
MODEL="${MODEL:-anthropic/claude-sonnet-4-6}"
INSTANCE="${INSTANCE:-django__django-10914}"
OUT="${OUT:-runs/api_smoke}"; rm -rf "$OUT"; mkdir -p "$OUT"
echo "[api-smoke] model=$MODEL instance=$INSTANCE  (stock default.yaml, function-calling)"

sweagent run-batch \
  --config external/SWE-agent/config/default.yaml \
  --instances.type swe_bench --instances.subset verified --instances.split test \
  --instances.filter "$INSTANCE" \
  --agent.model.name "$MODEL" \
  --agent.model.top_p null \
  --agent.model.per_instance_cost_limit 3.0 \
  --agent.model.total_cost_limit 5.0 \
  --num_workers 1 \
  --output_dir "$OUT" 2>&1 | tee "$OUT/smoke.log"
echo "[api-smoke] done -> $OUT"
