#!/usr/bin/env bash
# STEP 2 of methodology: OFFICIAL SWE-bench evaluation of the agent's predictions.
# Applies each model_patch to the repo at base_commit and runs FAIL_TO_PASS + PASS_TO_PASS.
# resolved = ALL FAIL_TO_PASS pass AND ALL PASS_TO_PASS still pass. This is the ground truth
# for "did the agent actually work" BEFORE we measure any perf.
# Uses a dedicated venv (.venv_swebench) so it can't perturb SWE-agent's env.
set -uo pipefail
cd "$(dirname "$0")"
PREDS="${1:-runs/agent/preds.json}"
RUN_ID="${RUN_ID:-agent_eval}"
[ -f "$PREDS" ] || { echo "FATAL: predictions not found: $PREDS"; exit 1; }
echo "[eval] predictions=$PREDS  run_id=$RUN_ID"
./.venv_swebench/bin/python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --split test \
  --predictions_path "$PREDS" \
  --max_workers 4 \
  --cache_level instance \
  --run_id "$RUN_ID"
echo "[eval] DONE — see report json (<model>.$RUN_ID.json) for resolved_instances"
