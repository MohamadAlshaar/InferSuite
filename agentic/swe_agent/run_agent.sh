#!/usr/bin/env bash
# STEP 1 of methodology: run SWE-agent on SWE-bench Verified to COMPLETION. NO perf here
# (capture is decoupled from measurement). Stock paper-style config (07_thought_action.yaml):
# windowed file viewer + line-range edit w/ flake8 lint guardrail + search + submit, thought_action
# parser, one-shot demonstration. Produces trajectories + merged preds.json for evaluation.
#
# Deviation from paper (documented): paper used temp 0.0 (tuned for GPT-4). This project's prior
# runs found Qwen2.5-Coder-7B degenerates into identical-action loops at temp 0.0, so we use the
# minimal adaptation for a weak open model: temp 0.2 + mild freq/presence penalties. Nothing else.
set -uo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen2.5-7b}"
export HOSTED_VLLM_API_BASE="$VLLM_API_BASE" HOSTED_VLLM_API_KEY="dummy" OPENAI_API_KEY="dummy"
# 10 instances: 5 LIGHT (django) + 5 HEAVY (numeric: sklearn/astropy/sympy/matplotlib/xarray)
FILTER="${FILTER:-(django__django-(10880|10914|10973|10999|11066)|scikit-learn__scikit-learn-25232|astropy__astropy-14096|sympy__sympy-14248|matplotlib__matplotlib-24627|pydata__xarray-6744)}"
OUT="${OUT:-runs/agent}"
WORKERS="${WORKERS:-4}"
mkdir -p "$OUT"
echo "[run_agent] config=07_thought_action.yaml (stock) | model=$MODEL | workers=$WORKERS"
echo "[run_agent] filter=$FILTER"

sweagent run-batch \
  --config external/SWE-agent/config/sweagent_0_7/07_thought_action.yaml \
  --instances.type swe_bench \
  --instances.subset verified \
  --instances.split test \
  --instances.filter "$FILTER" \
  --agent.model.name "hosted_vllm/${MODEL}" \
  --agent.model.api_base "$VLLM_API_BASE" \
  --agent.model.api_key dummy \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.per_instance_call_limit 30 \
  --agent.model.max_input_tokens 28000 \
  --agent.model.max_output_tokens 4096 \
  --agent.model.temperature 0.2 \
  --agent.model.completion_kwargs '{"frequency_penalty":0.2,"presence_penalty":0.1}' \
  --agent.tools.execution_timeout 300 \
  --num_workers "$WORKERS" \
  --output_dir "$OUT"
echo "[run_agent] DONE -> $OUT (preds.json + trajectories)"
