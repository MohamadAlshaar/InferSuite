#!/usr/bin/env bash
# HEAVY-task variant: scikit-learn SWE-bench instance (Ridge regression test = numpy/scipy/BLAS
# -> real CPU + AVX-FP when the agent runs the tests). Forced tool calls so the 7B fires tools.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
VLLM_API_BASE="${VLLM_API_BASE:-http://localhost:8000/v1}"
MODEL="${MODEL:-qwen2.5-32b}"
export HOSTED_VLLM_API_BASE="$VLLM_API_BASE"
export HOSTED_VLLM_API_KEY="dummy"; export OPENAI_API_KEY="dummy"
INSTANCE="${INSTANCE:-scikit-learn__scikit-learn-10297}"

sweagent run-batch \
  --config external/SWE-agent/config/default_backticks.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split test \
  --instances.filter "${INSTANCE}" \
  --agent.model.name "hosted_vllm/${MODEL}" \
  --agent.model.api_base "$VLLM_API_BASE" \
  --agent.model.api_key dummy \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.max_input_tokens ${MAX_INPUT:-28000} \
  --agent.model.max_output_tokens ${MAX_OUTPUT:-4096} \
  --agent.model.temperature 0.5 \
  --agent.model.completion_kwargs '{"frequency_penalty":0.2,"presence_penalty":0.1}' \
  --agent.tools.execution_timeout 120 \
  --agent.tools.max_consecutive_execution_timeouts 6 \
  --num_workers 1 \
  --output_dir runs/heavy
