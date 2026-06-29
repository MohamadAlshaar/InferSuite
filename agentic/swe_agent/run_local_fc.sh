#!/usr/bin/env bash
# Function-calling variant: vLLM has --enable-auto-tool-choice --tool-call-parser hermes,
# and we force tool_choice=required (guided decoding) so the weak 7B emits well-formed tool
# calls instead of XML-in-content. Uses default.yaml-derived fc_local.yaml (real edit tools,
# function_calling parser, cache_control removed for hosted_vllm compatibility).
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
export HOSTED_VLLM_API_KEY="dummy"
export OPENAI_API_KEY="dummy"
N="${1:-1}"

sweagent run-batch \
  --config external/SWE-agent/config/fc_local.yaml \
  --instances.type swe_bench \
  --instances.subset lite \
  --instances.split dev \
  --instances.slice ":${N}" \
  --agent.model.name hosted_vllm/qwen2.5-7b \
  --agent.model.api_base http://localhost:8000/v1 \
  --agent.model.api_key dummy \
  --agent.model.per_instance_cost_limit 0 \
  --agent.model.total_cost_limit 0 \
  --agent.model.max_input_tokens 24000 \
  --agent.model.max_output_tokens 4096 \
  --agent.model.temperature 0.4 \
  --agent.model.completion_kwargs '{"tool_choice":"required","frequency_penalty":0.5,"presence_penalty":0.3}' \
  --agent.tools.execution_timeout 90 \
  --agent.tools.max_consecutive_execution_timeouts 6 \
  --num_workers 1 \
  --output_dir runs/smoke
