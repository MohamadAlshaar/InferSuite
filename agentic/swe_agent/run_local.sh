#!/usr/bin/env bash
# Run SWE-agent against the LOCAL vLLM (Qwen2.5-7B-AWQ in minikube, port-forwarded
# to localhost:8000). Wiring/plumbing test — 0.5B->7B is a local stand-in; final
# runs use 14B on H100. Keep external clone/venv/runs gitignored.
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate

# vLLM endpoint (litellm hosted_vllm provider = native OpenAI-compatible vLLM)
export HOSTED_VLLM_API_BASE="http://localhost:8000/v1"
export HOSTED_VLLM_API_KEY="dummy"
export OPENAI_API_KEY="dummy"

N="${1:-1}"   # number of instances (default 1)

sweagent run-batch \
  --config external/SWE-agent/config/default_backticks.yaml \
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
  --agent.tools.execution_timeout 90 \
  --agent.tools.max_consecutive_execution_timeouts 6 \
  --num_workers 1 \
  --output_dir runs/smoke
