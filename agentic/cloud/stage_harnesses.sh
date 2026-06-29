#!/usr/bin/env bash
# After launch: copy the agentic harnesses + perf scripts onto the c7i.metal box.
# Excludes venvs and the big external clones (those are re-created/cloned on the box).
set -euo pipefail
cd "$(dirname "$0")"; . ./env.sh; . ./state.env
REPO=/home/mohamad/llm-service-kernel-latest
echo "Staging agentic harnesses -> metal box ($CPU_IP) ..."
tar -C "$REPO" \
  --exclude='**/.venv' --exclude='**/venv' --exclude='**/external' \
  --exclude='**/__pycache__' --exclude='**/runs' --exclude='**/output' \
  -czf /tmp/agentic_harness.tgz agentic
scp -o StrictHostKeyChecking=no -i "$PEM" /tmp/agentic_harness.tgz ubuntu@"$CPU_IP":/opt/agentic/
ssh -o StrictHostKeyChecking=no -i "$PEM" ubuntu@"$CPU_IP" \
  "cd /opt/agentic && tar xzf agentic_harness.tgz && echo staged: && ls agentic"
echo "Done. On the metal box, point harnesses at the GPU box: http://$GPU_PRIV:8000/v1"
echo "  (SWE-agent/BigCodeBench use Coder model; set the served name 'qwen2.5-32b')"
