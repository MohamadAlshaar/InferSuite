#!/usr/bin/env bash
# Runs the 4 CANONICAL OpenClaw tasks live (Sonnet via local litellm proxy) with per-task perf record.
# Requires ~/.anthropic_key (chmod 600). Starts/stops the proxy itself.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KEYFILE="$HOME/.anthropic_key"
[ -s "$KEYFILE" ] || { echo "ERROR: $KEYFILE missing"; exit 1; }
export ANTHROPIC_API_KEY="$(tr -d '[:space:]' < "$KEYFILE")"
cd "$REPO/agentic/openclaw"
./.venv_litellm/bin/litellm --config litellm_config.yaml --port 8000 > /tmp/litellm_oc.log 2>&1 & PROXY=$!
trap 'kill $PROXY 2>/dev/null' EXIT
for i in $(seq 1 30); do curl -sf localhost:8000/health/liveliness >/dev/null 2>&1 && break; sleep 2; done
curl -sf localhost:8000/health/liveliness >/dev/null || { echo "ERROR: proxy did not start"; tail -5 /tmp/litellm_oc.log; exit 1; }
echo "proxy up"
T=tasks/01_Productivity_Flow
TASK="$T/01_Productivity_Flow_task_6_calendar_scheduling.md"      LABEL=calendar   bash "$REPO/local_agents/scripts/oc_tool_record_local.sh"
TASK="$T/01_Productivity_Flow_task_1_arxiv_digest.md"             LABEL=web-digest bash "$REPO/local_agents/scripts/oc_tool_record_local.sh"
TASK="$T/01_Productivity_Flow_task_10_pdf_digest.md"              LABEL=pdf-digest bash "$REPO/local_agents/scripts/oc_tool_record_local.sh"
TASK="tasks/05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop.md" LABEL=image-crop bash "$REPO/local_agents/scripts/oc_tool_record_local.sh"
echo "OC-CHAIN-DONE"
