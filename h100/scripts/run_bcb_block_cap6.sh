#!/usr/bin/env bash
# Re-run the BCB captures at cap=6 under the CURRENT (evblock) serve. Sequential — they share the
# GPU serve and the PMU, so they must not overlap. Engine captures use capture_orchestration.sh;
# tool-exec captures use the perf-wrapped subprocess harness.
set -uo pipefail
cd /home/ubuntu/bcb
PY=/home/ubuntu/bcb/.venv/bin/python; CAP=6
pkill -9 -f agentic_bcb_toolperf 2>/dev/null; pkill -9 -f "agentic_bcb.py" 2>/dev/null; sleep 2
AGENT="MODEL=coder-32b VLLM=http://localhost:8000/v1 $PY /home/ubuntu/bcb/agentic_bcb.py 12 $CAP"
FPEV="cycles,instructions,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double"

echo "== [1/4] engine block core @${CAP} =="
bash capture_orchestration.sh bcb_block_core core "$AGENT"
echo "== [2/4] engine block fp @${CAP} =="
bash capture_orchestration.sh bcb_block_fp fp "$AGENT"
echo "== [3/4] tool-exec core @${CAP} =="
sudo sysctl -w kernel.perf_event_paranoid=-1 >/dev/null 2>&1; rm -f /tmp/bcb_agentic_markers.txt; rm -rf runs/tool_core
env TOOLPERF_DIR=/home/ubuntu/bcb/runs/tool_core MODEL=coder-32b VLLM=http://localhost:8000/v1 \
  $PY /home/ubuntu/bcb/agentic_bcb_toolperf.py 12 $CAP > runs/tool_core.log 2>&1
echo "tool_core rc=$? csvs=$(ls runs/tool_core/*.csv 2>/dev/null | wc -l)"; tail -1 runs/tool_core.log
echo "== [4/4] tool-exec fp @${CAP} =="
rm -f /tmp/bcb_agentic_markers.txt; rm -rf runs/tool_fp
env TOOLPERF_DIR=/home/ubuntu/bcb/runs/tool_fp TOOLPERF_EVENTS="$FPEV" MODEL=coder-32b VLLM=http://localhost:8000/v1 \
  $PY /home/ubuntu/bcb/agentic_bcb_toolperf.py 12 $CAP > runs/tool_fp.log 2>&1
echo "tool_fp rc=$? csvs=$(ls runs/tool_fp/*.csv 2>/dev/null | wc -l)"; tail -1 runs/tool_fp.log
echo "ALL_BLOCK_CAP6_DONE"
