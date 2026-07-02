#!/usr/bin/env bash
# BCB record-once-replay at cap=6 under the CURRENT (evblock) serve.
#   1 LIVE record run  -> engine CORE microarch + attribution + saves program corpus + request stream
#   ENGINE fp/mem/stall -> REPLAY the recorded requests (identical work per group)
#   TOOL core/fp/mem/stall -> REPLAY the recorded programs (identical work per group)
set -uo pipefail
cd /home/ubuntu/bcb
COND="${COND:-spin}"   # serve condition label (default serve = spin)
PY=/home/ubuntu/bcb/.venv/bin/python; CAP=6; CORPUS=/home/ubuntu/bcb/corpus
V="MODEL=coder-32b VLLM=http://localhost:8000/v1"
COREV="cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses"
FPV="cycles,instructions,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_double"
MEMV="cycles,instructions,mem_load_retired.l1_hit,mem_load_retired.l2_hit,mem_load_retired.l3_hit,mem_load_retired.l3_miss"
STALLV="cycles,instructions,cycle_activity.stalls_total,cycle_activity.stalls_l3_miss,l1d_pend_miss.pending,l1d_pend_miss.pending_cycles"
pkill -9 -f "record_bcb|replay_engine|agentic_bcb" 2>/dev/null; rm -rf "$CORPUS"; mkdir -p "$CORPUS"

echo "== LIVE RECORD (engine core + attribution + save corpus/requests) =="
bash capture_orchestration.sh bcb_${COND}_core core "$V CORPUS=$CORPUS $PY /home/ubuntu/bcb/record_bcb.py 12 $CAP"
echo "corpus: progs=$(ls $CORPUS/prog_*.py 2>/dev/null | wc -l) requests=$(wc -l < $CORPUS/requests.jsonl 2>/dev/null)"

echo "== ENGINE REPLAY (identical requests, forced decode) =="
bash capture_orchestration.sh bcb_${COND}_fp    fp    "$V $PY /home/ubuntu/bcb/replay_engine.py $CORPUS/requests.jsonl"
bash capture_orchestration.sh bcb_${COND}_mem   mem   "$V $PY /home/ubuntu/bcb/replay_engine.py $CORPUS/requests.jsonl"
bash capture_orchestration.sh bcb_${COND}_stall stall "$V $PY /home/ubuntu/bcb/replay_engine.py $CORPUS/requests.jsonl"

echo "== TOOL REPLAY (identical programs) =="
bash replay_tool.sh "$CORPUS" "$COREV"  runs/tool_core
bash replay_tool.sh "$CORPUS" "$FPV"    runs/tool_fp
bash replay_tool.sh "$CORPUS" "$MEMV"   runs/tool_mem
bash replay_tool.sh "$CORPUS" "$STALLV" runs/tool_stall
echo "ALL_RR_CAP6_DONE"
