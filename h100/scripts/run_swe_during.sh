#!/usr/bin/env bash
# SWE-agent DURING-inference engine microarch: replay the 3 recorded trajectories' request streams to
# Coder-32B under each counter group (aggregate SWE-during, same 4-group PMC-safe basis as BCB).
# Pure request replay -> no live agent, no Docker. Runs sequential (share GPU serve + PMU).
set -uo pipefail
cd /home/ubuntu/swe
PY=python3; V="MODEL=coder-32b VLLM=http://localhost:8000/v1"
T="/home/ubuntu/swe/traj/astropy__astropy-14096.traj /home/ubuntu/swe/traj/scikit-learn__scikit-learn-25232.traj /home/ubuntu/swe/traj/sympy__sympy-14248.traj"
AG="$V $PY /home/ubuntu/swe/scripts/traj_replay_engine.py $T"
CAP=/home/ubuntu/swe/scripts/capture_orchestration.sh
pkill -9 -f traj_replay_engine 2>/dev/null; sleep 2
for grp in core fp mem stall; do
  echo "== SWE DURING $grp =="
  bash "$CAP" swe_during_$grp "$grp" "$AG"
done
echo SWE_DURING_ALL_DONE