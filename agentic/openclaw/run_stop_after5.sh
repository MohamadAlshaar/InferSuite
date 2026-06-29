#!/usr/bin/env bash
# Watch the category run; once task_5 finishes, stop the driver cleanly (no task_6+).
set -uo pipefail
while ! grep -q "DONE task_5" /tmp/oc_cat_status 2>/dev/null; do sleep 10; done
sleep 1
# patterns built from pieces so this watcher never matches itself
P_DRV="run_category""_prod.sh"
P_PERF="run_perf"".sh"
P_AG="run_batch"".py"
pkill -9 -f "$P_DRV" 2>/dev/null
pkill -INT -f "$P_PERF" 2>/dev/null
pkill -INT -f "$P_AG" 2>/dev/null
sudo pkill -INT -x perf 2>/dev/null
sleep 2
docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
echo "$(date +%T) STOPPED_AFTER_5" >> /tmp/oc_cat_status
