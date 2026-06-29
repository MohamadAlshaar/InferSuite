#!/usr/bin/env bash
# Run ALL measurement passes for BigCodeBench code-execution through the shared lib.
# Deterministic GT eval re-run per pass (separate perf groups -> no multiplexing).
# Monitored, stdin-closed, hard-fail per pass, freq sampled during window.
set -uo pipefail
cd "$(dirname "$0")"
source ../common/perf_events.sh
source ../common/lib_perf.sh
PERF=$(perf_bin) || exit 1
SUDO=$(perf_sudo); perf_enable
OUT=runs/passes; mkdir -p "$OUT"; rm -f "$OUT"/group_*.txt "$OUT"/freq_*
SAMPLES="${SAMPLES:-gt_samples_small.jsonl}"
# NOTE: --split must be complete|instruct (NOT a dataset version). GT solutions were built
# from complete_prompt+canonical_solution, so use 'complete'. (Was bogusly '--split v0.1.4'.)
EVAL="python3 -m bigcodebench.evaluate --execution local --samples $SAMPLES --subset hard --split complete --parallel 4 --no_gt"
ST=/tmp/bcb_passes_status; : > "$ST"
. .venv/bin/activate

run_perf_group() {           # $1=name  $2=events
  local g="$1" ev="$2"
  echo "$(date +%T) pass=$g START" >> "$ST"
  rm -f "${SAMPLES%.jsonl}_eval_results.json" *_eval_results.json
  ( eval "$EVAL" </dev/null ) > "/tmp/bcb_pass_${g}.log" 2>&1 &
  local EV=$!
  sleep 8
  $SUDO "$PERF" stat -e "$ev" -a -o "$OUT/group_${g}.txt" </dev/null & local PP=$!
  ( while kill -0 "$EV" 2>/dev/null; do eff_freq_hz; sleep 2; done ) > "$OUT/freq_${g}" 2>/dev/null &
  wait "$EV"
  $SUDO kill -INT "$PP" 2>/dev/null
  for _ in $(seq 1 20); do [ -s "$OUT/group_${g}.txt" ] && break; sleep 0.5; done
  if assert_perf_ok "$OUT/group_${g}.txt" "$g" 2>>"$ST"; then echo "$(date +%T) pass=$g OK" >> "$ST"
  else echo "$(date +%T) pass=$g ASSERT_FAIL" >> "$ST"; fi
}

run_perf_group TMA   "$PG_TMA"
run_perf_group CACHE "$PG_CACHE"
run_perf_group FP    "$PG_FP"
run_perf_group MLP   "$PG_MLP"
run_perf_group IMC   "$PG_IMC"

# toplev -l2 pass (workload = wait until eval PID gone, so toplev flushes on clean exit)
echo "$(date +%T) pass=toplev START" >> "$ST"
rm -f "${SAMPLES%.jsonl}_eval_results.json" *_eval_results.json
( eval "$EVAL" </dev/null ) > /tmp/bcb_pass_toplev.log 2>&1 & EV=$!
sleep 8
$SUDO python3 ../swe_agent/external/pmu-tools/toplev.py -l2 --no-desc -a -o "$OUT/group_toplev.txt" \
  -- bash -c "while kill -0 $EV 2>/dev/null; do sleep 1; done" 2>/tmp/bcb_toplev.err & TP=$!
wait "$EV"; sleep 2; $SUDO kill -INT "$TP" 2>/dev/null; wait "$TP" 2>/dev/null
[ -s "$OUT/group_toplev.txt" ] && echo "$(date +%T) pass=toplev OK" >> "$ST" || echo "$(date +%T) pass=toplev EMPTY" >> "$ST"

FREQ=$(cat "$OUT"/freq_* 2>/dev/null | awk '{s+=$1;n++} END{if(n)printf "%d",s/n; else print 0}')
echo "FREQ=$FREQ" > "$OUT/freq.txt"
echo "$(date +%T) avg_freq=$FREQ ALLDONE" >> "$ST"
