#!/usr/bin/env bash
# validate_data.sh — post-hoc data validation (fixed: no pipefail false-positive on clean files).
# Checks every capture window under data/: multiplex tags <99.5%, not-counted rows, near-zero
# cycles/slots, empty attribution, and load verification. Exit 0 = all clean.
set -u
ROOT="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data}"
bad_total=0
for dir in "$ROOT"/tok* "$ROOT"/idle_control; do
  [ -d "$dir" ] || continue
  name=$(basename "$dir"); bad=0
  for f in "$dir"/group_*.txt; do
    [ -e "$f" ] || continue
    if grep -qE "<not counted>|<not supported>" "$f"; then
      echo "FAIL $name: $(basename "$f") not-counted/not-supported"; bad=1
    fi
    muxn=$(grep -oE '\([0-9]{1,2}\.[0-9]+%\)' "$f" 2>/dev/null | awk -F'[(%]' '$2+0 < 99.5 {c++} END {print c+0}')
    if [ "${muxn:-0}" -gt 0 ]; then
      echo "FAIL $name: $(basename "$f") $muxn multiplexed rows (<99.5%)"; bad=1
    fi
    if ! grep -E "cycles|slots" "$f" 2>/dev/null | grep -qE "[0-9][0-9,]{4,}"; then
      echo "FAIL $name: $(basename "$f") near-zero cycles/slots"; bad=1
    fi
  done
  for key in vllm fastapi; do
    if [ -e "$dir/rec_${key}.data" ]; then
      [ -s "$dir/${key}_flat.txt" ] || { echo "FAIL $name: ${key}_flat.txt empty"; bad=1; }
      sz=$(stat -c%s "$dir/rec_${key}.data" 2>/dev/null || echo 0)
      [ "$sz" -lt 20000 ] && { echo "FAIL $name: rec_${key}.data only ${sz}B"; bad=1; }
    fi
  done
  if [ "$name" != "idle_control" ] && [ -e "$dir/vllm_status_tail.txt" ]; then
    grep -qE "Running: [1-9]" "$dir/vllm_status_tail.txt" || { echo "FAIL $name: engine never Running>=1"; bad=1; }
  fi
  [ "$bad" = 0 ] && echo "OK   $name" || bad_total=1
done
exit $bad_total
