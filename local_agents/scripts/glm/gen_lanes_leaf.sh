#!/usr/bin/env bash
# gen_lanes_leaf.sh — derive per-CPU occupancy lanes + harness leaf-symbol tables from the
# banked full-episode perf records (rec_scope*.data). These inputs feed Fig 8 (hw-threads)
# and Fig 9 (harness anatomy); the campaign never wrote them (writer was never committed —
# found in the 2026-07-14 audit). Post-hoc derivation is exact: the records already carry
# the per-sample CPU and callchain.
#
#   ./gen_lanes_leaf.sh <data_root> [--leaf]     # --leaf also writes scope1_leaf.txt
#
# scopeN_cpulanes.tsv : "<time> <cpu>" per 99 Hz sample (scopes 1+2 = harness+tool fences)
# scope1_leaf.txt     : "<weight>\t<sym> (<dso>)" leaf frame of each harness sample
# Needs sudo (records are root-written on some runs; perf refuses silently otherwise).
set -o pipefail
PERF=$(ls -d /usr/lib/linux-tools-6.8*/perf 2>/dev/null | tail -1)
ROOT="${1:?usage: $0 <data_root> [--leaf]}"
WANT_LEAF="${2:-}"

for rd in "$ROOT"/*/run_*; do
  [ -d "$rd" ] || continue
  for sc in 1 2; do
    rec="$rd/rec_scope${sc}.data"; out="$rd/scope${sc}_cpulanes.tsv"
    [ -f "$rec" ] || continue
    [ -s "$out" ] && continue
    # perf prints "[cpu] time:" — swap to "time cpu"
    sudo -n "$PERF" script -f -i "$rec" -F time,cpu 2>/dev/null | \
      awk 'NF>=2 {gsub(/\[|\]/,"",$1); gsub(/:$/,"",$2); print $2, $1+0}' \
      > "$out.tmp"
    if [ -s "$out.tmp" ]; then mv "$out.tmp" "$out"; echo "  $out ($(wc -l < "$out") samples)"
    else rm -f "$out.tmp"; echo "  WARN empty extraction: $rec"; fi
  done
  if [ "$WANT_LEAF" = "--leaf" ]; then
    rec="$rd/rec_scope1.data"; out="$rd/scope1_leaf.txt"
    [ -f "$rec" ] || continue
    [ -s "$out" ] && continue
    sudo -n "$PERF" script -f -i "$rec" -F comm,period,ip,sym,dso 2>/dev/null | awk '
      /^\t/ { if (want) { line=$0; sub(/^\t[ ]*[0-9a-f]+ /,"",line); n[line]+=per; want=0 } next }
      NF>=2 { per=$NF; want=1 }
      END { for (x in n) printf "%d\t%s\n", n[x], x }' | sort -rn > "$out.tmp"
    if [ -s "$out.tmp" ]; then mv "$out.tmp" "$out"; echo "  $out ($(wc -l < "$out") leaf syms)"
    else rm -f "$out.tmp"; echo "  WARN empty leaf extraction: $rec"; fi
  fi
done
echo "done."
