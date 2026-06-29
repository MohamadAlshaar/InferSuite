#!/usr/bin/env bash
# Corrected FP measurement: full event set incl packed-DOUBLE, own perf instance.
set -uo pipefail
cd "$(dirname "$0")"
. .venv/bin/activate
PERF=perf
FP="cycles,instructions,fp_arith_inst_retired.scalar_single,fp_arith_inst_retired.scalar_double,fp_arith_inst_retired.128b_packed_single,fp_arith_inst_retired.128b_packed_double,fp_arith_inst_retired.256b_packed_single,fp_arith_inst_retired.256b_packed_double,fp_arith_inst_retired.512b_packed_single,fp_arith_inst_retired.512b_packed_double"
ST=/tmp/bcb_fp_status; RES=runs/perf/fp_result.txt
rm -f gt_samples_eval_results.json runs/perf/bcb_fp_full.txt "$RES"
docker ps -aq --filter ancestor=wildclawbench-ubuntu:v1.3 2>/dev/null | xargs -r docker rm -f >/dev/null 2>&1
echo "START $(date +%s)" > "$ST"
# stdin closed (</dev/null) so it can NEVER hang on a Y/N prompt
python3 -m bigcodebench.evaluate --execution local --samples gt_samples.jsonl \
  --subset hard --split complete --parallel 4 --no_gt </dev/null > /tmp/bcb_fp_eval.log 2>&1 &
EV=$!
sleep 10
echo "PERF_ON $(date +%s)" >> "$ST"
"$PERF" stat -e "$FP" -a -o runs/perf/bcb_fp_full.txt </dev/null & PP=$!
wait $EV
echo "EVAL_DONE $(date +%s)" >> "$ST"
kill -INT $PP 2>/dev/null
for i in $(seq 1 30); do [ -s runs/perf/bcb_fp_full.txt ] && break; sleep 0.5; done
# parse
/usr/bin/python3 - "$RES" <<'PY'
import re,sys
d={}
for l in open("runs/perf/bcb_fp_full.txt"):
    m=re.match(r'\s*([\d,]+)\s+(\S+)',l)
    if m: d[m.group(2)]=float(m.group(1).replace(",",""))
g=lambda k:d.get(f"fp_arith_inst_retired.{k}",0)
ss,sd=g("scalar_single"),g("scalar_double")
ps={128:g("128b_packed_single"),256:g("256b_packed_single"),512:g("512b_packed_single")}
pd={128:g("128b_packed_double"),256:g("256b_packed_double"),512:g("512b_packed_double")}
flops=ss+sd+ps[128]*4+ps[256]*8+ps[512]*16+pd[128]*2+pd[256]*4+pd[512]*8
cyc=d.get("cycles",1) or 1
out=[]
out.append(f"scalar_sp={ss:,.0f}  scalar_dp={sd:,.0f}")
out.append(f"packed_single 128/256/512 = {ps[128]:,.0f} / {ps[256]:,.0f} / {ps[512]:,.0f}")
out.append(f"packed_DOUBLE 128/256/512 = {pd[128]:,.0f} / {pd[256]:,.0f} / {pd[512]:,.0f}   <-- was INVISIBLE before")
out.append(f"total FLOPs(lane-weighted) = {flops:,.0f}   FLOP/cycle = {flops/cyc:.3f}")
out.append(f"VECTORIZED share = {(flops-ss-sd)/flops*100 if flops else 0:.0f}%   (old method said ~0%)")
out.append(f"AVX-512 share = {(pd[512]*8+ps[512]*16)/flops*100 if flops else 0:.0f}%")
open(sys.argv[1],"w").write("\n".join(out)+"\n")
print("\n".join(out))
PY
echo "DONE $(date +%s)" >> "$ST"
