#!/usr/bin/env bash
# Validate the record-replay cap-6 BCB run: (1) NO multiplexing / zeroing on every engine timeline
# and every tool-replay CSV; (2) print derived metrics so we can sanity-check they make sense.
# engine timeline -x, cols: time,value,unit,event,run-time,PCT,...     (pct = field 6)
# tool -x, cols:            value,unit,event,run-time,PCT,...           (pct = field 5)
set -uo pipefail
R=/home/ubuntu/bcb/runs; COND="${1:-spin}"
echo "########## ENGINE (during-inference) integrity ##########"
for g in core fp mem stall; do
  f=$R/bcb_${COND}_$g/engine_${g}_timeline.csv; [ "$g" = core ] && f=$R/bcb_${COND}_core/engine_timeline.csv
  [ -f "$f" ] || { echo "[$g] MISSING $f"; continue; }
  nc=$(grep -icE "not counted|not supported|<not" "$f" || true)
  awk -F, -v g="$g" -v nc="$nc" 'NF>=6 && $4!="" {tot++; if($6!="" && $6+0<99.0) mux++} $4=="cycles"&&$2+0==0{zc++}
    END{printf "  [%-5s] rows=%d  multiplexed(pct<99)=%d  zeroed-cycles=%d  notcounted=%s\n",g,tot,mux+0,zc+0,nc}' "$f"
done
echo
echo "########## TOOL-EXEC (outside-inference) integrity ##########"
for g in core fp mem stall; do
  d=$R/tool_$g; [ -d "$d" ] || { echo "[$g] MISSING $d"; continue; }
  n=$(ls "$d"/*.csv 2>/dev/null | wc -l)
  bad=$(grep -lE "not counted|not supported" "$d"/*.csv 2>/dev/null | wc -l)
  mux=$(awk -F, 'NF>=5 && $3!="" && $5!="" && $5+0<99.0{c++} END{print c+0}' "$d"/*.csv 2>/dev/null)
  echo "  [$g] programs=$n  notcounted-files=$bad  multiplexed-rows(pct<99)=$mux"
done
echo
echo "########## SANITY: derived metrics (do they make sense?) ##########"
python3 - "$R" "$COND" <<'PY'
import sys,glob,os
R,COND=sys.argv[1],sys.argv[2]
def st(f):
    t={}
    if not os.path.exists(f): return t
    for ln in open(f):
        if ln.startswith('#') or not ln.strip(): continue
        c=ln.split(',')
        if len(c)<4: continue
        v,e=c[1].strip(),c[3].strip()
        if not e or v in ('<not counted>','<not supported>',''): continue
        try: t[e]=t.get(e,0.0)+float(v)
        except: pass
    return t
def td(d):
    t={}
    for f in glob.glob(d+'/*.csv'):
        for ln in open(f):
            if ln.startswith('#') or not ln.strip(): continue
            c=ln.split(',')
            if len(c)<3: continue
            v,e=c[0].strip(),c[2].strip()
            if not e or v in ('<not counted>','<not supported>',''): continue
            try: t[e]=t.get(e,0.0)+float(v)
            except: pass
    return t
def m(t):
    ins=t.get('instructions',0); cyc=t.get('cycles',0)
    l1=t.get('mem_load_retired.l1_hit',0); l2=t.get('mem_load_retired.l2_hit',0)
    l3=t.get('mem_load_retired.l3_hit',0); l3m=t.get('mem_load_retired.l3_miss',0); tot=l1+l2+l3+l3m
    stl=t.get('cycle_activity.stalls_total',0); sl3=t.get('cycle_activity.stalls_l3_miss',0)
    mp=t.get('l1d_pend_miss.pending',0); mpc=t.get('l1d_pend_miss.pending_cycles',0)
    s=t.get('fp_arith_inst_retired.scalar_double',0); p1=t.get('fp_arith_inst_retired.128b_packed_double',0)
    p2=t.get('fp_arith_inst_retired.256b_packed_double',0); p5=t.get('fp_arith_inst_retired.512b_packed_double',0)
    fl=s+p1*2+p2*4+p5*8; vec=p1*2+p2*4+p5*8; o={}
    if cyc: o['IPC']=round(ins/cyc,2)
    if ins: o['cMPKI']=round(t.get('cache-misses',0)*1000/ins,2); o['bMPKI']=round(t.get('branch-misses',0)*1000/ins,2)
    if tot: o['L1/L2/L3/miss%']=f"{100*l1/tot:.0f}/{100*l2/tot:.0f}/{100*l3/tot:.0f}/{100*l3m/tot:.1f}"; o['AMATcyc']=round((l1*5+l2*15+l3*50+l3m*250)/tot,1)
    if l3m: o['DRAMrd_MB']=round(l3m*64/1e6,1)
    if cyc and stl: o['stall%']=round(100*stl/cyc,1)
    if cyc and sl3: o['memBound%']=round(100*sl3/cyc,1)
    if mpc: o['MLP']=round(mp/mpc,2)
    if fl: o['vec%']=round(100*vec/fl,1); o['avx512%']=round(100*p5*8/fl,1); o['MFLOP']=round(fl/1e6,1)
    return o
et={}
for g in ['core','fp','mem','stall']:
    f=f"{R}/bcb_{COND}_{g}/engine_{g}_timeline.csv"
    if g=='core': f=f"{R}/bcb_{COND}_core/engine_timeline.csv"
    et.update(st(f))
tt={}
for g in ['core','fp','mem','stall']: tt.update(td(f"{R}/tool_{g}"))
print("DURING  (engine): ",m(et))
print("OUTSIDE (tool)  : ",m(tt))
PY
