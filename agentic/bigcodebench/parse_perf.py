#!/usr/bin/env python3
"""Parse BigCodeBench code-execution perf (runs/perf/bcb_timeline.csv, perf -a -I -x,).
Aggregates TMA (fixed, clean) + cache/fp/mlp (multiplexed -> perf-scaled, ratios valid).
Emphasis: FP/SIMD — numpy/scipy/sklearn should show real FP (vs the integer-only agent runs)."""
import os, collections
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,"runs","perf")

agg=collections.Counter()
p=os.path.join(R,"bcb_timeline.csv")
if not os.path.exists(p): print("no bcb_timeline.csv — run run_perf.sh first"); raise SystemExit
for line in open(p):
    q=line.rstrip("\n").split(",")
    if len(q)<4 or not q[0][:1].isdigit(): continue
    try: agg[q[3]] += float(q[1].replace(",","")) if q[1] not in("","<not counted>") else 0.0
    except ValueError: pass

slots=agg.get("slots",0) or 1; cyc=agg.get("cycles",0) or 1; ins=agg.get("instructions",0)
print("=== BigCodeBench code-execution CPU (system-wide; vLLM idle during eval) ===")
print(f"  IPC={ins/cyc:.2f}")
if agg.get("topdown-retiring",0):  # Ice Lake+ slots form (c7i Sapphire Rapids)
    ret,fe,bad,be=(agg.get(k,0)/slots*100 for k in ("topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"))
else:  # LEGACY (Cascade Lake / g4dn.metal): SLOTS = 4 * cpu_clk_unhalted.thread
    S=4*(agg.get("cpu_clk_unhalted.thread",0) or cyc) or 1
    rt=agg.get("uops_retired.retire_slots",0); fb=agg.get("idq_uops_not_delivered.core",0)
    bs=max(agg.get("uops_issued.any",0)-rt+4*agg.get("int_misc.recovery_cycles",0),0)
    ret=rt/S*100; fe=fb/S*100; bad=bs/S*100; be=max(100-ret-fe-bad,0)
print(f"  TMA: Retiring {ret:.0f}% / Frontend {fe:.0f}% / Bad-spec {bad:.0f}% / Backend {be:.0f}%")
l1,l2,l3,miss=(agg.get(f"mem_load_retired.{k}",0) for k in ("l1_hit","l2_hit","l3_hit","l3_miss"))
tot=l1+l2+l3+miss
if tot:
    amat=(l1*4+l2*12+l3*40+miss*200)/tot
    print(f"  AMAT~{amat:.1f}cyc (L1 {l1/tot*100:.0f}%/L2 {l2/tot*100:.0f}%/L3 {l3/tot*100:.0f}%/miss {miss/tot*100:.1f}%)  LLC-MPKI={miss/(ins/1000):.2f}")
s1,s128,s256,s512,sd,d128,d256,d512=(agg.get(f"fp_arith_inst_retired.{k}",0) for k in ("scalar_single","128b_packed_single","256b_packed_single","512b_packed_single","scalar_double","128b_packed_double","256b_packed_double","512b_packed_double"))
flops=s1*1+s128*4+s256*8+s512*16+sd*1+d128*2+d256*4+d512*8
avx512=s512*16+d512*8
print(f"\n  *** FP / SIMD (the new thing vs integer-only agents) ***")
print(f"  scalar_sp={s1:,.0f} 128b_sp={s128:,.0f} 256b_sp={s256:,.0f} 512b_sp={s512:,.0f} | scalar_dp={sd:,.0f} 128b_dp={d128:,.0f} 256b_dp={d256:,.0f} 512b_dp={d512:,.0f}")
print(f"  FLOPs(lane-weighted)={flops:,.0f}  FLOP/cycle={flops/cyc:.3f}  AVX-512 share={avx512/flops*100 if flops else 0:.0f}%")
pend,pendc,uops=agg.get("l1d_pend_miss.pending",0),agg.get("l1d_pend_miss.pending_cycles",0),agg.get("uops_executed.thread",0) or agg.get("uops_executed.core",0)
if pendc: print(f"\n  MLP={pend/pendc:.2f}  ILP={uops/cyc:.2f}")
# coverage note
import statistics
print(f"\n  (cycles={cyc:,.0f}; deep GP events multiplexed -> perf-scaled, ratios valid)")
