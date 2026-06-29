#!/usr/bin/env python3
"""Parse the OpenClaw live-run perf outputs (runs/perf/) -> CPU characterization.
  container_timeline.csv : fixed-counter TMA over time (no mux)   -> overall TMA + IPC (tool side)
  container_deep.txt     : GP aggregate (cache/fp/mlp, multiplexed)-> AMAT/MPKI/FLOPs/MLP/ILP
  vllm_tma.txt           : vLLM EngineCore TMA (during-inference)
  gpu_timeline.csv + markers.txt : timing
NOTE: container_deep is MULTIPLEXED (one live run, no replay) -> perf scales it; treat as approximate.
"""
import os, re, collections
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "runs", "perf")

def counters(path):
    d = {}
    if not os.path.exists(path): return d
    for line in open(path):
        m = re.match(r"\s*([\d,]+)\s+([\w\.\-]+)", line)
        if m:
            try: d[m.group(2)] = int(m.group(1).replace(",", ""))
            except ValueError: pass
    return d

def tma_from_timeline(path):
    """sum fixed-counter topdown events over -I -x, intervals -> aggregate TMA."""
    agg = collections.Counter()
    if not os.path.exists(path): return agg
    for line in open(path):
        p = line.rstrip("\n").split(",")
        if len(p) < 4 or not p[0][:1].isdigit(): continue
        ev = p[3];
        try: v = float(p[1].replace(",", "")) if p[1] not in ("", "<not counted>") else 0.0
        except ValueError: v = 0.0
        agg[ev] += v
    return agg

def show_tma(agg, label):
    slots = agg.get("slots", 0) or 1
    ret, fe, bad, be = (agg.get(k,0) for k in ("topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"))
    cyc, ins = agg.get("cycles",0), agg.get("instructions",0)
    ipc = ins/cyc if cyc else 0
    print(f"  [{label}] IPC={ipc:.2f}  TMA: Retiring {ret/slots*100:.0f}% / FE {fe/slots*100:.0f}% / BadSpec {bad/slots*100:.0f}% / BE {be/slots*100:.0f}%")

print("=== OpenClaw OUTSIDE-inference CPU (task container = agent + tools) ===")
# combined timeline carries BOTH fixed TMA (clean) AND the GP cache/fp/mlp events
# (multiplexed). Aggregate it once and use for both TMA and the deep microarch.
c = {k: int(v) for k, v in tma_from_timeline(os.path.join(R, "container_timeline.csv")).items()}
show_tma(c, "container TMA")
if c:
    cyc = c.get("cycles",0) or 1; ins = c.get("instructions",0)
    print(f"  deep (multiplexed): IPC={ins/cyc:.2f}")
    l1,l2,l3,miss = (c.get(f"mem_load_retired.{k}",0) for k in ("l1_hit","l2_hit","l3_hit","l3_miss"))
    tot = l1+l2+l3+miss
    if tot:
        amat = (l1*4+l2*12+l3*40+miss*200)/tot
        print(f"  AMAT~{amat:.1f}cyc (L1 {l1/tot*100:.0f}%/L2 {l2/tot*100:.0f}%/L3 {l3/tot*100:.0f}%/miss {miss/tot*100:.1f}%)  LLC-MPKI={miss/(ins/1000):.2f}")
    s1,s128,s256,s512,sd,d128,d256,d512 = (c.get(f"fp_arith_inst_retired.{k}",0) for k in ("scalar_single","128b_packed_single","256b_packed_single","512b_packed_single","scalar_double","128b_packed_double","256b_packed_double","512b_packed_double"))
    flops = s1*1+s128*4+s256*8+s512*16+sd*1+d128*2+d256*4+d512*8; avx512=s512*16+d512*8
    print(f"  FP: scalar_sp={s1:,} 256b_sp={s256:,} 512b_sp={s512:,} dp={sd:,} 256b_dp={d256:,} 512b_dp={d512:,}  FLOP/cyc={flops/cyc:.3f}  AVX512%={avx512/flops*100 if flops else 0:.0f}")
    pend,pendc,uops = c.get("l1d_pend_miss.pending",0), c.get("l1d_pend_miss.pending_cycles",0), c.get("uops_executed.thread",0) or c.get("uops_executed.core",0)
    if pendc: print(f"  MLP={pend/pendc:.2f}  ILP={uops/cyc:.2f}")

print("\n=== OpenClaw DURING-inference CPU (vLLM EngineCore) ===")
v = counters(os.path.join(R, "vllm_tma.txt"))
if v:
    slots=v.get("slots",0) or 1; cyc=v.get("cycles",0) or 1
    ret,fe,bad,be=(v.get(k,0) for k in ("topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"))
    print(f"  IPC={v.get('instructions',0)/cyc:.2f}  TMA: Retiring {ret/slots*100:.0f}% / FE {fe/slots*100:.0f}% / BadSpec {bad/slots*100:.0f}% / BE {be/slots*100:.0f}%")
else:
    print("  (no vllm_tma.txt)")

# timing
mk = {}
for l in open(os.path.join(R,"markers.txt")) if os.path.exists(os.path.join(R,"markers.txt")) else []:
    p=l.split()
    if len(p)>=2:
        try: mk[p[1]]=float(p[0])
        except: pass
if "perf_start" in mk and "agent_done" in mk:
    print(f"\n=== timing ===  run wall-clock = {mk['agent_done']-mk['perf_start']:.0f}s")
