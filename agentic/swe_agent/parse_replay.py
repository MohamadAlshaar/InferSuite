#!/usr/bin/env python3
"""Aggregate the per-counter-group replay totals -> deep tool-exec CPU microarch.
Reads runs/replay/group_{cache,fp,mlp}.txt (aggregate `perf stat -G` output) and
prints IPC, AMAT, cache MPKI, FLOPs/SIMD mix, MLP/ILP for the tool-exec phase.
"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs", "replay")

def counters(path):
    d = {}
    if not os.path.exists(path): return d
    for line in open(path):
        m = re.match(r"\s*([\d,]+)\s+([\w\.\-]+)", line)
        if m:
            try: d[m.group(2)] = int(m.group(1).replace(",", ""))
            except ValueError: pass
    return d

cache = counters(os.path.join(R, "group_cache.txt"))
fp    = counters(os.path.join(R, "group_fp.txt"))
mlp   = counters(os.path.join(R, "group_mlp.txt"))
if not (cache or fp or mlp):
    print("no replay group data in", R); sys.exit(1)

def ipc(d):
    c, i = d.get("cycles", 0), d.get("instructions", 0)
    return i / c if c else 0

print("=== tool-exec phase deep microarch (replay, aggregate) ===")
for name, d in [("cache", cache), ("fp", fp), ("mlp", mlp)]:
    if d: print(f"  [{name}] cycles={d.get('cycles',0):,} instr={d.get('instructions',0):,} IPC={ipc(d):.2f}")

# ---- AMAT + cache MPKI (cache group) ----
if cache:
    l1, l2, l3, miss = (cache.get(f"mem_load_retired.{k}", 0) for k in ("l1_hit","l2_hit","l3_hit","l3_miss"))
    tot = l1 + l2 + l3 + miss
    instr = cache.get("instructions", 0) or 1
    if tot:
        amat = (l1*4 + l2*12 + l3*40 + miss*200) / tot  # SPR approx latencies (cycles)
        print(f"\n  AMAT ~= {amat:.1f} cycles  (L1 {l1/tot*100:.0f}% / L2 {l2/tot*100:.0f}% / L3 {l3/tot*100:.0f}% / miss {miss/tot*100:.1f}%)")
        print(f"  LLC-load-miss MPKI = {miss/(instr/1000):.2f}   L2 MPKI = {(l2+l3+miss)/(instr/1000):.2f}")

# ---- FLOPs / SIMD mix (fp group) ----
if fp:
    s1 = fp.get("fp_arith_inst_retired.scalar_single", 0)
    s128 = fp.get("fp_arith_inst_retired.128b_packed_single", 0)
    v256 = fp.get("fp_arith_inst_retired.256b_packed_single", 0)
    v512 = fp.get("fp_arith_inst_retired.512b_packed_single", 0)
    sd = fp.get("fp_arith_inst_retired.scalar_double", 0)
    d128 = fp.get("fp_arith_inst_retired.128b_packed_double", 0)
    d256 = fp.get("fp_arith_inst_retired.256b_packed_double", 0)
    d512 = fp.get("fp_arith_inst_retired.512b_packed_double", 0)
    # FLOPs = instrs * lanes (FMA x2 omitted; report as MACs-equivalent lower bound)
    flops = s1*1 + s128*4 + v256*8 + v512*16 + sd*1 + d128*2 + d256*4 + d512*8
    avx512 = v512*16 + d512*8
    cyc = fp.get("cycles", 0) or 1
    print(f"\n  FP ops: scalar_sp={s1:,} 256b_sp={v256:,} 512b_sp={v512:,} scalar_dp={sd:,} 256b_dp={d256:,} 512b_dp={d512:,}")
    print(f"  FLOPs(lane-weighted) = {flops:,}  ({flops/cyc:.3f} FLOP/cycle)  [AVX-512 share: {avx512/flops*100 if flops else 0:.0f}%]")
    print("  (arithmetic intensity needs DRAM bytes from IMC -> add a node-wide IMC replay)")

# ---- MLP / ILP (mlp group) ----
if mlp:
    pend = mlp.get("l1d_pend_miss.pending", 0)
    pendc = mlp.get("l1d_pend_miss.pending_cycles", 0)
    uops = mlp.get("uops_executed.thread", 0) or mlp.get("uops_executed.core", 0)
    cyc = mlp.get("cycles", 0) or 1
    print(f"\n  MLP = {pend/pendc:.2f} outstanding L1 misses/cycle" if pendc else "\n  MLP: n/a")
    print(f"  ILP = {uops/cyc:.2f} uops/cycle")

# ---- IMC DRAM bytes -> arithmetic intensity / roofline (imc group) ----
def imc_bytes(path):
    rd = wr = 0.0
    if not os.path.exists(path): return None
    for line in open(path):
        m = re.search(r"([\d,\.]+)\s+(MiB|GiB|KiB)?\s*uncore_imc/cas_count_(read|write)/", line)
        if m:
            v = float(m.group(1).replace(",", "")); unit = m.group(2) or ""
            mult = {"KiB":2**10, "MiB":2**20, "GiB":2**30, "":64}.get(unit, 64)  # raw count*64B if no unit
            (globals().__setitem__) if False else None
            if m.group(3) == "read": rd = v*mult
            else: wr = v*mult
    return rd, wr
imc = imc_bytes(os.path.join(R, "group_imc.txt"))
if imc:
    rd, wr = imc; dram = rd + wr
    print(f"\n  DRAM traffic (node-wide UPPER BOUND): read={rd/2**30:.2f} GiB  write={wr/2**30:.2f} GiB  total={dram/2**30:.2f} GiB")
    # IMC/uncore counters are a node-level PMU -> CANNOT be cgroup-scoped, so this is the
    # whole machine, not the tool. Verified: the node-wide window (~176G cycles, IPC 0.86)
    # is ~13x the sandbox tool's own cycles (~13G, IPC 1.45) -> dominated by scaffolding
    # (dockerd/containerd, resident vLLM, host orchestrator). Marker-fencing out container
    # start + git-reset changed it by only ~5% (51.7->49.3 GiB), so setup is NOT the cause.
    # => DRAM here is a loose upper bound; the authoritative tool-exec microarch is the
    #    cgroup-scoped cache/fp/mlp groups above (L1-resident, integer-bound).
    print(f"  NOTE: node-wide (uncore cannot be cgroup-scoped) -> upper bound, ~13x tool cycles,")
    print(f"        dominated by scaffolding (dockerd/vLLM/orchestrator); fencing reset = -5% only.")
    if fp and dram:
        s1=fp.get("fp_arith_inst_retired.scalar_single",0); s128=fp.get("fp_arith_inst_retired.128b_packed_single",0); v256=fp.get("fp_arith_inst_retired.256b_packed_single",0)
        v512=fp.get("fp_arith_inst_retired.512b_packed_single",0); sd=fp.get("fp_arith_inst_retired.scalar_double",0); d128=fp.get("fp_arith_inst_retired.128b_packed_double",0); d256=fp.get("fp_arith_inst_retired.256b_packed_double",0); d512=fp.get("fp_arith_inst_retired.512b_packed_double",0)
        flops = s1*1 + s128*4 + v256*8 + v512*16 + sd*1 + d128*2 + d256*4 + d512*8
        print(f"  Arithmetic intensity = {flops/dram:.4f} FLOP/byte  (integer-heavy tool + node-wide bytes -> ~0)")

# ---- deep TMA tree (toplev) ----
tl = os.path.join(R, "group_toplev.txt")
if os.path.exists(tl) and os.path.getsize(tl) > 0:
    print("\n  === deep TMA tree (toplev -l2, system-wide during replay) ===")
    for line in open(tl):
        # format: "FE   Frontend_Bound.Fetch_Latency   % Slots   19.9   [49.0%]<=="
        m = re.match(r"\s*\S+\s+([A-Za-z][\w.]+)\s+%.*?([0-9]+\.[0-9]+)", line)
        if m:
            flag = "  <== bottleneck" if "<==" in line else ""
            print(f"    {m.group(1):30s} {float(m.group(2)):5.1f}%{flag}")
    mux = re.search(r"MUX\s+%\s+([0-9.]+)", open(tl).read())
    if mux: print(f"    (MUX={mux.group(1)}% -> multiplexed, system-wide; cgroup cache/fp/mlp groups are the authoritative tool-exec numbers)")
