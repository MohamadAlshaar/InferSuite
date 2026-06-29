#!/usr/bin/env python3
"""Service CPU microarch signature heatmap (CPU OUTSIDE inference) from run_20260609_140052.
Parses per-pod perf passes (pass1=IPC/branch, 2a=cache, 2b=stalls/TLB, 4=MLP), sums counters per pod
across all tiers+cells, computes within-pass ratios, plots metrics x pods."""
import re, glob, os, math
from collections import defaultdict
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RUN = "/home/mohamad/llm-service-kernel-latest/benchmark_results/run_20260609_140052"
PODS = ["fastapi", "milvus", "mongodb", "seaweed_filer", "seaweed_volume", "llmd_gateway"]
LABEL = {"fastapi":"FastAPI\n(orchestr.)", "milvus":"Milvus\n(RAG+cache)", "mongodb":"MongoDB\n(cache)",
         "seaweed_filer":"SeaweedFS\nfiler", "seaweed_volume":"SeaweedFS\nvolume", "llmd_gateway":"llm-d\ngateway"}

S = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))     # [pod][pass][counter] = summed value
ln = re.compile(r'^\s*([\d,]+(?:\.\d+)?)\s+(?:msec\s+)?([A-Za-z][\w.:-]+)')
for pod in PODS:
    for f in glob.glob(f"{RUN}/tok*/*/perf_pass*_{pod}.txt"):
        m = re.search(rf'perf_pass([0-9][a-z]?)_{re.escape(pod)}\.txt$', os.path.basename(f))
        if not m: continue
        ps = m.group(1)
        for line in open(f, errors="ignore"):
            if "<not counted>" in line or "<not supported>" in line: continue
            mm = ln.match(line)
            if mm:
                S[pod][ps][mm.group(2)] += float(mm.group(1).replace(",", ""))

def g(pod, ps, name): return S[pod].get(ps, {}).get(name, 0.0)
def div(a, b): return a / b if b else float("nan")

# metric rows (name, fn->value, fmt, higher_is_"hot")
def metrics(pod):
    insn = g(pod,"1","instructions"); cyc1 = g(pod,"1","cycles")
    br = g(pod,"1","branches") or g(pod,"1","branch-instructions")
    return {
        "CPU-seconds (total)":      g(pod,"1","task-clock")/1000.0,
        "IPC":                      div(insn, cyc1),
        "Branch-miss %":            div(g(pod,"1","branch-misses"), br)*100,
        "LLC cache-miss %":         div(g(pod,"2a","cache-misses"), g(pod,"2a","cache-references"))*100,
        "Mem-bound stall %":        div(g(pod,"2b","cycle_activity.stalls_l3_miss"), g(pod,"2b","cycles"))*100,
        "Total stall %":            div(g(pod,"2b","cycle_activity.stalls_total"), g(pod,"2b","cycles"))*100,
        "dTLB MPKI":                div(g(pod,"2b","dTLB-load-misses"), insn)*1000,
        "MLP (L1 pend/cyc)":        div(g(pod,"4","l1d_pend_miss.pending"), g(pod,"4","l1d_pend_miss.pending_cycles")),
    }

M = {p: metrics(p) for p in PODS}
rows = list(next(iter(M.values())).keys())
data = [[M[p][r] for p in PODS] for r in rows]

# per-row min-max normalize for color (each metric comparable across pods)
import numpy as np
A = np.array(data, dtype=float)
norm = np.zeros_like(A)
for i in range(A.shape[0]):
    row = A[i]; lo, hi = np.nanmin(row), np.nanmax(row)
    norm[i] = (row - lo)/(hi - lo) if hi > lo else 0.5

fig, ax = plt.subplots(figsize=(8.6, 5.6))
im = ax.imshow(norm, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(PODS))); ax.set_xticklabels([LABEL[p] for p in PODS], fontsize=9)
ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=9.5)
for i in range(len(rows)):
    for j in range(len(PODS)):
        v = A[i][j]
        txt = f"{v:,.0f}" if rows[i].startswith("CPU-sec") else (f"{v:.2f}" if v < 100 else f"{v:.0f}")
        if math.isnan(v): txt = "—"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="black" if norm[i][j] < 0.6 else "white")
ax.set_title("Micro-architectural signature of the service CPU — work OUTSIDE inference\n"
             "(aggregated across all token tiers)", fontsize=11)
fig.text(0.5, -0.02, "Colour = per-metric min–max across pods (light = low, dark = high vs the other pods); "
         "numbers are true values.", ha="center", fontsize=8.2, style="italic", color="#666")
plt.tight_layout()
OUTDIR = "/home/mohamad/llm-service-kernel-latest/thesis_plots/figures/full_benchmark/cross_tier"
for ext in ("png", "pdf"):
    p = f"{OUTDIR}/cpu_signature.{ext}"; fig.savefig(p, dpi=140, bbox_inches="tight"); print("saved", p)
# also print the raw table
print("\nmetric".ljust(22) + "".join(p[:10].rjust(12) for p in PODS))
for i,r in enumerate(rows):
    print(r.replace(chr(10)," ").ljust(22) + "".join((f"{A[i][j]:,.2f}" if not math.isnan(A[i][j]) else "—").rjust(12) for j in range(len(PODS))))
