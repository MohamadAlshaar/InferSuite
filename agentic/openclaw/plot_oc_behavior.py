#!/usr/bin/env python3
"""OpenClaw arxiv_digest — CPU BEHAVIOR plots (recent clean run).
  1) cpu_timeline.png  — cores + IPC over wall-clock (the bursty inference-wait vs tool-exec rhythm)
  2) roofline.png      — arithmetic intensity vs achieved GFLOP/s against compute & memory ceilings
Data: timeline = runs/perf_cat_prod/task_1_arxiv_digest/container_timeline.csv (per-second, cycles is a
fixed counter so valid); roofline = clean per-group passes (FP + CACHE). Box = Intel Xeon w5-3425 (SPR).
Run with SYSTEM python3."""
import os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=f"{HERE}/runs/plots_oc_clean"; os.makedirs(OUT,exist_ok=True)
FREQ=4.6e9  # turbo; achieved & ceilings both use it so roofline position is freq-robust

# ---------- 1) CPU-over-time timeline ----------
TL=f"{HERE}/runs/perf_cat_prod/task_1_arxiv_digest/container_timeline.csv"
rows=collections.defaultdict(dict)
for line in open(TL):
    q=line.rstrip().split(",")
    if len(q)<4 or not q[0][:1].isdigit(): continue
    try: rows[q[0]][q[3]]=float(q[1]) if q[1] not in ("","<not counted>") else 0.0
    except: pass
ts=sorted(rows,key=float); t0=float(ts[0])
rel=[float(t)-t0 for t in ts]
cores=[rows[t].get("cycles",0)/FREQ for t in ts]
ipc=[(rows[t].get("instructions",0)/rows[t]["cycles"]) if rows[t].get("cycles",0)>1e7 else np.nan for t in ts]
tool_s=sum(1 for c in cores if c>0.05); tot_s=len(cores)

fig,ax=plt.subplots(2,1,figsize=(11,6),sharex=True,gridspec_kw=dict(height_ratios=[2,1]))
ax[0].fill_between(rel,cores,color="#1f77b4",alpha=0.5,step="mid")
ax[0].plot(rel,cores,color="#1f77b4",lw=0.6)
ax[0].axhline(1.0,color="gray",ls=":",lw=0.8); ax[0].text(rel[-1],1.02,"1 core",fontsize=8,color="gray",ha="right")
ax[0].set_ylabel("container CPU\n(busy cores)"); ax[0].set_ylim(0,max(1.2,max(cores)*1.1))
ax[0].set_title(f"OpenClaw arxiv_digest — CPU over the run (tool-exec is bursty; idle = waiting on the LLM)\n"
                f"tool-exec active {tool_s}s of {tot_s}s wall ({tool_s/tot_s*100:.0f}%) — the rest is off-box inference")
ax[1].plot(rel,ipc,color="#2ca02c",lw=0.8)
ax[1].set_ylabel("IPC\n(when active)"); ax[1].set_xlabel("seconds since start"); ax[1].set_ylim(0,4)
ax[1].axhline(np.nanmean(ipc),color="#2ca02c",ls="--",lw=0.8); ax[1].text(rel[-1],np.nanmean(ipc)+0.1,f"mean {np.nanmean(ipc):.2f}",fontsize=8,color="#2ca02c",ha="right")
fig.tight_layout(); fig.savefig(f"{OUT}/cpu_timeline.png",dpi=140); plt.close(fig)

# ---------- 2) Roofline ----------
def parse(p):
    a=collections.Counter()
    for line in open(p):
        q=line.split()
        if len(q)>=2:
            try: v=float(q[0].replace(",",""))
            except: continue
            if q[1] not in a: a[q[1]]=v
    return a
import sys; sys.path.insert(0, os.path.join(HERE,"..","common")); import microarch as M
f=M.parse(f"{HERE}/runs/passes_arxiv/group_FP_r1.txt"); c=M.parse(f"{HERE}/runs/passes_arxiv/group_CACHE_r1.txt")
flop=M.flops(f)                                   # CORRECT: precision lanes + FMA ×2
fcy=f["cycles"] or 1
bytes_dram=c["mem_load_retired.l3_miss"]*64        # cgroup-scoped DRAM-read proxy
AI=flop/bytes_dram if bytes_dram else 0
achieved=flop/fcy*FREQ/1e9                          # GFLOP/s = FLOP/cycle * freq
# single-core peaks @ FREQ (2 FMA ports)
PK={"scalar DP (4 F/cyc)":4*FREQ/1e9, "128-bit DP (8 F/cyc)":8*FREQ/1e9, "AVX-512 DP (32 F/cyc)":32*FREQ/1e9}
BW=25.0  # GB/s single-core est (Sapphire Rapids)
fig,ax=plt.subplots(figsize=(7.5,5.6))
xs=np.logspace(-3,2,200)
ax.plot(xs,np.minimum(BW*xs,PK["AVX-512 DP (32 F/cyc)"]),color="#888",lw=1)
for lab,pk in PK.items():
    ax.axhline(pk,ls="--",lw=1,color="#aaa"); ax.text(60,pk*1.03,lab,fontsize=8,ha="right",color="#666")
ax.plot(xs,BW*xs,ls=":",color="#d62728",lw=1); ax.text(0.02,BW*0.02*1.1,f"mem BW ~{BW:.0f} GB/s (1 core, est)",fontsize=8,color="#d62728",rotation=30)
ax.scatter([AI],[achieved],s=140,color="#1f77b4",zorder=5,edgecolor="k")
ax.annotate(f"arxiv_digest tool-exec\nAI={AI:.2f} F/B, {achieved*1e3:.0f} MFLOP/s\n(0% wide-AVX; packed=128-bit)",
            (AI,achieved),textcoords="offset points",xytext=(15,18),fontsize=9,
            arrowprops=dict(arrowstyle="->",lw=0.8))
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-3,1e2); ax.set_ylim(1e-3,500)
ax.set_xlabel("arithmetic intensity (FLOP / byte)"); ax.set_ylabel("performance (GFLOP/s)")
ax.set_title("OpenClaw arxiv_digest tool-exec — CPU roofline (single-core, est. ceilings)\n"
             "sits FAR below every roof → NOT compute/bandwidth-bound; latency/control-bound (cf. TMA FE+BE)")
fig.tight_layout(); fig.savefig(f"{OUT}/roofline.png",dpi=140); plt.close(fig)

print("WROTE -> cpu_timeline.png, roofline.png")
print(f"timeline: {tot_s}s wall, tool-exec active {tool_s}s ({tool_s/tot_s*100:.0f}%), mean active IPC {np.nanmean(ipc):.2f}")
print(f"roofline: FLOP {flop/1e6:.0f}M, AVX(packed) {M.avx_pct(f):.0f}% (all 128-bit SSE; 256/512=0), "
      f"AI {AI:.2f} F/B, achieved {achieved*1e3:.0f} MFLOP/s ({achieved/PK['128-bit DP (8 F/cyc)']*100:.2f}% of 128b peak)")
