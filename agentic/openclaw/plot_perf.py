#!/usr/bin/env python3
"""Plots for the OpenClaw live run (runs/perf/). Run with SYSTEM python3 (matplotlib)."""
import os, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE=os.path.dirname(os.path.abspath(__file__)); R=os.path.join(HERE,"runs","perf"); OUT=os.path.join(HERE,"figures"); os.makedirs(OUT,exist_ok=True)

def tl(path):
    rows=collections.defaultdict(dict)
    if not os.path.exists(path): return rows
    for line in open(path):
        p=line.rstrip("\n").split(",")
        if len(p)<4 or not p[0][:1].isdigit(): continue
        try: rows[float(p[0])][p[3]]=float(p[1].replace(",","")) if p[1] not in("","<not counted>") else 0.0
        except ValueError: pass
    return rows
def counters(path):
    d={}
    if not os.path.exists(path): return d
    import re
    for line in open(path):
        m=re.match(r"\s*([\d,]+)\s+([\w\.\-]+)",line)
        if m:
            try: d[m.group(2)]=int(m.group(1).replace(",",""))
            except: pass
    return d

rows=tl(os.path.join(R,"container_timeline.csv")); ts=sorted(rows)
rel=[t-ts[0] for t in ts] if ts else []
def col(e): return [rows[t].get(e,0.0) for t in ts]
cyc,ins,slots=col("cycles"),col("instructions"),col("slots")
ret,fe,bad,be=col("topdown-retiring"),col("topdown-fe-bound"),col("topdown-bad-spec"),col("topdown-be-bound")
ipc=[(ins[i]/cyc[i]) if cyc[i] else 0 for i in range(len(ts))]

# ---- Fig 1: container activity + IPC ----
if ts:
    fig,(a1,a2)=plt.subplots(2,1,figsize=(11,6),sharex=True)
    a1.fill_between(rel,[c/1e9 for c in cyc],color="#9467bd",alpha=0.4); a1.plot(rel,[c/1e9 for c in cyc],color="#9467bd",lw=1)
    a1.set_ylabel("container CPU\n(Gcycles/s)"); a1.set_title("OpenClaw task container (agent + tools) — CPU over the run")
    a2.plot(rel,ipc,color="#d62728",lw=1); a2.set_ylabel("IPC"); a2.set_xlabel("time (s)"); a2.set_ylim(0,max(2.5,(max(ipc) if ipc else 2)*1.1))
    for a in (a1,a2): a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/oc_01_activity_ipc.png",dpi=130); plt.close(fig)

# ---- Fig 2: TMA comparison — OUTSIDE (container) vs DURING (vLLM) ----
def agg_tma(c):
    s=c.get("slots",0) or 1
    return [c.get("topdown-retiring",0)/s*100,c.get("topdown-fe-bound",0)/s*100,c.get("topdown-bad-spec",0)/s*100,c.get("topdown-be-bound",0)/s*100], (c.get("instructions",0)/(c.get("cycles",0) or 1))
cont={"slots":sum(slots),"topdown-retiring":sum(ret),"topdown-fe-bound":sum(fe),"topdown-bad-spec":sum(bad),"topdown-be-bound":sum(be),"cycles":sum(cyc),"instructions":sum(ins)}
v=counters(os.path.join(R,"vllm_tma.txt"))
cont_tma,cont_ipc=agg_tma(cont); v_tma,v_ipc=agg_tma(v) if v else ([0,0,0,0],0)
labs=["Retiring","Frontend-bound","Bad-spec","Backend-bound"]; cols=["#2ca02c","#1f77b4","#7f7f7f","#ff7f0e"]
fig,ax=plt.subplots(figsize=(8,6))
for i,(name,tma,ipc_) in enumerate([("OUTSIDE inference\n(agent+tools, container)",cont_tma,cont_ipc),("DURING inference\n(vLLM EngineCore)",v_tma,v_ipc)]):
    b=0
    for j,k in enumerate(labs):
        ax.bar(i,tma[j],bottom=b,color=cols[j],width=0.6,label=k if i==0 else "")
        if tma[j]>5: ax.text(i,b+tma[j]/2,f"{tma[j]:.0f}%",ha="center",va="center",color="white",fontsize=10,fontweight="bold")
        b+=tma[j]
    ax.text(i,103,f"IPC {ipc_:.2f}",ha="center",fontsize=12,fontweight="bold")
ax.set_xticks([0,1]); ax.set_xticklabels(["OUTSIDE inference\n(agent+tools)","DURING inference\n(vLLM)"]); ax.set_ylim(0,112); ax.set_ylabel("% of pipeline slots")
ax.set_title("OpenClaw CPU TMA — outside vs during inference"); ax.legend(ncol=4,loc="lower center",bbox_to_anchor=(0.5,-0.16),fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/oc_02_tma_outside_vs_during.png",dpi=130); plt.close(fig)

# ---- Fig 3: container CPU(cores) + GPU util over time ----
mk={}
if os.path.exists(os.path.join(R,"markers.txt")):
    for l in open(os.path.join(R,"markers.txt")):
        p=l.split()
        if len(p)>=2:
            try: mk[p[1]]=float(p[0])
            except: pass
ps=mk.get("perf_start")
gpu={}
if os.path.exists(os.path.join(R,"gpu_timeline.csv")) and ps:
    for l in open(os.path.join(R,"gpu_timeline.csv")):
        p=l.strip().split(",")
        if len(p)>=2:
            try: gpu[float(p[0])-ps]=float(p[1])
            except: pass
FREQ=4.6e9
fig,ax=plt.subplots(figsize=(11,4)); axb=ax.twinx()
ax.fill_between(rel,[c/FREQ for c in cyc],color="#9467bd",alpha=0.4,label="container CPU (cores)")
ax.set_ylabel("container CPU (cores)"); ax.set_xlabel("time (s)"); ax.grid(alpha=0.3); ax.legend(loc="upper right",fontsize=8)
if gpu:
    gx=sorted(gpu); axb.plot(gx,[gpu[s] for s in gx],color="#d62728",lw=1.2); axb.set_ylabel("GPU util %",color="#d62728"); axb.set_ylim(0,105); axb.tick_params(axis="y",colors="#d62728")
ax.set_title("OpenClaw — container CPU (purple) vs GPU inference util (red)")
fig.tight_layout(); fig.savefig(f"{OUT}/oc_03_cpu_vs_gpu.png",dpi=130); plt.close(fig)
# ---- Fig 4: wall-clock time donut (GPU inference vs container CPU vs idle/latency) ----
# container cycles per integer second
cont_s = {int(round(rel[i])): cyc[i] for i in range(len(ts))}
# gpu util per integer second + forward-fill across dropped sampler ticks (drift fix)
gpu_i = {int(round(s)): u for s, u in gpu.items()}
gff = {}
if gpu_i:
    lo, hi, last = min(gpu_i), max(gpu_i), 0.0
    for s in range(lo, hi + 1):
        if s in gpu_i: last = gpu_i[s]
        gff[s] = last
allsec = sorted(set(cont_s) | set(gff))
BUSY = 1.5e8
inf = contb = idle = 0
for s in allsec:
    if gff.get(s, 0) > 50: inf += 1
    elif cont_s.get(s, 0) > BUSY: contb += 1
    else: idle += 1
n = len(allsec) or 1
vals = [inf, contb, idle]
labs = [f"GPU inference  {inf}s ({inf/n*100:.0f}%)",
        f"container CPU (agent+tools)  {contb}s ({contb/n*100:.0f}%)",
        f"idle / inter-turn latency  {idle}s ({idle/n*100:.0f}%)"]
cols2 = ["#d62728", "#9467bd", "#cccccc"]
keep = [i for i in range(3) if vals[i] > 0]
fig, ax = plt.subplots(figsize=(8, 5))
w = ax.pie([vals[i] for i in keep], colors=[cols2[i] for i in keep], startangle=90,
           wedgeprops=dict(width=0.42, edgecolor="white"))[0]
ax.legend(w, [labs[i] for i in keep], loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=False)
ax.text(0, 0, f"{n}s\nwall-clock", ha="center", va="center", fontsize=12, fontweight="bold")
ax.set_title("OpenClaw run — where the time goes (warmup-fenced)")
fig.tight_layout(); fig.savefig(f"{OUT}/oc_04_time_donut.png", dpi=130); plt.close(fig)
print(f"time split: GPU {inf}s / container {contb}s / idle {idle}s  (n={n})")

# ---- Fig 5: CPU core-seconds donut (container agent+tools vs vLLM serving) ----
FREQ2 = 4.6e9
cont_cs = sum(cyc) / FREQ2  # container core-seconds (from cycles)
vllm_cs = 0.0
vcsv = os.path.join(R, "vllm_timeline.csv")
if os.path.exists(vcsv):
    for line in open(vcsv):
        p = line.strip().split(",")
        if len(p) >= 2:
            try: vllm_cs += float(p[1])   # cores per ~1s sample = core-seconds
            except ValueError: pass
tot = cont_cs + vllm_cs
if tot > 0:
    vals = [vllm_cs, cont_cs]
    labs = [f"vLLM serving (during-inf)  {vllm_cs:.0f} core-s ({vllm_cs/tot*100:.0f}%)",
            f"agent+tools (outside-inf)  {cont_cs:.0f} core-s ({cont_cs/tot*100:.0f}%)"]
    colsz = ["#9467bd", "#1f77b4"]
    keep = [i for i in range(2) if vals[i] > 0.05]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = ax.pie([vals[i] for i in keep], colors=[colsz[i] for i in keep], startangle=90,
               wedgeprops=dict(width=0.42, edgecolor="white"))[0]
    ax.legend(w, [labs[i] for i in keep], loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=False)
    ax.text(0, 0, f"{tot:.0f}\ncore-s", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("OpenClaw — where the CPU goes (core-seconds)")
    fig.tight_layout(); fig.savefig(f"{OUT}/oc_05_cpu_breakdown_donut.png", dpi=130); plt.close(fig)
    print(f"CPU core-seconds: vLLM {vllm_cs:.0f} / container {cont_cs:.0f}")
print("wrote:", *[f for f in sorted(os.listdir(OUT)) if f.startswith("oc_")])
