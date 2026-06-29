#!/usr/bin/env python3
"""OpenClaw/WildClawBench — TWO-task CLEAN (non-multiplexed) tool-exec microarch comparison.
  calendar_scheduling (Productivity, constraint-solving)  vs  arxiv_digest (Productivity, web-fetch/digest)
Both from per-group live passes (one run per counter group; OpenClaw has no replay). cgroup-scoped on the
task container = pure tool-exec (LLM runs off-box via API). Run with SYSTEM python3."""
import os, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__)); P=f"{HERE}/runs/passes"; OUT=f"{HERE}/runs/plots_oc_clean"; os.makedirs(OUT,exist_ok=True)
def parse(p):
    a=collections.Counter()
    for line in open(p):
        q=line.split()
        if len(q)>=2:
            try: v=float(q[0].replace(",",""))
            except: continue
            if q[1] not in a: a[q[1]]=v
    return a
# ---- arxiv_digest: parse fresh per-group files ----
t=parse(f"{P}/group_TMA_r1.txt"); c=parse(f"{P}/group_CACHE_r1.txt"); f=parse(f"{P}/group_FP_r1.txt")
m=parse(f"{P}/group_MLP_r1.txt"); im=parse(f"{P}/group_IMC_r1.txt")
sl=t["slots"]or 1; cy=t["cycles"]or 1
def fpv(k): return f.get("fp_arith_inst_retired."+k,0)
s=fpv("scalar_double")+fpv("scalar_single"); p1=fpv("128b_packed_double")+fpv("128b_packed_single")
p2=fpv("256b_packed_double")+fpv("256b_packed_single"); p5=fpv("512b_packed_double")+fpv("512b_packed_single")
fl=s+p1*2+p2*4+p5*8
l1=c["mem_load_retired.l1_hit"];l2=c["mem_load_retired.l2_hit"];l3=c["mem_load_retired.l3_hit"];miss=c["mem_load_retired.l3_miss"]
ctot=l1+l2+l3+miss or 1; mcy=m["cycles"]or 1
ax_score=json.load(open(sorted(__import__('glob').glob(f"{HERE}/external/WildClawBench/output/openclaw/01_Productivity_Flow/01_Productivity_Flow_task_1_arxiv_digest/claude-sonnet-4-6_*/score.json"))[-1]))
ARX=dict(task="arxiv_digest (web-fetch)", score=ax_score.get("overall_score",0), sub="—",
    ipc=t["instructions"]/cy, tma=[t["topdown-"+k]/sl*100 for k in ("retiring","fe-bound","bad-spec","be-bound")],
    l1=l1/ctot*100,l2=l2/ctot*100,l3=l3/ctot*100,miss=miss/ctot*100,
    mflop=fl/1e6, avx=(p1*2+p2*4+p5*8)/fl*100 if fl else 0,
    mlp=m["l1d_pend_miss.pending"]/(m["l1d_pend_miss.pending_cycles"]or 1), ilp=m["uops_executed.thread"]/mcy,
    dram=(im.get("uncore_cha/unc_cha_imc_reads_count.normal/",0)+im.get("uncore_cha/unc_cha_imc_writes_count.full/",0))*64/1e9)

# ---- calendar_scheduling: validated clean numbers from its earlier per-group run ----
CAL=dict(task="calendar_scheduling (constraint-solve)", score=0.00, sub="12/16",
    ipc=1.86, tma=[32,31,11,30], l1=98.9,l2=0.93,l3=0.09,miss=0.08, mflop=1.0, avx=0, mlp=1.84, ilp=1.66, dram=17.3)

tasks=[CAL,ARX]
# ===== TMA comparison (2 stacked bars) =====
fig,ax=plt.subplots(figsize=(6.2,5.2)); COMP=["Retiring","Frontend-bound","Bad-spec","Backend-bound"];COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]
X=[0,1]
for xi,T in zip(X,tasks):
    bot=0
    for v,col in zip(T["tma"],COL): ax.bar(xi,v,bottom=bot,color=col,width=0.55); bot+=v
    ax.text(xi,bot+1.5,f"IPC {T['ipc']:.2f}",ha="center",fontweight="bold",fontsize=10)
ax.set_xticks(X); ax.set_xticklabels(["calendar\n(constraint-solve)","arxiv_digest\n(web-fetch)"])
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("OpenClaw tool-exec Top-down (TMA) — CLEAN, non-multiplexed\n2 Productivity tasks via Claude Sonnet 4.6")
ax.legend([plt.Rectangle((0,0),1,1,color=c) for c in COL],COMP,loc="upper center",bbox_to_anchor=(0.5,-0.07),ncol=2,fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma_2task.png",dpi=140); plt.close(fig)

# ===== microarch comparison table (2 rows) =====
cols=["task","score","sub","IPC","L1hit%","L2hit%","L3hit%","L3miss%","MFLOP","AVX%","MLP","ILP","DRAM_GB*"]
rows=[[T["task"],f"{T['score']:.2f}",T["sub"],f"{T['ipc']:.2f}",f"{T['l1']:.1f}",f"{T['l2']:.2f}",f"{T['l3']:.2f}",
       f"{T['miss']:.2f}",f"{T['mflop']:.1f}",f"{T['avx']:.0f}",f"{T['mlp']:.2f}",f"{T['ilp']:.2f}",f"{T['dram']:.1f}"] for T in tasks]
fig,ax=plt.subplots(figsize=(13,2.1)); ax.axis("off")
Tt=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center"); Tt.auto_set_font_size(False); Tt.set_fontsize(9); Tt.scale(1,1.9)
for j in range(len(cols)): Tt[0,j].set_facecolor("#2c3e50"); Tt[0,j].set_text_props(color="w",fontweight="bold")
ax.set_title("OpenClaw CLEAN tool-exec microarchitecture — calendar vs arxiv_digest (per-group live passes; *DRAM node-wide upper bound)",pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/microarch_table_2task.png",dpi=140); plt.close(fig)

print("WROTE ->",OUT); [print("  ",x) for x in sorted(os.listdir(OUT))]
for T in tasks:
    print(f"\n{T['task']}: score {T['score']:.2f}  IPC {T['ipc']:.2f}  TMA ret{T['tma'][0]:.0f}/fe{T['tma'][1]:.0f}/bad{T['tma'][2]:.0f}/be{T['tma'][3]:.0f}")
    print(f"   L1 {T['l1']:.1f}%  {T['mflop']:.1f}MFLOP AVX{T['avx']:.0f}%  MLP {T['mlp']:.2f} ILP {T['ilp']:.2f}  DRAM {T['dram']:.1f}GB")
