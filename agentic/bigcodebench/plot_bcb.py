#!/usr/bin/env python3
"""Agentic BigCodeBench-Hard via Claude Sonnet 4.6 — analysis plots.
  loops_and_completion.png : loops (turns) per task distribution + how many completed (solved)
  time_split_donut.png     : in-agent(inference) vs outside(tool-exec) wall-clock
  toolexec_tma.png         : tool-exec Top-down (TMA) + IPC/AVX
  microarch_table.png      : other measured microarch (cache, FP/AVX, MLP, ILP, DRAM; NO TMA, NO cost)
Run with SYSTEM python3.
"""
import os, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"runs","plots_bcb"); os.makedirs(OUT,exist_ok=True)
import sys; sys.path.insert(0, os.path.join(HERE,"..","common"))
import microarch as M   # verified-correct FP/DRAM/L2 derivations (shared single source of truth)
R=json.load(open(f"{HERE}/runs/agentic_claude/results.json"))
res=R["results"]; nsolved=R["solved"]; ntask=R["n_tasks"]
w=[float(l.split()[0]) for l in open(f"{HERE}/runs/agentic_claude/wall.txt")]
total=w[1]-w[0]; tool=sum(sum(r.get("exec_times",[])) for r in res); infer=max(total-tool,0)

def parse(p):
    a=collections.Counter()
    for line in open(p):
        q=line.split()
        if len(q)>=2:
            try: v=float(q[0].replace(",",""))
            except: continue
            if q[1] not in a: a[q[1]]=v
    return a
t=parse(f"{HERE}/runs/replay_perf/group_tma.txt"); c=parse(f"{HERE}/runs/replay_perf/group_cache.txt")
f=parse(f"{HERE}/runs/replay_perf/group_fp.txt"); m=parse(f"{HERE}/runs/replay_perf/group_mlp.txt")
td2=parse(f"{HERE}/runs/replay_perf/group_td2.txt")
sl=t["slots"]or 1; cy=t["cycles"]or 1
ipc=t["instructions"]/cy
tma=[t["topdown-"+k]/sl*100 for k in ("retiring","fe-bound","bad-spec","be-bound")]
l2tma=M.tma_l2(td2)                                   # clean L2 (system-wide, box idle)
l1=c["mem_load_retired.l1_hit"];l2=c["mem_load_retired.l2_hit"];l3=c["mem_load_retired.l3_hit"];miss=c["mem_load_retired.l3_miss"]
tot=l1+l2+l3+miss or 1; ins=c["instructions"]or 1
fl=M.flops(f); avx=M.avx_pct(f)                       # CORRECT: precision lanes + FMA ×2
mlp=m["l1d_pend_miss.pending"]/(m["l1d_pend_miss.pending_cycles"]or 1); ilp=m["uops_executed.thread"]/cy
dram=M.dram_gb_cgroup(c)                              # L3-miss×64 (not node-wide IMC)

# ===== loops + completion: stacked by turns (solved/unsolved) =====
maxt=max(r["turns"] for r in res)
solv=[sum(1 for r in res if r["turns"]==k and r["solved"]) for k in range(1,maxt+1)]
unsv=[sum(1 for r in res if r["turns"]==k and not r["solved"]) for k in range(1,maxt+1)]
fig,ax=plt.subplots(figsize=(6.5,4.6)); x=range(1,maxt+1)
ax.bar(x,solv,color="#2ca02c",label="solved"); ax.bar(x,unsv,bottom=solv,color="#d62728",label="not solved (hit turn cap)")
for i,k in enumerate(x):
    tt=solv[i]+unsv[i]
    if tt: ax.text(k,tt+0.5,str(tt),ha="center",fontsize=9)
ax.set_xticks(list(x)); ax.set_xlabel("loops (agent turns to finish the task)"); ax.set_ylabel("number of tasks")
ax.set_title(f"Loops per task & completion — BCB-Hard via Claude Sonnet 4.6\nCOMPLETED (solved): {nsolved}/{ntask} = {nsolved/ntask*100:.0f}%")
ax.legend(fontsize=9,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/loops_and_completion.png",dpi=140); plt.close(fig)

# ===== time split donut =====
fig,ax=plt.subplots(figsize=(5.2,4.8)); tt=infer+tool or 1
ax.pie([infer,tool],colors=["#9467bd","#1f77b4"],startangle=90,wedgeprops=dict(width=0.42),
       autopct=lambda p:f"{p:.0f}%",pctdistance=0.78,textprops=dict(fontsize=11,fontweight="bold"))
ax.text(0,0,f"{tt:.0f}s\nwall",ha="center",va="center",fontsize=10,fontweight="bold")
ax.legend(handles=[Patch(color="#9467bd",label=f"in-agent / inference (LLM): {infer:.0f}s"),
                   Patch(color="#1f77b4",label=f"outside / tool-exec (CPU): {tool:.0f}s")],
          loc="lower center",bbox_to_anchor=(0.5,-0.2),fontsize=9,frameon=False)
ax.set_title("Agent-loop time split — BCB-Hard (123 tasks)\nin-agent inference vs outside tool-exec")
fig.tight_layout(); fig.savefig(f"{OUT}/time_split_donut.png",dpi=140); plt.close(fig)

# ===== tool-exec TMA =====
fig,ax=plt.subplots(figsize=(4.6,4.8)); COMP=["Retiring","Frontend-bound","Bad-spec","Backend-bound"]; COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]
bot=0
for v,lab,col in zip(tma,COMP,COL): ax.bar(0,v,bottom=bot,color=col,label=lab,width=0.5); bot+=v
ax.text(0,bot+1,f"IPC {ipc:.2f} | AVX {avx:.0f}%",ha="center",fontweight="bold",fontsize=10)
ax.set_xticks([0]); ax.set_xticklabels(["BCB-Hard\ntool-exec"]); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("Tool-exec Top-down (TMA)")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma.png",dpi=140); plt.close(fig)

# ===== tool-exec TMA Level-2 drill-down (clean) =====
L2SEG=[("light_ops","Light-ops","#98df8a"),("heavy_ops","Heavy-ops","#2ca02c"),
       ("fetch_lat","Fetch-latency","#1f77b4"),("fetch_bw","Fetch-bandwidth","#aec7e8"),
       ("br_mispred","Branch-mispred","#d62728"),("machine_clears","Machine-clears","#ff9896"),
       ("mem_bound","Memory-bound","#ff7f0e"),("core_bound","Core-bound","#ffbb78")]
fig,ax=plt.subplots(figsize=(4.8,5.2)); bot=0
for key,lab,col in L2SEG: ax.bar(0,l2tma[key],bottom=bot,color=col,label=lab,width=0.5); bot+=l2tma[key]
ax.text(0,bot+1.5,f"IPC {ipc:.2f}",ha="center",fontweight="bold",fontsize=10)
ax.set_xticks([0]); ax.set_xticklabels(["BCB-Hard\ntool-exec"]); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("Tool-exec Top-down Level-2 (clean)\nbottleneck attribution within each L1 category")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=7.5,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma_l2.png",dpi=140); plt.close(fig)

# ===== microarch table (no TMA, no cost) =====
cols=["workload","tasks","solved","IPC","L1hit%","L2hit%","L3hit%","L3miss%","LLC-MPKI","MFLOP","AVX%","MLP","ILP","DRAM_GB*"]
row=["BCB-Hard (agentic)",str(ntask),f"{nsolved} ({nsolved/ntask*100:.0f}%)",f"{ipc:.2f}",f"{l1/tot*100:.1f}",
     f"{l2/tot*100:.2f}",f"{l3/tot*100:.2f}",f"{miss/tot*100:.2f}",f"{miss/(ins/1000):.2f}",f"{fl/1e6:.0f}",f"{avx:.0f}",f"{mlp:.2f}",f"{ilp:.2f}",f"{dram:.1f}"]
fig,ax=plt.subplots(figsize=(12,1.8)); ax.axis("off")
T=ax.table(cellText=[row],colLabels=cols,cellLoc="center",loc="center"); T.auto_set_font_size(False); T.set_fontsize(9); T.scale(1,1.8)
for j in range(len(cols)): T[0,j].set_facecolor("#2c3e50"); T[0,j].set_text_props(color="w",fontweight="bold")
ax.set_title("BCB-Hard tool-exec microarchitecture (replay) — cache, FP/AVX, MLP, ILP, DRAM  (TMA in its own plots)",pad=10)
ax.text(0.5,-0.55,"FLOPs precision-correct + FMA×2 | *DRAM = L3-miss×64 (read-miss lower bound; box-idle system-wide, not node-wide IMC)",
        ha="center",transform=ax.transAxes,fontsize=7.5,style="italic",color="#555")
fig.tight_layout(); fig.savefig(f"{OUT}/microarch_table.png",dpi=140); plt.close(fig)

print("WROTE ->",OUT); [print("  ",x) for x in sorted(os.listdir(OUT))]
print(f"\nsolved {nsolved}/{ntask} ({nsolved/ntask*100:.0f}%) | time: inference {infer:.0f}s ({infer/total*100:.0f}%) / tool {tool:.0f}s ({tool/total*100:.0f}%)")
print(f"microarch: IPC {ipc:.2f} | TMA ret{tma[0]:.0f}/fe{tma[1]:.0f}/bad{tma[2]:.0f}/be{tma[3]:.0f} | L1 {l1/tot*100:.1f}% | {fl/1e6:.0f}MFLOP AVX{avx:.0f}% | MLP {mlp:.2f} ILP {ilp:.2f} | DRAM {dram:.1f}GB")
