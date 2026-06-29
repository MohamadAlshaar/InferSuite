#!/usr/bin/env python3
"""OpenClaw/WildClawBench — CLEAN tool-exec microarch (corrected), 2 Productivity tasks.
Uses the shared verified microarch.py (precision-correct FP+FMA, cgroup DRAM, L2 drill-down).
Data:
  calendar_scheduling -> runs/passes_calendar/  (clean, coherent: cross-group IPC spread 0.35)
  arxiv_digest        -> runs/passes_arxiv/      (heavy TMA/CACHE/MLP cluster + old FP + new TD2;
                                                  per-group IPC variable 1.55-2.34 = task is bimodal)
Run with SYSTEM python3.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "common"))
import microarch as M
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=f"{HERE}/runs/plots_oc_clean"; os.makedirs(OUT,exist_ok=True)
TASKS=[("calendar_scheduling\n(constraint-solve)","runs/passes_calendar"),
       ("social_poster_crop\n(image SIMD)","runs/passes_task10"),
       ("arxiv_digest\n(web-fetch)","runs/passes_arxiv")]
def load(d):
    g=lambda x: M.parse(f"{HERE}/{d}/group_{x}_r1.txt")
    tma,td2,ca,fp,ml=g("TMA"),g("TD2"),g("CACHE"),g("FP"),g("MLP")
    ch=M.cache_hits(ca)
    return dict(ipc=M.ipc(tma), l1=M.tma_l1(tma), l2=M.tma_l2(td2),
                avx=M.avx_pct(fp), mflop=M.flops(fp)/1e6, dram=M.dram_gb_cgroup(ca),
                l1hit=ch["l1"], l2hit=ch["l2"], l3hit=ch["l3"], miss=ch["miss"], mpki=ch["mpki"],
                mlp=M.mlp(ml), ilp=M.ilp(ml))
data=[load(d) for _,d in TASKS]; labels=[l for l,_ in TASKS]

# ===== L1 TMA comparison =====
COMP=["retiring","fe-bound","bad-spec","be-bound"]; LBL=["Retiring","Frontend-bound","Bad-spec","Backend-bound"]; COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]
N=len(data); X=list(range(N))
fig,ax=plt.subplots(figsize=(2.4*N,5)); bot=[0]*N
for k,lab,col in zip(COMP,LBL,COL):
    vals=[d["l1"][k] for d in data]; ax.bar(X,vals,bottom=bot,label=lab,color=col,width=0.55); bot=[bot[i]+vals[i] for i in range(N)]
for i,d in enumerate(data): ax.text(i,bot[i]+1.5,f"IPC {d['ipc']:.2f}",ha="center",fontweight="bold",fontsize=9)
ax.set_xticks(X); ax.set_xticklabels(labels,fontsize=8); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("OpenClaw tool-exec Top-down L1 (clean, corrected)")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=4,fontsize=7.5,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma_l1.png",dpi=140); plt.close(fig)

# ===== L2 TMA drill-down comparison =====
L2SEG=[("light_ops","Light-ops","#98df8a"),("heavy_ops","Heavy-ops","#2ca02c"),
       ("fetch_lat","Fetch-latency","#1f77b4"),("fetch_bw","Fetch-bandwidth","#aec7e8"),
       ("br_mispred","Branch-mispred","#d62728"),("machine_clears","Machine-clears","#ff9896"),
       ("mem_bound","Memory-bound","#ff7f0e"),("core_bound","Core-bound","#ffbb78")]
fig,ax=plt.subplots(figsize=(2.4*N,5.2)); bot=[0]*N
for key,lab,col in L2SEG:
    vals=[d["l2"][key] for d in data]; ax.bar(X,vals,bottom=bot,label=lab,color=col,width=0.55); bot=[bot[i]+vals[i] for i in range(N)]
ax.set_xticks(X); ax.set_xticklabels(labels,fontsize=8); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("OpenClaw tool-exec Top-down L2 (clean)\nbottleneck attribution within each L1 category")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.10),ncol=4,fontsize=7,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma_l2.png",dpi=140); plt.close(fig)

# ===== microarch table =====
cols=["task","IPC","L1hit%","L2hit%","L3hit%","L3miss%","LLC-MPKI","MFLOP","AVX%","MLP","ILP","DRAM_GB*"]
rows=[[l.replace("\n"," "),f"{d['ipc']:.2f}",f"{d['l1hit']:.1f}",f"{d['l2hit']:.2f}",f"{d['l3hit']:.2f}",f"{d['miss']:.2f}",
       f"{d['mpki']:.2f}",f"{d['mflop']:.0f}",f"{d['avx']:.0f}",f"{d['mlp']:.2f}",f"{d['ilp']:.2f}",f"{d['dram']:.2f}"] for l,d in zip(labels,data)]
fig,ax=plt.subplots(figsize=(13,2.1)); ax.axis("off")
T=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center"); T.auto_set_font_size(False); T.set_fontsize(9); T.scale(1,1.9)
for j in range(len(cols)): T[0,j].set_facecolor("#2c3e50"); T[0,j].set_text_props(color="w",fontweight="bold")
ax.set_title("OpenClaw CLEAN tool-exec microarch (corrected FP+FMA, cgroup DRAM) — calendar vs arxiv_digest",pad=10)
ax.text(0.5,-0.32,"*DRAM = cgroup L3-miss×64 (read-miss lower bound). arxiv per-group IPC variable (1.55-2.34, bimodal task); calendar coherent (spread 0.35).",
        ha="center",transform=ax.transAxes,fontsize=7.5,style="italic",color="#555")
fig.tight_layout(); fig.savefig(f"{OUT}/microarch_table.png",dpi=140); plt.close(fig)

print("WROTE ->",OUT)
for l,d in zip(labels,data):
    print(f"  {l.replace(chr(10),' '):34s} IPC {d['ipc']:.2f} | AVX {d['avx']:.0f}% MFLOP {d['mflop']:.0f} | DRAM {d['dram']:.2f}GB | L2 BE mem{d['l2']['mem_bound']:.0f}/core{d['l2']['core_bound']:.0f}")
