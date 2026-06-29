#!/usr/bin/env python3
"""SWE-bench HEAVY via hosted Claude Sonnet 4.6 — analysis plots.
  loops_per_task.png    : loops per task (resolved heavy instances; sympy excluded)
  time_split_donuts.png : per-instance donut, in-agent(inference) vs outside(tool-exec)
  toolexec_tma.png      : tool-exec Top-down (TMA) + AVX
  microarch_table.png   : all OTHER microarch we measured (NO TMA breakdown, NO cost)
Run with SYSTEM python3.
"""
import os, json, glob, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
import sys; sys.path.insert(0, os.path.join(HERE, "..", "common"))
import microarch as M   # verified-correct FP/DRAM/L2 derivations (shared single source of truth)
OUT = os.path.join(HERE, "runs", "plots_api"); os.makedirs(OUT, exist_ok=True)
# microarch/time/loops plots use the resolved heavy instances only (sympy excluded everywhere)
INSTS = [
    ("scikit-learn__scikit-learn-25232", "scikit-learn-25232", "sklearn-25232"),
    ("astropy__astropy-14096",           "astropy-14096",      "astropy-14096"),
]
COMP=["retiring","fe-bound","bad-spec","be-bound"]; LBL=["Retiring","Frontend-bound","Bad-spec","Backend-bound"]
COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]

def parse_txt(path):
    a=collections.Counter()
    if not os.path.exists(path): return a
    for line in open(path):
        p=line.split()
        if len(p)<2: continue
        try: v=float(p[0].replace(",",""))
        except: continue
        if p[1] not in a: a[p[1]]=v
    return a
def tma(a):
    sl=a.get("slots",0) or 1; cyc=a.get("cycles",0) or 1; ins=a.get("instructions",0)
    return ins/cyc,[a.get("topdown-"+k,0)/sl*100 for k in COMP]
def fp(a):  # CORRECT: precision-specific lanes + FMA ×2 (shared module)
    return M.flops(a), M.avx_pct(a)
def cache(a):  # returns (l1%,l2%,l3%,miss%, llc_mpki)
    l1=a.get("mem_load_retired.l1_hit",0); l2=a.get("mem_load_retired.l2_hit",0)
    l3=a.get("mem_load_retired.l3_hit",0); miss=a.get("mem_load_retired.l3_miss",0)
    tot=l1+l2+l3+miss or 1; ins=a.get("instructions",0) or 1
    return l1/tot*100,l2/tot*100,l3/tot*100,miss/tot*100, miss/(ins/1000)
def mlp(a): return a.get("l1d_pend_miss.pending",0)/(a.get("l1d_pend_miss.pending_cycles",0) or 1)
def ilp(a): return a.get("uops_executed.thread",0)/(a.get("cycles",0) or 1)

rep=glob.glob(f"{HERE}/*.api_heavy.json")
resolved=set(json.load(open(rep[0]))["resolved_ids"]) if rep else set()

rows=[]
for iid,short,label in INSTS:
    tj=glob.glob(f"{HERE}/runs/api/{iid}/**/{iid}.traj", recursive=True)
    d=json.load(open(tj[0])); tr=d["trajectory"]
    texec=sum(s.get("execution_time",0) or 0 for s in tr)
    ws=float(open(f"{HERE}/runs/api/{iid}/wall_start.txt").read()); we=float(open(f"{HERE}/runs/api/{iid}/wall_end.txt").read())
    total=we-ws; infer=max(total-texec,0)
    rd=f"{HERE}/runs/api_replay_{short}"
    ipc,t=tma(parse_txt(f"{rd}/group_tma.txt")); fl,avx=fp(parse_txt(f"{rd}/group_fp.txt"))
    cad=parse_txt(f"{rd}/group_cache.txt"); l1,l2,l3,miss,mpki=cache(cad); mm=parse_txt(f"{rd}/group_mlp.txt")
    l2tma=M.tma_l2(parse_txt(f"{rd}/group_td2.txt"))           # clean L2 (cgroup-scoped, no-multiplex)
    dram_gb=M.dram_gb_cgroup(cad)                              # cgroup L3-miss×64 (not node-wide IMC)
    rows.append(dict(label=label,resolved=(iid in resolved),loops=len(tr),texec=texec,infer=infer,total=total,
        ipc=ipc,tma=t,l2tma=l2tma,fl=fl,avx=avx,l1=l1,l2=l2,l3=l3,miss=miss,mpki=mpki,mlp=mlp(mm),ilp=ilp(mm),dram=dram_gb))

# ===== loops per task (no sympy, no cost) =====
fig,ax=plt.subplots(figsize=(5.5,4.4))
x=range(len(rows)); loops=[r["loops"] for r in rows]
ax.bar(x,loops,color="#2ca02c",width=0.5)
for i,r in enumerate(rows): ax.text(i,r["loops"]+1.5,f"{r['loops']} loops\n✓ solved",ha="center",fontsize=10)
ax.set_xticks(list(x)); ax.set_xticklabels([r["label"] for r in rows]); ax.set_ylim(0,max(loops)*1.3)
ax.set_ylabel("loops (agent turns = LLM calls)")
ax.set_title("Loops per task — Claude Sonnet 4.6 (resolved heavy instances)")
fig.tight_layout(); fig.savefig(f"{OUT}/loops_per_task.png",dpi=140); plt.close(fig)

# ===== time split: TWO donuts =====
fig,axes=plt.subplots(1,len(rows),figsize=(5.2*len(rows),4.8))
for ax,r in zip(axes,rows):
    inf=r["infer"]; tl=r["texec"]; tot=inf+tl or 1
    ax.pie([inf,tl],colors=["#9467bd","#1f77b4"],startangle=90,wedgeprops=dict(width=0.42),
           autopct=lambda p:f"{p:.0f}%",pctdistance=0.78,textprops=dict(fontsize=10,fontweight="bold"))
    ax.text(0,0,f"{r['label']}\n{tot:.0f}s",ha="center",va="center",fontsize=10,fontweight="bold")
    ax.set_title(r["label"])
fig.legend(handles=[Patch(color="#9467bd",label="in-agent / inference (LLM)"),
                    Patch(color="#1f77b4",label="outside / tool-exec (CPU)")],
           loc="lower center",ncol=2,fontsize=9,frameon=False)
fig.suptitle("Per-instance time split: in-agent (inference) vs outside (tool-exec)",fontsize=12)
fig.tight_layout(rect=[0,0.06,1,1]); fig.savefig(f"{OUT}/time_split_donuts.png",dpi=140); plt.close(fig)

# ===== tool-exec TMA (kept) =====
fig,ax=plt.subplots(figsize=(6,4.8)); bot=[0]*len(rows)
for ci,(c,lab,col) in enumerate(zip(COMP,LBL,COL)):
    vals=[r["tma"][ci] for r in rows]; ax.bar(range(len(rows)),vals,bottom=bot,label=lab,color=col,width=0.55); bot=[bot[i]+vals[i] for i in range(len(rows))]
ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r['label']}\nIPC {r['ipc']:.2f} | AVX {r['avx']:.0f}%" for r in rows],fontsize=8)
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,108)
ax.set_title("Tool-exec Top-down (TMA) — resolved heavy instances")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.13),ncol=4,fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma.png",dpi=140); plt.close(fig)

# ===== tool-exec TMA Level-2 drill-down (clean, cgroup-scoped) =====
# 8 leaves grouped by parent: Retiring(greens) FE(blues) BadSpec(reds) BE(oranges)
L2SEG=[("light_ops","Light-ops","#98df8a"),("heavy_ops","Heavy-ops","#2ca02c"),
       ("fetch_lat","Fetch-latency","#1f77b4"),("fetch_bw","Fetch-bandwidth","#aec7e8"),
       ("br_mispred","Branch-mispred","#d62728"),("machine_clears","Machine-clears","#ff9896"),
       ("mem_bound","Memory-bound","#ff7f0e"),("core_bound","Core-bound","#ffbb78")]
fig,ax=plt.subplots(figsize=(6.5,5.2)); bot=[0]*len(rows)
for key,lab,col in L2SEG:
    vals=[r["l2tma"][key] for r in rows]
    ax.bar(range(len(rows)),vals,bottom=bot,label=lab,color=col,width=0.55)
    bot=[bot[i]+vals[i] for i in range(len(rows))]
ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"{r['label']}\nIPC {r['ipc']:.2f}" for r in rows],fontsize=8)
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,108)
ax.set_title("Tool-exec Top-down Level-2 (clean, cgroup-scoped)\nbottleneck attribution within each L1 category")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.10),ncol=4,fontsize=7.5,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma_l2.png",dpi=140); plt.close(fig)

# ===== microarch table (NO TMA, NO cost) — all the other measured microarch =====
cols=["instance","resolved","loops","IPC","L1hit%","L2hit%","L3hit%","L3miss%","LLC-MPKI","MFLOP","AVX%","MLP","ILP","DRAM_GB*"]
tbl=[[r['label'],"YES" if r['resolved'] else "no",str(r['loops']),f"{r['ipc']:.2f}",f"{r['l1']:.1f}",f"{r['l2']:.2f}",
      f"{r['l3']:.2f}",f"{r['miss']:.2f}",f"{r['mpki']:.2f}",f"{r['fl']/1e6:.0f}",f"{r['avx']:.0f}",
      f"{r['mlp']:.2f}",f"{r['ilp']:.2f}",f"{r['dram']:.1f}"] for r in rows]
fig,ax=plt.subplots(figsize=(12,2.4)); ax.axis("off")
T=ax.table(cellText=tbl,colLabels=cols,cellLoc="center",loc="center"); T.auto_set_font_size(False); T.set_fontsize(8.5); T.scale(1,1.7)
for j in range(len(cols)): T[0,j].set_facecolor("#2c3e50"); T[0,j].set_text_props(color="w",fontweight="bold")
ax.set_title("Tool-exec microarchitecture (replay) — cache, FP/AVX, MLP, ILP, DRAM  (TMA in its own plots)",pad=10)
ax.text(0.5,-0.30,"FLOPs precision-correct + FMA×2 | *DRAM = cgroup L3-miss×64 (container-only, read-miss lower bound; not node-wide IMC)",
        ha="center",transform=ax.transAxes,fontsize=7.5,style="italic",color="#555")
fig.tight_layout(); fig.savefig(f"{OUT}/microarch_table.png",dpi=140); plt.close(fig)

# remove superseded files
for stale in ["loops_and_completion.png","time_split_per_instance.png","everything_table.png",
              "1_time_inference_vs_toolexec.png","2_toolexec_tma.png","3_everything_table.png"]:
    p=os.path.join(OUT,stale)
    if os.path.exists(p): os.remove(p)

print("WROTE ->",OUT); [print("  ",f) for f in sorted(os.listdir(OUT))]
for r in rows: print(f"  {r['label']:14} loops={r['loops']} IPC={r['ipc']:.2f} L1={r['l1']:.1f}% AVX={r['avx']:.0f}% MFLOP={r['fl']/1e6:.0f} MLP={r['mlp']:.2f} ILP={r['ilp']:.2f} DRAM={r['dram']:.1f}GB")
