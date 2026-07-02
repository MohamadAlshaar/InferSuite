#!/usr/bin/env python3
"""Thesis-style per-task SWE-agent figures for the H100 co-located run (SYSTEM python3).
Puts the 3 SWE-bench tasks (astropy/scikit-learn/sympy) SIDE BY SIDE in the house style of
agentic/thesis_figures.py:
  - swe_time_allocation.png : per-task CPU(tool)-vs-GPU(inference) donuts (fig-06 style)
  - swe_signature_heatmap.png : tasks-as-rows micro-arch heatmap of tool-exec (fig-04 style)
  - swe_twoview_heatmap.png   : engine (DURING inference) + 3 tool rows -> during-invariant vs task-varying
Reads tool-exec CSVs h100/data_swe/tool_<inst>/tool_{core,fp,mem,stall}.csv, the SWE-during engine
timelines h100/data_swe/swe_during_{group}/, and infer_times.json (per-task GPU wall-time from the
forced-decode replay). Tool CPU wall-time = sum of trajectory `execution_time` (whole agent loop)."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_orchestration as P
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------- house style (copied from agentic/thesis_figures.py) ----------
plt.rcParams.update({
    "font.family":"serif", "font.serif":["DejaVu Serif"], "mathtext.fontset":"dejavuserif",
    "font.size":12, "axes.titlesize":13, "axes.labelsize":12, "xtick.labelsize":10.5, "ytick.labelsize":10.5,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.color":"#cccccc", "grid.linewidth":0.5, "grid.alpha":0.6,
    "axes.axisbelow":True, "legend.frameon":False, "legend.fontsize":10,
    "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})
SB_COL="#0072B2"          # SWE-bench task label colour (Okabe-Ito blue)
GPU_COL="#6a51a3"; CPU_COL="#1b9e77"

# the 3 tasks (dir suffix, pretty label). SAME tasks as the CANONICAL run.
TASKS=[("astropy__astropy-14096","astropy"),
       ("scikit-learn__scikit-learn-25232","scikit-learn"),
       ("sympy__sympy-14248","sympy")]
# whole-agent-loop tool CPU wall-time = sum of trajectory `execution_time` (measured, live H100 run)
TOOL_S={"astropy":29.0, "scikit-learn":144.3, "sympy":28.6}

ROOT=sys.argv[1] if len(sys.argv)>1 else "h100/data_swe"
OUT="h100/plots/swe"; os.makedirs(OUT,exist_ok=True)

def tool_metrics(inst):
    t={}
    for g in ["core","fp","mem","stall"]:
        t.update(P.parse_toolexec_csv(os.path.join(ROOT,f"tool_{inst}",f"tool_{g}.csv"))
                 if hasattr(P,"parse_toolexec_csv") else _parse(os.path.join(ROOT,f"tool_{inst}",f"tool_{g}.csv")))
    return P.full_metrics(t)
def _parse(path):
    t={}
    if not os.path.exists(path): return t
    for ln in open(path):
        if ln.startswith("#") or not ln.strip(): continue
        c=ln.split(",")
        if len(c)<3: continue
        v,e=c[0].strip(),c[2].strip()
        if not e or v in ("<not counted>","<not supported>",""): continue
        try: t[e]=t.get(e,0.0)+float(v)
        except ValueError: pass
    return t
def during_metrics():
    t={}
    for g,fn in [("core","engine_timeline.csv"),("fp","engine_fp_timeline.csv"),
                 ("mem","engine_mem_timeline.csv"),("stall","engine_stall_timeline.csv")]:
        t.update(P.parse_timeline(os.path.join(ROOT,f"swe_during_{g}",fn)))
    return P.full_metrics(t)

tool={lab:tool_metrics(inst) for inst,lab in TASKS}
during=during_metrics()

# ================= per-task CPU-vs-GPU time allocation (fig-06 style) =================
def fig_time_allocation():
    infp=os.path.join(os.path.dirname(ROOT),"..","infer_times.json")
    infp=infp if os.path.exists(infp) else "h100/data_swe/infer_times.json"
    if not os.path.exists(infp):
        print("[time] infer_times.json missing -> skipping time-allocation fig"); return
    infer=json.load(open(infp))
    fig,axes=plt.subplots(1,3,figsize=(11.5,4.6))
    for ax,(inst,lab) in zip(axes, TASKS):
        gpu=float(infer.get(lab,{}).get("infer_s",0)); cpu=TOOL_S[lab]; tot=gpu+cpu
        if tot<=0: ax.axis("off"); continue
        toolpct=100*cpu/tot; infpct=100-toolpct
        ax.pie([infpct,toolpct], colors=[GPU_COL,CPU_COL], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
               autopct=lambda p:f"{p:.0f}%", pctdistance=0.78,
               textprops=dict(fontsize=11, fontweight="bold", color="white"))
        ax.text(0,0.08,f"{lab} (SB)",ha="center",va="center",fontsize=10,fontweight="bold",color=SB_COL)
        ax.text(0,-0.14,f"{tot:.0f}s total",ha="center",va="center",fontsize=8.7,color="#666")
    fig.legend(handles=[Patch(color=GPU_COL,label="Inference  (GPU — LLM generation)"),
                        Patch(color=CPU_COL,label="Tool execution  (CPU — agent)")],
               loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.02), fontsize=11)
    fig.suptitle("SWE-agent time allocation: inference (GPU) vs tool execution (CPU)", fontsize=14, y=1.02)
    fig.text(0.5,0.95,"Per SWE-bench task on the co-located H100 (Coder-32B). Numerically-heavy tool work (scikit-learn) shifts time onto the CPU.",
             ha="center",fontsize=9.3,style="italic",color="#555")
    fig.tight_layout(rect=[0,0.06,1,0.92])
    fig.savefig(f"{OUT}/swe_time_allocation.png"); plt.close(fig)
    print("time-allocation ->", {lab:(round(float(infer.get(lab,{}).get('infer_s',0)),1),TOOL_S[lab]) for _,lab in TASKS})

# ================= tasks-as-rows micro-arch heatmap (fig-04 style) =================
METRICS=[("IPC","IPC"),("L1 hit %","L1_pct"),("L2 hit %","L2_pct"),("L3 hit %","L3_pct"),
         ("cache-MPKI","cache_MPKI"),("AMAT (cyc)","AMAT_cyc"),("MLP","MLP"),
         ("Vectorized %","vec_pct"),("MFLOP","MFLOP")]
def heatmap(rows, fname, title, subtitle=None):
    names=[r[0] for r in rows]; mets=[r[1] for r in rows]; rowcols=[r[2] for r in rows]
    Mtx=np.array([[m.get(k,0.0) for _,k in METRICS] for m in mets], dtype=float)
    Norm=(Mtx-Mtx.min(0))/(np.ptp(Mtx,0)+1e-9)
    fig,ax=plt.subplots(figsize=(10.5, 1.15*len(rows)+2.4))
    im=ax.imshow(Norm, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([n for n,_ in METRICS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names)
    for tk,c in zip(ax.get_yticklabels(),rowcols): tk.set_color(c)
    for i in range(len(rows)):
        for j in range(len(METRICS)):
            v=Mtx[i,j]; txt=f"{v:.2f}" if abs(v)<10 else f"{v:.0f}"
            ax.text(j,i,txt,ha="center",va="center",fontsize=8.7,
                    color="black" if Norm[i,j]<0.6 else "white")
    ax.grid(False); fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02,label="per-column min–max (relative)")
    top=1-1.4/(1.15*len(rows)+2.4)   # reserve ~1.4in headroom for title+subtitle regardless of #rows
    fig.suptitle(title, fontsize=14, y=0.985)
    if subtitle: fig.text(0.5, top+0.035, subtitle, ha="center", fontsize=9.3, style="italic", color="#555")
    fig.tight_layout(rect=[0,0,1, top])
    fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

if __name__=="__main__":
    print("tool-exec metrics:")
    for lab in ["astropy","scikit-learn","sympy"]:
        m=tool[lab]; print(" ",lab,{k:round(m.get(k,0),2) for k in ["IPC","L1_pct","cache_MPKI","AMAT_cyc","MLP","vec_pct","MFLOP"]})
    print("during-inference engine:",{k:round(during.get(k,0),2) for k in ["IPC","L1_pct","cache_MPKI","AMAT_cyc","MLP","vec_pct","MFLOP"]})
    # 1. per-task tool-exec heatmap (tasks side by side)
    heatmap([(lab, tool[lab], SB_COL) for _,lab in TASKS],
            "swe_signature_heatmap.png",
            "SWE-agent tool-execution micro-arch signature (per SWE-bench task)",
            "OUTSIDE inference — the CPU signature is strongly task-dependent (scikit-learn = AVX-512 BLAS; sympy = scalar symbolic).")
    # 2. two-view: engine (during) + the 3 tool rows -> during invariant vs tool varying
    heatmap([("engine · DURING inf", during, GPU_COL)]+[(f"{lab} · tool", tool[lab], SB_COL) for _,lab in TASKS],
            "swe_twoview_heatmap.png",
            "SWE-agent CPU: DURING inference vs OUTSIDE (tool execution)",
            "DURING-inference orchestration is one invariant signature; OUTSIDE, each task's tool work has its own.")
    # 3. per-task time donuts
    fig_time_allocation()
    print("figs ->", OUT)
