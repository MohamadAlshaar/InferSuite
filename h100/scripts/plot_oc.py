#!/usr/bin/env python3
"""OpenClaw (WildClawBench) figures + the GRAND cross-workload tool-exec heatmap (SYSTEM python3).
OpenClaw = one container = agent + tools; OUTSIDE-inference CPU = the container cgroup (agent loop +
browser/python/pdf/image tools). Live per-group runs (non-deterministic; derived metrics are
self-consistent within each group). Reads h100/data_oc/<label>_outside/tool_{core,fp,mem,stall}.csv.
Also assembles ALL self-hosted H100 tool-exec rows (BCB + SWE + OpenClaw) into one signature heatmap,
mirroring agentic/thesis_figures/04."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_orchestration as P
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":"serif", "font.serif":["DejaVu Serif"], "mathtext.fontset":"dejavuserif",
    "font.size":12, "axes.titlesize":13, "axes.labelsize":12, "xtick.labelsize":10.5, "ytick.labelsize":10.5,
    "axes.spines.top":False, "axes.spines.right":False, "axes.grid":True, "grid.color":"#cccccc",
    "grid.linewidth":0.5, "grid.alpha":0.6, "axes.axisbelow":True, "legend.frameon":False,
    "legend.fontsize":10, "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})
# Okabe-Ito benchmark colours (match agentic/thesis_figures)
SWE_COL="#0072B2"; BCB_COL="#D55E00"; OC_COL="#009E73"; GPU_COL="#6a51a3"

def parse_csv(path):
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
def metrics_from_dir(d, groups=("core","fp","mem","stall"), fmt="tool_{}.csv"):
    t={}
    for g in groups: t.update(parse_csv(os.path.join(d, fmt.format(g))))
    return P.full_metrics(t) if t else {}

OUT="h100/plots/oc"; os.makedirs(OUT, exist_ok=True)
METRICS=[("IPC","IPC"),("L1 hit %","L1_pct"),("L2 hit %","L2_pct"),("L3 hit %","L3_pct"),
         ("cache-MPKI","cache_MPKI"),("branch-MPKI","branch_MPKI"),("AMAT (cyc)","AMAT_cyc"),
         ("MLP","MLP"),("Vectorized %","vec_pct"),("MFLOP","MFLOP")]

def heatmap(rows, fname, title, subtitle=None):
    names=[r[0] for r in rows]; mets=[r[1] for r in rows]; cols=[r[2] for r in rows]
    Mtx=np.array([[m.get(k,0.0) for _,k in METRICS] for m in mets], dtype=float)
    Norm=(Mtx-Mtx.min(0))/(np.ptp(Mtx,0)+1e-9)
    fig,ax=plt.subplots(figsize=(11.5, 1.0*len(rows)+2.4))
    im=ax.imshow(Norm, aspect="auto", cmap="YlGnBu")
    ax.set_xticks(range(len(METRICS))); ax.set_xticklabels([n for n,_ in METRICS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(names)
    for tk,c in zip(ax.get_yticklabels(),cols): tk.set_color(c)
    for i in range(len(rows)):
        for j in range(len(METRICS)):
            v=Mtx[i,j]; txt=f"{v:.2f}" if abs(v)<10 else f"{v:.0f}"
            ax.text(j,i,txt,ha="center",va="center",fontsize=8.4,color="black" if Norm[i,j]<0.6 else "white")
    ax.grid(False); fig.colorbar(im,ax=ax,fraction=0.03,pad=0.02,label="per-column min–max (relative)")
    top=1-1.4/(1.0*len(rows)+2.4)
    fig.suptitle(title, fontsize=14, y=0.985)
    if subtitle: fig.text(0.5, top+0.03, subtitle, ha="center", fontsize=9.3, style="italic", color="#555")
    fig.tight_layout(rect=[0,0,1,top]); fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

# OpenClaw tasks (label -> pretty)
OC_TASKS=[("calendar","calendar"),("pdf","pdf-digest"),("arxiv","web-digest"),("social","image-crop")]

if __name__=="__main__":
    root="h100/data_oc"
    oc={}
    for lab,pretty in OC_TASKS:
        d=os.path.join(root,f"{lab}_outside")
        m=metrics_from_dir(d)
        if m: oc[pretty]=m; print(pretty, {k:round(m.get(k,0),2) for k in ["IPC","branch_MPKI","cache_MPKI","MLP","vec_pct","MFLOP"]})
        else: print(pretty, "-> NO DATA")
    if oc:
        heatmap([(p,oc[p],OC_COL) for p in oc], "oc_signature_heatmap.png",
                "OpenClaw out-of-inference micro-arch signature (per WildClawBench task)",
                "The in-container Node.js/V8 agent runtime (branch/cache-bound, 0-FP). The 32B's actual tools are light — see grand_tool_attribution.")
    # GRAND cross-workload tool-exec heatmap (all self-hosted H100 rows)
    rows=[]
    # BCB tool-exec dirs are h100/data/tool_<g>/ (each a dir of prog_*.csv) -> aggregate
    def bcb_metrics():
        t={}
        for g in ("core","fp","mem","stall"):
            for f in glob.glob(f"h100/data/tool_{g}/*.csv"):
                for k,v in parse_csv(f).items(): t[k]=t.get(k,0.0)+v
        return P.full_metrics(t) if t else {}
    bm=bcb_metrics()
    if bm: rows.append(("code-gen · BCB", bm, BCB_COL))
    SWE=[("astropy__astropy-14096","astropy"),("scikit-learn__scikit-learn-25232","scikit-learn"),("sympy__sympy-14248","sympy")]
    for inst,pretty in SWE:
        m=metrics_from_dir(f"h100/data_swe/tool_{inst}")
        if m: rows.append((f"{pretty} · SWE", m, SWE_COL))
    # OpenClaw agent runs IN the measured container, so its OUTSIDE row = Node.js/V8 AGENT RUNTIME
    # (+ light tools), NOT a pure tool payload like BCB/SWE (whose agents are off-process). Label honestly.
    for p in oc: rows.append((f"{p} · OC agent-rt", oc[p], OC_COL))
    if rows:
        heatmap(rows, "grand_toolexec_heatmap.png",
                "Self-hosted agentic out-of-inference micro-arch (H100, Coder/Instruct-32B)",
                "BCB/SWE = pure tool payload (agent off-process): scalar-symbolic → AVX-512 BLAS. OpenClaw rows = the in-container Node.js/V8 agent runtime (branch-bound, 0-FP), not tools.")
    # OpenClaw two-view: engine DURING (cgroup-scoped, no-mux) + the 4 OUTSIDE (container) rows
    dur=metrics_from_dir("h100/data_oc/calendar_during")
    if dur and oc:
        tv=[("engine · DURING inf", dur, GPU_COL)]+[(f"{p} · OC agent-rt", oc[p], OC_COL) for p in oc]
        heatmap(tv, "oc_twoview_heatmap.png",
                "OpenClaw CPU: DURING inference vs OUTSIDE (container: agent runtime + tools)",
                "DURING = app-invariant engine orchestration (IPC 2.97). OUTSIDE = the Node.js/V8 agent runtime (branch/cache-bound, 0-FP); the 32B's actual tools are light.")
        print("during engine:", {k:round(dur.get(k,0),2) for k in ["IPC","L1_pct","cache_MPKI","MLP","vec_pct","MFLOP"]})
    print("figs ->", OUT)
