#!/usr/bin/env python3
"""Publication-quality figures for the agentic CPU-characterization thesis chapter.
Reads the authoritative CANONICAL/ data via the shared microarch.py and produces a unified,
cross-workload figure set. No internal-methodology jargon on the figure faces (that goes in the
thesis text). Run with SYSTEM python3 -> writes to thesis_figures/.
"""
import os, sys
HERE=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE,"CANONICAL"))
import microarch as M
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT=os.path.join(HERE,"thesis_figures"); os.makedirs(OUT,exist_ok=True)

# ---------- publication style ----------
plt.rcParams.update({
    "font.family":"serif", "font.serif":["DejaVu Serif"], "mathtext.fontset":"dejavuserif",
    "font.size":12, "axes.titlesize":13, "axes.labelsize":12, "xtick.labelsize":10.5, "ytick.labelsize":10.5,
    "axes.spines.top":False, "axes.spines.right":False,
    "axes.grid":True, "grid.color":"#cccccc", "grid.linewidth":0.5, "grid.alpha":0.6,
    "axes.axisbelow":True, "legend.frameon":False, "legend.fontsize":10,
    "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})
# colourblind-safe categorical palette (Okabe-Ito) for benchmark grouping
BENCH_COL={"SWE-bench":"#0072B2","BigCodeBench":"#D55E00","OpenClaw":"#009E73"}
# TMA Level-1 palette (consistent everywhere)
TMA1=[("retiring","Retiring","#009E73"),("fe-bound","Front-end bound","#0072B2"),
      ("bad-spec","Bad speculation","#D55E00"),("be-bound","Back-end bound","#E69F00")]
# TMA Level-2 palette
TMA2=[("light_ops","Light operations","#66c2a5"),("heavy_ops","Heavy operations","#1b9e77"),
      ("fetch_lat","Fetch latency","#8da0cb"),("fetch_bw","Fetch bandwidth","#c6dbef"),
      ("br_mispred","Branch mispredict","#fc8d62"),("machine_clears","Machine clears","#fdd0a2"),
      ("mem_bound","Memory bound","#e78ac3"),("core_bound","Core bound","#f4cae4")]

# ---------- load all workloads from CANONICAL ----------
SPEC=[("astropy","SWE-bench","swe_bench/data/astropy-14096",False),
      ("scikit-learn","SWE-bench","swe_bench/data/scikit-learn-25232",False),
      ("sympy","SWE-bench","swe_bench/data/sympy-14248",False),
      ("code-gen","BigCodeBench","bigcodebench/data",False),
      ("calendar","OpenClaw","openclaw/data/calendar",True),
      ("image-crop","OpenClaw","openclaw/data/social_poster_crop",True),
      ("web-digest","OpenClaw","openclaw/data/arxiv",True),
      ("pdf-digest","OpenClaw","openclaw/data/pdf_digest",True)]
def load(d, up):
    suf="_r1" if up else ""; g=lambda x: M.parse(f"{HERE}/CANONICAL/{d}/group_{x.upper() if up else x}{suf}.txt")
    tma,td2,ca,fp,ml=g("tma"),g("td2"),g("cache"),g("fp"),g("mlp"); ch=M.cache_hits(ca)
    return dict(ipc=M.ipc(tma), l1=M.tma_l1(tma), l2=M.tma_l2(td2), avx=M.avx_pct(fp),
                flop=M.flops(fp), mflop=M.flops(fp)/1e6, dram=M.dram_gb_cgroup(ca),
                l1hit=ch["l1"], l2hit=ch["l2"], l3hit=ch["l3"], mpki=ch["mpki"], mlp=M.mlp(ml), ilp=M.ilp(ml),
                amat=(ch["l1"]*5 + ch["l2"]*15 + ch["l3"]*50 + ch["miss"]*250)/100,
                cyc=tma.get("cycles",1), l3miss=ca.get("mem_load_retired.l3_miss",0))
TAG={"SWE-bench":"SB","BigCodeBench":"SBCB","OpenClaw":"OC"}
def tagged(task,bench): return f"{task} ({TAG[bench]})"
W=[dict(task=t, bench=b, col=BENCH_COL[b], **load(d,up)) for t,b,d,up in SPEC]
labels=[tagged(w["task"],w["bench"]) for w in W]; X=np.arange(len(W))
FREQ=4.6e9

def bench_xticklabels(ax):
    ax.set_xticks(X); ax.set_xticklabels(labels, rotation=20, ha="right")
    for tk,w in zip(ax.get_xticklabels(),W): tk.set_color(w["col"])
def bench_legend(ax, loc="upper right"):
    ax.legend(handles=[Patch(color=c,label=b) for b,c in BENCH_COL.items()], loc=loc, title="Benchmark")

# ================= 1. Cross-workload pipeline breakdown (TMA L1) =================
fig,ax=plt.subplots(figsize=(9,5.2)); bot=np.zeros(len(W))
for key,lab,col in TMA1:
    v=np.array([w["l1"][key] for w in W]); ax.bar(X,v,bottom=bot,label=lab,color=col,width=0.7,edgecolor="white",linewidth=0.5); bot+=v
for i,w in enumerate(W): ax.text(i,bot[i]+1.2,(f"IPC {w['ipc']:.2f}" if i==0 else f"{w['ipc']:.2f}"),ha="center",fontsize=9.5)
ax.set_ylabel("Share of pipeline slots (%)"); ax.set_ylim(0,116)
ax.set_title("Where the CPU spends cycles during agentic tool execution", pad=14)
bench_xticklabels(ax); ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.32))
fig.savefig(f"{OUT}/01_pipeline_breakdown.png"); plt.close(fig)

# ================= 2. Bottleneck attribution (TMA L2) =================
fig,ax=plt.subplots(figsize=(9,5.4)); bot=np.zeros(len(W))
for key,lab,col in TMA2:
    v=np.array([w["l2"][key] for w in W]); ax.bar(X,v,bottom=bot,label=lab,color=col,width=0.7,edgecolor="white",linewidth=0.4); bot+=v
ax.set_ylabel("Share of pipeline slots (%)"); ax.set_ylim(0,112)
ax.set_title("Micro-architectural bottleneck attribution")
bench_xticklabels(ax); ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5,-0.34))
fig.savefig(f"{OUT}/02_bottleneck_attribution.png"); plt.close(fig)

# ================= 3. Cross-workload roofline =================
fig,ax=plt.subplots(figsize=(8,6))
xs=np.logspace(-3,2.2,300)
PEAKS=[("Scalar (double precision)",4),("128-bit SIMD",8),("256-bit SIMD (AVX2)",16),("512-bit SIMD (AVX-512)",32)]
BW=25.0
for lab,fc in PEAKS:
    pk=fc*FREQ/1e9; ax.plot(xs,np.minimum(BW*xs,pk),color="#999",lw=1)
    ax.text(xs[-1],pk*1.02,lab,ha="right",fontsize=8.5,color="#666")
ax.plot(xs,BW*xs,"--",color="#888",lw=1); ax.text(2e-3,BW*2e-3*1.15,f"DRAM bandwidth ≈ {BW:.0f} GB/s",fontsize=8.5,color="#888",rotation=34)
for w in W:
    ai=w["flop"]/(w["l3miss"]*64) if w["l3miss"] else 1e-3
    perf=w["flop"]/w["cyc"]*FREQ/1e9
    ax.scatter(ai,perf,s=90,color=w["col"],edgecolor="k",linewidth=0.6,zorder=5)
    ax.annotate(tagged(w["task"],w["bench"]),(ai,perf),textcoords="offset points",xytext=(6,4),fontsize=8.5)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-3,1.5e2); ax.set_ylim(1e-3,400)
ax.set_xlabel("Arithmetic intensity (FLOP / byte)"); ax.set_ylabel("Performance (GFLOP/s)")
ax.set_title("Operational roofline of agentic tool execution")
bench_legend(ax, loc="upper left")
fig.savefig(f"{OUT}/03_roofline.png"); plt.close(fig)

# ================= 4. Microarchitecture signature heatmap =================
metrics=[("IPC","ipc"),("L1 hit %","l1hit"),("L2 hit %","l2hit"),("L3 hit %","l3hit"),
         ("MPKI","mpki"),("AMAT (est.)","amat"),("DRAM (GB)","dram"),
         ("MLP","mlp"),("ILP","ilp"),("Vectorized %","avx")]
def mval(w,key,name):
    return w[key]
Mtx=np.array([[mval(w,k,n) for n,k in metrics] for w in W],dtype=float)
Norm=(Mtx-Mtx.min(0))/(np.ptp(Mtx,0)+1e-9)
fig,ax=plt.subplots(figsize=(11.5,5.2))
im=ax.imshow(Norm,aspect="auto",cmap="YlGnBu")
ax.set_xticks(range(len(metrics))); ax.set_xticklabels([n for n,_ in metrics],rotation=25,ha="right")
ax.set_yticks(range(len(W))); ax.set_yticklabels(labels)
for tk,w in zip(ax.get_yticklabels(),W): tk.set_color(w["col"])
for i in range(len(W)):
    for j in range(len(metrics)):
        v=Mtx[i,j]; txt=f"{v:.2f}" if v<10 else f"{v:.0f}"
        ax.text(j,i,txt,ha="center",va="center",fontsize=8.5,color="black" if Norm[i,j]<0.6 else "white")
ax.set_title("Micro-architectural signature across agentic workloads")
ax.grid(False); fig.colorbar(im,ax=ax,fraction=0.025,pad=0.02,label="per-column min–max (relative)")
fig.savefig(f"{OUT}/04_signature_heatmap.png"); plt.close(fig)

# ================= 5. Vectorization spectrum =================
order=sorted(range(len(W)),key=lambda i:W[i]["avx"])
fig,ax=plt.subplots(figsize=(8.5,4.8))
ax.bar(range(len(W)),[W[i]["avx"] for i in order],color=[W[i]["col"] for i in order],width=0.7,edgecolor="white")
for r,i in enumerate(order): ax.text(r,W[i]["avx"]+1.5,f"{W[i]['avx']:.0f}%",ha="center",fontsize=9.5)
ax.set_xticks(range(len(W))); ax.set_xticklabels([labels[i] for i in order],rotation=20,ha="right")
for tk,i in zip(ax.get_xticklabels(),order): tk.set_color(W[i]["col"])
ax.set_ylabel("Vectorized floating-point operations (%)"); ax.set_ylim(0,100)
ax.set_title("Vectorization intensity spans the full spectrum")
bench_legend(ax, loc="upper left")
fig.savefig(f"{OUT}/05_vectorization.png"); plt.close(fig)

# ================= 6. Time allocation: inference (GPU) vs tool execution (CPU) =================
import json
TS=json.load(open(f"{HERE}/CANONICAL/timesplit.json"))["workloads"]
GPU_COL="#6a51a3"; CPU_COL="#1b9e77"
ncol=4; nrow=2
fig,axes=plt.subplots(nrow,ncol,figsize=(13.5,7.2))
for ax,w in zip(axes.flat, TS):
    tool=w["tool_pct"]; inf=100-tool
    ax.pie([inf,tool], colors=[GPU_COL,CPU_COL], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
           autopct=lambda p:f"{p:.0f}%", pctdistance=0.78,
           textprops=dict(fontsize=11, fontweight="bold", color="white"))
    ax.text(0,0.06,tagged(w["task"],w["bench"]),ha="center",va="center",fontsize=9.3,fontweight="bold",color=BENCH_COL[w["bench"]])
    ax.text(0,-0.16,f"{w['wall_s']:.0f}s total",ha="center",va="center",fontsize=8.5,color="#666")
for ax in axes.flat[len(TS):]: ax.axis("off")
fig.legend(handles=[Patch(color=GPU_COL,label="Inference  (GPU — LLM generation)"),
                    Patch(color=CPU_COL,label="Tool execution  (CPU — agent)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5,0.0), fontsize=11)
fig.suptitle("Agent-loop time allocation: inference (GPU) vs tool execution (CPU)", fontsize=14, y=0.98)
fig.text(0.5,0.925,"Agentic workloads are inference-dominated; compute-heavy tool work (numerical tests, image processing) raises the CPU share.",
         ha="center",fontsize=9.5,style="italic",color="#555")
fig.tight_layout(rect=[0,0.05,1,0.91])
fig.savefig(f"{OUT}/06_time_allocation.png"); plt.close(fig)

# ================= 7 & 8. Agent behaviour: prompt sizes + loops (per-agent distributions) =================
AB=json.load(open(f"{HERE}/CANONICAL/agent_behavior.json"))
AORDER=["SWE-bench","BigCodeBench","OpenClaw"]
ALAB=[f"{a}\n({TAG[a]})" for a in AORDER]
def boxfig(field, ylabel, title, fname, logy=False, fmt="{:.0f}"):
    fig,ax=plt.subplots(figsize=(7,5))
    data=[AB[field][a] for a in AORDER]
    bp=ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                  medianprops=dict(color="black",lw=1.6), whiskerprops=dict(color="#555"), capprops=dict(color="#555"))
    for patch,a in zip(bp["boxes"],AORDER): patch.set_facecolor(BENCH_COL[a]); patch.set_alpha(0.85); patch.set_edgecolor("#333")
    import numpy as _np
    for i,a in enumerate(AORDER):
        med=_np.median(AB[field][a])
        ax.text(i+1.34, med, "median "+fmt.format(med), ha="left", va="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#333", linewidth=0.8, alpha=0.95))
    if logy: ax.set_yscale("log")
    ax.set_xlim(0.4, len(AORDER)+1.25)
    ax.set_xticks(range(1,len(AORDER)+1)); ax.set_xticklabels(ALAB)
    for tk,a in zip(ax.get_xticklabels(),AORDER): tk.set_color(BENCH_COL[a])
    ax.set_ylabel(ylabel); ax.set_title(title)
    fig.savefig(f"{OUT}/{fname}"); plt.close(fig)

boxfig("prompt_tokens","Input context (tokens per LLM call)",
       "Agentic prompts are large and context-heavy",
       "07_prompt_sizes.png", logy=True, fmt="{:.0f} tok")
boxfig("loops","Agent loops (LLM calls per task)",
       "Agent iteration depth varies widely by task type",
       "08_loops.png", logy=False, fmt="{:.0f}")

print("WROTE ->",OUT); [print("  ",f) for f in sorted(os.listdir(OUT))]
