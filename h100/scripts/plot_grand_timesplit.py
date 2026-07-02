#!/usr/bin/env python3
"""Grand CPU-vs-GPU time-allocation across ALL self-hosted H100 agentic tasks (SYSTEM python3).
Mirrors agentic/thesis_figures/06 (per-task inference-GPU vs tool-CPU donuts), but from the co-located
H100 runs: BCB (markers), SWE (forced-decode replay wall vs Σ execution_time), OpenClaw (chat.jsonl
timestamp decomposition, h100/data_oc/oc_timesplit.json). Writes h100/plots/grand_timesplit.png."""
import os, sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":"serif", "font.serif":["DejaVu Serif"], "mathtext.fontset":"dejavuserif",
    "font.size":12, "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})
GPU_COL="#6a51a3"; CPU_COL="#1b9e77"
BENCH_COL={"SWE-bench":"#0072B2","BigCodeBench":"#D55E00","OpenClaw":"#009E73"}
TAG={"SWE-bench":"SB","BigCodeBench":"SBCB","OpenClaw":"OC"}

# --- assemble per-task (task, bench, tool_pct, wall_s) ---
# SWE: GPU = forced-decode replay wall (infer_times.json); CPU = Σ trajectory execution_time (measured)
SWE_GPU={"astropy":244.14,"scikit-learn":59.7,"sympy":599.6}
SWE_TOOL={"astropy":29.0,"scikit-learn":144.3,"sympy":28.6}
W=[]
for t in ["astropy","scikit-learn","sympy"]:
    tot=SWE_GPU[t]+SWE_TOOL[t]; W.append((t,"SWE-bench",100*SWE_TOOL[t]/tot,tot))
# BCB: from markers (RUN wall vs Σ toolexec pairs)
W.append(("code-gen","BigCodeBench",8.5,811.9))
# OpenClaw: from chat.jsonl decomposition
ocp=os.path.join(os.path.dirname(__file__),"..","data_oc","oc_timesplit.json")
oc=json.load(open(ocp))
for t in ["calendar","pdf-digest","web-digest","image-crop"]:
    d=oc[t]; W.append((t,"OpenClaw",d["tool_pct"],d["mean_wall_s"]))

# order: row1 = the 4 code tasks (SWE x3 + BCB), row2 = the 4 OpenClaw tasks
order=[0,1,2,3,4,5,6,7]
fig,axes=plt.subplots(2,4,figsize=(14,7.4))
for ax,i in zip(axes.flat,order):
    task,bench,tool,wall=W[i]; inf=100-tool
    ax.pie([inf,tool], colors=[GPU_COL,CPU_COL], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5),
           autopct=lambda p:(f"{p:.0f}%" if p>=6 else ""), pctdistance=0.78,
           textprops=dict(fontsize=11, fontweight="bold", color="white"))
    ax.text(0,0.08,f"{task} ({TAG[bench]})",ha="center",va="center",fontsize=10.5,
            fontweight="bold",color=BENCH_COL[bench])
    ax.text(0,-0.16,f"{wall:.0f}s total",ha="center",va="center",fontsize=8.7,color="#666")
    # explicit exact split below the ring (uniform readout — esp. for the tiny-CPU OpenClaw donuts)
    ax.text(0,-1.32,f"CPU {tool:.1f}%  ·  GPU {inf:.1f}%",ha="center",va="center",fontsize=8.8,color="#444")
fig.legend(handles=[Patch(color=GPU_COL,label="Inference  (GPU — LLM generation)"),
                    Patch(color=CPU_COL,label="Tool execution  (CPU — agent)")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.5,-0.01), fontsize=11, frameon=False)
fig.suptitle("Agentic time allocation on the co-located H100: inference (GPU) vs tool execution (CPU)",
             fontsize=14, y=0.99)
fig.text(0.5,0.945,"Self-hosted Coder/Instruct-32B. Mostly inference-dominated; only scikit-learn's "
         "AVX-512 test suite flips the loop to CPU-bound.", ha="center",fontsize=9.5,style="italic",color="#555")
fig.tight_layout(rect=[0,0.04,1,0.92])
OUT="h100/plots"; os.makedirs(OUT,exist_ok=True)
fig.savefig(f"{OUT}/grand_timesplit.png"); plt.close(fig)
for t,b,tool,wall in W: print(f"{t:14} ({TAG[b]:4}) CPU={tool:4.1f}% GPU={100-tool:4.1f}%  {wall:.0f}s")
print("fig -> h100/plots/grand_timesplit.png")
