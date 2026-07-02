#!/usr/bin/env python3
"""Grand DURING-inference engine CPU attribution (SYSTEM python3): where the recorded (perf record,
task-clock call-graph) engine CPU time goes, per benchmark, as role donuts in the house style.
Parses each benchmark's engine perf_flat.txt into NATIVE_ROLES buckets. Mirrors bcb_02 / swe_02 but
unified across BCB / SWE / OpenClaw. Writes h100/plots/grand_during_attribution.png."""
import os, sys, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_orchestration as P
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, math
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":"serif", "font.serif":["DejaVu Serif"], "mathtext.fontset":"dejavuserif",
    "font.size":12, "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})
BENCH_COL={"SWE-bench":"#0072B2","BigCodeBench":"#D55E00","OpenClaw":"#009E73"}

def parse_flat(path):
    """perf report flat line: '  30.44%  comm  dso  [.] symbol' -> (pct, full-line-text)."""
    pairs=[]
    if not os.path.exists(path): return pairs
    for ln in open(path):
        m=re.match(r"\s*([0-9.]+)%\s+(.*)", ln)
        if not m: continue
        try: pairs.append((float(m.group(1)), m.group(2)))
        except ValueError: pass
    return pairs

# (label, bench, perf_flat path). All three use the CLEAN, py-spy-free engine record (task-clock,
# cgroup-scoped). BCB was re-captured clean (its old core record was contaminated by a concurrent py-spy
# ptrace on EngineCore, which faked "80% thread-scheduling"). Coder-32B for BCB/SWE; OpenClaw live.
SPECS=[
    ("code-gen", "BigCodeBench", "h100/data/bcb_during_record_clean/perf_flat.txt"),
    ("SWE-agent","SWE-bench",     "h100/data_swe/swe_during_core/perf_flat.txt"),
    ("OpenClaw", "OpenClaw",      "h100/data_oc/oc_during_record/perf_flat.txt"),
]
ITEMS=[]
for name,bench,path in SPECS:
    pairs=parse_flat(path)
    if not pairs: print(f"{name}: NO DATA ({path})"); continue
    roles=P._roles(pairs)
    ITEMS.append((name,bench,roles))
    top=sorted(roles.items(), key=lambda x:-x[1])[:3]
    print(f"{name:10}", " | ".join(f"{k.split(' — ')[-1]} {v:.0f}%" for k,v in top if v>1))

if ITEMS:
    colmap={n:c for n,c,_ in P.NATIVE_ROLES}; colmap["other"]=P.NEUTRAL
    n=len(ITEMS)
    fig,axes=plt.subplots(1,n,figsize=(4.6*n,5.4))
    if n==1: axes=[axes]
    used=[]
    for ax,(name,bench,roles) in zip(axes,ITEMS):
        items=sorted([(k,v) for k,v in roles.items() if v>0.5], key=lambda x:-x[1])
        vals=[v for _,v in items]; cols=[colmap.get(k,P.NEUTRAL) for k,_ in items]
        for k,_ in items:
            if k not in used: used.append(k)
        ax.pie(vals,colors=cols,startangle=90,counterclock=False,
               wedgeprops=dict(width=0.44,edgecolor="white",linewidth=1.5))
        cum=0.0
        for v in vals:
            if v>=7:
                a=math.radians(90-(cum+v/2)/sum(vals)*360)
                ax.text(0.78*math.cos(a),0.78*math.sin(a),f"{v:.0f}%",ha="center",va="center",
                        color="white",fontweight="bold",fontsize=12)
            cum+=v
        ax.text(0,0.05,name,ha="center",va="center",fontsize=11.5,fontweight="bold",color=BENCH_COL[bench])
        # headline: the dominant role
        dom=max(roles.items(),key=lambda x:x[1])
        ax.text(0,-0.14,f"{dom[0].split(' — ')[-1]} {dom[1]:.0f}%",ha="center",va="center",fontsize=8.6,color="#555")
    handles=[Patch(color=colmap.get(k,P.NEUTRAL),label=k) for k in used]
    fig.legend(handles=handles,loc="lower center",ncol=min(4,len(used)),bbox_to_anchor=(0.5,-0.02),fontsize=9.5,frameon=False)
    fig.suptitle("What the CPU orchestrates DURING inference (recorded engine, perf task-clock)",fontsize=14,y=1.0)
    fig.text(0.5,0.95,"App-invariant: ~70-90% is the host thread BUSY-WAITING on the GPU (default cudaEventSynchronize "
             "spin — libcuda + vdso clock-poll), doing no real work. Same across code-gen, code-repair, and computer-use.",
             ha="center",fontsize=9.0,style="italic",color="#555")
    fig.tight_layout(rect=[0,0.08,1,0.92])
    os.makedirs("h100/plots",exist_ok=True)
    fig.savefig("h100/plots/grand_during_attribution.png"); plt.close(fig)
    print("fig -> h100/plots/grand_during_attribution.png")
