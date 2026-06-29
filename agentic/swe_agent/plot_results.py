#!/usr/bin/env python3
"""SWE-bench agentic CPU characterization plots (local Qwen2.5-Coder-7B, SWE-bench Verified).
Data: runs/live/<inst>/ (live during/outside-inference probes) + runs/replay_<inst>/ (per-group microarch).
Run with SYSTEM python3 (matplotlib not in .venv):  python3 plot_results.py
"""
import os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "plots"); os.makedirs(OUT, exist_ok=True)
FREQ = 4.6e9
# instances with substantial live work (used for during/outside aggregation)
LIVE = {"django-10880":"django__django-10880","django-10973":"django__django-10973",
        "scikit-learn-25232":"scikit-learn__scikit-learn-25232","xarray-6744":"pydata__xarray-6744"}
# all replayed instances (microarch table), with class
REPLAY = [("django-10880","light"),("django-10973","light"),("django-10999","light"),
          ("astropy-14096","heavy"),("matplotlib-24627","heavy"),
          ("scikit-learn-25232","heavy"),("xarray-6744","heavy")]
COMP=["retiring","fe-bound","bad-spec","be-bound"]; LBL=["Retiring","Frontend-bound","Bad-spec","Backend-bound"]
COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]

def parse_timeline(path):
    agg=collections.Counter()
    if not os.path.exists(path): return agg
    for line in open(path):
        q=line.rstrip("\n").split(",")
        if len(q)<4 or not q[0][:1].isdigit(): continue
        ev=q[3]
        try: agg[ev]+= float(q[1]) if q[1] not in ("","<not counted>") else 0.0
        except ValueError: pass
    return agg
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
def tma_pct(a):
    sl=a.get("slots",0) or 1; cyc=a.get("cycles",0) or 1; ins=a.get("instructions",0)
    return ins/cyc, [a.get("topdown-"+k,0)/sl*100 for k in COMP]

# ---------- aggregate live during (vLLM) vs outside (tool-exec) ----------
during=collections.Counter(); outside=collections.Counter()
vllm_coreS=tool_coreS=agent_coreS=0.0; gpu_busy=[]
for nm,d in LIVE.items():
    base=f"{HERE}/runs/live/{d}"
    during+=parse_timeline(f"{base}/vllm_perf.csv")
    outside+=parse_timeline(f"{base}/sandbox_perf.csv")
    tool_coreS += parse_timeline(f"{base}/sandbox_perf.csv").get("cycles",0)/FREQ
    agent_coreS+= parse_timeline(f"{base}/agent_perf.csv").get("cycles",0)/FREQ
    # vLLM cores sampler
    p=f"{base}/vllm_cores.csv"
    if os.path.exists(p):
        for line in open(p):
            q=line.strip().split(",")
            if len(q)==2:
                try: vllm_coreS+=float(q[1])
                except: pass
    g=f"{base}/gpu_timeline.csv"
    if os.path.exists(g):
        for line in open(g):
            q=line.strip().split(",")
            if len(q)>=2:
                try: gpu_busy.append(float(q[1]))
                except: pass
ipc_d,t_d=tma_pct(during); ipc_o,t_o=tma_pct(outside)
gpu_pct=(sum(1 for u in gpu_busy if u>5)/len(gpu_busy)*100) if gpu_busy else 0

# ===== FIG 1: TMA side-by-side during vs outside inference =====
fig,ax=plt.subplots(figsize=(6,4.8))
bars=[("CPU during inference\n(vLLM engine)",t_d,ipc_d),("CPU outside inference\n(tool-exec)",t_o,ipc_o)]
x=range(len(bars)); bot=[0]*len(bars)
for ci,(c,lab,col) in enumerate(zip(COMP,LBL,COL)):
    vals=[b[1][ci] for b in bars]
    ax.bar(x,vals,bottom=bot,label=lab,color=col,width=0.5); bot=[bot[i]+vals[i] for i in range(len(bars))]
ax.set_xticks(list(x)); ax.set_xticklabels([b[0] for b in bars])
for i,b in enumerate(bars): ax.text(i,101,f"IPC {b[2]:.2f}",ha="center",fontweight="bold",fontsize=9)
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,114)
ax.set_title("Top-down (TMA): CPU during vs outside inference\nlocal Qwen2.5-Coder-7B on SWE-bench Verified")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.10),ncol=2,fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/1_tma_during_vs_outside.png",dpi=140); plt.close(fig)

# ===== FIG 2: two donuts side by side =====
fig,(a1,a2)=plt.subplots(1,2,figsize=(10,4.8))
# (a) wall-clock time GPU vs CPU
a1.pie([gpu_pct,100-gpu_pct],labels=[f"GPU active\n(inference)\n{gpu_pct:.0f}%",f"CPU-only\n{100-gpu_pct:.0f}%"],
       colors=["#9467bd","#8c8c8c"],startangle=90,wedgeprops=dict(width=0.42))
a1.set_title("Wall-clock time: GPU vs CPU")
# (b) CPU core-seconds: vLLM(during-inf) vs tool-exec vs agent  (legend: tiny slices would overlap)
vals=[vllm_coreS,tool_coreS,agent_coreS]; tot=sum(vals) or 1
cols=["#9467bd","#1f77b4","#2ca02c"]
w,_=a2.pie(vals,colors=cols,startangle=90,wedgeprops=dict(width=0.42))
a2.legend(w,[f"vLLM inference (during): {vllm_coreS:.0f} core-s  ({vllm_coreS/tot*100:.1f}%)",
             f"tool-exec (outside): {tool_coreS:.1f} core-s  ({tool_coreS/tot*100:.1f}%)",
             f"agent brain (outside): {agent_coreS:.1f} core-s  ({agent_coreS/tot*100:.1f}%)"],
          loc="lower center",bbox_to_anchor=(0.5,-0.28),fontsize=8,frameon=False)
a2.text(0,0,f"{tot:.0f}\ncore-s",ha="center",va="center",fontsize=10,fontweight="bold")
a2.set_title("CPU core-seconds split")
fig.suptitle("Where the work goes: inference dominates time AND CPU",fontsize=11)
fig.tight_layout(); fig.savefig(f"{OUT}/2_time_and_coreseconds_donuts.png",dpi=140); plt.close(fig)

# ===== FIG 3: microarch table (replays) =====
def fp_share(a):
    s=a.get("fp_arith_inst_retired.scalar_double",0)+a.get("fp_arith_inst_retired.scalar_single",0)
    p1=a.get("fp_arith_inst_retired.128b_packed_double",0)+a.get("fp_arith_inst_retired.128b_packed_single",0)
    p2=a.get("fp_arith_inst_retired.256b_packed_double",0)+a.get("fp_arith_inst_retired.256b_packed_single",0)
    p5=a.get("fp_arith_inst_retired.512b_packed_double",0)+a.get("fp_arith_inst_retired.512b_packed_single",0)
    fl=s+p1*2+p2*4+p5*8; pk=p1*2+p2*4+p5*8
    return fl,(pk/fl*100 if fl else 0)
def cache_l1(a):
    l1=a.get("mem_load_retired.l1_hit",0); tot=l1+a.get("mem_load_retired.l2_hit",0)+a.get("mem_load_retired.l3_hit",0)+a.get("mem_load_retired.l3_miss",0)
    return l1/tot*100 if tot else 0
def mlp(a): return a.get("l1d_pend_miss.pending",0)/(a.get("l1d_pend_miss.pending_cycles",0) or 1)
def ilp(a): return a.get("uops_executed.thread",0)/(a.get("cycles",0) or 1)
rows=[]
for nm,cls in REPLAY:
    d=f"{HERE}/runs/replay_{nm}"
    ipc,t=tma_pct(parse_txt(f"{d}/group_tma.txt"))
    fl,avx=fp_share(parse_txt(f"{d}/group_fp.txt"))
    l1=cache_l1(parse_txt(f"{d}/group_cache.txt"))
    m=parse_txt(f"{d}/group_mlp.txt")
    rows.append([nm,cls,f"{ipc:.2f}",f"{t[0]:.0f}",f"{t[1]:.0f}",f"{t[2]:.0f}",f"{t[3]:.0f}",
                 f"{l1:.1f}",f"{fl/1e6:.0f}",f"{avx:.1f}",f"{mlp(m):.2f}",f"{ilp(m):.2f}"])
cols=["instance","class","IPC","Ret%","FE%","BS%","BE%","L1hit%","MFLOP","AVX%","MLP","ILP"]
fig,ax=plt.subplots(figsize=(11,4.4)); ax.axis("off")
tbl=ax.table(cellText=rows,colLabels=cols,cellLoc="center",loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(8.5); tbl.scale(1,1.5)
for j in range(len(cols)): tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="w",fontweight="bold")
for i,(nm,cls) in enumerate(REPLAY,1):
    c="#eaf4ea" if cls=="light" else "#fdecea"
    for j in range(len(cols)): tbl[i,j].set_facecolor(c)
ax.set_title("Tool-exec microarchitecture (replay, per counter group) — light=django, heavy=numeric",pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/3_microarch_table.png",dpi=140); plt.close(fig)

print("WROTE ->",OUT)
for f in sorted(os.listdir(OUT)): print("  ",f)
print(f"\nDuring-inf (vLLM) TMA: IPC {ipc_d:.2f} ret{t_d[0]:.0f} fe{t_d[1]:.0f} bs{t_d[2]:.0f} be{t_d[3]:.0f}")
print(f"Outside-inf (tool)  TMA: IPC {ipc_o:.2f} ret{t_o[0]:.0f} fe{t_o[1]:.0f} bs{t_o[2]:.0f} be{t_o[3]:.0f}")
print(f"GPU active wall-clock: {gpu_pct:.0f}%")
print(f"core-s: vLLM {vllm_coreS:.0f} | tool {tool_coreS:.1f} | agent {agent_coreS:.1f}  -> vLLM {vllm_coreS/(vllm_coreS+tool_coreS+agent_coreS)*100:.1f}%")
