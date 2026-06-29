#!/usr/bin/env python3
"""OpenClaw / WildClawBench single-task plots (Productivity task_6_calendar via Claude Sonnet 4.6).
Microarch = ONE live multiplexed cgroup perf pass (coarser than SWE-bench/BCB clean replays; OpenClaw's
non-deterministic long-horizon run isn't deterministically replayable). Run with SYSTEM python3."""
import os, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,"runs","plots_oc"); os.makedirs(OUT,exist_ok=True)
FREQ=4.6e9
# parse the live cgroup timeline (interval,value,,event,...)
rows=collections.defaultdict(dict)
for line in open(f"{HERE}/runs/perf/container_timeline.csv"):
    q=line.rstrip().split(",")
    if len(q)<4 or not q[0][:1].isdigit(): continue
    try: rows[q[0]][q[3]]=float(q[1]) if q[1] not in ("","<not counted>") else 0.0
    except: pass
ts=sorted(rows,key=float); agg=collections.Counter()
for t in ts:
    for ev,v in rows[t].items(): agg[ev]+=v
cyc=[rows[t].get("cycles",0) for t in ts]
total_s=len(ts); tool_s=sum(1 for c in cyc if c>FREQ*0.05); infer_s=total_s-tool_s
sl=agg["slots"]or 1; cy=agg["cycles"]or 1; ins=agg["instructions"]
ipc=ins/cy
tma=[agg["topdown-"+k]/sl*100 for k in ("retiring","fe-bound","bad-spec","be-bound")]
l1=agg["mem_load_retired.l1_hit"];tot=l1+agg["mem_load_retired.l2_hit"]+agg["mem_load_retired.l3_hit"]+agg["mem_load_retired.l3_miss"]
def fp(k): return agg.get("fp_arith_inst_retired."+k,0)
s=fp("scalar_double")+fp("scalar_single"); p2=fp("256b_packed_double")+fp("256b_packed_single"); p5=fp("512b_packed_double")+fp("512b_packed_single")
fl=s+p2*4+p5*8; avx=(p2*4+p5*8)/fl*100 if fl else 0
# score
sc=json.load(open(f"{HERE}/external/WildClawBench/output/openclaw/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling/"+
    sorted(os.listdir(f"{HERE}/external/WildClawBench/output/openclaw/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling"))[-1]+"/score.json"))
overall=sc.get("overall_score",0); passed=sum(1 for k,v in sc.items() if k!="overall_score" and v==1.0); ntot=len([k for k in sc if k!="overall_score"])

# ===== time split donut =====
fig,ax=plt.subplots(figsize=(5.2,4.8)); tt=total_s or 1
ax.pie([infer_s,tool_s],colors=["#9467bd","#1f77b4"],startangle=90,wedgeprops=dict(width=0.42),
       autopct=lambda p:f"{p:.0f}%",pctdistance=0.78,textprops=dict(fontsize=11,fontweight="bold"))
ax.text(0,0,f"{tt:.0f}s\nwall",ha="center",va="center",fontsize=10,fontweight="bold")
ax.legend(handles=[Patch(color="#9467bd",label=f"in-agent / inference (LLM): {infer_s}s"),
                   Patch(color="#1f77b4",label=f"outside / tool-exec (CPU): {tool_s}s")],
          loc="lower center",bbox_to_anchor=(0.5,-0.2),fontsize=9,frameon=False)
ax.set_title("OpenClaw/WildClawBench time split (calendar task)\nin-agent inference vs outside tool-exec")
fig.tight_layout(); fig.savefig(f"{OUT}/time_split_donut.png",dpi=140); plt.close(fig)

# ===== TMA bar =====
fig,ax=plt.subplots(figsize=(4.6,4.8)); COMP=["Retiring","Frontend-bound","Bad-spec","Backend-bound"];COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]
bot=0
for v,lab,col in zip(tma,COMP,COL): ax.bar(0,v,bottom=bot,color=col,label=lab,width=0.5); bot+=v
ax.text(0,bot+1,f"IPC {ipc:.2f}",ha="center",fontweight="bold",fontsize=10)
ax.set_xticks([0]); ax.set_xticklabels(["calendar\ntool-exec"]); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("Tool-exec Top-down (TMA)\n(single multiplexed live pass)")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.08),ncol=2,fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma.png",dpi=140); plt.close(fig)

# ===== summary table =====
cols=["task","score","sub-checks","agent_s","tool%","infer%","IPC","L1hit%","MFLOP","AVX%"]
row=["calendar_scheduling",f"{overall:.2f}",f"{passed}/{ntot}",f"{total_s}",f"{tool_s/tt*100:.0f}",f"{infer_s/tt*100:.0f}",
     f"{ipc:.2f}",f"{l1/(tot or 1)*100:.1f}",f"{fl/1e6:.1f}",f"{avx:.0f}"]
fig,ax=plt.subplots(figsize=(11,1.7)); ax.axis("off")
T=ax.table(cellText=[row],colLabels=cols,cellLoc="center",loc="center"); T.auto_set_font_size(False); T.set_fontsize(9); T.scale(1,1.8)
for j in range(len(cols)): T[0,j].set_facecolor("#2c3e50"); T[0,j].set_text_props(color="w",fontweight="bold")
ax.set_title("OpenClaw/WildClawBench task_6 — Claude Sonnet 4.6 (microarch = single multiplexed live pass)",pad=10)
fig.tight_layout(); fig.savefig(f"{OUT}/summary_table.png",dpi=140); plt.close(fig)

print("WROTE ->",OUT); [print("  ",x) for x in sorted(os.listdir(OUT))]
print(f"\nscore {overall:.2f} ({passed}/{ntot} sub-checks) | time {total_s}s: tool {tool_s}s ({tool_s/tt*100:.0f}%) / inference {infer_s}s ({infer_s/tt*100:.0f}%)")
print(f"IPC {ipc:.2f} | TMA ret{tma[0]:.0f}/fe{tma[1]:.0f}/bad{tma[2]:.0f}/be{tma[3]:.0f} | L1 {l1/(tot or 1)*100:.1f}% | {fl/1e6:.1f}MFLOP AVX{avx:.0f}%")
