#!/usr/bin/env python3
"""OpenClaw/WildClawBench calendar_scheduling — CLEAN (non-multiplexed) tool-exec microarch.
One live agent run per counter group (TMA/CACHE/FP/MLP/IMC), cgroup-scoped on the task container
(= pure tool-exec; the LLM runs off-box via API so the container is idle during inference).
Each group is a separate run (OpenClaw has no replay); ratios are kept within-group for consistency.
Run with SYSTEM python3."""
import os, json, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE=os.path.dirname(os.path.abspath(__file__)); P=f"{HERE}/runs/passes"; OUT=f"{HERE}/runs/plots_oc_clean"; os.makedirs(OUT,exist_ok=True)
def parse(p):
    a=collections.Counter()
    for line in open(p):
        q=line.split()
        if len(q)>=2:
            try: v=float(q[0].replace(",",""))
            except: continue
            if q[1] not in a: a[q[1]]=v
    return a
t=parse(f"{P}/group_TMA_r1.txt"); c=parse(f"{P}/group_CACHE_r1.txt"); f=parse(f"{P}/group_FP_r1.txt")
m=parse(f"{P}/group_MLP_r1.txt"); im=parse(f"{P}/group_IMC_r1.txt")
sl=t["slots"]or 1; cy=t["cycles"]or 1
ipc=t["instructions"]/cy
tma=[t["topdown-"+k]/sl*100 for k in ("retiring","fe-bound","bad-spec","be-bound")]
l1=c["mem_load_retired.l1_hit"];l2=c["mem_load_retired.l2_hit"];l3=c["mem_load_retired.l3_hit"];miss=c["mem_load_retired.l3_miss"]
tot=l1+l2+l3+miss or 1; cins=c["instructions"]or 1
def fpv(k): return f.get("fp_arith_inst_retired."+k,0)
s=fpv("scalar_double")+fpv("scalar_single"); p1=fpv("128b_packed_double")+fpv("128b_packed_single")
p2=fpv("256b_packed_double")+fpv("256b_packed_single"); p5=fpv("512b_packed_double")+fpv("512b_packed_single")
fl=s+p1*2+p2*4+p5*8; avx=(p1*2+p2*4+p5*8)/fl*100 if fl else 0
mcy=m["cycles"]or 1
mlp=m["l1d_pend_miss.pending"]/(m["l1d_pend_miss.pending_cycles"]or 1); ilp=m["uops_executed.thread"]/mcy
dram=(im.get("uncore_cha/unc_cha_imc_reads_count.normal/",0)+im.get("uncore_cha/unc_cha_imc_writes_count.full/",0))*64/1e9

# score (hard-gated; count passed sub-checks)
rd=sorted([d for d in os.listdir(f"{HERE}/external/WildClawBench/output/openclaw/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling")])[-1]
sc=json.load(open(f"{HERE}/external/WildClawBench/output/openclaw/01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling/{rd}/score.json"))
overall=sc.get("overall_score",0); passed=sum(1 for k,v in sc.items() if k!="overall_score" and v==1.0); ntot=len([k for k in sc if k!="overall_score"])

# ===== TMA bar (clean) =====
fig,ax=plt.subplots(figsize=(4.8,5.0)); COMP=["Retiring","Frontend-bound","Bad-spec","Backend-bound"];COL=["#2ca02c","#1f77b4","#d62728","#ff7f0e"]
bot=0
for v,lab,col in zip(tma,COMP,COL): ax.bar(0,v,bottom=bot,color=col,label=f"{lab} {v:.0f}%",width=0.5); bot+=v
ax.text(0,bot+1.5,f"IPC {ipc:.2f}",ha="center",fontweight="bold",fontsize=11)
ax.set_xticks([0]); ax.set_xticklabels(["calendar\ntool-exec"]); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("OpenClaw calendar_scheduling — tool-exec Top-down\n(CLEAN: one live run per counter group, non-multiplexed)")
ax.legend(loc="upper center",bbox_to_anchor=(0.5,-0.07),ncol=2,fontsize=8,frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/toolexec_tma_clean.png",dpi=140); plt.close(fig)

# ===== microarch table (clean) =====
cols=["task","score","sub-checks","IPC","L1hit%","L2hit%","L3hit%","L3miss%","MFLOP","AVX%","MLP","ILP","DRAM_GB*"]
row=["calendar_scheduling",f"{overall:.2f}",f"{passed}/{ntot}",f"{ipc:.2f}",f"{l1/tot*100:.1f}",
     f"{l2/tot*100:.2f}",f"{l3/tot*100:.2f}",f"{miss/tot*100:.2f}",f"{fl/1e6:.1f}",f"{avx:.0f}",f"{mlp:.2f}",f"{ilp:.2f}",f"{dram:.1f}"]
fig,ax=plt.subplots(figsize=(12,1.9)); ax.axis("off")
T=ax.table(cellText=[row],colLabels=cols,cellLoc="center",loc="center"); T.auto_set_font_size(False); T.set_fontsize(9); T.scale(1,1.9)
for j in range(len(cols)): T[0,j].set_facecolor("#2c3e50"); T[0,j].set_text_props(color="w",fontweight="bold")
ax.set_title("OpenClaw calendar_scheduling — CLEAN tool-exec microarchitecture (per-group live passes; *DRAM node-wide upper bound)",pad=12)
fig.tight_layout(); fig.savefig(f"{OUT}/microarch_table_clean.png",dpi=140); plt.close(fig)

print("WROTE ->",OUT); [print("  ",x) for x in sorted(os.listdir(OUT))]
print(f"\ncalendar CLEAN tool-exec microarch (non-multiplexed):")
print(f"  score {overall:.2f} ({passed}/{ntot} sub-checks)")
print(f"  IPC {ipc:.2f} | TMA ret{tma[0]:.0f}/fe{tma[1]:.0f}/bad{tma[2]:.0f}/be{tma[3]:.0f}")
print(f"  L1 {l1/tot*100:.1f}% L2 {l2/tot*100:.2f}% L3 {l3/tot*100:.2f}% miss {miss/tot*100:.2f}%")
print(f"  {fl/1e6:.1f} MFLOP, AVX {avx:.0f}% | MLP {mlp:.2f} ILP {ilp:.2f} | DRAM {dram:.1f} GB (node-wide)")
