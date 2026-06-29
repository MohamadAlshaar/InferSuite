#!/usr/bin/env python3
"""SWE-bench agentic tool-exec CPU plots — astropy Verified rerun (2026-06-25).
Provenance: LOCAL Qwen2.5-Coder-7B-Instruct-AWQ via vLLM, Xeon w5-3425 (Sapphire Rapids).
Data: runs/replay_13033/ + runs/replay_13453/ (clean cgroup-scoped per-group replay) and
runs/verified/perf/ (live capture). BCB column = agentic/bigcodebench/runs/passes (same local-7B box).
Run with SYSTEM python3 (matplotlib not in .venv):  python3 plot_astropy_rerun.py
"""
import os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "runs", "plots_astropy"); os.makedirs(OUT, exist_ok=True)
BCB = os.path.normpath(os.path.join(HERE, "..", "bigcodebench", "runs", "passes"))

def parse_perf_txt(path):
    agg = collections.Counter()
    if not os.path.exists(path): return agg
    for line in open(path):
        p = line.split()
        if len(p) < 2: continue
        try: v = float(p[0].replace(",", ""))
        except ValueError: continue
        if p[1] not in agg: agg[p[1]] = v
    return agg

def tma(agg):
    sl = agg.get("slots", 0) or 1; cyc = agg.get("cycles", 0) or 1; ins = agg.get("instructions", 0)
    g = lambda k: agg.get("topdown-"+k, 0)/sl*100
    return dict(ipc=ins/cyc, ret=g("retiring"), fe=g("fe-bound"), bad=g("bad-spec"), be=g("be-bound"))

def fp_share(agg):
    # FLOP-weighted: scalar=1, 128b=2, 256b=4, 512b=8 (double-precision lanes)
    s = agg.get("fp_arith_inst_retired.scalar_double",0)+agg.get("fp_arith_inst_retired.scalar_single",0)
    p128 = agg.get("fp_arith_inst_retired.128b_packed_double",0)+agg.get("fp_arith_inst_retired.128b_packed_single",0)
    p256 = agg.get("fp_arith_inst_retired.256b_packed_double",0)+agg.get("fp_arith_inst_retired.256b_packed_single",0)
    p512 = agg.get("fp_arith_inst_retired.512b_packed_double",0)+agg.get("fp_arith_inst_retired.512b_packed_single",0)
    flops = s*1 + p128*2 + p256*4 + p512*8
    packed = (p128*2 + p256*4 + p512*8)
    return dict(scalar=s, p128=p128, p256=p256, p512=p512, flops=flops,
                avx_pct=(packed/flops*100 if flops else 0))

# ---------- load ----------
R33 = {g: parse_perf_txt(f"{HERE}/runs/replay_13033/group_{g}.txt") for g in ("tma","cache","fp","mlp","imc")}
R53 = {g: parse_perf_txt(f"{HERE}/runs/replay_13453/group_{g}.txt") for g in ("tma","cache","fp","mlp","imc")}
BCBt = parse_perf_txt(f"{BCB}/group_TMA.txt"); BCBf = parse_perf_txt(f"{BCB}/group_FP.txt")
t33, t53, tb = tma(R33["tma"]), tma(R53["tma"]), tma(BCBt)
COMP = ["ret","fe","bad","be"]; LBL = ["Retiring","Frontend-bound","Bad-spec","Backend-bound"]
COL = ["#2ca02c","#1f77b4","#d62728","#ff7f0e"]

# ---------- 1. SWE-bench TMA stacked bars (per trajectory + mean) ----------
mean = {k: (t33[k]+t53[k])/2 for k in COMP}
bars = [("ap-13033", t33), ("ap-13453", t53), ("mean", mean)]
fig, ax = plt.subplots(figsize=(6.2,4.6))
x = range(len(bars)); bot = [0]*len(bars)
for c, lab, col in zip(COMP, LBL, COL):
    vals = [b[1][c] for b in bars]
    ax.bar(x, vals, bottom=bot, label=lab, color=col, width=0.6)
    bot = [bot[i]+vals[i] for i in range(len(bars))]
ax.set_xticks(list(x)); ax.set_xticklabels([b[0] for b in bars])
ipcs=[t33['ipc'], t53['ipc'], (t33['ipc']+t53['ipc'])/2]
for i,v in enumerate(ipcs): ax.text(i, 101, f"IPC {v:.2f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,112)
ax.set_title("SWE-bench agentic tool-exec — Top-down (TMA)\nlocal Qwen2.5-Coder-7B · astropy Verified")
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.12), ncol=4, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/01_swebench_tma.png", dpi=140); plt.close(fig)

# ---------- 2. Cross-workload TMA: SWE-bench vs BCB (same local-7B box) ----------
groups = [("SWE-bench\ntool-exec", mean, (t33['ipc']+t53['ipc'])/2),
          ("BigCodeBench\ntool-exec (GT)", tb, tb['ipc'])]
fig, ax = plt.subplots(figsize=(5.6,4.6))
x = range(len(groups)); bot=[0]*len(groups)
for c, lab, col in zip(COMP, LBL, COL):
    vals=[g[1][c] for g in groups]
    ax.bar(x, vals, bottom=bot, label=lab, color=col, width=0.55)
    bot=[bot[i]+vals[i] for i in range(len(groups))]
ax.set_xticks(list(x)); ax.set_xticklabels([g[0] for g in groups])
for i,g in enumerate(groups): ax.text(i, 103, f"IPC {g[2]:.2f}", ha="center", fontsize=8.5, fontweight="bold")
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0,114)
ax.set_title("Cross-workload tool-exec Top-down (local 7B)\nSWE-bench = frontend-bound · BCB = backend-bound")
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.12), ncol=4, fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{OUT}/02_cross_workload_tma.png", dpi=140); plt.close(fig)

# ---------- 3. FP / vectorization: SWE-bench (~0 AVX) vs BCB (real AVX) ----------
f33, f53, fb = fp_share(R33["fp"]), fp_share(R53["fp"]), fp_share(BCBf)
fig, (a1,a2) = plt.subplots(1,2, figsize=(8.4,4.3))
names=["SWE\nap-13033","SWE\nap-13453","BCB\nGT"]
flops=[f33["flops"], f53["flops"], fb["flops"]]
a1.bar(names, flops, color=["#1f77b4","#1f77b4","#ff7f0e"])
a1.set_yscale("log"); a1.set_ylabel("FLOPs (lane-weighted, log)")
a1.set_title("Floating-point volume")
for i,v in enumerate(flops): a1.text(i, max(v,1)*1.3, f"{v:.0f}", ha="center", fontsize=8)
avx=[f33["avx_pct"], f53["avx_pct"], fb["avx_pct"]]
a2.bar(names, avx, color=["#1f77b4","#1f77b4","#ff7f0e"])
a2.set_ylabel("% of FLOPs from packed/AVX"); a2.set_title("Vectorization (AVX share)")
for i,v in enumerate(avx): a2.text(i, v+0.5, f"{v:.1f}%", ha="center", fontsize=8)
fig.suptitle("SWE-bench tool-exec has near-zero FP/AVX; BCB runs real numeric tests", fontsize=10)
fig.tight_layout(); fig.savefig(f"{OUT}/03_fp_vectorization.png", dpi=140); plt.close(fig)

# ---------- 4. Cache hierarchy (SWE-bench, mean of two) ----------
def cache_rates(agg):
    l1=agg.get("mem_load_retired.l1_hit",0); l2=agg.get("mem_load_retired.l2_hit",0)
    l3=agg.get("mem_load_retired.l3_hit",0); miss=agg.get("mem_load_retired.l3_miss",0)
    tot=l1+l2+l3+miss or 1
    return [l1/tot*100,l2/tot*100,l3/tot*100,miss/tot*100]
c33=cache_rates(R33["cache"])
fig, ax = plt.subplots(figsize=(5.2,4.2))
labs=["L1 hit","L2 hit","L3 hit","L3 miss"]; cols=["#2ca02c","#98df8a","#ffbb78","#d62728"]
ax.bar(labs, c33, color=cols)
for i,v in enumerate(c33): ax.text(i, v+ (1 if v<90 else -6), f"{v:.2f}%", ha="center", fontsize=9)
ax.set_ylabel("% of retired loads"); ax.set_ylim(0,105)
ax.set_title("SWE-bench tool-exec cache hierarchy (ap-13033)\nL1-resident: interpreter loop fits in cache")
fig.tight_layout(); fig.savefig(f"{OUT}/04_cache_hierarchy.png", dpi=140); plt.close(fig)

# ---------- 5. Wall-clock GPU-gen vs CPU-tool-exec donut (live capture) ----------
gpu=[]
gp=f"{HERE}/runs/verified/perf/gpu_timeline.csv"
if os.path.exists(gp):
    for l in open(gp):
        x=l.split(",")
        if len(x)>1 and x[1].strip().replace(".","").isdigit(): gpu.append(float(x[1]))
busy=(sum(1 for u in gpu if u>5)/len(gpu)*100) if gpu else 0
fig, ax = plt.subplots(figsize=(5,4.4))
ax.pie([busy,100-busy], labels=[f"GPU generation\n{busy:.0f}%", f"CPU tool-exec /\nidle\n{100-busy:.0f}%"],
       colors=["#9467bd","#8c8c8c"], autopct="", startangle=90, wedgeprops=dict(width=0.42))
ax.set_title("SWE-bench agentic loop — wall-clock\n(live: 10 astropy × 4 workers, GPU-active fraction)")
fig.tight_layout(); fig.savefig(f"{OUT}/05_walltime_donut.png", dpi=140); plt.close(fig)

# ---------- 6. Microarch summary table ----------
def mlp(agg):
    pend=agg.get("l1d_pend_miss.pending",0); pc=agg.get("l1d_pend_miss.pending_cycles",0) or 1
    return pend/pc
def ilp(agg):
    return agg.get("uops_executed.thread",0)/(agg.get("cycles",0) or 1)
def dram_gb(agg):
    r=agg.get("uncore_cha/unc_cha_imc_reads_count.normal/",0)*64/1e9
    w=agg.get("uncore_cha/unc_cha_imc_writes_count.full/",0)*64/1e9
    return r,w
rows=[
 ("IPC", f"{t33['ipc']:.2f}", f"{t53['ipc']:.2f}", f"{tb['ipc']:.2f}"),
 ("Retiring %", f"{t33['ret']:.0f}", f"{t53['ret']:.0f}", f"{tb['ret']:.0f}"),
 ("Frontend-bound %", f"{t33['fe']:.0f}", f"{t53['fe']:.0f}", f"{tb['fe']:.0f}"),
 ("Bad-spec %", f"{t33['bad']:.0f}", f"{t53['bad']:.0f}", f"{tb['bad']:.0f}"),
 ("Backend-bound %", f"{t33['be']:.0f}", f"{t53['be']:.0f}", f"{tb['be']:.0f}"),
 ("FLOPs (lane-wt)", f"{f33['flops']:.0f}", f"{f53['flops']:.0f}", f"{fb['flops']:.2e}"),
 ("AVX share %", f"{f33['avx_pct']:.1f}", f"{f53['avx_pct']:.1f}", f"{fb['avx_pct']:.1f}"),
 ("MLP", f"{mlp(R33['mlp']):.2f}", f"{mlp(R53['mlp']):.2f}", "—"),
 ("ILP (uops/cyc)", f"{ilp(R33['mlp']):.2f}", f"{ilp(R53['mlp']):.2f}", "—"),
 ("DRAM read GB", f"{dram_gb(R33['imc'])[0]:.2f}", f"{dram_gb(R53['imc'])[0]:.2f}", "—"),
 ("DRAM write GB", f"{dram_gb(R33['imc'])[1]:.2f}", f"{dram_gb(R53['imc'])[1]:.2f}", "—"),
]
fig, ax = plt.subplots(figsize=(7.2,4.6)); ax.axis("off")
tbl=ax.table(cellText=[list(r) for r in rows],
             colLabels=["metric","SWE ap-13033","SWE ap-13453","BCB GT"],
             cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1,1.45)
for j in range(4): tbl[0,j].set_facecolor("#2c3e50"); tbl[0,j].set_text_props(color="w", fontweight="bold")
ax.set_title("Tool-exec microarchitecture summary (local 7B, replay)", pad=14)
fig.tight_layout(); fig.savefig(f"{OUT}/06_microarch_table.png", dpi=140); plt.close(fig)

print("WROTE plots ->", OUT)
for f in sorted(os.listdir(OUT)): print("  ", f)
print(f"\nGPU-active wall-clock: {busy:.1f}%")
print(f"SWE-bench mean TMA: ret {mean['ret']:.0f} fe {mean['fe']:.0f} bad {mean['bad']:.0f} be {mean['be']:.0f} | IPC {(t33['ipc']+t53['ipc'])/2:.2f}")
print(f"BCB TMA: ret {tb['ret']:.0f} fe {tb['fe']:.0f} bad {tb['bad']:.0f} be {tb['be']:.0f} | IPC {tb['ipc']:.2f}")
print(f"AVX share: SWE {f33['avx_pct']:.1f}% / {f53['avx_pct']:.1f}%  vs  BCB {fb['avx_pct']:.1f}%")
