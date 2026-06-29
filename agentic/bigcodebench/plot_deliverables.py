#!/usr/bin/env python3
"""BigCodeBench LOCAL GT-execution — deliverables + the SWE-agent-vs-BCB contrast.
Provenance: local workstation (Xeon w5-3425 SPR, 24 cores), 2026-06-24. BCB v0.1.4 Hard subset,
148 ground-truth solutions executed locally (--check_gt_only, --parallel 4); vLLM idle (0% CPU).
NOT cloud/32B data. Reads runs/perf/bcb_timeline.csv (system-wide, multiplexed -> ratios valid)
+ the SWE-agent local-7B numbers for contrast. Run with SYSTEM python3."""
import os, collections
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results_local_gt"); os.makedirs(OUT, exist_ok=True)
FREQ = 4.0e9

def parse_timeline(path):
    agg = collections.Counter(); ts = []
    for line in open(path):
        q = line.rstrip("\n").split(",")
        if len(q) < 4 or not q[0][:1].isdigit(): continue
        ts.append(float(q[0]))
        try: agg[q[3]] += float(q[1]) if q[1] not in ("", "<not counted>") else 0.0
        except ValueError: pass
    dur = (max(ts)-min(ts)+1) if ts else 1
    return agg, dur

bcb, dur = parse_timeline(os.path.join(HERE, "runs", "perf", "bcb_timeline.csv"))
sl = bcb.get("slots", 0) or 1; cyc = bcb.get("cycles", 0) or 1; ins = bcb.get("instructions", 0)
B = dict(ipc=ins/cyc, cores=cyc/FREQ/dur,
         ret=bcb["topdown-retiring"]/sl*100, fe=bcb["topdown-fe-bound"]/sl*100,
         bad=bcb["topdown-bad-spec"]/sl*100, be=bcb["topdown-be-bound"]/sl*100)
g = lambda k: bcb.get("fp_arith_inst_retired."+k, 0)
s1,s128,s256,s512 = g("scalar_single"),g("128b_packed_single"),g("256b_packed_single"),g("512b_packed_single")
sd,d128,d256,d512 = g("scalar_double"),g("128b_packed_double"),g("256b_packed_double"),g("512b_packed_double")
flops = s1+s128*4+s256*8+s512*16+sd+d128*2+d256*4+d512*8
avx512 = (s512*16+d512*8)/flops*100 if flops else 0
scalar = (s1+sd)/flops*100 if flops else 0
B["flops"], B["avx"], B["scalar"] = flops, avx512, scalar
l1,l2,l3,ms = (bcb.get("mem_load_retired."+k,0) for k in ("l1_hit","l2_hit","l3_hit","l3_miss")); tot=l1+l2+l3+ms or 1
B["amat"] = (l1*4+l2*12+l3*40+ms*200)/tot; B["l1"]=l1/tot*100; B["miss"]=ms/tot*100
B["mlp"] = bcb.get("l1d_pend_miss.pending",0)/(bcb.get("l1d_pend_miss.pending_cycles",0) or 1)

# SWE-agent local-7B tool-exec (from swe_agent/results_local7b — the deep replay)
S = dict(cores=0.02, ipc=1.59, ret=27, fe=32, bad=16, be=25, flops=0.2e6, avx=0, scalar=100)

# ---- 1: CPU vs GPU time donut (parallel to SWE-agent's; here GPU is idle -> pure CPU) ----
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
gpu_pct = 0  # measured: GPU util 0% for the entire run (no inference; vLLM idle)
ax[0].pie([gpu_pct, 100-gpu_pct], labels=[f"GPU inference\n{gpu_pct:.0f}%", f"CPU code-exec\n{100-gpu_pct:.0f}%"],
          colors=["#4C72B0", "#C44E52"], startangle=90, wedgeprops=dict(width=0.42))
ax[0].set_title("Wall-clock time\n(GPU busy = inference)")
ic, ac = 0.0, B["cores"]; tot = ic + ac or 1
ax[1].pie([ic/tot*100, ac/tot*100],
          labels=[f"Inference CPU\n(vLLM engine)\n{ic/tot*100:.0f}%", f"Code-exec CPU\n{ac/tot*100:.0f}%"],
          colors=["#55A868", "#C44E52"], startangle=90, wedgeprops=dict(width=0.42))
ax[1].set_title("CPU core-seconds\n(during vs outside inference)")
fig.suptitle("① CPU vs GPU — BigCodeBench GT code-execution", fontsize=11, weight="bold")
fig.text(0.5, 0.05, f"GPU idle 0% (no inference) · {ac:.2f} CPU cores over {dur:.0f}s of code-execution",
         ha="center", fontsize=9, weight="bold")
fig.text(0.5, 0.005, "inverse of SWE-agent (90% GPU-inference wall-clock; 96% of its CPU was during inference)",
         ha="center", fontsize=8, style="italic", color="#555")
plt.tight_layout(rect=[0, 0.04, 1, 0.95]); plt.savefig(os.path.join(OUT, "1_cpu_vs_gpu_donut.png"), dpi=130); plt.close()

# ---- 2: heavy-CPU contrast (cores) ----
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(["SWE-agent\ntool-exec\n(file edits)", "BigCodeBench\nGT code-exec\n(numpy/scipy/sklearn)"],
              [S["cores"], B["cores"]], color=["#C44E52", "#55A868"], width=0.55)
for b, v in zip(bars, [S["cores"], B["cores"]]):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.03, f"{v:.2f} cores", ha="center", fontsize=12, weight="bold")
ax.set_ylabel("avg CPU cores busy"); ax.set_ylim(0, B["cores"]*1.25)
ax.set_title(f"② Agent file-edits vs real code-execution CPU\n(BCB ≈ {B['cores']/max(S['cores'],0.01):.0f}× the agent tool-exec)",
             fontsize=11, weight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "2_cpu_cores_contrast.png"), dpi=130); plt.close()

# ---- 2: TMA stacked, side by side (SWE-agent tool-exec vs BCB code-exec) ----
cats = ["Retiring", "Frontend", "Bad-spec", "Backend"]; tcol = ["#55A868", "#DD8452", "#C44E52", "#4C72B0"]
fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)
for ax, (lbl, d) in zip(axes, [("SWE-agent tool-exec\n(file edits, 0.02 cores)", S),
                                ("BigCodeBench GT code-exec\n(numpy/scipy, 1.51 cores)", B)]):
    vals = [d["ret"], d["fe"], d["bad"], d["be"]]; bottom = 0
    for v, c, n in zip(vals, tcol, cats):
        ax.bar(0, v, bottom=bottom, color=c, width=0.5, edgecolor="white", linewidth=1)
        if v > 3: ax.text(0, bottom+v/2, f"{n}\n{v:.0f}%", ha="center", va="center", fontsize=9.5, weight="bold", color="white")
        bottom += v
    ax.set_title(f"{lbl}\nIPC {d['ipc']:.2f}", fontsize=10.5, weight="bold")
    ax.set_ylim(0, 100); ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
axes[0].set_ylabel("% of pipeline slots")
fig.suptitle("③ Top-down (TMA): agent tool-exec vs real code-execution", fontsize=11, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94]); plt.savefig(os.path.join(OUT, "3_tma_swe_vs_bcb.png"), dpi=130); plt.close()

# ---- 3: FP / SIMD breakdown (the FP fix reveals BCB's AVX the old harness missed) ----
fig, ax = plt.subplots(figsize=(8, 5))
lanes = ["scalar\ndouble", "128b\npacked-dp", "256b\npacked-dp", "512b\npacked-dp\n(AVX-512)"]
counts = [sd, d128, d256, d512]
bars = ax.bar(lanes, counts, color=["#8172B3", "#4C72B0", "#4C72B0", "#C44E52"])
ax.set_ylabel("FP ops retired (count)"); ax.set_yscale("log")
for b, v in zip(bars, counts):
    if v > 0: ax.text(b.get_x()+b.get_width()/2, v*1.15, f"{v/1e6:.1f}M", ha="center", fontsize=9, weight="bold")
ax.set_title(f"④ BCB FP/SIMD — real vectorization (AVX-512 {B['avx']:.0f}% of FLOPs)\n"
             f"vs SWE-agent tool-exec: ~0 FP (file edits). The fixed harness now sees packed-double.",
             fontsize=10.5, weight="bold")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "4_bcb_fp_simd.png"), dpi=130); plt.close()

# ---- 4: BCB microarch table ----
rows = [
    ("avg CPU cores (over 148 GT tasks)", f"{B['cores']:.2f}  (SWE-agent tool-exec: 0.02)"),
    ("IPC", f"{B['ipc']:.2f}"),
    ("TMA Retiring / FE / BadSpec / BE", f"{B['ret']:.0f}% / {B['fe']:.0f}% / {B['bad']:.0f}% / {B['be']:.0f}%"),
    ("AMAT (cyc) · L1 hit% · miss%", f"{B['amat']:.1f} · {B['l1']:.0f}% · {B['miss']:.1f}%"),
    ("FP lane-weighted FLOPs", f"{B['flops']/1e6:.0f}M  (SWE-agent: 0.2M → ~{B['flops']/0.2e6:.0f}×)"),
    ("FP mix: scalar / AVX-512 share", f"{B['scalar']:.0f}% scalar · {B['avx']:.0f}% AVX-512"),
    ("MLP", f"{B['mlp']:.2f}"),
]
fig, ax = plt.subplots(figsize=(9, 0.6+0.5*len(rows))); ax.axis("off")
tbl = ax.table(cellText=rows, colLabels=["BCB code-execution metric", "value"], cellLoc="left", colLoc="left",
               loc="center", colWidths=[0.55, 0.45])
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
for j in range(2): tbl[0,j].set_facecolor("#55A868"); tbl[0,j].set_text_props(color="w", weight="bold")
ax.set_title("⑤ CPU microarchitecture — BigCodeBench GT code-execution (148 Hard, local)",
             fontsize=11, weight="bold", pad=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "5_bcb_microarch_table.png"), dpi=130, bbox_inches="tight"); plt.close()

print("=== wrote BCB deliverables to", OUT, "===")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"): print("  ", f)
print(f"\nBCB: {B['cores']:.2f} cores, IPC {B['ipc']:.2f}, {B['flops']/1e6:.0f}M FLOPs, AVX-512 {B['avx']:.0f}%, "
      f"TMA BE{B['be']:.0f}/FE{B['fe']:.0f}/Ret{B['ret']:.0f}")
