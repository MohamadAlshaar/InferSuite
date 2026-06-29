#!/usr/bin/env python3
"""GPU top-down figures for the inference regimes, built from Nsight Compute warp-state + Speed-of-Light
counters (RTX A2000, Qwen2.5-7B-AWQ, enforce_eager, FlashAttention-2, NVTX-fenced, enable_prefix_caching
=False so prefill is real). Dominant kernels only (-k filter), duration-weighted.

Regimes: Prefill-focused (large prompt, out=1 -> real compute-bound prefill),
         Decode-focused  (1-token prompt, long gen -> pure M=1 memory-bound decode),
         Prompts         (a real agent turn = the measured time-weighted blend of the two; the prompt's
                          prefill dominates GPU time -> prefill_frac 0.75 at a 13.5K-tok / 128-tok turn).
Run with SYSTEM python3."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = "/home/mohamad/llm-service-kernel-latest/agentic/inference/runs/ncu"
G = json.load(open(f"{SRC}/gpu_tma.json"))
TS = json.load(open(f"{SRC}/timing_split.json"))
W_PF = TS["prefill_frac_O128"]; W_DEC = 1 - W_PF        # blend weight for the realistic "Prompts" turn

# ---- derive the "Prompts" regime as the time-weighted blend of prefill + decode ----
def blend(a, b):
    if isinstance(a, dict):
        return {k: blend(a[k], b.get(k, 0) if isinstance(b, dict) else 0) for k in a}
    return W_PF * a + W_DEC * b
pf, dec = G["prefill"], G["decode"]
prompts = {"by_reason": blend(pf["by_reason"], dec["by_reason"]),
           "by_category": blend(pf["by_category"], dec["by_category"]),
           "sol_compute_pct": blend(pf["sol_compute_pct"], dec["sol_compute_pct"]),
           "sol_memory_pct": blend(pf["sol_memory_pct"], dec["sol_memory_pct"]),
           "issue_eff_per_cycle": blend(pf["issue_eff_per_cycle"], dec["issue_eff_per_cycle"])}
# blended kernel-class time%
allcls = set(pf["by_kernel_class"]) | set(dec["by_kernel_class"])
prompts["by_kernel_class"] = {c: {"time_pct": W_PF * pf["by_kernel_class"].get(c, {}).get("time_pct", 0)
                                          + W_DEC * dec["by_kernel_class"].get(c, {}).get("time_pct", 0)}
                              for c in allcls}
prompts["lane_eff_pct"] = blend(pf["lane_eff_pct"], dec["lane_eff_pct"])
prompts["uarch"] = {nm: blend(pf["uarch"][nm], dec["uarch"][nm]) for nm in pf["uarch"]}
R = {"prefill": pf, "decode": dec, "prompts": prompts}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
REG = [("prefill", "Prefill-focused", "#0072B2"), ("decode", "Decode-focused", "#D55E00"),
       ("prompts", "Agent prompts", "#009E73")]
COMPUTE, MEMORY, LAT = "#0072B2", "#D55E00", "#6a51a3"

# ===================== Fig 1: Speed-of-Light bottleneck map =====================
fig, ax = plt.subplots(figsize=(7.0, 6.4))
ax.axhspan(60, 100, color=COMPUTE, alpha=0.05, zorder=0)
ax.axvspan(60, 100, color=MEMORY, alpha=0.05, zorder=0)
ax.add_patch(plt.Rectangle((0, 0), 50, 50, color=LAT, alpha=0.06, zorder=0))
ax.plot([0, 100], [0, 100], ls="--", color="#bbbbbb", lw=1.0, zorder=1)
ax.text(82, 92, "compute-bound", color=COMPUTE, fontsize=10.5, ha="center", style="italic")
ax.text(92, 8, "memory-\nbound", color=MEMORY, fontsize=10.5, ha="center", va="center", style="italic")
ax.text(11, 9, "latency-bound", color=LAT, fontsize=10.5, ha="left", va="center", style="italic")
LBLPOS = {"prefill": (49, 93), "decode": (86, 55), "prompts": (67, 78)}
for key, lab, col in REG:
    x, y = R[key]["sol_memory_pct"], R[key]["sol_compute_pct"]
    ax.scatter(x, y, s=300, color=col, zorder=5, edgecolor="white", linewidth=2)
    tx, ty = LBLPOS[key]
    ax.annotate(f"{lab}\ncompute {y:.0f}% / mem {x:.0f}%\n{R[key]['issue_eff_per_cycle']:.2f} issue/cyc",
                (x, y), xytext=(tx, ty), fontsize=10, fontweight="bold", color=col, zorder=6, ha="center",
                arrowprops=dict(arrowstyle="-", color=col, lw=1.0, alpha=0.7))
ax.set_xlim(0, 100); ax.set_ylim(0, 100)
ax.set_xlabel("Memory throughput  (% of peak)")
ax.set_ylabel("Compute / SM throughput  (% of peak)")
ax.set_title("GPU Speed-of-Light: prefill is compute-bound, decode is memory-bound")
fig.text(0.5, 0.005, "Prefill saturates the SM math pipes (81 %); decode is weight-bandwidth-bound (61 %). "
         "A real agent turn is prefill-dominated.", ha="center", fontsize=8.6, style="italic", color="#666")
fig.savefig(f"{HERE}/gpu_01_speed_of_light.png"); plt.close(fig)

# ===================== Fig 2: warp-scheduler top-down =====================
def rollup(c):
    mem = sum(c.get(k, 0) for k in c if k.startswith("Mem"))
    other = c.get("Front-end (fetch)", 0) + c.get("Branch resolve", 0) + c.get("Other", 0)
    return [("Issued (useful work)", c.get("Issued", 0), "#009E73"),
            ("Latency hidden (warp swap)", c.get("Scheduler-covered", 0), "#a6dba0"),
            ("Stall · execution latency", c.get("Latency (fixed deps)", 0), "#E69F00"),
            ("Stall · math pipe (compute)", c.get("Compute · math pipe", 0), "#0072B2"),
            ("Stall · memory", mem, "#D55E00"),
            ("Stall · synchronization", c.get("Synchronization", 0), "#CC79A7"),
            ("Stall · fetch / branch / other", other, "#999999")]
order = [rollup(R[k]["by_category"]) for k, _, _ in REG]
cats = [c[0] for c in order[0]]; cols = [c[2] for c in order[0]]
fig, ax = plt.subplots(figsize=(7.6, 6.4))
xs = range(len(REG))
for ci, (cat, col) in enumerate(zip(cats, cols)):
    bot = [sum(order[ri][cj][1] for cj in range(ci)) for ri in range(len(REG))]
    vals = [order[ri][ci][1] for ri in range(len(REG))]
    ax.bar(list(xs), vals, bottom=bot, color=col, width=0.62, edgecolor="white", linewidth=0.8, label=cat)
    for ri, v in enumerate(vals):
        if v >= 4:
            ax.text(ri, bot[ri] + v / 2, f"{v:.0f}", ha="center", va="center", color="white", fontweight="bold", fontsize=10.5)
for ri, (key, lab, _) in enumerate(REG):
    ax.text(ri, 101.5, f"{R[key]['issue_eff_per_cycle']:.2f} issue/cyc", ha="center", fontsize=9.5, style="italic", color="#444")
ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l, _ in REG], fontsize=10.5)
ax.set_ylabel("Share of warp-scheduler issue cycles (%)"); ax.set_ylim(0, 108)
ax.set_title("GPU warp-scheduler top-down: where the issue slots go")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=10); ax.grid(axis="x", visible=False)
fig.savefig(f"{HERE}/gpu_02_warpstate_topdown.png"); plt.close(fig)

# ===================== Fig 3a: GPU-time composition by kernel class =====================
CLS = ["AWQ GEMM (Marlin)", "Attention (flash)", "RMSNorm", "RoPE", "KV-cache write", "Activation (SwiGLU)", "Elementwise/index", "Other"]
CCOL = {"AWQ GEMM (Marlin)": "#D55E00", "Attention (flash)": "#6a51a3", "RMSNorm": "#0072B2",
        "RoPE": "#56B4E9", "KV-cache write": "#E69F00", "Activation (SwiGLU)": "#009E73",
        "Elementwise/index": "#CC79A7", "Other": "#bbbbbb"}
fig, axA = plt.subplots(figsize=(8.4, 4.8))
for ri, (key, lab, _) in enumerate(REG):
    kc = R[key]["by_kernel_class"]; left = 0
    for cls in CLS:
        v = kc.get(cls, {}).get("time_pct", 0)
        if v <= 0: continue
        axA.barh(ri, v, left=left, color=CCOL[cls], edgecolor="white", linewidth=0.8)
        if v >= 7:
            axA.text(left + v / 2, ri, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
        left += v
axA.set_yticks(range(len(REG))); axA.set_yticklabels([l for _, l, _ in REG]); axA.invert_yaxis()
axA.set_xlim(0, 100); axA.set_xlabel("Share of GPU time (%)")
axA.set_title("Where GPU time goes: the AWQ GEMM dominates every regime")
handles = [Patch(color=CCOL[c], label=c) for c in CLS if any(R[k]["by_kernel_class"].get(c, {}).get("time_pct", 0) >= 1 for k, _, _ in REG)]
axA.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=9.5)
axA.grid(axis="y", visible=False)
fig.savefig(f"{HERE}/gpu_03a_kernel_time.png"); plt.close(fig)

# ===================== Fig 3b: the dominant kernel's bottleneck FLIPS with regime =====================
import numpy as np
gpf = pf["by_kernel_class"]["AWQ GEMM (Marlin)"]; gdec = dec["by_kernel_class"]["AWQ GEMM (Marlin)"]
groups = [("Prefill GEMM\n(compute-bound)", gpf["sol_compute"], gpf["sol_memory"]),
          ("Decode GEMV\n(memory-bound)", gdec["sol_compute"], gdec["sol_memory"])]
fig, axB = plt.subplots(figsize=(6.6, 5.2))
xpos = np.arange(len(groups)); w = 0.36
comp = [g[1] for g in groups]; memo = [g[2] for g in groups]
axB.bar(xpos - w / 2, comp, w, color=COMPUTE, label="Compute / SM", edgecolor="white")
axB.bar(xpos + w / 2, memo, w, color=MEMORY, label="Memory", edgecolor="white")
for i, (cc, mm) in enumerate(zip(comp, memo)):
    axB.text(i - w / 2, cc + 1.5, f"{cc:.0f}%", ha="center", fontsize=11, fontweight="bold")
    axB.text(i + w / 2, mm + 1.5, f"{mm:.0f}%", ha="center", fontsize=11, fontweight="bold")
axB.axhline(100, ls=":", color="#999", lw=1); axB.text(1.45, 97, "peak", fontsize=9, color="#999", va="top")
axB.set_xticks(xpos); axB.set_xticklabels([g[0] for g in groups], fontsize=11)
axB.set_ylim(0, 100); axB.set_ylabel("% of peak (Speed-of-Light)")
axB.set_title("Same kernel, opposite bottleneck: AWQ GEMM across regimes")
axB.legend(loc="upper center", fontsize=10.5); axB.grid(axis="x", visible=False)
fig.savefig(f"{HERE}/gpu_03b_kernel_bottleneck.png"); plt.close(fig)

# ===================== Fig 4: MICROARCH hardware signature heatmap =====================
# Real microarch measures (NOT the warp-state top-down) grouped like the CPU IPC/ILP/MLP/cache/AVX signature.
ROWS = [("prefill", "AWQ GEMM (Marlin)", "Prefill · AWQ GEMM"),
        ("prefill", "Attention (flash)", "Prefill · Attention"),
        ("prefill", "Activation (SwiGLU)", "Prefill · Activation"),
        ("decode", "AWQ GEMM (Marlin)", "Decode · AWQ GEMV"),
        ("decode", "Attention (flash)", "Decode · Attention"),
        ("decode", "RMSNorm", "Decode · RMSNorm")]
GROUPS = [
    ("Throughput / latency-hiding", [("IPC", "IPC", "f1"), ("Occupancy", "Occ\n%", "i"), ("Eligible warps", "Elig\nwarps", "f1")]),
    ("Compute pipes (cycles-active)", [("Tensor pipe", "Tensor\n%", "i"), ("FMA pipe", "FMA\n%", "i"), ("ALU pipe", "ALU\n%", "i")]),
    ("SIMT", [("SIMT eff", "Lanes\nact %", "simt")]),
    ("Memory hierarchy", [("L1 hit", "L1 hit\n%", "i"), ("L2 hit", "L2 hit\n%", "i"), ("DRAM BW", "DRAM\nBW %", "i")]),
    ("Res.", [("Registers", "Regs/\nthr", "i")]),
]
flat = [(m, lab, fmt) for _, cols in GROUPS for m, lab, fmt in cols]
COLS = [lab for _, lab, _ in flat]
def uval(reg, cls, m): return G[reg]["by_kernel_class"].get(cls, {}).get("uarch", {}).get(m, 0.0)
def disp(v, fmt): return f"{v/32*100:.0f}" if fmt == "simt" else (f"{v:.1f}" if fmt == "f1" else f"{v:.0f}")
M = np.array([[uval(r, c, m) for m, _, _ in flat] for r, c, _ in ROWS])
Mn = (M - M.min(0)) / (np.ptp(M, 0) + 1e-9)            # per-column min-max for colour
fig, ax = plt.subplots(figsize=(13.0, 5.6))
ax.imshow(Mn, cmap="YlGnBu", aspect="auto", vmin=0, vmax=1)
ax.set_xticks(range(len(COLS))); ax.set_xticklabels(COLS, fontsize=9)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[2] for r in ROWS], fontsize=10.5)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0)
for i in range(len(ROWS)):
    for j, (m, _, fmt) in enumerate(flat):
        ax.text(j, i, disp(M[i, j], fmt), ha="center", va="center", fontsize=9.5,
                color="black" if Mn[i, j] < 0.6 else "white", fontweight="bold")
b = 0                                                   # group separators + group labels on top
for gname, cols in GROUPS:
    n = len(cols)
    if b > 0: ax.axvline(b - 0.5, color="white", lw=3)
    ax.text(b + (n - 1) / 2.0, -0.82, gname, ha="center", va="bottom", fontsize=9.3, fontweight="bold", color="#333")
    b += n
ax.axhline(2.5, color="white", lw=3)                   # prefill / decode blocks
ax.set_ylim(len(ROWS) - 0.5, -1.45)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title("Micro-architectural signature of the dominant GPU kernels", pad=52)
fig.text(0.5, -0.02, "Hardware measures (not the warp-state top-down). Colour = per-column min–max (light = low, "
         "dark = high vs the other kernels); numbers are true values. IPC max = 4/SM; Lanes = active threads/warp.",
         ha="center", fontsize=8.2, style="italic", color="#666")
fig.savefig(f"{HERE}/gpu_04_signature_heatmap.png"); plt.close(fig)

# ===================== Fig 5: 2-level GPU top-down (Intel-style, corrected) =====================
# L1: Retiring / Front-end / Back-end{Core,Memory} / Covered. Sync = GPU-specific side bucket (not Backend).
# Bad-spec has NO GPU analog (no speculation) -> shown as divergence(spatial)+replays(temporal), both ~0.
RET_C, CORE_C, MEM_C, FE_C, SYNC_C, COV_C, OTH_C = "#009E73", "#E69F00", "#D55E00", "#0072B2", "#CC79A7", "#9e9ac8", "#dddddd"
SEGS = [("Retiring (issued)", "Issued", RET_C, "white"),
        ("Back-end · Core (math-pipe + latency)", "_core", CORE_C, "white"),
        ("Back-end · Memory (scoreboard + DRAM)", "_mem", MEM_C, "white"),
        ("Front-end (i-cache / branch)", "_fe", FE_C, "white"),
        ("Synchronization (barrier — GPU-only)", "Synchronization", SYNC_C, "white"),
        ("Covered — latency hidden (GPU-only)", "Scheduler-covered", COV_C, "#333"),
        ("Other", "Other", OTH_C, "#333")]
def bval(cat, key):
    if key == "_core": return cat.get("Compute · math pipe", 0) + cat.get("Latency (fixed deps)", 0)
    if key == "_mem": return sum(cat.get(k, 0) for k in cat if k.startswith("Mem"))
    if key == "_fe": return cat.get("Front-end (fetch)", 0) + cat.get("Branch resolve", 0)
    return cat.get(key, 0)
Bk = {k: [bval(R[k]["by_category"], s[1]) for s in SEGS] for k, _, _ in REG}
fig, ax = plt.subplots(figsize=(10.0, 6.4))
xs = range(len(REG))
for idx, (lab, key, col, tc) in enumerate(SEGS):
    bot = [sum(Bk[k][j] for j in range(idx)) for k, _, _ in REG]
    vals = [Bk[k][idx] for k, _, _ in REG]
    ax.bar(list(xs), vals, bottom=bot, color=col, width=0.54, edgecolor="white", linewidth=0.9, label=lab)
    for ri, v in enumerate(vals):
        if v >= 3.5: ax.text(ri, bot[ri] + v / 2, f"{v:.0f}", ha="center", va="center", color=tc, fontweight="bold", fontsize=10.5)
for ri, (key, lab, _) in enumerate(REG):                 # back-end bracket + heavy-op annotation
    be = Bk[key][1] + Bk[key][2]; tp = R[key]["uarch"]["Tensor pipe"]
    ax.text(ri, 101, f"Back-end {be:.0f}%", ha="center", fontsize=9.6, fontweight="bold", color="#8a2a2a")
    ax.text(ri, 105, f"heavy-op (tensor) {tp:.0f}%", ha="center", fontsize=8.2, style="italic", color="#555")
ax.set_xticks(list(xs)); ax.set_xticklabels([l for _, l, _ in REG], fontsize=11)
ax.set_ylabel("Share of warp-scheduler issue slots (%)"); ax.set_ylim(0, 110); ax.set_xlim(-0.6, 3.05)
ax.set_title("GPU top-down in Intel-style buckets — Retiring / Front-end / Back-end{Core, Memory} / Covered")
ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8.6); ax.grid(axis="x", visible=False)
fig.text(0.5, -0.04, "Bad-speculation = N/A on GPU (no speculative execution): its analogs are SIMT divergence (spatial) "
         "and instruction replays (temporal) — both ≈ 0 here (lanes 100%).  Covered (latency-hidden) "
         "is small because occupancy is low (16–18%), so stalls are uncovered → genuinely back-end-bound.",
         ha="center", fontsize=8.1, style="italic", color="#666", wrap=True)
fig.savefig(f"{HERE}/gpu_05_gpu_tma_intel.png"); plt.close(fig)

# ===================== Fig 6: GPU throughput is 2-D (issue slots x SIMT lanes) =====================
fig, ax = plt.subplots(figsize=(6.8, 6.4))
ax.axhspan(97, 103, color=RET_C, alpha=0.10)
ax.axhline(100, ls="--", color=RET_C, lw=1.3)
ax.text(99, 93, "lane ceiling — no divergence", color="#00735a", fontsize=8.6, ha="right", style="italic")
ax.text(50, 42, "divergent / sparse kernels\nwould fall here\n(wasted SIMT lanes ↓)", ha="center", va="center",
        fontsize=9.5, color="#c8c8c8", style="italic")
LPOS = {"prefill": (25, 82), "decode": (56, 82), "prompts": (31, 64)}   # absolute label anchors, spread below ceiling
for key, lab, col in REG:
    x = R[key]["issue_eff_per_cycle"] * 100; y = R[key]["lane_eff_pct"]
    ax.scatter(x, y, s=320, color=col, edgecolor="white", linewidth=2, zorder=5)
    ax.annotate(f"{lab}\nissue {x:.0f}%  ×  lanes {y:.0f}%", (x, y), xytext=LPOS[key], fontsize=9.2,
                fontweight="bold", color=col, ha="left", arrowprops=dict(arrowstyle="-", color=col, alpha=0.65))
ax.set_xlim(0, 100); ax.set_ylim(0, 110)
ax.set_xlabel("Issue efficiency  (temporal — % of active scheduler cycles issuing)")
ax.set_ylabel("Lane efficiency  (spatial — % of 32 SIMT lanes active)")
ax.set_title("GPU throughput is 2-D: issue slots × SIMT lanes", pad=12)
fig.text(0.5, 0.002, "Useful work = issue-eff × lane-eff (lane-weighted Retiring). All regimes sit on the lane ceiling "
         "(divergence-free kernels) → no spatial waste; all loss is temporal (low issue efficiency — too few warps to hide latency).",
         ha="center", fontsize=8.2, style="italic", color="#666", wrap=True)
fig.savefig(f"{HERE}/gpu_06_throughput_2d.png"); plt.close(fig)

print(f"Prompts blend weights: prefill {W_PF:.2f} / decode {W_DEC:.2f} (O=128, {TS['prompt_tokens']}-tok prompt)")
print("WROTE GPU figures ->", HERE)
for f in sorted(os.listdir(HERE)):
    if f.endswith(".png"): print("  ", f)
