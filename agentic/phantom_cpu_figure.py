#!/usr/bin/env python3
"""Thesis figure (agents chapter): the host-CPU 'phantom busy-wait' during single-stream inference.
Measured 2026-06-28 on local A2000 + Qwen2.5-7B-AWQ via vLLM, default spin-sync vs an event-blocking
LD_PRELOAD shim (cudaEventBlockingSync). Shows the engine CPU cost collapsing while every outcome
(GPU work, throughput, agent progress) is unchanged -> the 'busy' core was doing nothing.
Run with SYSTEM python3 -> writes agentic/thesis_figures/phantom_cpu.png"""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "thesis_figures"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12, "xtick.labelsize": 10.5, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
SPIN_C, BLOCK_C = "#D55E00", "#0072B2"   # spin = vermillion (the waste), block = blue (after fix)

# measured spin -> block (live SWE-agent over 420 s, except throughput = isolated single-stream A/B)
# (label, spin_raw, block_raw, spin_txt, block_txt, group)
M = [
    ("Engine CPU busy",        1.00, 0.24, "1.00 core", "0.24 core", "cost"),
    ("Engine IPC",             3.46, 0.59, "3.46",      "0.59",      "cost"),
    ("GPU work (inference)",   395.0, 387.0, "395 s",   "387 s",     "out"),
    ("Throughput",           48.8, 48.6, "48.8 tok/s", "48.6 tok/s", "out"),
    ("Agent progress",         17.0, 16.0, "17 steps", "16 steps",  "out"),
]
M = M[::-1]                                # plot bottom-up so 'cost' group ends on top
fig, ax = plt.subplots(figsize=(9.2, 5.6))
y = 0; yt = []; ylab = []; h = 0.34
for i, (lab, sp, bl, spt, blt, grp) in enumerate(M):
    if i > 0 and M[i - 1][5] != grp:       # gap between outcome and cost groups
        y += 0.9
    ax.barh(y + h / 2, 1.0, height=h, color=SPIN_C, alpha=0.9, edgecolor="white")
    ax.barh(y - h / 2, bl / sp, height=h, color=BLOCK_C, alpha=0.95, edgecolor="white")
    ax.text(1.0 + 0.015, y + h / 2, spt, va="center", ha="left", fontsize=9.5, color=SPIN_C, fontweight="bold")
    ax.text(bl / sp + 0.015, y - h / 2, blt, va="center", ha="left", fontsize=9.5, color=BLOCK_C, fontweight="bold")
    yt.append(y); ylab.append(lab); y += 1.0
ax.set_yticks(yt); ax.set_yticklabels(ylab)
ax.set_xlim(0, 1.32); ax.set_xlabel("relative to spin-mode (spin = 1.0)")
ax.axvline(1.0, color="#bbbbbb", lw=0.8, ls=":")
ax.grid(axis="y", visible=False)
# group brackets / labels on the right
ax.set_title("Spin versus blocking GPU synchronization")
ax.legend(handles=[Patch(color=SPIN_C, label="spin-sync (default) — polls the GPU"),
                   Patch(color=BLOCK_C, label="block-sync (one-flag fix) — sleeps")],
          loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2)
fig.savefig(os.path.join(OUT, "phantom_cpu.png")); plt.close(fig)
print("wrote", os.path.join(OUT, "phantom_cpu.png"))
