#!/usr/bin/env python3
"""Three conceptual figures explaining the GPU top-down (run with SYSTEM python3).
Outputs next to this file:  gpu_topdown_pipeline.png / _roofline.png / _mirror.png"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mc
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 200, "savefig.bbox": "tight",
})
GREEN, BLUE, ORANGE, PINK = "#009E73", "#0072B2", "#E69F00", "#CC79A7"
GREY, LGREEN, NA, RED = "#9aa0a6", "#3aa884", "#c8ccd1", "#b03030"


def light(col, f=0.16):
    """Blend a colour toward white (f = fraction of the original colour kept)."""
    r, g, b = mc.to_rgb(col)
    return (1 - f + f * r, 1 - f + f * g, 1 - f + f * b)


def box(ax, x, y, w, h, fc, ec, lw=1.4, r=0.05, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={r}",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))


def arr(ax, x1, y1, x2, y2, c="#555", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                 color=c, lw=lw, zorder=1, shrinkA=1, shrinkB=1))


# ---------------- Figure 1: the pipeline ----------------
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(12.2, 9)); ax.set_xlim(0, 12); ax.set_ylim(0, 11); ax.axis("off")
    ax.text(6, 10.68, "GPU top-down — what happens in one scheduler slot",
            ha="center", fontsize=16, fontweight="bold")
    box(ax, 2.7, 9.5, 6.6, 0.82, light("#666", 0.10), "#777", r=0.10)
    ax.text(6, 10.06, "ONE SCHEDULER CYCLE  =  a “slot”", ha="center", fontsize=13, fontweight="bold")
    ax.text(6, 9.70, "every cycle, the scheduler tries to issue an instruction to a ready warp",
            ha="center", fontsize=9.4, style="italic", color="#555")
    arr(ax, 5.1, 9.5, 3.0, 8.98); arr(ax, 6.9, 9.5, 9.3, 8.98)
    # issued branch
    box(ax, 1.0, 8.05, 4.0, 0.92, light(GREEN, 0.13), GREEN, r=0.07)
    ax.text(3.0, 8.66, "ISSUED  →  RETIRING", ha="center", fontsize=12, fontweight="bold", color=GREEN)
    ax.text(3.0, 8.27, "a warp got work this cycle — real progress", ha="center", fontsize=9.3,
            color="#444", style="italic")
    # stalled branch
    box(ax, 7.0, 8.05, 4.6, 0.92, light(RED, 0.10), RED, r=0.07)
    ax.text(9.3, 8.66, "NO ISSUE — every warp stalled", ha="center", fontsize=12, fontweight="bold", color=RED)
    ax.text(9.3, 8.27, "the bottleneck — split by WHY  ↓", ha="center", fontsize=9.3,
            color="#444", style="italic")
    arr(ax, 9.3, 8.05, 9.3, 7.62, RED)
    rows = [
        ("no_instruction",     "recipe not ready  (instruction fetch / decode)",        "Front-end",         BLUE),
        ("long_scoreboard",    "waiting on FAR memory  (DRAM, KV-cache, weights)",       "Back-end · Memory", PINK),
        ("short_scoreboard",   "waiting on NEAR memory  (shared mem / L1)",              "Back-end · Memory", PINK),
        ("math_pipe_throttle", "math units are FULL — the good “busy” = the ceiling", "Back-end · Core", ORANGE),
        ("mio_throttle",       "the memory-IO pipe is full",                             "Back-end · Core",   ORANGE),
        ("wait",               "previous instruction's result isn't back yet",           "Back-end · Core",   ORANGE),
        ("barrier / membar",   "teams waiting for each other at a checkpoint",           "sync",              GREY),
        ("not_selected",       "a warp WAS ready, scheduler chose another → plenty",  "not a stall",       LGREEN),
        ("bad speculation",    "a GPU doesn't run wrong branches and roll back",         "N/A on a GPU",      GREY),
    ]
    y0, dy = 7.35, 0.78
    for i, (word, mean, binlab, col) in enumerate(rows):
        yy = y0 - i * dy
        faded = word == "bad speculation"
        ls = "--" if faded else "-"
        chipfc = light(NA, 0.4) if faded else light(col, 0.18)
        tagfc = light(NA, 0.4) if faded else light(col, 0.34)
        wcol = GREY if faded else col
        box(ax, 0.7, yy - 0.28, 3.05, 0.56, chipfc, col if not faded else GREY, r=0.06, ls=ls)
        ax.text(2.22, yy, word, ha="center", va="center", fontsize=10.0, fontweight="bold",
                color=wcol, family="monospace")
        ax.text(4.0, yy, mean, ha="left", va="center", fontsize=9.7, color="#555" if faded else "#333")
        box(ax, 9.5, yy - 0.24, 2.2, 0.48, tagfc, col if not faded else GREY, r=0.22, ls=ls)
        ax.text(10.6, yy, binlab, ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="#777" if faded else "#222")
    fig.savefig(f"{OUT}/gpu_topdown_pipeline.png"); plt.close(fig); print("wrote gpu_topdown_pipeline.png")


# ---------------- Figure 2: the two roofs ----------------
def fig_roofline():
    fig, ax = plt.subplots(figsize=(8.6, 6.3))
    AI = np.logspace(-1, 2, 400); BW, PEAK = 10.0, 100.0
    roof = np.minimum(BW * AI, PEAK); ridge = PEAK / BW
    ax.axvspan(0.1, ridge, color=PINK, alpha=0.07); ax.axvspan(ridge, 100, color=ORANGE, alpha=0.07)
    ax.plot(AI, roof, color="#222", lw=2.4, zorder=5)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.1, 100); ax.set_ylim(1, 220)
    ax.set_xlabel("arithmetic intensity  (FLOP per byte)  —  “how much math per byte fetched”", fontsize=10)
    ax.set_ylabel("achievable performance  (FLOP/s)", fontsize=10.5)
    ax.text(33, 112, "compute roof  —  math-unit peak", ha="center", fontsize=9.6, color="#222", style="italic")
    ax.text(0.74, 11.5, "memory roof\nslope = bandwidth", ha="center", fontsize=9.6, color="#222",
            style="italic", rotation=33)
    ax.text(0.62, 1.7, "MEMORY-BOUND", fontsize=11.5, fontweight="bold", color=PINK)
    ax.text(78, 1.7, "COMPUTE-BOUND", fontsize=11.5, fontweight="bold", color="#b07000", ha="right")
    ax.scatter([1.6], [12], s=170, color=PINK, edgecolor="k", lw=0.8, zorder=10)
    ax.annotate("DECODE\nskinny GEMV + attention\nlong_scoreboard  ·  near the memory roof",
                (1.6, 12), xytext=(2.2, 2.4), fontsize=8.9, color=PINK, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=PINK, lw=1.4))
    ax.scatter([28], [72], s=170, color=ORANGE, edgecolor="k", lw=0.8, zorder=10)
    ax.annotate("PREFILL   (≈ agent prompts)\nbig GEMM  ·  math_pipe_throttle\nnear the compute roof",
                (28, 72), xytext=(34, 17), ha="center", fontsize=8.9, color="#b07000", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.4))
    ax.set_title("The two roofs — which ceiling is the kernel hitting?", fontsize=14, fontweight="bold")
    ax.grid(True, which="both", color="#dddddd", lw=0.5, alpha=0.6); ax.set_axisbelow(True)
    fig.savefig(f"{OUT}/gpu_topdown_roofline.png"); plt.close(fig); print("wrote gpu_topdown_roofline.png")


# ---------------- Figure 3: CPU <-> GPU mirror ----------------
def fig_mirror():
    fig, ax = plt.subplots(figsize=(10, 6.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
    ax.text(5, 6.85, "Why the GPU figure mirrors the CPU top-down", ha="center", fontsize=15, fontweight="bold")
    ax.text(2.5, 6.15, "CPU pipeline slot", ha="center", fontsize=12.5, fontweight="bold", color="#333")
    ax.text(7.5, 6.15, "GPU scheduler slot", ha="center", fontsize=12.5, fontweight="bold", color="#333")
    pairs = [
        ("Retiring",          "Issued",                     GREEN,  "-", False),
        ("Front-end bound",   "no_instruction",             BLUE,   "-", True),
        ("Back-end · Core",   "math/pipe throttle · wait", ORANGE, "-", True),
        ("Back-end · Memory", "long / short scoreboard",    PINK,   "-", True),
        ("Bad speculation",   "(none — N/A on a GPU)",   GREY,   "--", False),
    ]
    y0, dy = 5.3, 1.04
    for i, (l, r, col, ls, mono) in enumerate(pairs):
        yy = y0 - i * dy
        faded = ls == "--"
        fc = light(NA, 0.4) if faded else light(col, 0.16)
        tcol = GREY if faded else col
        box(ax, 0.5, yy - 0.34, 4.0, 0.68, fc, col, r=0.08, ls=ls)
        ax.text(2.5, yy, l, ha="center", va="center", fontsize=11, fontweight="bold", color=tcol)
        box(ax, 5.5, yy - 0.34, 4.0, 0.68, fc, col, r=0.08, ls=ls)
        ax.text(7.5, yy, r, ha="center", va="center", fontsize=10.5, fontweight="bold", color=tcol,
                family="monospace" if mono else "serif")
        arr(ax, 4.55, yy, 5.45, yy, col)
    ax.text(5, 0.30, "same question on both machines: did a useful instruction issue — "
            "and if not, what was everyone waiting on?", ha="center", fontsize=9.4, style="italic", color="#555")
    fig.savefig(f"{OUT}/gpu_topdown_mirror.png"); plt.close(fig); print("wrote gpu_topdown_mirror.png")


fig_pipeline(); fig_roofline(); fig_mirror()
