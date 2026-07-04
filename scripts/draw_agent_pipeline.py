#!/usr/bin/env python3
"""Agent-loop pipeline diagram, v3 — disciplined like docs/service_pipeline.png:
stacked full-width planes, strictly orthogonal arrows, loop channel on the right.
Output: docs/agent_pipeline_draft.png (draft, not in thesis)."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import matplotlib.patheffects as pe

plt.rcParams.update({"font.family": "DejaVu Sans", "figure.dpi": 150, "savefig.dpi": 300})
fig, ax = plt.subplots(figsize=(13.6, 9.6))
ax.set_xlim(0, 13.6); ax.set_ylim(0, 9.6); ax.axis("off")

P = {"agent": dict(fc="#eaf3ea", ec="#2e7d32", tc="#1b5e20"),
     "tool":  dict(fc="#fdf4de", ec="#c77f00", tc="#8a5a00"),
     "infer": dict(fc="#f1ebf8", ec="#6a51a3", tc="#4a3577"),
     "hw":    dict(fc="#e9eff7", ec="#4a6fa5", tc="#2d4a75")}

def plane(y, h, key, label, sub):
    st = P[key]
    ax.add_patch(FancyBboxPatch((2.8, y), 8.9, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=st["fc"], ec=st["ec"], lw=1.6, zorder=1))
    ax.text(3.02, y + h/2 + 0.16, label, fontsize=11.5, fontweight="bold", color=st["tc"],
            va="center", zorder=5)
    ax.text(3.02, y + h/2 - 0.22, sub, fontsize=8, color=st["tc"], va="center", zorder=5)

def box(x, y, w, h, title, sub, fs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                fc="#fdfdfb", ec="#4d4d4d", lw=1.4, zorder=3))
    ax.text(x + w/2, y + h - 0.34, title, ha="center", fontsize=fs, fontweight="bold", zorder=5)
    ax.text(x + w/2, y + 0.30, sub, ha="center", fontsize=7.8, color="#444444", zorder=5)

def vline(x, y0, y1, label=None, color="#333333", dashed=False, lx=None, ly=None):
    ax.add_patch(FancyArrowPatch((x, y0), (x, y1), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.7, color=color, ls=(0, (5, 3)) if dashed else "-", zorder=4))
    if label:
        ax.text(lx if lx else x, ly if ly else (y0+y1)/2, label, fontsize=8.2, ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#999999", lw=0.8))

def hline(x0, x1, y, label=None, color="#333333", dashed=False, lx=None, ly=None, arrow=True):
    if arrow:
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>", mutation_scale=15,
                                     lw=1.7, color=color, ls=(0, (5, 3)) if dashed else "-", zorder=4))
    else:
        ax.plot([x0, x1], [y, y], color=color, lw=1.7, ls=(0, (5, 3)) if dashed else "-", zorder=4)
    if label:
        ax.text(lx if lx else (x0+x1)/2, ly if ly else y + 0.24, label, fontsize=8.2, ha="center",
                va="center", zorder=6,
                bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#999999", lw=0.8))

def fence(x, y, w, h):
    ax.add_patch(Rectangle((x, y), w, h, fc="none", ec="#c0392b", lw=1.2, ls=(0, (2, 2)), zorder=2))

# ---------------- outer frame ----------------
ax.add_patch(FancyBboxPatch((2.55, 0.6), 10.6, 8.55, boxstyle="round,pad=0.08,rounding_size=0.18",
                            fc="#f5f8fb", ec="#8aa0bb", lw=1.8, zorder=0))
ax.text(2.9, 8.82, "Workstation (single node)", fontsize=13.5, fontweight="bold", color="#2d4a75")
ax.text(7.7, 8.42, "one turn:  generate → dispatch → execute → observe        repeated until submit or context limit",
        fontsize=9, ha="center", color="#555555")

# ---------------- planes (full-width bands) ----------------
plane(6.55, 1.6, "agent", "Agent plane", "orchestration")
plane(4.45, 1.6, "tool",  "Tool plane",  "delegated work")
plane(2.35, 1.6, "infer", "Inference plane", "token generation")
plane(0.85, 1.1, "hw",    "Hardware", "")

# ---------------- boxes on a strict grid ----------------
AX, AW = 5.0, 3.1     # left column (harness / sandbox / engine share x)
BX, BW = 8.6, 2.6     # right column (history / workspace / api)
box(AX, 6.75, AW, 1.2, "Agent harness", "SWE-agent / BCB driver / OpenClaw\nparses replies, dispatches tools")
box(BX, 6.75, BW, 1.2, "History / context", "grows every turn until\nthe window fills")
box(AX, 4.65, AW, 1.2, "Tool sandbox", "container: shell, edit, pytest,\nbrowser, file I/O")
box(BX, 4.65, BW, 1.2, "Workspace", "repo checkout,\nartifacts, results")
box(AX, 2.55, AW, 1.2, "vLLM engine", "self-served 7B / 32B\n(or remote frontier API)")
box(AX, 0.95, AW, 0.9, "CPU cores", "harness + tools", fs=10)
box(BX, 0.95, BW, 0.9, "GPU", "generation", fs=10)

# ---------------- measurement fences ----------------
fence(AX-0.12, 6.63, AW+0.24, 1.44)
fence(AX-0.12, 4.53, AW+0.24, 1.44)
fence(AX-0.12, 2.43, AW+0.24, 1.44)
ax.text(11.55, 2.18, "red dashes: perf cgroup fences (harness / tools / engine), measured in the same windows",
        fontsize=7.4, color="#c0392b", ha="right", style="italic", zorder=6,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

# ---------------- task client (outside, like the service figure) ----------------
box(0.35, 6.75, 1.75, 1.2, "Benchmark\ntask", "SWE-bench / BCB /\nWildClawBench", fs=9.5)
hline(2.1, AX, 7.75, "task prompt", ly=8.0)
hline(AX, 2.1, 7.0, "patch / answer", dashed=True, ly=6.62)

# ---------------- loop arrows, orthogonal, routed around boxes ----------------
# 1. context assembled in history feeds the engine, via the right corridor
CH1, CH2 = 12.05, 12.55
ax.plot([11.2, CH1], [7.15, 7.15], color="#6a51a3", lw=1.7, zorder=4)
ax.plot([CH1, CH1], [7.15, 3.35], color="#6a51a3", lw=1.7, zorder=4)
ax.add_patch(FancyArrowPatch((CH1, 3.35), (8.1, 3.35), arrowstyle="-|>", mutation_scale=15,
                             lw=1.7, color="#6a51a3", zorder=4))
ax.text(CH1, 5.3, "1. prompt + context", fontsize=8.2, ha="center", va="center", rotation=90, zorder=6,
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#999999", lw=0.8))
# 2. generated action returns to the harness from below
ax.plot([8.1, CH2], [2.95, 2.95], color="#2e7d32", lw=1.7, zorder=4)
ax.plot([CH2, CH2], [2.95, 6.3], color="#2e7d32", lw=1.7, zorder=4)
ax.plot([CH2, 7.75], [6.3, 6.3], color="#2e7d32", lw=1.7, zorder=4)
ax.add_patch(FancyArrowPatch((7.75, 6.3), (7.75, 6.75), arrowstyle="-|>", mutation_scale=15,
                             lw=1.7, color="#2e7d32", zorder=4))
ax.text(CH2, 5.3, "2. next action (tool call)", fontsize=8.2, ha="center", va="center", rotation=90, zorder=6,
        bbox=dict(boxstyle="round,pad=0.24", fc="white", ec="#999999", lw=0.8))
# 3./4. harness <-> sandbox
vline(5.9, 6.75, 5.85, "3. execute command", lx=5.35, ly=6.28)
vline(7.0, 5.85, 6.75, "4. observation", lx=8.6, ly=6.42)
# sandbox -> workspace
hline(8.1, 8.6, 5.25, None, color="#888888")
# hardware drops, routed around the engine column
ax.plot([5.0, 4.55], [4.95, 4.95], color="#c77f00", lw=1.7, ls=(0, (5, 3)), zorder=4)
ax.plot([4.55, 4.55], [4.95, 1.4], color="#c77f00", lw=1.7, ls=(0, (5, 3)), zorder=4)
ax.add_patch(FancyArrowPatch((4.55, 1.4), (5.0, 1.4), arrowstyle="-|>", mutation_scale=15,
                             lw=1.7, color="#c77f00", ls=(0, (5, 3)), zorder=4))
ax.plot([8.1, 9.9], [2.75, 2.75], color="#6a51a3", lw=1.7, ls=(0, (5, 3)), zorder=4)
ax.add_patch(FancyArrowPatch((9.9, 2.75), (9.9, 1.85), arrowstyle="-|>", mutation_scale=15,
                             lw=1.7, color="#6a51a3", ls=(0, (5, 3)), zorder=4))

# ---------------- legend ----------------
lx = 2.9
for key, lab, sub in [("agent", "Agent plane", "orchestration"),
                      ("tool", "Tool plane", "delegated work"),
                      ("infer", "Inference plane", "token generation"),
                      ("hw", "Hardware", "CPU + GPU")]:
    st = P[key]
    ax.add_patch(FancyBboxPatch((lx, 0.12), 0.34, 0.3, boxstyle="round,pad=0.02",
                                fc=st["fc"], ec=st["ec"], lw=1.4))
    ax.text(lx + 0.45, 0.36, lab, fontsize=9, fontweight="bold", va="center")
    ax.text(lx + 0.45, 0.14, sub, fontsize=7.5, color="#555555", va="center")
    lx += 2.6

fig.savefig("docs/agent_pipeline_draft.png", bbox_inches="tight", facecolor="white")
print("wrote docs/agent_pipeline_draft.png")
