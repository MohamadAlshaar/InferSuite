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

# fills/borders sampled from docs/service_pipeline.png; ic = strong accent for icons/arrows
P = {"agent": dict(fc="#eff5ed", ec="#90a78d", tc="#1b5e20", ic="#2e7d32", cb="#577759"),
     "tool":  dict(fc="#fef7ea", ec="#e4c690", tc="#8a5a00", ic="#c77f00", cb="#9c7a3a"),
     "infer": dict(fc="#f5f1f8", ec="#a998c7", tc="#4a3577", ic="#6a51a3", cb="#4a3577"),
     "hw":    dict(fc="#eaf1f9", ec="#7d98b8", tc="#2d4a75", ic="#4a6fa5", cb="#5d646a")}

def plane(y, h, key, label, sub):
    st = P[key]
    ax.add_patch(FancyBboxPatch((2.86, y - 0.06), 8.9, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc="#00000018", ec="none", zorder=0.8))
    ax.add_patch(FancyBboxPatch((2.8, y), 8.9, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=st["fc"], ec=st["ec"], lw=1.6, zorder=1))
    ax.text(3.02, y + h/2 + 0.16, label, fontsize=11.5, fontweight="bold", color=st["tc"],
            va="center", zorder=5)
    ax.text(3.02, y + h/2 - 0.22, sub, fontsize=8, color=st["tc"], va="center", zorder=5)

def box(x, y, w, h, title, sub, fs=10.5, icon=None, ic="#4d4d4d", bec="#4d4d4d"):
    ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                fc="#0000001c", ec="none", zorder=2.5))
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.10",
                                fc="#fdfdfb", ec=bec, lw=1.4, zorder=3))
    if icon:
        title_with_icon(x, w, y + h - 0.34, title, fs, icon, ic)
    else:
        ax.text(x + w/2, y + h - 0.34, title, ha="center", fontsize=fs, fontweight="bold", zorder=5)
    ax.text(x + w/2, y + 0.30, sub, ha="center", fontsize=7.8, color="#444444", zorder=5)

import numpy as np
from matplotlib.path import Path as MPath
from matplotlib.patches import PathPatch, Polygon as MPolygon

def rounded_poly(pts, r):
    """Closed path through pts with rounded corners (quadratic bezier at each vertex)."""
    verts, codes = [], []
    n = len(pts)
    for i in range(n):
        p0, p1, p2 = np.array(pts[i - 1]), np.array(pts[i]), np.array(pts[(i + 1) % n])
        v1, v2 = p1 - p0, p2 - p1
        a = p1 - v1 / np.hypot(*v1) * min(r, np.hypot(*v1) / 2)
        b = p1 + v2 / np.hypot(*v2) * min(r, np.hypot(*v2) / 2)
        verts.append(a); codes.append(MPath.MOVETO if i == 0 else MPath.LINETO)
        verts.append(p1); codes.append(MPath.CURVE3)
        verts.append(b); codes.append(MPath.CURVE3)
    verts.append(verts[0]); codes.append(MPath.CLOSEPOLY)
    return MPath(verts, codes)

def cuboid(x, y, w, h, title, sub, fs=10.5, dx=0.18, dy=0.11, icon=None, ic="#4d4d4d"):
    sil = rounded_poly([(x, y), (x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy),
                        (x + dx, y + h + dy), (x, y + h)], 0.07)
    ax.add_patch(FancyBboxPatch((x + 0.05, y - 0.05), w + dx, h + dy,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc="#0000001c", ec="none", zorder=2.5))
    base = PathPatch(sil, fc="#f6f8fa", ec="none", zorder=3)
    ax.add_patch(base)
    top = MPolygon([(x, y + h), (x + dx, y + h + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                   closed=True, fc="#e9eef4", ec="none", zorder=3.05)
    side = MPolygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)],
                    closed=True, fc="#dae2ec", ec="none", zorder=3.05)
    for f in (top, side):
        ax.add_patch(f); f.set_clip_path(base)
    for seg in ([(x, x + w), (y + h, y + h)], [(x + w, x + w), (y, y + h)],
                [(x + w, x + w + dx), (y + h, y + h + dy)]):
        ax.plot(seg[0], seg[1], color="#5d646a", lw=1.1, solid_capstyle="round", zorder=3.2)
    ax.add_patch(PathPatch(sil, fc="none", ec="#5d646a", lw=1.4, joinstyle="round", zorder=3.3))
    if icon:
        title_with_icon(x, w, y + h - 0.24, title, fs, icon, ic)
    else:
        ax.text(x + w/2, y + h - 0.24, title, ha="center", fontsize=fs, fontweight="bold", zorder=5)
    ax.text(x + w/2, y + 0.09, sub, ha="center", fontsize=7.2, color="#444444", zorder=5)

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

# ---------------- icon glyphs (service-figure style: thin line art in plane color) ----------------
import matplotlib.patches as mpat

def ic_person(cx, cy, c):
    ax.add_patch(mpat.Circle((cx, cy + 0.075), 0.075, fc="none", ec=c, lw=1.6, zorder=6))
    ax.add_patch(mpat.Arc((cx, cy - 0.17), 0.32, 0.3, theta1=0, theta2=180, ec=c, lw=1.6, zorder=6))

def ic_tree(cx, cy, c):  # orchestrator: one node fanning to two
    s = 0.055
    ax.add_patch(mpat.Rectangle((cx - s, cy + 0.02), 2*s, 2*s, fc="none", ec=c, lw=1.4, zorder=6))
    for dx in (-0.12, 0.12):
        ax.add_patch(mpat.Rectangle((cx + dx - s, cy - 0.17), 2*s, 2*s, fc="none", ec=c, lw=1.4, zorder=6))
    ax.plot([cx, cx], [cy + 0.02, cy - 0.03], color=c, lw=1.2, zorder=6)
    ax.plot([cx - 0.12, cx + 0.12], [cy - 0.03, cy - 0.03], color=c, lw=1.2, zorder=6)
    for dx in (-0.12, 0.12):
        ax.plot([cx + dx, cx + dx], [cy - 0.03, cy - 0.06], color=c, lw=1.2, zorder=6)

def ic_doc(cx, cy, c):  # document with text lines
    ax.add_patch(mpat.Rectangle((cx - 0.09, cy - 0.15), 0.18, 0.3, fc="none", ec=c, lw=1.4, zorder=6))
    for dy in (0.06, 0.0, -0.06):
        ax.plot([cx - 0.05, cx + 0.05], [cy + dy, cy + dy], color=c, lw=1.1, zorder=6)

def ic_term(cx, cy, c):  # terminal >_
    ax.add_patch(FancyBboxPatch((cx - 0.14, cy - 0.12), 0.28, 0.24,
                                boxstyle="round,pad=0.015,rounding_size=0.03",
                                fc="none", ec=c, lw=1.4, zorder=6))
    ax.plot([cx - 0.09, cx - 0.04, cx - 0.09], [cy + 0.05, cy, cy - 0.05], color=c, lw=1.3, zorder=6)
    ax.plot([cx + 0.01, cx + 0.09], [cy - 0.05, cy - 0.05], color=c, lw=1.3, zorder=6)

def ic_folder(cx, cy, c):
    ax.add_patch(mpat.Rectangle((cx - 0.13, cy - 0.11), 0.26, 0.18, fc="none", ec=c, lw=1.4, zorder=6))
    ax.plot([cx - 0.13, cx - 0.13, cx - 0.04, cx - 0.01],
            [cy + 0.07, cy + 0.105, cy + 0.105, cy + 0.07], color=c, lw=1.4, zorder=6)

def ic_cube(cx, cy, c):  # 3D box (model workers glyph in the service figure)
    r = 0.115
    w = r * 0.87
    pts = [(cx, cy + r), (cx + w, cy + r/2), (cx + w, cy - r/2), (cx, cy - r), (cx - w, cy - r/2), (cx - w, cy + r/2)]
    ax.add_patch(mpat.Polygon(pts, closed=True, fc="none", ec=c, lw=1.4, zorder=6))
    ax.plot([cx - w, cx, cx + w], [cy + r/2, cy, cy + r/2], color=c, lw=1.2, zorder=6)
    ax.plot([cx, cx], [cy, cy - r], color=c, lw=1.2, zorder=6)

def ic_chip(cx, cy, c):  # processor with pins
    s = 0.085
    ax.add_patch(mpat.Rectangle((cx - s, cy - s), 2*s, 2*s, fc="none", ec=c, lw=1.4, zorder=6))
    ax.add_patch(mpat.Rectangle((cx - 0.035, cy - 0.035), 0.07, 0.07, fc="none", ec=c, lw=1.1, zorder=6))
    for d in (-0.05, 0.0, 0.05):
        ax.plot([cx + d, cx + d], [cy + s, cy + s + 0.045], color=c, lw=1.1, zorder=6)
        ax.plot([cx + d, cx + d], [cy - s, cy - s - 0.045], color=c, lw=1.1, zorder=6)
        ax.plot([cx + s, cx + s + 0.045], [cy + d, cy + d], color=c, lw=1.1, zorder=6)
        ax.plot([cx - s, cx - s - 0.045], [cy + d, cy + d], color=c, lw=1.1, zorder=6)

def ic_monitor(cx, cy, c):  # workstation glyph for the frame title
    ax.add_patch(mpat.Rectangle((cx - 0.15, cy - 0.06), 0.30, 0.22, fc="none", ec=c, lw=1.6, zorder=6))
    ax.plot([cx, cx], [cy - 0.06, cy - 0.13], color=c, lw=1.6, zorder=6)
    ax.plot([cx - 0.08, cx + 0.08], [cy - 0.13, cy - 0.13], color=c, lw=1.6, zorder=6)

CHAR_W = 0.0086  # approx half-char width per fontsize unit (DejaVu bold), axis units

def title_with_icon(x, w, ty, title, fs, icon, ic_color):
    """Centered title with an icon glyph to its left, group-centered like the service cards."""
    half = len(title) * CHAR_W * fs / 2
    tdx = 0.21
    ax.text(x + w/2 + tdx, ty, title, ha="center", fontsize=fs, fontweight="bold", zorder=5)
    icon(x + w/2 + tdx - half - 0.42, ty + 0.02, ic_color)

# ---------------- outer frame: extruded slab, smooth silhouette ----------------
import matplotlib.patches as mpat
DX, DY = 0.30, 0.20
FX0, FY0, FX1, FY1 = 2.55, 0.6, 13.15, 9.15
fsil = rounded_poly([(FX0, FY0), (FX1, FY0), (FX1 + DX, FY0 + DY), (FX1 + DX, FY1 + DY),
                     (FX0 + DX, FY1 + DY), (FX0, FY1)], 0.14)
fbase = PathPatch(fsil, fc="#ccd9e8", ec="none", zorder=0)
ax.add_patch(fbase)
ftop = MPolygon([(FX0, FY1), (FX0 + DX, FY1 + DY), (FX1 + DX, FY1 + DY), (FX1, FY1)],
                closed=True, fc="#dde7f2", ec="none", zorder=0.05)
ax.add_patch(ftop); ftop.set_clip_path(fbase)
ax.plot([FX1, FX1 + DX], [FY1, FY1 + DY], color="#7c99bf", lw=1.2, solid_capstyle="round", zorder=0.1)
ax.add_patch(PathPatch(fsil, fc="none", ec="#7c99bf", lw=1.8, joinstyle="round", zorder=0.1))
ax.add_patch(FancyBboxPatch((FX0, FY0), FX1 - FX0, FY1 - FY0, boxstyle="round,pad=0.02,rounding_size=0.10",
                            fc="#f5f8fb", ec="#7c99bf", lw=1.8, zorder=0.2))
ic_monitor(3.02, 8.86, "#2d4a75")
ax.text(3.32, 8.82, "Workstation (single node)", fontsize=13.5, fontweight="bold", color="#2d4a75")
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
box(AX, 6.75, AW, 1.2, "Agent harness", "SWE-agent / BCB driver / OpenClaw\nparses replies, dispatches tools",
    icon=ic_tree, ic=P["agent"]["ic"], bec=P["agent"]["cb"])
box(BX, 6.75, BW, 1.2, "History / context", "grows every turn until\nthe window fills",
    icon=ic_doc, ic=P["agent"]["ic"], bec=P["agent"]["cb"])
box(AX, 4.65, AW, 1.2, "Tool sandbox", "container: shell, edit, pytest,\nbrowser, file I/O",
    icon=ic_term, ic=P["tool"]["ic"], bec=P["tool"]["cb"])
box(BX, 4.65, BW, 1.2, "Workspace", "repo checkout,\nartifacts, results",
    icon=ic_folder, ic=P["tool"]["ic"], bec=P["tool"]["cb"])
box(AX, 2.55, AW, 1.2, "vLLM engine", "self-served 7B / 32B\n(or remote frontier API)",
    icon=ic_cube, ic=P["infer"]["ic"], bec=P["infer"]["cb"])
cuboid(5.55, 1.12, 2.0, 0.55, "CPU cores", "harness / tools / engine", fs=9.5, icon=ic_chip, ic=P["hw"]["ic"])
cuboid(9.05, 1.12, 1.7, 0.55, "GPU", "generation", fs=9.5, icon=ic_chip, ic=P["hw"]["ic"])

# ---------------- measurement fences ----------------
fence(AX-0.12, 6.63, AW+0.24, 1.44)
fence(AX-0.12, 4.53, AW+0.24, 1.44)
fence(AX-0.12, 2.43, AW+0.24, 1.44)
ax.text(11.55, 2.18, "red dashes: perf cgroup fences (harness / tools / engine), measured in the same windows",
        fontsize=7.4, color="#c0392b", ha="right", style="italic", zorder=6,
        path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

# ---------------- client (outside, like the service figure) ----------------
ax.add_patch(FancyBboxPatch((0.405, 6.545), 1.75, 1.5, boxstyle="round,pad=0.05,rounding_size=0.10",
                            fc="#00000026", ec="none", zorder=2.5))
ax.add_patch(FancyBboxPatch((0.35, 6.6), 1.75, 1.5, boxstyle="round,pad=0.05,rounding_size=0.10",
                            fc="#fdfdfb", ec="#4d4d4d", lw=1.4, zorder=3))
ic_person(1.225, 7.72, "#333333")
ax.text(1.225, 7.22, "Client /\nBenchmark", ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=5)
ax.text(1.225, 6.78, "SWE-bench / BCB /\nWildClawBench", ha="center", va="center", fontsize=7.2,
        color="#444444", zorder=5)
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
# hardware drops: every plane's scope runs on the CPU cores — one dashed drop per plane,
# left corridor, plane accent colors (harness / tools / engine host)
for x_c, y_exit, y_in, col in [(4.45, 7.35, 1.52, P["agent"]["ic"]),
                               (4.65, 4.95, 1.40, P["tool"]["ic"]),
                               (4.78, 3.00, 1.28, P["infer"]["ic"])]:
    ax.plot([5.0, x_c], [y_exit, y_exit], color=col, lw=1.7, ls=(0, (5, 3)), zorder=4)
    ax.plot([x_c, x_c], [y_exit, y_in], color=col, lw=1.7, ls=(0, (5, 3)), zorder=4)
    ax.add_patch(FancyArrowPatch((x_c, y_in), (5.55, y_in), arrowstyle="-|>", mutation_scale=15,
                                 lw=1.7, color=col, ls=(0, (5, 3)), zorder=4))
ax.plot([8.1, 9.9], [2.75, 2.75], color="#6a51a3", lw=1.7, ls=(0, (5, 3)), zorder=4)
ax.add_patch(FancyArrowPatch((9.9, 2.75), (9.9, 1.82), arrowstyle="-|>", mutation_scale=15,
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
