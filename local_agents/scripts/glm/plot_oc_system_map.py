#!/usr/bin/env python3
"""plot_oc_system_map.py — how OpenClaw works and is packaged, + where our fences sit."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "..", "glm_plots", "oc")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

fig, ax = plt.subplots(figsize=(13.6, 7.2)); ax.axis("off")
ax.set_xlim(0, 13.6); ax.set_ylim(0, 7.2)

def box(x, y, w, h, color, title, lines, alpha=0.14, ls="-", title_size=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.07",
                                facecolor=color, alpha=alpha, edgecolor=color,
                                linewidth=1.6, linestyle=ls))
    ax.text(x + w/2, y + h - 0.14, title, ha="center", va="top",
            fontsize=title_size, fontweight="bold", color=color)
    ax.text(x + w/2, y + h - 0.52, lines, ha="center", va="top", fontsize=7.8, color="#333333")

def arrow(x1, y1, x2, y2, lab, dy=0.14, color="#555555"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, linewidth=1.2, shrinkA=2, shrinkB=2))
    ax.text((x1+x2)/2, (y1+y2)/2 + dy, lab, ha="center", fontsize=7.7, color="#333333")

# ---- the container (packaging) ----
ax.add_patch(FancyBboxPatch((3.6, 0.5), 6.6, 6.2, boxstyle="round,pad=0.1",
                            facecolor="#f2f0f7", alpha=0.5, edgecolor="#555555", linewidth=2))
ax.text(6.9, 6.55, "ONE docker container — image wildclawbench-ubuntu (~28 GB):\n"
        "node runtime + OpenClaw + browsers + python, everything pre-installed",
        ha="center", va="top", fontsize=8.6, color="#333333")

# gateway = the heart
box(4.0, 3.4, 5.8, 2.2, "#6a51a3", "OpenClaw GATEWAY (node.js server — always running)",
    "the assistant's brain-stem:\n"
    "· holds the session + conversation history\n"
    "· message loop: prompt -> model -> parse reply\n"
    "· BUILT-IN tools run in-process: read/write files, memory, notes\n"
    "· dispatcher: spawns EXTERNAL tools as child processes")
box(4.0, 2.1, 2.7, 1.0, "#6a51a3", "agent worker (node)",
    "runs the reasoning loop\nfor this session", alpha=0.10)
box(7.1, 2.1, 2.7, 1.0, "#1b9e77", "spawned tools",
    "chromium / playwright,\nshell, python — on demand", alpha=0.10)

# our fences (dashed overlays)
ax.add_patch(FancyBboxPatch((3.85, 1.95), 3.1, 3.8, boxstyle="round,pad=0.05",
                            facecolor="none", edgecolor="#6a51a3", linewidth=1.4, linestyle="--"))
ax.text(3.95, 1.75, "/agent sub-cgroup (our fence: node family)", fontsize=7.4, color="#6a51a3")
ax.add_patch(FancyBboxPatch((7.0, 1.95), 2.95, 1.35, boxstyle="round,pad=0.05",
                            facecolor="none", edgecolor="#1b9e77", linewidth=1.4, linestyle="--"))
ax.text(7.1, 1.28, "/toolexec sub-cgroup (our fence:\neverything non-node, watcher-sorted)",
        fontsize=7.4, color="#1b9e77")

# host-side boxes (all on the left — flows never cross the container)
box(0.2, 5.0, 2.9, 1.6, "#888888", "WildClawBench runner (host)",
    "sends task brief + workspace\nfiles; collects artifacts;\ngrades -> score")
box(0.2, 2.7, 2.9, 1.5, "#d95f02", "litellm proxy (host scope)",
    "our fence #3: relays +\nlogs every model call")
box(0.2, 0.5, 2.9, 1.5, "#333333", "z.ai GLM-5.2 (cloud)",
    "the model: all thinking\nhappens off-machine", alpha=0.08)

# flows — left rail only
arrow(3.1, 5.9, 4.0, 5.5, "1. task + files", dy=0.16)
arrow(4.0, 5.0, 3.1, 6.15, "6. artifacts -> grade", dy=0.18)
arrow(4.0, 4.2, 3.1, 3.6, "2. model call", dy=0.16)
arrow(1.65, 2.7, 1.65, 2.0, "3. HTTPS; thinking\nstreams back", dy=-0.05)
arrow(6.9, 3.4, 8.2, 3.1, "5. spawn external tool", dy=0.14, color="#1b9e77")
ax.text(6.9, 3.62, "4. built-in tool use (read/write files, memory) executes in-process — inside this fence",
        fontsize=7.6, color="#6a51a3", ha="center")

fig.suptitle("How OpenClaw works and is packaged — one container, gateway-centric; "
             "our measurement fences drawn dashed", fontsize=12.5, y=0.98)
fig.savefig(f"{OUT}/oc_system_map.png"); plt.close(fig)
print("wrote oc_system_map.png")
