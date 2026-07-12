#!/usr/bin/env python3
"""plot_oc_calendar.py — first-look figures for the OpenClaw calendar episode (GLM-5.2).
ONE certified episode (glm_oc_calendar/run_1); single-episode caveat in every title.
House palette + locked vocabulary. Run with system python3."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RD = os.path.join(HERE, "..", "..", "data", "glm_oc_calendar", "run_1")
OUT = os.path.join(HERE, "..", "..", "glm_plots", "oc"); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35})
C_AGENT, C_TOOL, C_PROXY, C_WAIT = "#6a51a3", "#1b9e77", "#d95f02", "#cccccc"

def series(i):
    rows = [(float(p[0]), float(p[2])) for p in (l.split() for l in open(f"{RD}/cpustat_scope{i}.tsv"))
            if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0]
    t = np.array([r[0] for r in rows]); u = np.array([r[1] for r in rows])
    mid = (t[1:] + t[:-1]) / 2
    rate = np.maximum(0, np.diff(u) / 1e6 / np.maximum(np.diff(t), 1e-9))
    return mid, rate, (u[-1] - u[0]) / 1e6, t[-1] - t[0]

S = {i: series(i) for i in (1, 2, 3)}
wall = max(S[i][3] for i in S)

# ---- O1: two-view (single episode) --------------------------------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
agent_active = float(np.sum(S[1][1] > 0.05) * 0.1)
tool_active = float(np.sum(S[2][1] > 0.05) * 0.1)
wait = max(wall - agent_active - tool_active, 0)
a1.pie([wait, agent_active, tool_active], colors=[C_WAIT, C_AGENT, C_TOOL], startangle=90,
       counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
       autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", pctdistance=0.76)
a1.text(0, 0, f"{wall/60:.1f} min", ha="center", va="center", fontsize=12, fontweight="bold")
a1.set_title("share of wall time", fontsize=11)
cs = [S[1][2], S[2][2], S[3][2]]
a2.pie([cs[0], cs[1], cs[2]], colors=[C_AGENT, C_TOOL, C_PROXY], startangle=90,
       counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
       autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", pctdistance=0.76)
a2.text(0, 0.12, f"{sum(cs):.0f}\ncore-sec", ha="center", va="center", fontsize=11, fontweight="bold")
a2.text(0, -0.34, f"= {sum(cs)/wall:.3f} cores\navg usage", ha="center", va="center",
        fontsize=8, color="#666666")
a2.set_title("share of CPU work", fontsize=11)
for ax in (a1, a2): ax.set_aspect("equal")
fig.legend(handles=[Patch(fc=C_WAIT, label="model round-trip (thinking)"),
                    Patch(fc=C_AGENT, label="agent runtime (gateway, node)"),
                    Patch(fc=C_TOOL, label="tool execution"),
                    Patch(fc=C_PROXY, label="API proxy (streaming relay)")],
           ncol=2, loc="lower center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.10))
fig.suptitle("OpenClaw calendar, GLM-5.2 — one certified episode (solved, score 1.00): "
             "tool phase absent, proxy dominates CPU", fontsize=11.5, y=1.03)
fig.savefig(f"{OUT}/oc_two_view.png"); plt.close(fig)

# ---- O2: fence timeline --------------------------------------------------------------------------
fig, axs = plt.subplots(3, 1, figsize=(12.0, 5.6), sharex=True)
t0 = min(S[i][0][0] for i in S)
for ax, (i, name, col) in zip(axs, [(3, "API proxy (streams GLM thinking)", C_PROXY),
                                    (1, "agent runtime (node family)", C_AGENT),
                                    (2, "tool execution", C_TOOL)]):
    x = (S[i][0] - t0) / 60
    ax.fill_between(x, 0, S[i][1], color=col, linewidth=0.4, alpha=0.9)
    ax.set_ylabel(name.split(" (")[0], fontsize=9, rotation=0, ha="right", va="center")
    ax.set_ylim(0, max(0.12, float(S[i][1].max()) * 1.15))
    ax.grid(axis="x", alpha=0.3)
axs[-1].set_xlabel("episode time (minutes) — each step = one 0.1 s sample")
fig.supylabel("CPU usage (cores) per 0.1 s sample", fontsize=10, x=0.02)
fig.suptitle("OpenClaw calendar episode timeline — proxy streaming bouts mirror GLM's thinking; "
             "agent blips between turns; tools silent", fontsize=11.5, y=0.97)
fig.savefig(f"{OUT}/oc_timeline.png"); plt.close(fig)

# ---- O3: what ran in each fence ------------------------------------------------------------------
def comm_table(i):
    rows = []
    for ln in open(f"{RD}/scope{i}_comm.txt"):
        p = ln.split()
        if p and p[0].endswith("%"):
            share = float(p[0].rstrip("%")); name = " ".join(p[1:])
            if name.startswith(":"): name = "node worker threads"
            rows.append((name, share))
    agg = {}
    for n, s in rows: agg[n] = agg.get(n, 0) + s
    return sorted(agg.items(), key=lambda kv: -kv[1])[:6]

fig, axs = plt.subplots(1, 2, figsize=(11.0, 3.4))
for ax, (i, ttl, col) in zip(axs, [(1, "agent fence — processes (share of CPU samples)", C_AGENT),
                                   (3, "proxy fence — processes (share of CPU samples)", C_PROXY)]):
    tab = comm_table(i)
    ax.barh(range(len(tab)), [s for _, s in tab], color=col, height=0.55, edgecolor="white")
    ax.set_yticks(range(len(tab))); ax.set_yticklabels([n for n, _ in tab], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_title(ttl, fontsize=10)
    for j, (_, s) in enumerate(tab):
        ax.text(s, j, f" {s:.0f}%", va="center", fontsize=8.5)
fig.suptitle("What ran inside the fences (tool fence omitted: 0.0 core-sec — one 10 ms python blip)",
             fontsize=11.5, y=1.04)
fig.savefig(f"{OUT}/oc_what_ran.png"); plt.close(fig)
print("wrote oc figures")
