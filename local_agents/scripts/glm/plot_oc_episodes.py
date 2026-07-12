#!/usr/bin/env python3
"""plot_oc_episodes.py — OC verification episodes (calendar + link-a-pix), GLM-5.2.
Two certified episodes; single-episode caveat in titles. System python3."""
import json, os
from datetime import datetime, timezone
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "glm_plots", "oc"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 11, "figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.35})
C_AGENT, C_TOOL, C_PROXY, C_WAIT = "#6a51a3", "#1b9e77", "#d95f02", "#cccccc"

EPS = [("calendar (solved 1.00)", "glm_oc_calendar/run_1"),
       ("link-a-pix easy (0.41: image 0, description 0.83)", "glm_oc_linkapix/run_1"),
       ("jigsaw medium (0.85; agent-side E4-bounded)", "glm_oc_jigsaw-med/run_1")]

def series(rd, i):
    rows = [(float(p[0]), float(p[2])) for p in (l.split() for l in open(f"{rd}/cpustat_scope{i}.tsv"))
            if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0]
    t = np.array([r[0] for r in rows]); u = np.array([r[1] for r in rows])
    return (t[1:] + t[:-1]) / 2, np.maximum(0, np.diff(u) / 1e6 / np.maximum(np.diff(t), 1e-9)), \
           (u[-1] - u[0]) / 1e6, t[-1] - t[0], t[0]

# ---- two-view: both episodes ----
fig, axes = plt.subplots(3, 2, figsize=(9.6, 11.0))
for row, (name, rd0) in enumerate(EPS):
    rd = f"{DATA}/{rd0}"
    S = {i: series(rd, i) for i in (1, 2, 3)}
    wall = max(S[i][3] for i in S)
    aa = float(np.sum(S[1][1] > 0.05) * 0.1); ta = float(np.sum(S[2][1] > 0.05) * 0.1)
    ax = axes[row, 0]
    ax.pie([max(wall-aa-ta, 0), aa, ta], colors=[C_WAIT, C_AGENT, C_TOOL], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", pctdistance=0.76)
    ax.text(0, 0, f"{wall/60:.1f} min", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_ylabel(name.split(" (")[0], fontsize=11, fontweight="bold")
    if row == 0: ax.set_title("share of wall time", fontsize=11)
    cs = [S[1][2], S[2][2], S[3][2]]
    ax = axes[row, 1]
    ax.pie(cs, colors=[C_AGENT, C_TOOL, C_PROXY], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           autopct=lambda p: f"{p:.0f}%" if p >= 5 else "", pctdistance=0.76)
    ax.text(0, 0.12, f"{sum(cs):.0f}\ncore-sec", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0, -0.35, f"= {sum(cs)/wall:.3f} cores avg", ha="center", va="center", fontsize=7.5, color="#666666")
    if row == 0: ax.set_title("share of CPU work", fontsize=11)
for ax in axes.flat: ax.set_aspect("equal")
fig.legend(handles=[Patch(fc=C_WAIT, label="model round-trip (thinking)"),
                    Patch(fc=C_AGENT, label="agent runtime"),
                    Patch(fc=C_TOOL, label="tool execution"),
                    Patch(fc=C_PROXY, label="API proxy (streaming relay)")],
           ncol=4, loc="lower center", frameon=False, fontsize=8.8, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("OpenClaw x GLM-5.2 — three certified episodes: pure reasoning / code loop / toolchain build",
             fontsize=12, y=0.955)
fig.savefig(f"{OUT}/oc_two_view.png"); plt.close(fig)

# ---- link-a-pix orchestration timeline with transcript events ----
rd = f"{DATA}/glm_oc_linkapix/run_1"
S = {i: series(rd, i) for i in (1, 2, 3)}
t0 = min(S[i][4] for i in S)
CHAT = ("/home/mohamad/llm-service-kernel-latest/agentic/openclaw/external/WildClawBench/output/openclaw/"
        "02_Code_Intelligence/02_Code_Intelligence_task_9_link_a_pix_color_easy_zh/glm-5.2_20260710_1330_40e2e1/chat.jsonl")
ev = []
for ln in open(CHAT):
    m = json.loads(ln)
    msg = m.get("message")
    if not msg: continue
    c = msg.get("content")
    for b in (c if isinstance(c, list) else []):
        if isinstance(b, dict) and b.get("type") == "toolCall":
            ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")).timestamp()
            kind = "exec" if b.get("name") == "exec" else "builtin"
            ev.append(((ts - t0) / 60, kind, b.get("name")))

fig, axs = plt.subplots(3, 1, figsize=(12.2, 6.0), sharex=True)
for ax, (i, name, col) in zip(axs, [(3, "API proxy", C_PROXY), (1, "agent runtime", C_AGENT),
                                    (2, "tool execution", C_TOOL)]):
    x = (S[i][0] - t0) / 60
    ax.fill_between(x, 0, S[i][1], color=col, linewidth=0.4, alpha=0.9)
    ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
    ax.set_ylim(0, max(0.12, float(S[i][1].max()) * 1.3))
    ax.grid(axis="x", alpha=0.3)
ymax = axs[2].get_ylim()[1]
for x, kind, name in ev:
    ax = axs[2] if kind == "exec" else axs[1]
    ax.axvline(x, color="#333333", linewidth=0.7, alpha=0.5)
    ax.plot([x], [ax.get_ylim()[1] * 0.92], marker="v", color="#333333", markersize=4)
for x, kind, name in ev:
    if name == "exec": continue
axs[0].annotate("4-minute thinking bout\n(model reasons remotely; proxy streams it)",
                xy=(2.2, axs[0].get_ylim()[1]*0.55), fontsize=8.5, color="#666666", ha="center")
axs[-1].set_xlabel("episode time (minutes) — markers = tool calls from the agent's own transcript "
                   "(on agent strip: built-ins; on tool strip: execs)")
fig.supylabel("CPU usage (cores) per 0.1 s sample", fontsize=10, x=0.015)
fig.suptitle("link-a-pix orchestration — transcript events joined onto fence activity:\n"
             "read -> think 4 min -> write solver -> exec -> edit -> exec -> inspect -> describe",
             fontsize=11.5, y=1.0)
fig.savefig(f"{OUT}/oc_timeline_linkapix.png"); plt.close(fig)
print("wrote oc figures")


# ---- jigsaw orchestration timeline ----
import glob as _glob
rd = f"{DATA}/glm_oc_jigsaw-med/run_1"
S = {i: series(rd, i) for i in (1, 2, 3)}
t0 = min(S[i][4] for i in S)
cdir = sorted(_glob.glob("/home/mohamad/llm-service-kernel-latest/agentic/openclaw/external/WildClawBench/"
              "output/openclaw/02_Code_Intelligence/02_Code_Intelligence_task_4_jigsaw_puzzle_medium_zh/glm-5.2_*"))[-1]
ev = []
for ln in open(f"{cdir}/chat.jsonl"):
    m = json.loads(ln)
    msg = m.get("message") or {}
    for b in (msg.get("content") if isinstance(msg.get("content"), list) else []):
        if isinstance(b, dict) and b.get("type") == "toolCall":
            ts = datetime.fromisoformat(m["timestamp"].replace("Z", "+00:00")).timestamp()
            ev.append(((ts - t0) / 60, "exec" if b.get("name") == "exec" else "builtin"))
fig, axs = plt.subplots(3, 1, figsize=(12.6, 6.2), sharex=True)
for ax, (i, name, col) in zip(axs, [(3, "API proxy", C_PROXY), (1, "agent runtime", C_AGENT),
                                    (2, "tool execution", C_TOOL)]):
    x = (S[i][0] - t0) / 60
    ax.fill_between(x, 0, S[i][1], color=col, linewidth=0.4, alpha=0.9)
    ax.set_ylabel(name, fontsize=9, rotation=0, ha="right", va="center")
    ax.set_ylim(0, max(0.12, float(S[i][1].max()) * 1.3))
    ax.grid(axis="x", alpha=0.3)
for x, kind in ev:
    ax = axs[2] if kind == "exec" else axs[1]
    ax.axvline(x, color="#333333", linewidth=0.6, alpha=0.45)
    ax.plot([x], [ax.get_ylim()[1] * 0.92], marker="v", color="#333333", markersize=3.5)
axs[-1].set_xlabel("episode time (minutes) — markers: built-ins on agent strip, execs (33: pip/apt installs,"
                   " OCR, solvers) on tool strip")
fig.supylabel("CPU usage (cores) per 0.1 s sample", fontsize=10, x=0.015)
fig.suptitle("jigsaw-medium orchestration — GLM builds itself an OCR toolchain, then iterates solvers "
             "(score 0.85, 18.4 min)", fontsize=11.5, y=0.98)
fig.savefig(f"{OUT}/oc_timeline_jigsaw.png"); plt.close(fig)

# ---- spectrum: who owns the CPU, per episode ----
fig, ax = plt.subplots(figsize=(10.6, 3.4))
labels, vals = [], []
for name, rd0 in EPS:
    rd = f"{DATA}/{rd0}"
    S = {i: series(rd, i) for i in (1, 2, 3)}
    labels.append(name.split(" (")[0] + f"\n{max(S[i][3] for i in S)/60:.0f} min")
    vals.append([S[1][2], S[2][2], S[3][2]])
vals = np.array(vals)
left = np.zeros(len(EPS))
for k, (lab, col) in enumerate([("agent runtime", C_AGENT), ("tool execution", C_TOOL),
                                ("API proxy", C_PROXY)]):
    ax.barh(range(len(EPS)), vals[:, k], left=left, color=col, height=0.55,
            edgecolor="white", linewidth=0.8, label=lab)
    left += vals[:, k]
for i, tot in enumerate(left):
    ax.text(tot, i, f" {tot:.0f} core-s", va="center", fontsize=9, color="#333333")
ax.set_yticks(range(len(EPS))); ax.set_yticklabels(labels, fontsize=9.5)
ax.invert_yaxis(); ax.set_xlabel("CPU work (core-seconds)")
ax.legend(ncol=3, fontsize=9, frameon=False, loc="upper right")
ax.set_title("One agent, one model, three tasks — the task flips the CPU regime "
             "(proxy-dominated -> tool-dominated)", fontsize=11.5, pad=10)
fig.savefig(f"{OUT}/oc_spectrum.png"); plt.close(fig)
print("wrote jigsaw + spectrum figures")
