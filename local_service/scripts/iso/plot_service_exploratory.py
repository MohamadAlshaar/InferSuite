#!/usr/bin/env python3
"""Service exploratory figures (adopted 2026-07-17) — NOT thesis-featured; banked under
plots_iso/extra/. All three read the priv counter-group windows (task-clock,
context-switches, cycles user/kernel split) per pod cgroup, summed over every cell of the
isolated campaign (12 cells x 3 repeats):

  svc_avg_util.png      per-cell CPU usage stacked by pod, as cores and % of the partition
  svc_ctx_switches.png  context switches per CPU-second, per pod
  svc_os_share.png      kernel share of cycles, per pod

Run with SYSTEM python3."""
import json, os, re
from glob import glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data_iso")
OUT = os.path.join(HERE, "..", "..", "plots_iso", "extra")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})
BUCKETS = ["short", "medium", "long", "very_long"]
TIERS = [64, 192, 320]
PODS = ["vllm", "fastapi", "milvus", "mongodb", "seaweed_filer", "seaweed_volume"]
PCOL = {"vllm": "#c9c9c9", "fastapi": "#6a51a3", "milvus": "#CC79A7",
        "mongodb": "#e6ab02", "seaweed_filer": "#66a61e", "seaweed_volume": "#1b9e77"}
PLAB = {"vllm": "vLLM host (busy-wait)", "fastapi": "fastapi (RAG + embed)", "milvus": "milvus",
        "mongodb": "mongodb", "seaweed_filer": "seaweed filer", "seaweed_volume": "seaweed volume"}
NCORES, WINSEC = 20, 10
LINE = re.compile(r"\s*([\d,.]+)\s+(msec task-clock|task-clock|context-switches|cycles:u|cycles:k)\s+(\S+)")

# accumulate priv-group counters per pod: [task-clock s, ctx, cycles:u, cycles:k, n_windows]
ACC = {p: [0.0, 0.0, 0.0, 0.0, 0] for p in PODS}
CELL = {}   # (bucket,tier) -> {pod: [task-clock s, windows]}
for b in BUCKETS:
    for t in TIERS:
        cell = {p: [0.0, 0] for p in PODS}
        for rd in sorted(glob(f"{DATA}/svc_{b}_tok{t}/run_*")):
            if not os.path.exists(f"{rd}/DONE"): continue
            cg2pod = {v: k for k, v in json.load(open(f"{rd}/metadata.json"))["pods"].items()}
            for f in glob(f"{rd}/group_priv_w*.txt"):
                txt = open(f).read()
                if "<not counted>" in txt: continue
                for ln in txt.splitlines():
                    m = LINE.match(ln)
                    if not m: continue
                    pod = cg2pod.get(m.group(3))
                    if not pod: continue
                    v, ev = float(m.group(1).replace(",", "")), m.group(2)
                    if "task-clock" in ev:
                        ACC[pod][0] += v / 1000.0; ACC[pod][4] += 1
                        cell[pod][0] += v / 1000.0; cell[pod][1] += 1
                    elif ev == "context-switches": ACC[pod][1] += v
                    elif ev == "cycles:u": ACC[pod][2] += v
                    elif ev == "cycles:k": ACC[pod][3] += v
        CELL[(b, t)] = cell

# ---------------- 1: per-cell utilization, stacked by pod ----------------
fig, ax = plt.subplots(figsize=(12.4, 4.6))
xs = np.arange(len(BUCKETS) * len(TIERS)); bot = np.zeros(len(xs))
for p in PODS:
    v = []
    for b in BUCKETS:
        for t in TIERS:
            tc, nw = CELL[(b, t)][p]
            v.append(tc / (nw * WINSEC) if nw else 0.0)   # mean cores in the pod's windows
    v = np.array(v)
    ax.bar(xs, v, 0.62, bottom=bot, color=PCOL[p], label=PLAB[p], edgecolor="white", linewidth=0.6)
    bot += v
for i, tot in enumerate(bot):
    ax.text(i, tot + 0.05, f"{tot:.2f}\n({100*tot/NCORES:.0f}%)", ha="center", fontsize=7.4, color="#333333")
ax.set_xticks(xs)
ax.set_xticklabels([f"{t}out" for b in BUCKETS for t in TIERS], fontsize=8)
for i, b in enumerate(BUCKETS):
    ax.text(i * 3 + 1, -0.62, {"short": "9-26 tokens in", "medium": "~150 tokens in",
                               "long": "~435 tokens in", "very_long": "~720 tokens in"}[b],
            ha="center", fontsize=9, color="#555555")
    if i: ax.axvline(i * 3 - 0.5, color="#cccccc", linewidth=0.7)
ax.set_ylabel("CPU usage (cores)")
ax.set_ylim(0, max(bot) * 1.3)
sec = ax.secondary_yaxis("right", functions=(lambda v: 100 * v / NCORES, lambda p: p * NCORES / 100))
sec.set_ylabel("utilization of the 20-core partition (%)")
ax.legend(ncol=3, fontsize=8.2, frameon=False, loc="upper left")
ax.set_title("Service CPU utilization under load, per cell", pad=12)
fig.savefig(f"{OUT}/svc_avg_util.png"); plt.close(fig)
print("wrote svc_avg_util.png")

# ---------------- 2: context switches per CPU-second, per pod ----------------
fig, ax = plt.subplots(figsize=(8.8, 4.2))
xs = np.arange(len(PODS))
v = [ACC[p][1] / ACC[p][0] if ACC[p][0] > 0.05 else 0 for p in PODS]
ax.bar(xs, v, 0.6, color=[PCOL[p] for p in PODS], edgecolor="white")
for xi, vi in zip(xs, v):
    ax.text(xi, vi * 1.02, f"{vi:,.0f}", ha="center", fontsize=8.4, color="#333333")
ax.set_xticks(xs); ax.set_xticklabels([PLAB[p] for p in PODS], fontsize=8, rotation=12)
ax.set_ylabel("context switches per CPU-second")
ax.set_title("Context-switch rate per pod (all cells, priv windows)", pad=10)
fig.savefig(f"{OUT}/svc_ctx_switches.png"); plt.close(fig)
print("wrote svc_ctx_switches.png")

# ---------------- 3: OS share of cycles, per pod ----------------
fig, ax = plt.subplots(figsize=(8.8, 4.2))
v = [100 * ACC[p][3] / (ACC[p][2] + ACC[p][3]) if ACC[p][2] + ACC[p][3] else 0 for p in PODS]
ax.bar(xs, v, 0.6, color=[PCOL[p] for p in PODS], edgecolor="white")
for xi, vi in zip(xs, v):
    ax.text(xi, vi + 0.6, f"{vi:.0f}%", ha="center", fontsize=8.6, color="#333333")
ax.set_xticks(xs); ax.set_xticklabels([PLAB[p] for p in PODS], fontsize=8, rotation=12)
ax.set_ylabel("share of cycles spent in the OS (%)")
ax.set_title("OS share per pod (all cells, priv windows)", pad=10)
fig.savefig(f"{OUT}/svc_os_share.png"); plt.close(fig)
print("wrote svc_os_share.png")
print(f"service exploratory -> {OUT}")
