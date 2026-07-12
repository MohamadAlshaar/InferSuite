#!/usr/bin/env python3
"""plot_thread_lanes.py — CANDIDATE figure: thread-level timeline per fence.
Each row = one software thread (top-N by CPU time); each mark = a profiler sample
(thread on-CPU at that instant), colored by WHICH logical CPU it ran on.
Reading: white gaps = thread off-CPU (asleep/blocked — indistinguishable in this data);
color changes within a row = the scheduler migrated the thread to another CPU;
many simultaneously-active rows = real parallelism.
Run with system python3 (matplotlib not in .venv)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "glm_plots", "swe")

plt.rcParams.update({"font.size": 10, "figure.dpi": 150, "savefig.dpi": 300,
                     "savefig.bbox": "tight", "axes.grid": False})

PANELS = [
    ("scikit-learn — tool fence", "glm_swe_scikit-learn/run_1/scope2_threads.tsv",
     "1199 threads total: pytest workers + a fresh OpenBLAS pool per test burst"),
    ("astropy — tool fence", "glm_swe_astropy/run_1/scope2_threads.tsv",
     "667 threads total: parallel compiler/pytest processes, sub-second lives"),
    ("sympy — harness fence", "glm_swe_sympy/run_1/scope1_threads.tsv",
     "8 threads: single-threaded Python + idle helpers"),
    ("django — harness fence", "glm_swe_django-lite/run_1/scope1_threads.tsv",
     "8 threads: one worker doing everything"),
]
TOPN = 40
LANES = [c for pair in zip(range(2, 12), range(14, 24)) for c in pair]
cpu_color = {c: plt.cm.tab20(i / 20) for i, c in enumerate(LANES)}

fig, axes = plt.subplots(4, 1, figsize=(12.4, 12.8))
fig.subplots_adjust(hspace=0.45)
for ax, (title, path, note) in zip(axes, PANELS):
    arr = np.loadtxt(os.path.join(DATA, path), ndmin=2)  # tid, time, cpu
    t0 = arr[:, 1].min()
    tids, counts = np.unique(arr[:, 0], return_counts=True)
    keep = tids[np.argsort(counts)[::-1][:TOPN]]
    # order lanes by first appearance so thread lifecycle reads top-to-bottom
    first = {tid: arr[arr[:, 0] == tid, 1].min() for tid in keep}
    order = sorted(keep, key=lambda k: first[k])
    shown = 0
    for row, tid in enumerate(order):
        sel = arr[arr[:, 0] == tid]
        shown += len(sel)
        ax.scatter((sel[:, 1] - t0) / 60, np.full(len(sel), row), s=1.2,
                   c=[cpu_color.get(int(c), (0.6, 0.6, 0.6, 1)) for c in sel[:, 2]],
                   marker="|", linewidths=0.9)
    nthr = len(tids)
    ax.set_ylim(-1, min(TOPN, nthr))
    ax.invert_yaxis()
    ax.set_ylabel(f"threads (top {min(TOPN, nthr)} of {nthr}\nby CPU time)", fontsize=8.5)
    cov = 100 * shown / len(arr)
    ax.set_title(f"{title} — {note}   [rows cover {cov:.0f}% of samples]",
                 loc="left", fontsize=9.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
axes[-1].set_xlabel("episode time (minutes)")
fig.legend(handles=[Patch(fc=cpu_color[c], label=f"cpu {c}") for c in LANES],
           ncol=10, fontsize=6.5, frameon=False, loc="upper center",
           bbox_to_anchor=(0.5, 0.955))
fig.suptitle("Thread-level timeline — each row a software thread, marks colored by the logical CPU it ran on\n"
             "white gap = off-CPU (asleep/blocked); color change within a row = scheduler migration; "
             "stacked active rows = parallelism", fontsize=12, y=1.0)
fig.text(0.01, -0.01, "From full-episode 99 Hz records (tid + cpu per sample). Off-CPU cause "
         "(sleeping vs runnable vs blocked) is not distinguishable from samples; would need sched "
         "tracepoints — noted as a future-capture addition.", fontsize=7.5, color="#888888")
fig.savefig(f"{OUT}/glm_thread_lanes.png")
print("wrote glm_thread_lanes.png")
