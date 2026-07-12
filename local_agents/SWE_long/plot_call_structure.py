#!/usr/bin/env python3
"""Call structure + parallelism (SWE_long): per-burst peak CPU usage for the TOOL fence
(top row) and the AGENT HARNESS fence (bottom row). Spike = max over one 0.1 s sample
(capped at 20 = partition size); sustained = max 1 s rolling mean. Bursts from the 10 Hz
cpu.stat lanes (tool thr 0.005 cores, harness thr 0.02 — same as the timeline figure).
Run with SYSTEM python3."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TASKS = [("django (Python)", f"{BASE}/data/glm_swe_django/run_1"),
         ("sympy (Python)", f"{BASE}/data/glm_swe_sympy-light/run_1"),
         ("babel (JavaScript)", f"{BASE}/data/glm_swe_babel/run_1"),
         ("fmt (C++)", f"{BASE}/data/glm_swe_fmtlib/run_1")]
OUT = f"{BASE}/plots"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})
C_HARN, C_TOOL = "#6a51a3", "#1b9e77"

def bursts(rd, scope, thr):
    """[(duration_s, core_s, spike, sustained)] from the 10 Hz cpu.stat lane."""
    rows = []
    for ln in open(f"{rd}/cpustat_scope{scope}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec":
            rows.append((float(p[0]), int(p[2])))
    rates = []   # (t, rate) per 0.1 s sample
    for (t0, u0), (t1, u1) in zip(rows, rows[1:]):
        dt = max(t1 - t0, 1e-6)
        rates.append((t0, t1, min((u1 - u0) / 1e6 / dt, 20.0), (u1 - u0) / 1e6))
    out, cur = [], None
    for t0, t1, r, cs in rates:
        if r > thr:
            if cur and t0 - cur[1] < 0.4:
                cur = [cur[0], t1, cur[2] + cs, cur[3] + [r]]
            else:
                if cur: out.append(cur)
                cur = [t0, t1, cs, [r]]
        # gaps end bursts implicitly via the 0.4 s merge rule above
    if cur: out.append(cur)
    res = []
    for s0, s1, cs, rr in out:
        if cs <= 0.001: continue
        spike = max(rr)
        sust = max((np.mean(rr[i:i+10]) for i in range(max(1, len(rr) - 9))), default=spike)
        res.append((s1 - s0, cs, spike, sust))
    return res

fig, axes = plt.subplots(2, len(TASKS), figsize=(3.3 * len(TASKS), 7.6), sharey="row")
fig.subplots_adjust(hspace=0.42, wspace=0.10, top=0.86)
for col, (label, rd) in enumerate(TASKS):
    for row, (scope, thr, color, rolename) in enumerate(
            [(2, 0.005, C_TOOL, "Tool calls"), (1, 0.02, C_HARN, "Harness bursts")]):
        ax = axes[row][col]
        B = bursts(rd, scope, thr)
        spikes = np.array([b[2] for b in B])
        heavy = int((spikes > 0.3).sum())
        x = np.random.default_rng(7).uniform(-0.30, 0.30, len(spikes))
        ax.scatter(x, spikes, s=13, alpha=0.45, color=color,
                   edgecolors="white", linewidths=0.4)
        med, p95 = (np.median(spikes), np.percentile(spikes, 95)) if len(spikes) else (0, 0)
        ax.hlines(med, -0.40, 0.40, color="#222222", lw=1.6, zorder=4)
        ax.hlines(p95, -0.40, 0.40, color="#222222", lw=1.1, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xlim(-0.62, 0.62); ax.set_xticks([])
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_title(label, fontsize=11, pad=26)
            ax.text(0.5, 1.045, f"{len(spikes)} calls · {heavy} heavy (>0.3 cores)",
                    transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(0, 23)
            ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20])
            ax.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
            ax.axhline(20, color="#aaaaaa", lw=0.8, linestyle=":", zorder=1)
            if len(spikes) and spikes.max() > 2:   # selective: label only a notable max
                ax.annotate(f"max {spikes.max():.0f}", (x[spikes.argmax()], spikes.max()),
                            textcoords="offset points", xytext=(7, -2),
                            fontsize=8, color="#555555")
        else:
            ax.text(0.5, 1.045, f"{len(spikes)} bursts · max {spikes.max() if len(spikes) else 0:.2f} cores",
                    transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")
            ax.set_ylim(0, 1.28)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
            ax.axhline(1.0, color="#aaaaaa", lw=0.8, linestyle=":", zorder=1)
        if col == 0:
            ax.set_ylabel(f"{rolename}\npeak CPU usage (cores, 0.1 s spike)")
axes[0][-1].annotate("20-core partition", xy=(0.62, 20), xycoords="data",
                     xytext=(-2, 4), textcoords="offset points",
                     ha="right", fontsize=7.5, color="#888888")
axes[1][-1].annotate("1 core", xy=(0.62, 1.0), xycoords="data",
                     xytext=(-2, 4), textcoords="offset points",
                     ha="right", fontsize=7.5, color="#888888")
fig.suptitle("Per-burst parallelism — tool calls fan out per payload; "
             "the harness never exceeds one core", fontsize=13.5)
fig.text(0.5, 0.005, "solid line = median · dashed = p95 · bursts from the 10 Hz fence lanes",
         ha="center", fontsize=8, color="#777777")
fig.savefig(f"{OUT}/glm_call_structure.png"); plt.close(fig)
print(f"wrote {OUT}/glm_call_structure.png")
