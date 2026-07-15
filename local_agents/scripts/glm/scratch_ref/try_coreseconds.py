#!/usr/bin/env python3
"""EXPERIMENT (revertible): honest core-seconds views.
  1. per-burst -> y = core-seconds (total cost), color = peak parallelism (secondary)
  2. timeline  -> cumulative core-seconds curve (slope = old cores rate; flat = idle)
Reads the same SWE_long cpu.stat data. Writes to this scratch dir. SYSTEM python3."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWE = "/home/mohamad/llm-service-kernel-latest/local_agents/SWE_long"
OUT = os.path.dirname(os.path.abspath(__file__))
TASKS = [("django (Python)", f"{SWE}/data/glm_swe_django/run_1"),
         ("sympy (Python)", f"{SWE}/data/glm_swe_sympy-light/run_1"),
         ("babel (JavaScript)", f"{SWE}/data/glm_swe_babel/run_1"),
         ("fmt (C++)", f"{SWE}/data/glm_swe_fmtlib/run_1")]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5,
    "grid.alpha": 0.6, "axes.axisbelow": True})
C_HARN, C_TOOL = "#6a51a3", "#1b9e77"


def rows(rd, scope):
    r = []
    for ln in open(f"{rd}/cpustat_scope{scope}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and int(p[2]) >= 0:
            r.append((float(p[0]), int(p[2])))
    return r


def bursts(rd, scope, thr):
    """returns list of (dur_s, core_seconds, peak_cores) per burst."""
    r = rows(rd, scope)
    rates = []
    for (t0, u0), (t1, u1) in zip(r, r[1:]):
        dt = max(t1 - t0, 1e-6)
        cs = (u1 - u0) / 1e6
        rates.append((t0, t1, min(cs / dt, 20.0), cs))
    out, cur = [], None
    for t0, t1, rate, cs in rates:
        if rate > thr:
            if cur and t0 - cur[1] < 0.4:
                cur = [cur[0], t1, cur[2] + cs, cur[3] + [rate]]
            else:
                if cur: out.append(cur)
                cur = [t0, t1, cs, [rate]]
    if cur: out.append(cur)
    res = []
    for s0, s1, cs, rr in out:
        if cs <= 0.001: continue
        res.append((s1 - s0, cs, max(rr)))
    return res


# ============ PLOT 1: per-burst core-seconds =====================================================
Bt = {lab: bursts(rd, 2, 0.005) for lab, rd in TASKS}
Bh = {lab: bursts(rd, 1, 0.02) for lab, rd in TASKS}
fig, axes = plt.subplots(2, len(TASKS), figsize=(3.3 * len(TASKS), 7.6), sharey="row")
axes = np.atleast_2d(axes)
fig.subplots_adjust(hspace=0.42, wspace=0.10, top=0.85)
rng = np.random.default_rng(7)
for col, (lab, rd) in enumerate(TASKS):
    for row, (B, color, role) in enumerate(
            [(Bt[lab], C_TOOL, "Tool bursts"), (Bh[lab], C_HARN, "Harness bursts")]):
        ax = axes[row][col]
        cs = np.array([b[1] for b in B])          # core-seconds  (the honest total)
        pk = np.array([b[2] for b in B])          # peak cores    (secondary encoding = color)
        x = rng.uniform(-0.30, 0.30, len(cs))
        sc = ax.scatter(x, cs, c=pk, cmap="viridis", vmin=1, vmax=8,
                        s=16, alpha=0.7, edgecolors="white", linewidths=0.4)
        med = np.median(cs) if len(cs) else 0
        p95 = np.percentile(cs, 95) if len(cs) else 0
        tot = cs.sum()
        ax.hlines(med, -0.40, 0.40, color="#222222", lw=1.6, zorder=4)
        ax.hlines(p95, -0.40, 0.40, color="#222222", lw=1.1, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xlim(-0.62, 0.62); ax.set_xticks([])
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1e2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_title(lab, fontsize=11, pad=24)
        ax.text(0.5, 1.02, f"{len(cs)} bursts · Σ={tot:.0f} core-s · med {med:.2f}",
                transform=ax.transAxes, ha="center", fontsize=8.3, color="#555555")
        if col == 0:
            ax.set_ylabel(f"{role}\nCPU per burst (core-seconds)")
cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
cb.set_label("peak parallelism (cores)", fontsize=8.5)
fig.suptitle("Per-burst CPU COST (core-seconds, additive) — color = peak parallelism", fontsize=13, y=0.94)
fig.text(0.5, 0.005, "solid = median · dashed = p95 · Σ over bursts = fence total (donut)",
         ha="center", fontsize=8, color="#777777")
fig.savefig(f"{OUT}/try_per_burst_coreseconds.png"); plt.close(fig)
print("wrote try_per_burst_coreseconds.png")


# ============ PLOT 2: cumulative core-seconds timeline ===========================================
fig = plt.figure(figsize=(11.8, 2.6 * len(TASKS)))
gs = fig.add_gridspec(len(TASKS), 1, hspace=0.45)
for pnl, (lab, rd) in enumerate(TASKS):
    ax = fig.add_subplot(gs[pnl])
    rt, rh = rows(rd, 2), rows(rd, 1)
    t0 = min(rt[0][0] if rt else 1e18, rh[0][0] if rh else 1e18)
    for r, color, name in ((rt, C_TOOL, "tool"), (rh, C_HARN, "harness")):
        if not r: continue
        t = np.array([(x[0] - t0) / 60 for x in r])
        c = np.array([(x[1] - r[0][1]) / 1e6 for x in r])
        ax.plot(t, c, color=color, lw=1.8, label=f"{name} (Σ={c[-1]:.0f} core-s)")
        ax.fill_between(t, 0, c, color=color, alpha=0.10)
    ax.set_ylabel(f"{lab.split(' (')[0]}\ncumulative core-s", fontsize=9)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.grid(True, alpha=0.4)
    if pnl == len(TASKS) - 1: ax.set_xlabel("Episode time (minutes)")
fig.suptitle("Cumulative CPU work (core-seconds) — flat = idle (model round-trip), steep = CPU busy; slope = cores",
             fontsize=12, y=0.995)
fig.savefig(f"{OUT}/try_timeline_cumulative.png"); plt.close(fig)
print("wrote try_timeline_cumulative.png")
