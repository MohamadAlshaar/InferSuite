#!/usr/bin/env python3
"""Per-burst parallelism: peak CPU usage per burst for the TOOL fence (top row) and the
HARNESS/agent fence (bottom row). Spike = max over one 0.1 s sample (capped at 20 = partition);
sustained = max 1 s rolling mean. Bursts from the 10 Hz cpu.stat lanes (tool thr 0.005, harness
thr 0.02). Configure via PLOT_SPEC=<spec.json> (same file the main plotter uses); without it,
defaults to the SWE_long set. Run with SYSTEM python3.

Harness y-axis is ADAPTIVE: SWE harness is Python/GIL (~1 core), OC harness is node/V8
(multi-threaded, bursts past 1 core) — the panel scales to the measured peak either way."""
import os, sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "SWE_long")
if os.environ.get("PLOT_SPEC"):
    spec = json.load(open(os.environ["PLOT_SPEC"]))
    DATA, OUT = spec["data"], spec["out"]
    TASKS = [(x[0], f"{DATA}/{x[1]}/{x[2][0]}") for x in spec["resolved"]]
else:
    TASKS = [("django (Python)", f"{SWE}/data/glm_swe_django/run_1"),
             ("sympy (Python)", f"{SWE}/data/glm_swe_sympy-light/run_1"),
             ("babel (JavaScript)", f"{SWE}/data/glm_swe_babel/run_1"),
             ("fmt (C++)", f"{SWE}/data/glm_swe_fmtlib/run_1")]
    OUT = f"{SWE}/plots"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})
C_HARN, C_TOOL = "#6a51a3", "#1b9e77"

def bursts(rd, scope, thr):
    rows = []
    for ln in open(f"{rd}/cpustat_scope{scope}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and int(p[2]) >= 0:
            rows.append((float(p[0]), int(p[2])))
    rates = []
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
    if cur: out.append(cur)
    res = []
    for s0, s1, cs, rr in out:
        if cs <= 0.001: continue
        res.append((s1 - s0, cs, max(rr)))
    return res

# precompute so the harness row can scale to the true peak across all panels
H = {label: np.array([b[2] for b in bursts(rd, 1, 0.02)]) for label, rd in TASKS}
T = {label: np.array([b[2] for b in bursts(rd, 2, 0.005)]) for label, rd in TASKS}
harn_peak = max((h.max() for h in H.values() if len(h)), default=1.0)
harn_top = max(1.28, harn_peak * 1.12)

fig, axes = plt.subplots(2, len(TASKS), figsize=(3.3 * len(TASKS), 7.6), sharey="row")
axes = np.atleast_2d(axes)
fig.subplots_adjust(hspace=0.42, wspace=0.10, top=0.86)
for col, (label, rd) in enumerate(TASKS):
    for row, (spikes, color, rolename) in enumerate(
            [(T[label], C_TOOL, "Tool bursts"), (H[label], C_HARN, "Harness bursts")]):
        ax = axes[row][col]
        heavy = int((spikes > 0.3).sum())
        x = np.random.default_rng(7).uniform(-0.30, 0.30, len(spikes))
        ax.scatter(x, spikes, s=13, alpha=0.45, color=color, edgecolors="white", linewidths=0.4)
        med, p95 = (np.median(spikes), np.percentile(spikes, 95)) if len(spikes) else (0, 0)
        ax.hlines(med, -0.40, 0.40, color="#222222", lw=1.6, zorder=4)
        ax.hlines(p95, -0.40, 0.40, color="#222222", lw=1.1, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xlim(-0.62, 0.62); ax.set_xticks([])
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        mx = float(spikes.max()) if len(spikes) else 0.0
        if row == 0:
            ax.set_title(label, fontsize=11, pad=26)
            pk = f"{mx:.0f}" if mx >= 3 else f"{mx:.1f}"
            ax.text(0.5, 1.045, f"{len(spikes)} bursts · {heavy} heavy · peak {pk} cores",
                    transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")
            ax.set_yscale("symlog", linthresh=1.0)
            ax.set_ylim(0, 23)
            ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20])
            ax.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20"])
            ax.axhline(20, color="#aaaaaa", lw=0.8, linestyle=":", zorder=1)
        else:
            ax.text(0.5, 1.045, f"{len(spikes)} bursts · max {mx:.2f} cores",
                    transform=ax.transAxes, ha="center", fontsize=8.5, color="#555555")
            ax.set_ylim(0, harn_top)
            ax.axhline(1.0, color="#aaaaaa", lw=0.8, linestyle=":", zorder=1)
        if col == 0:
            ax.set_ylabel(f"{rolename}\npeak CPU usage (cores, 0.1 s spike)")
axes[0][-1].annotate("20-core partition", xy=(0.62, 20), xycoords="data",
                     xytext=(-2, 4), textcoords="offset points", ha="right", fontsize=7.5, color="#888888")
axes[1][-1].annotate("1 core", xy=(0.62, 1.0), xycoords="data",
                     xytext=(-2, 4), textcoords="offset points", ha="right", fontsize=7.5, color="#888888")
# title states what the data shows: harness ceiling differs by runtime (Python GIL vs node V8)
harn_note = ("harness stays ~1 core (Python/GIL)" if harn_peak < 1.3
             else f"harness bursts to {harn_peak:.1f} cores (node/V8, multi-threaded)")
fig.suptitle(f"Per-burst parallelism — tool calls fan out per payload; {harn_note}", fontsize=13)
fig.text(0.5, 0.005, "solid line = median · dashed = p95 · bursts from the 10 Hz fence lanes",
         ha="center", fontsize=8, color="#777777")
fig.savefig(f"{OUT}/glm_call_structure.png"); plt.close(fig)
print(f"wrote {OUT}/glm_call_structure.png")
