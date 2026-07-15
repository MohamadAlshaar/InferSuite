#!/usr/bin/env python3
"""Per-burst CPU COST (adopted 2026-07-14, replaces the peak-cores view): y = core-seconds
per burst — an exact, ADDITIVE total (Σ over bursts = fence total = the cpu_work donut).
Peak parallelism is kept as the dot color. The old y-axis ("peak cores, 0.1 s spike") read
like a core headcount and was not additive.

Bursts from the 10 Hz cpu.stat lanes with exact usec integration (tool thr 0.005, harness
thr 0.02, sub-0.4 s gaps merged, heavy > 0.3 — ONE vocabulary, see the plots MANIFEST).
Configure via PLOT_SPEC=<spec.json> (same file the main plotter uses); without it, defaults
to the SWE_clean set. Run with SYSTEM python3."""
import os, sys, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "SWE_clean")
if os.environ.get("PLOT_SPEC"):
    spec = json.load(open(os.environ["PLOT_SPEC"]))
    DATA, OUT = spec["data"], spec["out"]
    TASKS = [(x[0], f"{DATA}/{x[1]}/{x[2][0]}") for x in spec["resolved"]]
else:
    TASKS = [("django (Python)", f"{SWE}/data/glm_swe_django/run_1"),
             ("sympy (Python)", f"{SWE}/data/glm_swe_sympy/run_1"),
             ("babel (JavaScript)", f"{SWE}/data/glm_swe_babel/run_1"),
             ("fmt (C++)", f"{SWE}/data/glm_swe_fmtlib/run_1")]
    OUT = f"{SWE}/plots"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})
C_TOOL, C_HARN = "#1b9e77", "#6a51a3"
THR_TOOL, THR_HARN, THR_HEAVY, GAP_S = 0.005, 0.02, 0.3, 0.4

def bursts(rd, scope, thr):
    """[(duration_s, core_seconds, peak_cores)] — exact usec integration, GAP_S merge."""
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
            if cur and t0 - cur[1] < GAP_S:
                cur = [cur[0], t1, cur[2] + cs, cur[3] + [r]]
            else:
                if cur: out.append(cur)
                cur = [t0, t1, cs, [r]]
    if cur: out.append(cur)
    return [(s1 - s0, cs, max(rr)) for s0, s1, cs, rr in out if cs > 0.001]

B_TOOL = {lab: bursts(rd, 2, THR_TOOL) for lab, rd in TASKS}
B_HARN = {lab: bursts(rd, 1, THR_HARN) for lab, rd in TASKS}
harn_peak = max((b[2] for B in B_HARN.values() for b in B), default=1.0)

fig, axes = plt.subplots(2, len(TASKS), figsize=(3.3 * len(TASKS), 7.6), sharey="row")
axes = np.atleast_2d(axes)
fig.subplots_adjust(hspace=0.42, wspace=0.10, top=0.85)
rng = np.random.default_rng(7)
sc = None
for col, (lab, rd) in enumerate(TASKS):
    for row, (B, role) in enumerate([(B_TOOL[lab], "Tool bursts"), (B_HARN[lab], "Harness bursts")]):
        ax = axes[row][col]
        cs = np.array([b[1] for b in B])          # core-seconds  (the honest, additive total)
        pk = np.array([b[2] for b in B])          # peak cores    (secondary encoding = color)
        heavy = int((pk > THR_HEAVY).sum())
        x = rng.uniform(-0.30, 0.30, len(cs))
        sc = ax.scatter(x, cs, c=pk, cmap="viridis", vmin=1, vmax=8,
                        s=16, alpha=0.7, edgecolors="white", linewidths=0.4)
        med = np.median(cs) if len(cs) else 0
        p95 = np.percentile(cs, 95) if len(cs) else 0
        ax.hlines(med, -0.40, 0.40, color="#222222", lw=1.6, zorder=4)
        ax.hlines(p95, -0.40, 0.40, color="#222222", lw=1.1, linestyle=(0, (4, 3)), zorder=4)
        ax.set_xlim(-0.62, 0.62); ax.set_xticks([])
        ax.set_yscale("log"); ax.set_ylim(1e-3, 1e2)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        if row == 0:
            ax.set_title(lab, fontsize=11, pad=26)
            note = f"{len(cs)} bursts ({heavy} heavy) · Σ={cs.sum():.0f} cs"
        else:
            note = f"{len(cs)} bursts · Σ={cs.sum():.0f} cs · med {med:.2f}"
        ax.text(0.5, 1.02, note, transform=ax.transAxes, ha="center",
                fontsize=7.3, color="#555555")
        if col == 0:
            ax.set_ylabel(f"{role}\nCPU per burst (core-seconds)")
cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.018, pad=0.015)
cb.set_label("peak parallelism (cores)", fontsize=8.5)
harn_note = ("harness stays ~1 core (Python/GIL)" if harn_peak < 1.3
             else f"harness bursts to {harn_peak:.1f} cores (node/V8, multi-threaded)")
fig.suptitle(f"Per-burst CPU cost (core-seconds, additive) — color = peak parallelism; {harn_note}",
             fontsize=12.5, y=0.94)
fig.text(0.5, 0.005, "solid = median · dashed = p95 · Σ over bursts = fence total (cpu_work donut)",
         ha="center", fontsize=8, color="#777777")
fig.savefig(f"{OUT}/glm_call_structure.png"); plt.close(fig)
print(f"wrote {OUT}/glm_call_structure.png")
