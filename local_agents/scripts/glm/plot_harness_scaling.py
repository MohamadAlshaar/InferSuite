#!/usr/bin/env python3
"""Harness cost vs conversation length — CROSS-CAMPAIGN power-law figure (adopted 2026-07-15).

Every live SWE episode from both campaigns (SWE_long + SWE_clean; replays and rejected loop
episodes excluded): x = turns (STEP markers), y = exact harness fence core-seconds (cpu.stat
delta). Log-log fit printed on-figure and dumped to glm_harness_scaling_values.json for the
audit. Result: core-s ∝ turns^~2.7 (R² ~0.998) across 4 repos (68-531 KLOC) — orchestration
cost is set by conversation length, not codebase size. Run with SYSTEM python3."""
import glob, json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
LA = os.path.join(HERE, "..", "..")
OUT = os.path.join(LA, "SWE_clean", "plots")
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})
TCOL = {"django": "#6a51a3", "sympy": "#1b9e77", "sympy-light": "#1b9e77",
        "babel": "#d95f02", "fmtlib": "#0072B2"}

pts = []
for root in ("../archive/glm_softiso_long_campaigns/SWE_long", "SWE_clean"):
    for rd in glob.glob(os.path.join(LA, root, "data", "glm_swe_*", "run_*")):
        if "replay" in rd or "rejected" in rd:
            continue
        try:
            steps = open(f"{rd}/agent.log", errors="ignore").read().count("STEP ")
            s = [(float(p[0]), float(p[2])) for p in (l.split() for l in open(f"{rd}/cpustat_scope1.tsv"))
                 if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0]
            cs = (s[-1][1] - s[0][1]) / 1e6
            task = rd.split("/")[-2].replace("glm_swe_", "")
            if steps > 20 and cs > 1:
                pts.append((task, steps, cs))
        except (OSError, IndexError):
            pass

x = np.log([p[1] for p in pts]); y = np.log([p[2] for p in pts])
b, a = np.polyfit(x, y, 1)
r2 = 1 - np.sum((y - (a + b * x))**2) / np.sum((y - np.mean(y))**2)

fig, ax = plt.subplots(figsize=(7.6, 5.6))
xs = np.linspace(min(x) - 0.1, max(x) + 0.1, 50)
ax.plot(np.exp(xs), np.exp(a + b * xs), color="#444444", lw=1.6, ls="--", zorder=2,
        label=f"fit: core-s ∝ turns$^{{{b:.2f}}}$  (R² = {r2:.3f})")
seen = set()
for task, st, cs in pts:
    base = task.replace("-light", "")
    lab = base if base not in seen else None
    seen.add(base)
    ax.scatter(st, cs, s=64, marker="o", color=TCOL[task], edgecolors="white",
               linewidths=0.8, zorder=3, label=lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.xaxis.set_minor_formatter(NullFormatter()); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xticks([70, 100, 150, 200, 300, 400]); ax.set_xticklabels(["70", "100", "150", "200", "300", "400"])
ax.set_yticks([10, 30, 100, 300, 800]); ax.set_yticklabels(["10", "30", "100", "300", "800"])
ax.set_xlabel("conversation length (turns)")
ax.set_ylabel("harness CPU work (core-seconds)")
ax.set_title("Harness CPU work vs conversation length", fontsize=12)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.annotate("django & sympy:\nsame code size (477k vs 531k lines),\n22× the harness work at 3.1× the turns",
            xy=(389, 632), xytext=(120, 340), fontsize=8, color="#555555",
            arrowprops=dict(arrowstyle="->", color="#999999", lw=0.8))
fig.savefig(f"{OUT}/glm_harness_scaling.png"); plt.close(fig)
json.dump({"exponent": round(b, 3), "r2": round(r2, 4), "n": len(pts),
           "points": [{"task": t, "turns": s, "harness_core_s": round(c, 1)} for t, s, c in pts]},
          open(f"{OUT}/glm_harness_scaling_values.json", "w"), indent=1)
print(f"wrote {OUT}/glm_harness_scaling.png  (exponent {b:.2f}, R² {r2:.3f}, n={len(pts)})")
