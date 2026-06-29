#!/usr/bin/env python3
"""Precise timing decomposition from instrumented data (no util-thresholding for tool).
 - tool wall-clock  = sum of trajectory[].execution_time  (SWE-agent instrumented, per step)
 - total wall-clock = agent_done - perf_start              (markers.txt)
 - inference        = GPU-util>50 seconds                  (gpu_timeline.csv, from the live run)
 - rest             = total - tool - inference             (host-agent + idle/wait)
Writes figures/07_timing_donut.png. NOTE: execution_time INCLUDES timed-out/hung commands,
so on a flaily trajectory the tool slice is inflated by the 90s hangs (flagged below).
"""
import os, re, json, glob, sys
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figures"); os.makedirs(OUT, exist_ok=True)

# --- tool wall-clock from the trajectory ---
trajs = glob.glob(os.path.join(HERE, "runs", "smoke", "*", "*.traj"))
if not trajs: print("no trajectory found"); sys.exit(1)
d = json.load(open(trajs[0]))
steps = d.get("trajectory") or []
exec_times = [float(s.get("execution_time", 0) or 0) for s in steps if isinstance(s, dict)]
tool = sum(exec_times)
n_steps = len(exec_times)
# how much of tool is hung (a timeout ~ execution_time near the 90s limit)
hung = sum(t for t in exec_times if t >= 85)

# --- total from markers ---
mk = {}
for line in open(os.path.join(HERE, "runs", "perf", "markers.txt")):
    p = line.split()
    if len(p) >= 2:
        try: mk[p[1]] = float(p[0])
        except ValueError: pass
total = (mk.get("agent_done", 0) - mk.get("perf_start", 0)) if mk else 0

# --- inference from GPU util (live run) ---
# Forward-fill across dropped ticks: the sampler drifts (~1.03s/iter) so ~1 integer
# second every ~31s has no sample; counting raw samples undercounts inference time.
gcsv = os.path.join(HERE, "runs", "perf", "gpu_timeline.csv")
util = {}
if os.path.exists(gcsv) and mk.get("perf_start"):
    for line in open(gcsv):
        p = line.strip().split(",")
        if len(p) >= 2:
            try: util[int(round(float(p[0]) - mk["perf_start"]))] = float(p[1])
            except ValueError: pass
inf = 0
if util:
    lo, hi, last = min(util), max(util), 0.0
    for s in range(lo, hi + 1):
        if s in util: last = util[s]
        if last > 50: inf += 1

rest = max(total - tool - inf, 0)
print(f"steps={n_steps}  total={total:.0f}s")
print(f"  tool wall-clock (Σ execution_time) = {tool:.0f}s   (of which hung/timeout ≈ {hung:.0f}s)")
print(f"  inference (GPU util>50)            = {inf}s")
print(f"  rest (host-agent + idle/wait)      = {rest:.0f}s")

# --- timing donut (precise tool + util inference) ---
vals = [inf, tool - hung, hung, rest]
labs = [f"inference (GPU)  {inf}s", f"tool-exec (real)  {tool-hung:.0f}s",
        f"tool hung/timeout  {hung:.0f}s", f"host-agent + idle  {rest:.0f}s"]
cols = ["#d62728", "#1f77b4", "#8c564b", "#cccccc"]
keep = [i for i in range(4) if vals[i] > 0]
fig, ax = plt.subplots(figsize=(8, 5))
w = ax.pie([vals[i] for i in keep], colors=[cols[i] for i in keep], startangle=90,
           wedgeprops=dict(width=0.42, edgecolor="white"))[0]
ax.legend(w, [labs[i] for i in keep], loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=False)
ax.text(0, 0, f"{total:.0f}s\nwall-clock", ha="center", va="center", fontsize=12, fontweight="bold")
ax.set_title("SWE-bench run — precise timing decomposition (instrumented tool time)")
fig.tight_layout(); fig.savefig(f"{OUT}/07_timing_donut.png", dpi=130); plt.close(fig)
print("wrote", f"{OUT}/07_timing_donut.png")
