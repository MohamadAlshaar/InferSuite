#!/usr/bin/env python3
"""Calls-vs-bursts table (SWE_long): tool calls (actions in the agent trajectory) against
measured CPU bursts (contiguous activity in each fence, from the 10 Hz cpu.stat timeline,
sub-0.4 s gaps bridged). Explains why the counts differ — light actions make no tool burst,
while the harness fires once per turn. Renders a table image. Run with SYSTEM python3."""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "SWE_clean")
if os.environ.get("PLOT_SPEC"):
    _spec = json.load(open(os.environ["PLOT_SPEC"]))
    BASE = os.path.dirname(_spec["data"])
    TASKS = [(x[0], x[1]) for x in _spec["resolved"]]
    OUT = _spec["out"]
else:
    TASKS = [("django (Python)", "glm_swe_django"),
             ("sympy (Python)", "glm_swe_sympy-light"),
             ("babel (JavaScript)", "glm_swe_babel"),
             ("fmt (C++)", "glm_swe_fmtlib")]
    OUT = f"{BASE}/plots"
os.makedirs(OUT, exist_ok=True)

def n_bursts(rd, scope, thr):
    """Same burst definition as the call-structure figure: exact usec integration, sub-0.4 s
    gaps merged, bursts with <=0.001 core-s of work dropped as numerical dust."""
    rows = []
    for ln in open(f"{rd}/cpustat_scope{scope}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and int(p[2]) >= 0:
            rows.append((float(p[0]), int(p[2])))
    out, cur = [], None
    for (t0, u0), (t1, u1) in zip(rows, rows[1:]):
        dt = max(t1 - t0, 1e-6)
        cs = (u1 - u0) / 1e6
        if cs / dt > thr:
            if cur and t0 - cur[1] < 0.4:
                cur = [cur[0], t1, cur[2] + cs]
            else:
                if cur: out.append(cur)
                cur = [t0, t1, cs]
    if cur: out.append(cur)
    return sum(1 for _s0, _s1, cs in out if cs > 0.001)

rows = []
for name, cfg in TASKS:
    rd = f"{BASE}/data/{cfg}/run_1"
    tj = glob.glob(f"{rd}/traj/*/*.traj")
    calls = len(json.load(open(tj[0]))["trajectory"]) if tj else 0
    rows.append([name, str(calls), str(n_bursts(rd, 2, 0.005)), str(n_bursts(rd, 1, 0.02))])
    print(f"{name:20s} calls={rows[-1][1]:>4}  tool_bursts={rows[-1][2]:>4}  harness_bursts={rows[-1][3]:>4}")

# ---- table figure ----
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "savefig.dpi": 300, "savefig.bbox": "tight"})
cols = ["task", "tool calls", "tool bursts", "harness bursts"]
fig, ax = plt.subplots(figsize=(7.2, 2.4)); ax.axis("off")
t = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center")
t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.7)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor("#dddddd"); cell.set_linewidth(0.8)
    if r == 0:
        cell.set_facecolor("#f0f0f0"); cell.set_text_props(color="#333333", fontweight="bold")
    if c == 0 and r > 0:
        cell.set_text_props(ha="left"); cell._loc = "left"; cell.PAD = 0.06
t.auto_set_column_width([0, 1, 2, 3])
ax.set_title("Tool calls (from the agent log)  vs  bursts (measured CPU activity), per SWE-agent episode",
             fontsize=10.5, color="#333333", pad=14)
fig.text(0.5, 0.02, "calls = actions the agent took · bursts = contiguous stretches of measured CPU work "
         "(a light action makes no tool burst; the harness fires once per turn)",
         ha="center", fontsize=7.6, color="#777777")
fig.savefig(f"{OUT}/glm_calls_vs_bursts.png"); plt.close(fig)
print(f"wrote {OUT}/glm_calls_vs_bursts.png")
