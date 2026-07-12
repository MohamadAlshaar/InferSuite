#!/usr/bin/env python3
"""Internal-tools vs payload split inside the TOOL fence (SWE_long).

SWE-agent's own tool scripts (editor/viewer/search/submit — "internal tools") execute
INSIDE the sandbox container, sharing the tool fence with the task's payload commands
(test runs, git, arbitrary bash). This script attributes fence CPU to per-call classes
by joining the trajectory's ordered calls (with execution_time) onto the fence's 10 Hz
cpu.stat activity spans, matched in order with duration tolerance. Attribution coverage
is printed and stamped on the figure. Run with SYSTEM python3."""
import json, glob, os, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
TASKS = [("django (Python)", f"{BASE}/data/glm_swe_django/run_1"),
         ("sympy (Python)", f"{BASE}/data/glm_swe_sympy-light/run_1"),
         ("babel (JavaScript)", f"{BASE}/data/glm_swe_babel/run_1"),
         ("fmt (C++)", f"{BASE}/data/glm_swe_fmtlib/run_1")]
OUT = f"{BASE}/plots"; os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})

INTERNAL_PREFIXES = ("str_replace_editor", "open ", "goto ", "scroll_up", "scroll_down",
                     "create ", "search_file", "search_dir", "find_file", "submit", "edit ")
BUILD_TEST_PAT = ("runtests.py", "pytest", "bin/test", "-m django test", "reproduce",
                  "jest", "yarn", "npm test", "npm run", "make", "cmake", "ctest",
                  "ninja", "g++", "gcc", "cc1", "node ")
def classify(action):
    a = action.strip()
    if not a: return "internal"                       # requery/format retries: harness-side chatter
    if a.startswith(INTERNAL_PREFIXES): return "internal"
    low = a.lower()
    if any(p in low for p in BUILD_TEST_PAT) or ("python" in low and "test" in low):
        return "payload: build/tests"
    if a.startswith("git ") or " git " in a[:30]: return "payload: git"
    return "payload: other bash"

def spans_of(rd):
    """Activity spans (start, end, core_s) from the tool-fence 10 Hz cpu.stat lane."""
    rows = []
    for ln in open(f"{rd}/cpustat_scope2.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec":
            rows.append((float(p[0]), int(p[2])))
    spans, cur = [], None
    for (t0, u0), (t1, u1) in zip(rows, rows[1:]):
        rate = (u1 - u0) / 1e6 / max(t1 - t0, 1e-6)
        if rate > 0.005:
            if cur and t0 - cur[1] < 0.4: cur = [cur[0], t1, cur[2] + (u1 - u0) / 1e6]
            elif cur: spans.append(tuple(cur)); cur = [t0, t1, (u1 - u0) / 1e6]
            else: cur = [t0, t1, (u1 - u0) / 1e6]
    if cur: spans.append(tuple(cur))
    return [s for s in spans if s[2] > 0.001]

results = {}
for label, rd in TASKS:
    traj = json.load(open(glob.glob(f"{rd}/traj/**/*.traj", recursive=True)[0]))["trajectory"]
    calls = [(classify(s.get("action") or ""), float(s.get("execution_time") or 0.0)) for s in traj]
    spans = spans_of(rd)
    total_cs = sum(s[2] for s in spans)
    agg = {}
    for cls, et in calls:
        a = agg.setdefault(cls, dict(calls=0, wall=0.0, cpu=0.0))
        a["calls"] += 1; a["wall"] += et
    # ---- anchor join: >5s calls and >5s spans are unambiguous 1:1 anchors (only long
    # payload commands run that long); short spans between anchors are distributed over
    # the same segment's short calls, weighted by execution_time.
    LONG = 5.0
    long_calls = [i for i, (c, et) in enumerate(calls) if et > LONG]
    long_spans = [j for j, s in enumerate(spans) if s[1] - s[0] > LONG]
    n_anchor = min(len(long_calls), len(long_spans))
    print(f"  anchors: {len(long_calls)} long calls vs {len(long_spans)} long spans "
          f"(pairing {n_anchor}); long-call classes: "
          f"{sorted(set(calls[i][0] for i in long_calls))}")
    matched_cs = 0.0
    for k in range(n_anchor):                                # 1:1 anchor attribution
        cls = calls[long_calls[k]][0]
        agg[cls]["cpu"] += spans[long_spans[k]][2]; matched_cs += spans[long_spans[k]][2]
    # segment fill between anchors
    call_bounds = [-1] + long_calls[:n_anchor] + [len(calls)]
    span_bounds = [-1] + long_spans[:n_anchor] + [len(spans)]
    for seg in range(len(call_bounds) - 1):
        seg_calls = [calls[i] for i in range(call_bounds[seg] + 1, call_bounds[seg + 1])]
        seg_cs = sum(spans[j][2] for j in range(span_bounds[seg] + 1, span_bounds[seg + 1]))
        wall = sum(et for _, et in seg_calls)
        if seg_cs <= 0 or not seg_calls: continue
        matched_cs += seg_cs
        for cls, et in seg_calls:                            # weight by call wall time
            agg[cls]["cpu"] += seg_cs * ((et / wall) if wall > 0 else 1.0 / len(seg_calls))
    cov = 100 * matched_cs / total_cs if total_cs else 0
    method = f"anchored on {n_anchor} long calls" if n_anchor else "duration-weighted (no long calls)"
    results[label] = (agg, total_cs, cov, method)
    print(f"\n== {label}  (fence total {total_cs:.1f} core-s, attribution coverage {cov:.0f}%)")
    for cls in sorted(agg, key=lambda c: -agg[c]["cpu"]):
        a = agg[cls]
        print(f"  {cls:22s} calls={a['calls']:4d}  wall={a['wall']:7.1f}s  cpu={a['cpu']:7.1f} core-s")

# ---------------- figure -------------------------------------------------------------------
CLS_ORDER = ["internal", "payload: build/tests", "payload: git", "payload: other bash"]
CLS_LBL   = {"internal": "internal tools\n(editor/viewer/submit)", "payload: build/tests": "build / tests",
             "payload: git": "git", "payload: other bash": "other bash"}
CLS_COL   = {"internal": "#6a51a3", "payload: build/tests": "#1b9e77",
             "payload: git": "#66c2a5", "payload: other bash": "#a6dcc9"}
ROWS = ["payload: other bash", "payload: git", "payload: build/tests", "internal"]  # bottom -> top
fig, axes = plt.subplots(1, len(results), figsize=(3.6 * len(results), 3.4), sharey=True)
fig.subplots_adjust(wspace=0.14, top=0.76)
for k, (ax, (label, (agg, total_cs, cov, method))) in enumerate(
        zip(np.atleast_1d(axes), results.items())):
    ypos = range(len(ROWS))
    cpu = [agg.get(c, {}).get("cpu", 0.0) for c in ROWS]
    ax.barh(ypos, cpu, height=0.62, color=[CLS_COL[c] for c in ROWS])
    for i, c in enumerate(ROWS):
        v, n = agg.get(c, {}).get("cpu", 0.0), agg.get(c, {}).get("calls", 0)
        ax.text(v + max(cpu) * 0.03, i,
                f"{v:.0f} ({n} calls)" if v >= 10 else f"{v:.1f} ({n} calls)",
                va="center", fontsize=8.5, color="#333333")
    ax.set_yticks(list(ypos))
    if k == 0:
        ax.set_yticklabels([CLS_LBL[c] for c in ROWS], fontsize=9)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(-0.55, len(ROWS) - 0.45)
    ax.set_xlim(0, max(cpu) * 1.55)
    ax.set_xlabel("CPU amount (core-s)", fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title(label, fontsize=10.5, pad=24)
    ax.text(0.5, 1.07, f"coverage {cov:.0f}% · {method.split(' (')[0]}",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555555")
fig.suptitle("Inside the tool fence: SWE-agent internal tools vs task payload (CPU core-seconds)",
             fontsize=13)
fig.savefig(f"{OUT}/glm_internal_tools.png"); plt.close(fig)
print(f"\nwrote {OUT}/glm_internal_tools.png")
