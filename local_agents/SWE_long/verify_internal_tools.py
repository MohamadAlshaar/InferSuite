#!/usr/bin/env python3
"""Verify the internal-tools attribution against the REPLAYS.

Replays re-execute the live trajectory's commands verbatim (no model), so the per-class
CPU split (internal tools vs payload) must reproduce independently in each replay capture.
Live uses merge gap 0.4 s (model gaps are seconds); replays use a tight gap because calls
run back-to-back. Reports per-class CPU for live vs each replay + replay dispersion."""
import json, glob, sys

BASE = "/home/mohamad/llm-service-kernel-latest/local_agents/SWE_long/data"
INTERNAL_PREFIXES = ("str_replace_editor", "open ", "goto ", "scroll_up", "scroll_down",
                     "create ", "search_file", "search_dir", "find_file", "submit", "edit ")
BUILD_TEST_PAT = ("runtests.py", "pytest", "bin/test", "-m django test", "reproduce",
                  "jest", "yarn", "npm test", "npm run", "make", "cmake", "ctest",
                  "ninja", "g++", "gcc", "cc1", "node ")

def classify(action):
    a = action.strip()
    if not a: return "internal"
    if a.startswith(INTERNAL_PREFIXES): return "internal"
    low = a.lower()
    if any(p in low for p in BUILD_TEST_PAT) or ("python" in low and "test" in low):
        return "build/tests"
    if a.startswith("git ") or " git " in a[:30]: return "git"
    return "other bash"

def spans_of(rd, merge_gap):
    rows = []
    for ln in open(f"{rd}/cpustat_scope2.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and int(p[2]) >= 0:
            rows.append((float(p[0]), int(p[2])))
    spans, cur = [], None
    for (t0, u0), (t1, u1) in zip(rows, rows[1:]):
        rate = (u1 - u0) / 1e6 / max(t1 - t0, 1e-6)
        if rate > 0.005:
            if cur and t0 - cur[1] < merge_gap: cur = [cur[0], t1, cur[2] + (u1 - u0) / 1e6]
            elif cur: spans.append(tuple(cur)); cur = [t0, t1, (u1 - u0) / 1e6]
            else: cur = [t0, t1, (u1 - u0) / 1e6]
    if cur: spans.append(tuple(cur))
    return [s for s in spans if s[2] > 0.001]

def attribute(rd, calls, merge_gap, long_s=5.0):
    spans = spans_of(rd, merge_gap)
    total = sum(s[2] for s in spans)
    agg = {}
    for cls, et in calls:
        agg.setdefault(cls, 0.0)
    long_calls = [i for i, (c, et) in enumerate(calls) if et > long_s]
    long_spans = [j for j, s in enumerate(spans) if s[1] - s[0] > long_s]
    n = min(len(long_calls), len(long_spans))
    for k in range(n):
        agg[calls[long_calls[k]][0]] += spans[long_spans[k]][2]
    cb = [-1] + long_calls[:n] + [len(calls)]
    sb = [-1] + long_spans[:n] + [len(spans)]
    for seg in range(len(cb) - 1):
        seg_calls = [calls[i] for i in range(cb[seg] + 1, cb[seg + 1])]
        seg_cs = sum(spans[j][2] for j in range(sb[seg] + 1, sb[seg + 1]))
        wall = sum(et for _, et in seg_calls)
        if seg_cs <= 0 or not seg_calls: continue
        for cls, et in seg_calls:
            agg[cls] += seg_cs * ((et / wall) if wall > 0 else 1.0 / len(seg_calls))
    return agg, total, len(long_calls), len(long_spans)

for task, live_cfg, rep_cfg, reps in [
        ("django-16560", "glm_swe_django", "glm_replay_swe_django", ["run_1", "run_4", "run_5"]),
        ("sympy-13878",  "glm_swe_sympy",  "glm_replay_swe_sympy",  ["run_1", "run_4"])]:
    traj = json.load(open(glob.glob(f"{BASE}/{live_cfg}/run_1/traj/**/*.traj", recursive=True)[0]))
    calls = [(classify(s.get("action") or ""), float(s.get("execution_time") or 0.0))
             for s in traj["trajectory"]]
    print(f"\n=== {task} ({len(calls)} calls) ===")
    print(f"{'class':12s} {'LIVE':>8s}", *[f"{r:>8s}" for r in reps])
    results = {}
    live_agg, live_tot, lc, ls = attribute(f"{BASE}/{live_cfg}/run_1", calls, 0.4)
    for r in reps:
        results[r] = attribute(f"{BASE}/{rep_cfg}/{r}", calls, 0.15)
    for cls in ("internal", "build/tests", "git", "other bash"):
        vals = [results[r][0].get(cls, 0.0) for r in reps]
        print(f"{cls:12s} {live_agg.get(cls,0):8.1f}", *[f"{v:8.1f}" for v in vals])
    print(f"{'TOTAL':12s} {live_tot:8.1f}", *[f"{results[r][1]:8.1f}" for r in reps])
    print(f"anchors live: {lc} long calls vs {ls} long spans; "
          f"replays: " + ", ".join(f"{r}={results[r][2]}v{results[r][3]}" for r in reps))
EOF_MARKER_UNUSED = None
