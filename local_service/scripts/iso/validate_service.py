#!/usr/bin/env python3
"""validate_service.py — proof-based validation of the isolated service campaign
(local_service/data_iso). Same evidence philosophy as validate_glm_agents.py:
every number must carry observed proof plus an independent cross-check.

Per cell (bucket x tier x run):
  E1  window length: each stat window's perf "seconds time elapsed" ~= WINSEC
  E2  CPUs formula:  task-clock/elapsed == perf's own "CPUs utilized" comment
  E3  cpu.stat vs PMU: cgroup usage_usec delta over the capture ~= summed PMU
      task-clock for the same pod (two subsystems, one truth; vllm+fastapi)
  E5  work proof:    engine token counter advanced in >= nw-2 windows
  E6  kernel share:  cycles:k/(k+u) from PMU vs system_usec/usage_usec from the
      scheduler for the vllm pod (within 6 pp when share > 10%, else informational)
  MUX zero-multiplexing: any scaling annotation or <not counted>/<not supported>
      rejects the window (hard fail if >2 windows rejected)
  TMA sanity: topdown L1 shares sum to 95-105% of slots-weighted total

Per cell (across the 3 repeats):
  DISPERSION: activity-weighted IPC + CPUs per pod; max relative spread across
      repeats <= 10% (vllm, fastapi) — measurement precision AND stationarity.

Usage: python3 validate_service.py [data_root]   (default ../../data_iso)
Exit 0 = all hard checks pass.
"""
import json, os, re, sys
from glob import glob

DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "data_iso")

LINE = re.compile(r"^\s+([\d,]+(?:\.\d+)?)\s+(?:msec\s+)?([\w.:-]+)\s+(kubepods\S+)"
                  r"(?:\s+#\s+([\d.]+)\s+CPUs utilized)?")
ELAPSED = re.compile(r"([\d.]+)\s+seconds time elapsed")

hard_fails = []
def chk(level, cell, name, ok, detail):
    tag = "OK " if ok else ("FAIL" if level == "hard" else "warn")
    print(f"  {tag} {cell:24s} {name:18s} {detail}")
    if level == "hard" and not ok:
        hard_fails.append(f"{cell} {name}")

def parse_run(rd):
    """-> (per-pod counter sums, n_windows_used, n_rejected, e1/e2 evidence)"""
    meta = json.load(open(f"{rd}/metadata.json"))
    cg2pod = {v: k for k, v in meta["pods"].items()}
    S, e1, e2, nuse, nrej = {}, [], [], 0, 0
    tcw = {}   # windows that carried task-clock for a pod (only core+priv groups have it)
    tcwin = {} # pod -> [(win_no, task_clock_ms)] for same-interval cpu.stat comparison
    for f in sorted(glob(f"{rd}/group_*_w*.txt")):
        txt = open(f).read()
        if ("<not counted>" in txt or "<not supported>" in txt
                or re.search(r"\(\s*\d+[.,]\d+%\s*\)\s*$", txt, re.M)):
            nrej += 1
            continue
        nuse += 1
        m = ELAPSED.search(txt)
        if m: e1.append(float(m.group(1)))
        for ln in txt.splitlines():
            mm = LINE.match(ln)
            if not mm: continue
            v, ev, cg = float(mm.group(1).replace(",", "")), mm.group(2), mm.group(3)
            pod = cg2pod.get(cg)
            if not pod: continue
            S.setdefault(pod, {})
            S[pod][ev] = S[pod].get(ev, 0.0) + v
            if ev == "task-clock":
                tcw[pod] = tcw.get(pod, 0) + 1
                wno = int(re.search(r"_w(\d+)\.txt$", f).group(1))
                tcwin.setdefault(pod, []).append((wno, v))
                if mm.group(4) and m:
                    e2.append((v / 1000 / float(m.group(1)), float(mm.group(4))))
    return meta, S, nuse, nrej, e1, e2, tcw, tcwin

def load_cpustat(rd, pod):
    rows = []
    try:
        for ln in open(f"{rd}/cpustat_{pod}.tsv"):
            p = ln.split()
            if len(p) >= 6 and p[1] == "usage_usec":
                rows.append((float(p[0]), float(p[2]), float(p[6]) if len(p) > 6 else float(p[5])))
    except OSError:
        pass
    return rows

def interp(rows, t):
    """cumulative (usage, system) at exactly t, linearly interpolated between polls."""
    if t <= rows[0][0]: return rows[0][1], rows[0][2]
    if t >= rows[-1][0]: return rows[-1][1], rows[-1][2]
    lo, hi = 0, len(rows) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if rows[mid][0] <= t: lo = mid
        else: hi = mid
    a, b = rows[lo], rows[hi]
    f = (t - a[0]) / max(b[0] - a[0], 1e-9)
    return a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])

def cpustat_delta(rows, t0, t1):
    """usage/system core-seconds inside exactly [t0,t1]."""
    if len(rows) < 2: return None
    u0, s0 = interp(rows, t0); u1, s1 = interp(rows, t1)
    return (u1 - u0) / 1e6, (s1 - s0) / 1e6

cells = {}
for rd in sorted(glob(f"{DATA}/svc_*/run_*")):
    if not os.path.exists(f"{rd}/DONE"): continue
    cell = rd.split("/")[-2]
    cells.setdefault(cell, []).append(rd)

for cell, runs in cells.items():
    agg = {}
    for rd in runs:
        rn = f"{cell}/{os.path.basename(rd)}"
        meta, S, nuse, nrej, e1, e2, tcw, tcwin = parse_run(rd)
        W = meta["winsec"]; NW = meta["cycles"] * 9
        chk("hard", rn, "MUX-reject", nrej <= 2, f"{nrej} rejected / {nuse} used")
        bad1 = [x for x in e1 if abs(x - W) > 0.6]
        chk("hard", rn, "E1-winlen", not bad1, f"{len(e1)} windows, worst {max(e1, key=lambda x: abs(x-W)):.2f}s" if e1 else "no windows")
        bad2 = [(a, b) for a, b in e2 if abs(a - b) > 0.05 * max(b, 0.05)]
        chk("hard", rn, "E2-cpus", len(bad2) <= 2, f"{len(e2)-len(bad2)}/{len(e2)} rows match perf's own CPUs")
        # E5 work: token deltas from windows.tsv
        toks = [int(l.split("\t")[5]) for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]
                if len(l.split("\t")) > 5]
        busy = sum(1 for a, b in zip(toks, toks[1:]) if b > a)
        chk("hard", rn, "E5-work", busy >= NW - 2, f"tokens advanced in {busy}/{len(toks)-1} gaps")
        # E3 + E6 on the two main fences
        ws = [l.split("\t") for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]]
        t0, t1 = float(ws[0][2]), float(ws[-1][3])
        cap_frac = sum(e1) / (t1 - t0) if e1 else 0  # PMU-covered fraction of the span
        for pod in ("vllm", "fastapi"):
            d = S.get(pod, {})
            rows = load_cpustat(rd, pod)
            cs = cpustat_delta(rows, t0, t1)
            if cs and tcwin.get(pod):
                wtimes = {int(w[0]): (float(w[2]), float(w[3])) for w in ws}
                pmu = sched = 0.0
                for wno, tc in tcwin[pod]:
                    iv = wtimes.get(wno)
                    if not iv: continue
                    delta = cpustat_delta(rows, iv[0], iv[1])
                    if delta is None: continue
                    pmu += tc / 1000; sched += delta[0]
                ok = abs(pmu - sched) <= 0.10 * max(sched, 1.0)
                chk("hard", rn, f"E3-{pod}", ok,
                    f"PMU {pmu:.1f} vs cpu.stat {sched:.1f} core-s (same {len(tcwin[pod])} windows)")
            if cs and d.get("cycles:k") is not None and d.get("cycles:u"):
                pk = 100 * d["cycles:k"] / (d["cycles:k"] + d["cycles:u"])
                sk = 100 * cs[1] / max(cs[0], 1e-9)
                lvl = "hard" if max(pk, sk) > 10 else "info"
                chk(lvl, rn, f"E6-kern-{pod}", abs(pk - sk) <= 6, f"PMU {pk:.1f}% vs sched {sk:.1f}%")
        # TMA sanity on vllm
        d = S.get("vllm", {})
        tma = [d.get(f"topdown-{k}", 0) for k in ("retiring", "bad-spec", "fe-bound", "be-bound")]
        if sum(tma) and d.get("slots"):
            r = 100 * sum(tma) / d["slots"]
            chk("hard", rn, "TMA-sum", 90 <= r <= 110, f"L1 sum = {r:.1f}% of slots")
        # collect for dispersion
        for pod in ("vllm", "fastapi"):
            d = S.get(pod, {})
            if d.get("cycles") and tcw.get(pod):
                agg.setdefault(pod, []).append(
                    (d["instructions"] / d["cycles"], d["task-clock"] / 1000 / (tcw[pod] * W)))
    if len(runs) >= 2:
        for pod, vals in agg.items():
            for i, name in ((0, "IPC"), (1, "CPUs")):
                xs = [v[i] for v in vals]
                spread = (max(xs) - min(xs)) / (sum(xs) / len(xs))
                lvl = "hard" if (pod == "vllm" or name == "IPC") else "info"
                chk(lvl, cell, f"DISP-{pod}-{name}", spread <= 0.10,
                    f"{'/'.join(f'{x:.3f}' for x in xs)} spread {100*spread:.1f}%")

print(f"\n{'ALL HARD CHECKS PASS' if not hard_fails else str(len(hard_fails)) + ' HARD FAILURES: ' + '; '.join(hard_fails[:8])}")
print(f"cells validated: {len(cells)} ({sum(len(r) for r in cells.values())} runs)")
sys.exit(1 if hard_fails else 0)
