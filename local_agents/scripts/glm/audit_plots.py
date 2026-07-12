#!/usr/bin/env python3
"""audit_plots.py — verify the rendered figures match the raw data (all 4 tasks).

Independent recomputation: fresh regex parsing (NOT importing the plotter or the
validator), raw cpustat/traj/group files -> every displayed number, compared against
DISPLAYED values transcribed from the rendered PNGs. Tolerances = display rounding.
Exit 0 = all match."""
import bisect, json, re, sys
from glob import glob
from statistics import median

DATA = "../../data"
TASKS = {"astropy": "glm_swe_astropy", "scikit-learn": "glm_swe_scikit-learn",
         "sympy": "glm_swe_sympy", "django": "glm_swe_django-lite"}

# ---- values as DISPLAYED in the rendered PNGs (transcribed 2026-07-09) -----------------------
D = {
 #        wall  cpu_s  toolCPU% harnCPU% | calls med  peak  wall% | IPCt  DSBt kernt fpT | IPCh | ret fe | topslice kernC | ecdf_t ecdf_h | heavy acts
 "astropy":      (15, 147, 92, 6,  35, 0.4, 16.6,  9.5, 1.84, 52, 40, 17, 2.47, 42, 35, 61, 40,  82,  73,  35,  83),
 "scikit-learn": ( 7, 1771, 99, 0,  36, 0.7, 20.0, 27.4, 0.69, 82, 10, 70, 2.45, 43, 24, 85, 10,  72,  62,  36,  75),
 "sympy":        (31, 142, 33, 59, 111, 0.3,  1.3,  2.4, 1.71, 61, 12,  0, 2.83, 29, 34, 75, 12, 188, 188, 111, 188),
 "django":       (41, 506,  6, 87, 317, 0.1,  1.0,  1.3, 1.46, 60, 20,  0, 2.89, 24, 39, 58, 20, 351, 348, 317, 351),
}

SUST = {"astropy": 2.7, "scikit-learn": 20.0, "sympy": 1.0, "django": 0.3}
# harness anatomy figure — displayed values transcribed from the rendered PNG
INTERP_D = {"astropy": 29, "scikit-learn": 28, "sympy": 27, "django": 30}   # (a) interpreter-loop %
SUSTH_D  = {"astropy": 0.61, "scikit-learn": 0.53, "sympy": 0.99, "django": 1.00}  # (b) sustained peak
HBURST_D = {"astropy": 73, "scikit-learn": 62, "sympy": 188, "django": 348}  # (c) burst counts
AVG_D = {"astropy": 0.16, "scikit-learn": 4.36, "sympy": 0.08, "django": 0.21}  # transcribed from donut centers

LINE = re.compile(r"^\s+([\d,]+(?:\.\d+)?)\s+(?:msec\s+)?([\w.:-]+)\s+(\S*(?:\.scope|/agent|/toolexec))")

def sums(rd, role_of):
    S = {}
    for f in glob(f"{rd}/group_*_w*.txt"):
        txt = open(f).read()
        if "<not counted>" in txt or "<not supported>" in txt: continue
        if re.search(r"\(\s*\d+[.,]\d+%\s*\)\s*$", txt, re.M): continue
        for ln in txt.splitlines():
            m = LINE.match(ln)
            if not m: continue
            v, ev, cg = float(m.group(1).replace(",", "")), m.group(2), m.group(3)
            r = role_of(cg)
            if r: S.setdefault(r, {}); S[r][ev] = S[r].get(ev, 0.0) + v
    return S

def cpustat(rd, i):
    out = []
    for ln in open(f"{rd}/cpustat_scope{i}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0:
            out.append((float(p[0]), float(p[2])))
    return out

def series(rd, i):
    s = cpustat(rd, i)
    return [((a[0]+b[0])/2, max(0.0, (b[1]-a[1])/max((b[0]-a[0])*1e6, 1e-9))) for a, b in zip(s, s[1:])]

def bursts(rd):
    out, cur = [], None
    for t, v in series(rd, 2):
        if v > 0.3:
            if cur is None: cur = [t, t, 0.0]
            cur[1] = t; cur[2] = max(cur[2], v)
        elif cur and t - cur[1] > 2.0:
            out.append(cur); cur = None
    if cur: out.append(cur)
    return out

def spans(rd, i, thr):
    out, cur = [], None
    for t, v in series(rd, i):
        if v > thr:
            if cur is None: cur = [t, t]
            cur[1] = t
        elif cur and t - cur[1] > 2.0:
            out.append(cur); cur = None
    if cur: out.append(cur)
    return out

fails = 0
def chk(task, name, got, want, tol):
    global fails
    ok = abs(got - want) <= tol
    if not ok: fails += 1
    print(f"  {'OK ' if ok else 'FAIL'} {task:13s} {name:22s} plotted={want:<8g} recomputed={round(got,3):<10g} tol={tol}")

for task, cfg in TASKS.items():
    rd = f"{DATA}/{cfg}/run_1"
    (wall_d, cpus_d, tshare_d, hshare_d, calls_d, med_d, peak_d, wshare_d,
     ipct_d, dsbt_d, kernt_d, fpt_d, ipch_d, ret_d, fe_d, top_d, kernc_d,
     et_d, eh_d, heavy_d, acts_d) = D[task]
    meta = json.load(open(f"{rd}/metadata.json"))
    ex = meta["extra"]
    role_of = lambda cg, ex=ex: ("harness" if cg == ex.get("harness_cg") else
                                 "tool" if cg == ex.get("tool_cg") else
                                 "proxy" if cg == ex.get("proxy_cg") else None)
    # F1 two-view
    tt = cpustat(rd, 2); hh = cpustat(rd, 1); pp = cpustat(rd, 3)
    wall = (tt[-1][0] - tt[0][0]) / 60
    cs = [(x[-1][1] - x[0][1]) / 1e6 for x in (hh, tt, pp)]
    chk(task, "wall (min)", wall, wall_d, 0.6)
    chk(task, "core-s total (displayed)", sum(cs), cpus_d, 1.5)
    chk(task, "avg CPUs (displayed)", sum(cs)/(wall*60), AVG_D[task], 0.005)
    chk(task, "tool CPU share %", 100*cs[1]/sum(cs), tshare_d, 1.0)
    chk(task, "harness CPU share %", 100*cs[0]/sum(cs), hshare_d, 1.0)
    # F3 tool calls
    B = bursts(rd)
    durs = [b[1]-b[0]+0.1 for b in B]
    tool_active = sum(1 for _, v in series(rd, 2) if v > 0.3) * 0.1
    chk(task, "tool calls (heavy)", len(B), calls_d, 0)
    chk(task, "tool calls total (displayed)", len(json.load(open(tj0))["trajectory"]) if (tj0 := glob(f"{rd}/traj/*/*.traj")[0]) else 0, acts_d, 0)
    chk(task, "median dur (s)", median(durs), med_d, 0.06)
    chk(task, "peak CPUs (0.1s, cap 20)", min(max(b[2] for b in B), 20.0), peak_d, 0.06)
    chk(task, "tool-active wall %", 100*tool_active/(wall*60), wshare_d, 0.3)
    # F4/F5 counters (independent parse)
    S = sums(rd, role_of)
    t, h = S["tool"], S["harness"]
    chk(task, "tool IPC", t["instructions"]/t["cycles"], ipct_d, 0.01)
    chk(task, "harness IPC", h["instructions"]/h["cycles"], ipch_d, 0.01)
    ut = t["idq.dsb_uops"] + t["idq.mite_uops"] + t["idq.ms_uops"] + t.get("lsd.uops", 0)
    chk(task, "tool DSB %", 100*t["idq.dsb_uops"]/ut, dsbt_d, 0.6)
    chk(task, "tool kernel %", 100*t["cycles:k"]/(t["cycles:k"]+t["cycles:u"]), kernt_d, 0.6)
    fpk = sum(v for k, v in t.items() if k.startswith("fp_arith") and "packed" in k)
    fps = sum(v for k, v in t.items() if k.startswith("fp_arith") and "scalar" in k)
    chk(task, "tool packed FP %", 100*fpk/(fpk+fps) if fpk+fps else 0, fpt_d, 0.6)
    tma = [t.get(f"topdown-{k}", 0) for k in ("retiring", "bad-spec", "fe-bound", "be-bound")]
    chk(task, "tool TMA retiring %", 100*tma[0]/sum(tma), ret_d, 0.6)
    chk(task, "tool TMA FE %", 100*tma[2]/sum(tma), fe_d, 0.6)
    # F6 software (top slice share of categorized samples, python-or-blas)
    rows = [(float(p[0].rstrip('%')), p[-1]) for p in
            (l.split() for l in open(f"{rd}/scope2_dso.txt")) if p and p[0].endswith('%')]
    tot = sum(r[0] for r in rows)
    py = sum(r[0] for r in rows if re.search(r"python|\.cpython-|libpython", r[1], re.I))
    blas = sum(r[0] for r in rows if re.search(r"openblas|libgomp", r[1], re.I))
    chk(task, "software top slice %", 100*max(py, blas)/tot, top_d, 1.0)
    chk(task, "software kernel ctr %", 100*t["cycles:k"]/(t["cycles:k"]+t["cycles:u"]), kernc_d, 0.6)
    # F7 ECDF ns + F2 counts
    tj = glob(f"{rd}/traj/*/*.traj")[0]
    steps = json.load(open(tj))["trajectory"]
    ets = [s["execution_time"] for s in steps if s.get("execution_time", 0) > 0]
    chk(task, "ECDF tool n", len(ets), et_d, 0)
    chk(task, "ECDF harness n", len(spans(rd, 1, 0.05)), eh_d, 0)
    chk(task, "timeline heavy", len(B), heavy_d, 0)
    chk(task, "timeline actions", len(steps), acts_d, 0)
    # F3 sustained peak (1 s window, step 0.25, cap 20) — independent recompute
    s2 = cpustat(rd, 2); T2 = [r[0] for r in s2]
    sp, t = 0.0, s2[0][0]
    while t + 1.0 <= s2[-1][0]:
        i = bisect.bisect_left(T2, t); j = bisect.bisect_left(T2, t + 1.0)
        if 0 <= i < j < len(s2) and s2[j][0] > s2[i][0]:
            sp = max(sp, (s2[j][1]-s2[i][1])/1e6/(s2[j][0]-s2[i][0]))
        t += 0.25
    chk(task, "peak CPUs sustained", min(sp, 20.0), SUST[task], 0.06)
    # F8 lanes: samples exist and land ONLY on the 20 pinned logical CPUs
    MEAS = set(range(2, 12)) | set(range(14, 24))
    lanes_cpus = set()
    nsamp = 0
    for ln in open(f"{rd}/scope2_cpulanes.tsv"):
        p = ln.split()
        if len(p) == 2:
            lanes_cpus.add(int(p[1])); nsamp += 1
    chk(task, "hw-lanes samples", 1 if nsamp > 500 else 0, 1, 0)
    chk(task, "hw-lanes pinned-only", 0 if lanes_cpus <= MEAS else 1, 0, 0)
    # F9 harness anatomy — independent recompute of the three panels
    interp = tot = 0
    for ln in open(f"{rd}/scope1_leaf.txt"):
        m2 = re.match(r"\s*(\d+)\s+\t?\s*(.+?)\s+\((.+)\)\s*$", ln)
        if not m2: continue
        tot += int(m2.group(1))
        if "_PyEval_EvalFrameDefault" in m2.group(2): interp += int(m2.group(1))
    chk(task, "harness interp-loop %", 100*interp/tot, INTERP_D[task], 0.6)
    s1 = cpustat(rd, 1); T1 = [r[0] for r in s1]
    sp1, t = 0.0, s1[0][0]
    while t + 1.0 <= s1[-1][0]:
        i = bisect.bisect_left(T1, t); j = bisect.bisect_left(T1, t + 1.0)
        if 0 <= i < j < len(s1) and s1[j][0] > s1[i][0]:
            sp1 = max(sp1, (s1[j][1]-s1[i][1])/1e6/(s1[j][0]-s1[i][0]))
        t += 0.25
    chk(task, "harness sustained peak", min(sp1, 20.0), SUSTH_D[task], 0.006)
    hb, cur = 0, None
    hs = series(rd, 1)
    for tt2, vv2 in hs:
        if vv2 > 0.05:
            if cur is None: cur = [tt2, tt2]
            cur[1] = tt2
        elif cur and tt2 - cur[1] > 2.0:
            hb += 1; cur = None
    if cur: hb += 1
    chk(task, "harness burst count", hb, HBURST_D[task], 0)

print(f"\n{'ALL MATCH — figures faithfully represent the data' if fails == 0 else f'{fails} MISMATCHES'}")
sys.exit(1 if fails else 0)
