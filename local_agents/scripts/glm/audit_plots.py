#!/usr/bin/env python3
"""audit_plots.py — verify the rendered figures match the raw data (PLOT_SPEC-driven).

Rewritten 2026-07-14: the old version hardcoded the archived certified campaign and values
hand-transcribed from its PNGs. Now: `plot_glm_results.py` dumps every number it renders to
<out>/values_dump.json at render time; this script INDEPENDENTLY recomputes each one from the
raw capture files (fresh regex parsing — no imports from the plotter or the validator) and
compares. Tolerances = display rounding. Exit 0 = all match.

  PLOT_SPEC=local_agents/SWE_long/plot_spec.json python3 audit_plots.py
  (or)  python3 audit_plots.py <plot_spec.json>

Burst vocabulary under test (must equal the MANIFEST): tool active >0.005 cores, harness
>0.02, heavy peak >0.3, gaps <0.4 s merged, dust <=0.001 core-s dropped, exact usec deltas.
"""
import json, os, re, sys, bisect
from glob import glob
from statistics import median

SPEC_PATH = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("PLOT_SPEC")
if not SPEC_PATH:
    sys.exit("need PLOT_SPEC (env or argv[1])")
SPEC = json.load(open(SPEC_PATH))
DATA, OUT = SPEC["data"], SPEC["out"]
DUMP = json.load(open(f"{OUT}/values_dump.json"))
TASKS = [(x[0], x[1], list(x[2])) for x in SPEC["resolved"]]

THR_TOOL, THR_HARN, THR_HEAVY, GAP, DUST = 0.005, 0.02, 0.3, 0.4, 0.001
LINE = re.compile(r"^\s+([\d,]+(?:\.\d+)?|<not counted>)\s+(?:msec\s+)?([\w.:-]+)\s+(\S*(?:\.scope|/agent|/toolexec))")

def cpustat(rd, i):
    out = []
    for ln in open(f"{rd}/cpustat_scope{i}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0:
            out.append((float(p[0]), float(p[2])))
    return out

def samples(rd, i):
    s = cpustat(rd, i)
    return [(a[0], b[0], max(b[1]-a[1], 0.0)/1e6/max(b[0]-a[0], 1e-9), max(b[1]-a[1], 0.0)/1e6)
            for a, b in zip(s, s[1:])]

def bursts(rd, i, thr):
    out, cur = [], None
    for t0, t1, r, cs in samples(rd, i):
        if r > thr:
            if cur and t0 - cur[1] < GAP:
                cur[1] = t1; cur[2] += cs; cur[3] = max(cur[3], r)
            else:
                if cur: out.append(cur)
                cur = [t0, t1, cs, r]
    if cur: out.append(cur)
    return [b for b in out if b[2] > DUST]

def sums(rd, role_of):
    """Independent re-implementation of the published window semantics: a fully idle scope
    (all rows <not counted>) is dropped from that window; a scope MIXING counts and
    <not counted> rejects the whole window; multiplex annotation or <not supported>
    rejects the whole window. Counted scopes in a window with idle siblings are KEPT."""
    S, CO = {}, {}
    for f in glob(f"{rd}/group_*_w*.txt"):
        txt = open(f).read()
        if "<not supported>" in txt: continue
        if re.search(r"\(\s*\d+[.,]\d+%\s*\)\s*$", txt, re.M): continue
        raw = {}
        for ln in txt.splitlines():
            m = LINE.match(ln)
            if not m: continue
            v, ev, cg = m.group(1), m.group(2), m.group(3)
            raw.setdefault(cg, {})[ev] = None if v.startswith("<") else float(v.replace(",", ""))
        if any((any(x is None for x in evs.values()) and not all(x is None for x in evs.values()))
               for evs in raw.values()):
            continue                     # partial scope -> genuine PMU failure, reject window
        for cg, evs in raw.items():
            if all(x is None for x in evs.values()):
                continue                 # idle scope (model-wait) -> drop scope, keep window
            r = role_of(cg)
            if r:
                S.setdefault(r, {}); CO.setdefault(r, {})
                wI, wC = evs.get("instructions", 0.0) or 0.0, evs.get("cycles", 0.0) or 0.0
                for ev, x in evs.items():
                    S[r][ev] = S[r].get(ev, 0.0) + x
                    e = CO[r].setdefault(ev, {"I": 0.0, "C": 0.0})
                    e["I"] += wI; e["C"] += wC
    # hardened-rerun episodes: continuous TMA census csv (time,count,,event,cgroup,run,pct)
    tc = f"{rd}/tma_cont.csv"
    if os.path.exists(tc):
        for ln in open(tc):
            if ln.startswith("#") or not ln.strip(): continue
            p = [x.strip() for x in ln.split(",")]
            if len(p) < 7 or p[1].startswith("<"): continue
            r = role_of(p[4])
            if not r: continue
            try:
                S.setdefault(r, {})[p[3]] = S[r].get(p[3], 0.0) + float(p[1])
            except ValueError:
                pass
    return S, CO

fails = 0
def chk(task, name, got, want, tol):
    global fails
    ok = abs(got - want) <= tol
    if not ok: fails += 1
    print(f"  {'OK ' if ok else 'FAIL'} {task:22s} {name:22s} plotted={round(want,3):<10g} recomputed={round(got,3):<10g} tol={tol}")

for name, cfg, runs in TASKS:
    rd = f"{DATA}/{cfg}/{runs[0]}"
    D = DUMP[name]
    meta = json.load(open(f"{rd}/metadata.json"))
    ex = meta.get("extra", {})
    if meta.get("workload") == "oc":
        c = ex.get("container_cg", "")
        role_of = lambda cg, c=c, ex=ex: ("harness" if cg == f"{c}/agent" else
                                          "tool" if cg == f"{c}/toolexec" else
                                          "proxy" if cg == ex.get("proxy_cg") else None)
    else:
        role_of = lambda cg, ex=ex: ("harness" if cg == ex.get("harness_cg") else
                                     "tool" if cg == ex.get("tool_cg") else
                                     "proxy" if cg == ex.get("proxy_cg") else None)

    # ---- Fig 1a/1b: wall, exact fence totals, active-wall (NOTE: cs may carry the OC
    # lineage correction — recompute the RAW totals and allow the moved share as tolerance)
    tt = cpustat(rd, 2)
    chk(name, "wall (min)", (tt[-1][0]-tt[0][0])/60, D["wall_min"], 0.05)
    raw = []
    for i in (1, 2, 3):
        s = cpustat(rd, i)
        raw.append((s[-1][1]-s[0][1])/1e6 if len(s) > 1 else 0.0)
    chk(name, "core-s total", sum(raw), D["cs_total"], 0.5)   # correction moves, never adds
    tool_act = sum(t1-t0 for t0, t1, r, _ in samples(rd, 2) if r > THR_TOOL)
    harn_act = sum(t1-t0 for t0, t1, r, _ in samples(rd, 1) if r > THR_HARN)
    chk(name, "tool active wall (s)", tool_act, D["tool_active_s"], 0.5)
    if meta.get("workload") != "oc":   # OC harness active-wall is lineage-scaled in the dump
        chk(name, "harness active wall(s)", harn_act, D["harn_active_s"], 0.5)

    # ---- Fig 3 / call structure: heavy bursts, med duration, peak, tool wall share
    B = bursts(rd, 2, THR_HEAVY)
    chk(name, "heavy bursts", len(B), D["heavy_bursts"], 0)
    chk(name, "median heavy dur (s)", median([b[1]-b[0] for b in B]) if B else 0, D["med_heavy_dur"], 0.06)
    chk(name, "peak spike (cores)", min(max(b[3] for b in B), 20.0) if B else 0, D["peak_spike"], 0.06)
    chk(name, "tool wall share (%)", 100*tool_act/(tt[-1][0]-tt[0][0]), D["tool_wall_pct"], 0.3)
    # sustained peak (1 s window / 0.25 step, cap 20) — independent recompute
    s2 = cpustat(rd, 2); T2 = [r[0] for r in s2]
    sp, t = 0.0, s2[0][0]
    while t + 1.0 <= s2[-1][0]:
        i = bisect.bisect_left(T2, t); j = bisect.bisect_left(T2, t + 1.0)
        if 0 <= i < j < len(s2) and s2[j][0] > s2[i][0]:
            sp = max(sp, (s2[j][1]-s2[i][1])/1e6/(s2[j][0]-s2[i][0]))
        t += 0.25
    chk(name, "sustained peak", min(sp, 20.0), D["sust"], 0.06)

    # ---- Figs 4/5: counter-derived card (independent parse of the group files)
    S, CO = sums(rd, role_of)
    for role in ("tool", "harness"):
        d, DD = S.get(role, {}), D[role]
        I, cyc = d.get("instructions", 0) or 1, d.get("cycles", 0)
        chk(name, f"{role} IPC", I/cyc if cyc else 0, DD["IPC"], 0.01)
        ut = d.get("idq.dsb_uops", 0)+d.get("idq.mite_uops", 0)+d.get("idq.ms_uops", 0)+d.get("lsd.uops", 0)
        chk(name, f"{role} DSB %", 100*d.get("idq.dsb_uops", 0)/ut if ut else 0, DD["DSB"], 0.6)
        ck, cu = d.get("cycles:k", 0), d.get("cycles:u", 0)
        chk(name, f"{role} OS share %", 100*ck/(ck+cu) if ck+cu else 0, DD["kern"], 0.6)
        fpk = sum(v for k, v in d.items() if k.startswith("fp_arith")
                  and ("packed" in k or k.endswith(".vector")))
        fps = sum(v for k, v in d.items() if k.startswith("fp_arith") and "scalar" in k)
        chk(name, f"{role} packed FP %", 100*fpk/(fpk+fps) if fpk+fps else 0, DD["vecFP"], 0.6)
        co = CO.get(role, {})
        def coI(k, d=d, co=co):
            e = co.get(k)
            return (e or {}).get("I", 0) or (d.get("instructions", 0) or 1)
        chk(name, f"{role} brMPKI", 1000*d.get("branch-misses", 0)/coI("branch-misses"), DD["brMPKI"], 0.06)
        chk(name, f"{role} L1I MPKI", 1000*d.get("l2_rqsts.all_code_rd", 0)/coI("l2_rqsts.all_code_rd"), DD["L1I_MPKI"], 0.06)
        tma = [d.get(f"topdown-{k}", 0) for k in ("retiring", "bad-spec", "fe-bound", "be-bound")]
        ts = sum(tma) or 1
        for lab, idx in (("retiring", 0), ("bad", 1), ("fe", 2), ("be", 3)):
            chk(name, f"{role} TMA {lab} %", 100*tma[idx]/ts, D[f"tma_{role}"][lab], 0.6)
        # TMA L2: child + remainder per L1 bucket, same order as the figure
        l2 = []
        for l1k, subk in (("topdown-retiring", "topdown-heavy-ops"),
                          ("topdown-fe-bound", "topdown-fetch-lat"),
                          ("topdown-bad-spec", "topdown-br-mispredict"),
                          ("topdown-be-bound", "topdown-mem-bound")):
            sub = d.get(subk, 0.0)
            if l1k == "topdown-fe-bound":     # figure order: SUB then REMAIN differs per bucket
                l2 += [100*sub/ts, 100*max(d.get(l1k, 0)-sub, 0)/ts]
            elif l1k == "topdown-retiring":
                l2 += [100*max(d.get(l1k, 0)-sub, 0)/ts, 100*sub/ts]
            else:
                l2 += [100*sub/ts, 100*max(d.get(l1k, 0)-sub, 0)/ts]
        for seg_i, seg_v in enumerate(l2):
            chk(name, f"{role} TMA-L2 seg{seg_i}", seg_v, D[f"tma_l2_{role}"][seg_i], 0.6)

    # ---- Fig 4c TMA L3/L4 drill (hardened-rerun sets only)
    for role in ("tool", "harness"):
        DL3 = D.get(f"l3_{role}")
        if not DL3:
            continue
        d = S.get(role, {})
        co = CO.get(role, {})
        for lab, key in (("icache", "icache_data.stalls"), ("itlb", "icache_tag.stalls"),
                         ("resteer", "int_misc.clear_resteer_cycles"),
                         ("div", "arith.div_active"),
                         ("dram_ge1", "offcore_requests_outstanding.cycles_with_data_rd")):
            e = co.get(key)
            cyc = (e or {}).get("C", 0) or (d.get("cycles", 0) or 1)
            chk(name, f"{role} L3 {lab} %cyc", 100 * d.get(key, 0) / cyc, DL3[lab], 0.15)

    # ---- Fig 4d TMA tree (independent recompute of all 16 leaf segments)
    for role in ("tool", "harness"):
        DT = D.get(f"tma_tree_{role}")
        if not DT:
            continue
        d = S.get(role, {}); co = CO.get(role, {})
        t = {k: d.get(f"topdown-{k}", 0) for k in ("retiring", "bad-spec", "fe-bound", "be-bound")}
        ts = sum(t.values()) or 1
        L1 = {k: 100*v/ts for k, v in t.items()}
        heavy = 100*d.get("topdown-heavy-ops", 0)/ts
        br = 100*d.get("topdown-br-mispredict", 0)/ts
        flat = 100*d.get("topdown-fetch-lat", 0)/ts
        mem = 100*d.get("topdown-mem-bound", 0)/ts
        core = max(L1["be-bound"] - mem, 0)
        def cycp(k, d=d, co=co):
            e = co.get(k) or {}
            c = e.get("C", 0) or (d.get("cycles", 0) or 1)
            return 100*d.get(k, 0)/c
        kids = {"ic": cycp("icache_data.stalls"), "it": cycp("icache_tag.stalls"),
                "rs": cycp("int_misc.clear_resteer_cycles")}
        sc = min(1.0, flat/max(sum(kids.values()), 1e-9))
        fl = {k: v*sc for k, v in kids.items()}
        fbw = max(L1["fe-bound"] - flat, 0)
        dsb, mite = d.get("idq.dsb_uops", 0), d.get("idq.mite_uops", 0)
        fbw_dsb = fbw*dsb/max(dsb+mite, 1)
        w = {"l1": 5*d.get("mem_load_retired.l1_hit", 0), "l2": 15*d.get("mem_load_retired.l2_hit", 0),
             "l3": 50*d.get("mem_load_retired.l3_hit", 0), "dr": 250*d.get("mem_load_retired.l3_miss", 0)}
        tw = sum(w.values()) or 1
        div = min(cycp("arith.div_active"), core)
        vec = [max(L1["retiring"]-heavy, 0), heavy, fl["ic"], fl["it"], fl["rs"],
               max(flat-sum(fl.values()), 0), fbw_dsb, max(fbw-fbw_dsb, 0),
               br, max(L1["bad-spec"]-br, 0),
               mem*w["l1"]/tw, mem*w["l2"]/tw, mem*w["l3"]/tw, mem*w["dr"]/tw,
               div, max(core-div, 0)]
        for si, x in enumerate(vec):
            chk(name, f"{role} tree seg{si}", x, DT[si], 0.6)

    # ---- Fig 7b cumulative sums = raw fence totals
    for nm, idx in (("tool", 1), ("harness", 0), ("litellm", 2)):
        if nm in D.get("cumsum", {}):
            chk(name, f"cumsum {nm}", raw[idx], D["cumsum"][nm], 0.5)

    # ---- Fig 7 ECDF ns (SWE only)
    if "ecdf_tool_n" in D:
        tj = glob(f"{rd}/traj/*/*.traj")
        ets = [s for s in json.load(open(tj[0]))["trajectory"] if s.get("execution_time", 0) > 0] if tj else []
        chk(name, "ECDF tool n", len(ets), D["ecdf_tool_n"], 0)
        chk(name, "ECDF harness bursts", len(bursts(rd, 1, THR_HARN)), D["ecdf_harn_n"], 2)

    # ---- Fig 8 lanes: samples land ONLY on the pinned partition
    MEAS = set(range(2, 12)) | set(range(14, 24))
    lf = f"{rd}/scope2_cpulanes.tsv"
    if os.path.exists(lf):
        cpus = set()
        for ln in open(lf):
            p = ln.split()
            if len(p) == 2: cpus.add(int(float(p[1])))
        chk(name, "lanes pinned-only", 0 if cpus <= MEAS else 1, 0, 0)

    # ---- Fig 9 card (SWE only): sustained harness peak recompute
    if "card" in D and "sustained peak (cores)" in D["card"]:
        s1 = cpustat(rd, 1); T1 = [r[0] for r in s1]
        sp1, t = 0.0, s1[0][0]
        while t + 1.0 <= s1[-1][0]:
            i = bisect.bisect_left(T1, t); j = bisect.bisect_left(T1, t + 1.0)
            if 0 <= i < j < len(s1) and s1[j][0] > s1[i][0]:
                sp1 = max(sp1, (s1[j][1]-s1[i][1])/1e6/(s1[j][0]-s1[i][0]))
            t += 0.25
        chk(name, "harness sustained", min(sp1, 20.0), D["card"]["sustained peak (cores)"], 0.006)

print(f"\n{'ALL MATCH — figures faithfully represent the data' if fails == 0 else f'{fails} MISMATCHES'}")
sys.exit(1 if fails else 0)
