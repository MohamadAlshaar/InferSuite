#!/usr/bin/env python3
"""validate_glm_agents.py <data_root> <tier_prefix> — proof-based validation for glm_* runs.

Layer 1  collection truth : window rejected on multiplexing annotation, <not supported>,
                            MIXED <not counted> (idle scopes dropped, not rejected),
                            unparseable file, TMA nesting >105%.
Layer 2  plausibility     : per-run medians (IPC/CPUs/kernel%/cs-per-s per ROLE).
Layer 3  behavior         : cross-run dispersion of per-run median IPC per role (flag >25%).

EVIDENCE (house rule: every validation claim ships with observed proof from an
INDEPENDENT source — two subsystems agreeing, never the same code saying OK twice):
  E1 window length   : perf's own 'seconds time elapsed' vs configured WINSEC
  E2 CPUs formula    : our task-clock derivation vs perf's own '# N CPUs utilized' comment
  E3 cpu.stat vs PMU : kernel cgroup accounting (usage_usec deltas over each core window)
                       vs PMU task-clock for the same scope+window
  E4 watcher (OC)    : record sample comms — /agent must be node-dominated, /toolexec not
  E5 work (SWE)      : STEP markers + trajectory presence prove the agent actually worked

Exit 0 = all runs usable; exit 1 = any hard failure. stdlib only (no matplotlib here).
"""
import bisect, json, os, re, sys
from glob import glob
from statistics import median

MUX_RE     = re.compile(r"\(\s*\d+[.,]\d+%\s*\)\s*$", re.M)
# value [msec] event cgroup — task-clock rows carry a 'msec' unit token; the
# 'seconds time elapsed' footer must not parse as a scope (cgroup-shape guard below)
LINE_RE    = re.compile(r"^\s+([\d,]+(?:\.\d+)?|<[^>]+>)\s+(?:msec\s+)?(\S+)\s+(\S+)")
ELAPSED_RE = re.compile(r"([\d.]+)\s+seconds time elapsed")
CPUS_CMT_RE = re.compile(r"msec\s+task-clock\s+(\S+)\s+#\s+([\d.]+)\s+CPUs utilized")
TMA_NEST = {"topdown-heavy-ops": "topdown-retiring", "topdown-br-mispredict": "topdown-bad-spec",
            "topdown-fetch-lat": "topdown-fe-bound", "topdown-mem-bound": "topdown-be-bound"}
L1 = ["topdown-retiring", "topdown-bad-spec", "topdown-fe-bound", "topdown-be-bound"]


def parse_group(path):
    """-> ({cgroup: {event: count}}, reject_reason|None). Idle scopes (ALL rows
    <not counted>) are dropped; a scope MIXING counts and <not counted> rejects
    the window (genuine PMU scheduling failure)."""
    try:
        txt = open(path).read()
    except OSError:
        return {}, "unreadable"
    if not txt.strip():
        return {}, "empty"
    if "<not supported>" in txt:
        return {}, "<not supported> event"
    if MUX_RE.search(txt):
        return {}, "multiplexing annotation"
    raw = {}
    for ln in txt.splitlines():
        m = LINE_RE.match(ln)
        if not m:
            continue
        val, ev, cg = m.groups()
        if "/" not in cg and not cg.endswith(".scope"):
            continue                       # footer lines ('seconds time elapsed'), not a scope
        raw.setdefault(cg, {})[ev] = None if val.startswith("<") else float(val.replace(",", ""))
    out = {}
    for cg, evs in raw.items():
        vals = list(evs.values())
        if all(v is None for v in vals):
            continue
        if any(v is None for v in vals):
            return {}, f"partial <not counted> in {cg.split('/')[-1]}"
        out[cg] = evs
    if not out:
        return {}, "idle"      # ALL scopes slept the whole window (model-wait) — a true
                               # observation for the timeline, NOT a collection failure
    return out, None


def roles_of(meta):
    ex = meta.get("extra", {})
    m = {}
    if str(meta.get("workload", "")).startswith("swe"):
        m[ex.get("harness_cg")] = "harness"
        m[ex.get("tool_cg")] = "tool"
    else:
        c = ex.get("container_cg")
        if c:
            m[f"{c}/agent"] = "harness"
            m[f"{c}/toolexec"] = "tool"
    m[ex.get("proxy_cg")] = "proxy"
    return {k: v for k, v in m.items() if k}


def load_meta(rd):
    try:
        meta = json.load(open(f"{rd}/metadata.json"))
        return meta, roles_of(meta), int(meta.get("winsec", 10))
    except Exception:
        return {}, {}, 10


def check_run(rd):
    fails, warns = [], []
    meta, roles, winsec = load_meta(rd)
    if not meta:
        fails.append("metadata.json missing/unparseable")
    if not os.path.exists(f"{rd}/DONE"):
        fails.append("no DONE marker")
    wins = glob(f"{rd}/group_*_w*.txt")
    if len(wins) < 8:
        fails.append(f"only {len(wins)} windows")
    rejected, idle, ipc, clocks, kshare, csps, wsum, mix, gwin = 0, 0, {}, {}, {}, {}, {}, {}, {}
    for w in sorted(wins):
        g, why = parse_group(w)
        if why == "idle":
            idle += 1
            continue
        if why:
            rejected += 1
            if rejected <= 3:
                warns.append(f"{os.path.basename(w)}: {why}")
            continue
        base = os.path.basename(w)
        if base.startswith("group_tma_"):        # legacy windowed-TMA runs; new runs use tma_cont.csv
            for cg, ev in g.items():
                if sum(ev.get(k, 0) for k in L1) <= 0:
                    continue
                for c, p in TMA_NEST.items():
                    if ev.get(c, 0) > 1.05 * max(ev.get(p, 0), 1):
                        warns.append(f"{base}:{roles.get(cg, cg.split('/')[-1])} L2 {c} > 105% of {p}")
        # metric collection is EVENT-PRESENCE based (2026-07-14): cycles+instructions live in
        # every group, branches/fp moved groups in the reworked rotation — prefix-keyed
        # collection would silently lose them on new data.
        for cg, ev in g.items():
            role = roles.get(cg, cg.split("/")[-1][:40])
            if ev.get("cycles", 0) > 0 and "instructions" in ev:
                ipc.setdefault(role, []).append(ev["instructions"] / ev["cycles"])
                sums = wsum.setdefault(role, [0.0, 0.0])
                sums[0] += ev["instructions"]; sums[1] += ev["cycles"]
                gw = gwin.setdefault(role, {})
                grp = base.split("_w")[0].replace("group_", "")
                gw.setdefault(grp, []).append((int(base.split("_w")[1].split(".")[0]),
                                               ev["instructions"], ev["cycles"]))
            if "branches" in ev and ev.get("instructions", 0) > 0:
                mx = mix.setdefault(role, {"br": 0.0, "bri": 0.0, "fp": 0.0, "fpi": 0.0})
                mx["br"] += ev.get("branches", 0); mx["bri"] += ev["instructions"]
            if any(k.startswith("fp_arith") for k in ev):
                mx = mix.setdefault(role, {"br": 0.0, "bri": 0.0, "fp": 0.0, "fpi": 0.0})
                mx["fp"] += sum(v for k, v in ev.items() if k.startswith("fp_arith"))
                mx["fpi"] += ev.get("instructions", 0)
            if "task-clock" in ev:
                clocks.setdefault(role, []).append(ev["task-clock"])
            ck, cu = ev.get("cycles:k", 0), ev.get("cycles:u", 0)
            if ck + cu > 0:
                kshare.setdefault(role, []).append(ck / (ck + cu))
                if ev.get("task-clock", 0) > 0:
                    csps.setdefault(role, []).append(ev.get("context-switches", 0) / (ev["task-clock"] / 1000.0))
    # ---- per-group coverage + split-half stability gates (2026-07-14, hardened rerun) ------
    # busy window = the fence actually executed (>~1.5 ms of cycles). HARD on shuffled-rotation
    # runs (the new contract); informational on legacy fixed-rotation data.
    # floors: WARN below 10 (round target); HARD below 7 — the SMARTS sizing bound
    # n >= (1.96*CV/0.03)^2 ~= 7 at our measured cross-window CV <=4% (split-half evidence
    # printed per run). Episodes are NOT re-rolled for being short: retrying until an episode
    # is long selects for struggling trajectories (behavioral bias); short-but-clean episodes
    # are kept and their wider CI documented.
    BUSY_CYC = 5e6
    WARN_BUSY = int(os.environ.get("GATE_MIN_BUSY", 10))
    HARD_BUSY = int(os.environ.get("GATE_HARD_BUSY", 7))
    # hard only for figure-source episodes: replays are determinism anchors (often <2 min,
    # structurally too short for per-group window counts) — warn there, never fail
    # figure-source episodes only: replays (anchors) and single-group dedicated captures
    # (method probes, GORDER_OVERRIDE) are instruments — their gates are the anchor section
    # and the probe-correlation analysis, not per-group coverage/stability floors.
    hard_gates = (meta.get("rotation") == "shuffled"
                  and "replay" not in str(meta.get("workload", ""))
                  and len(str(meta.get("gorder", "x y")).split()) > 1)
    for role in ("tool", "harness"):
        n_busy = {grp: sum(1 for r in rows_g if r[2] > BUSY_CYC)
                  for grp, rows_g in gwin.get(role, {}).items()}
        total_busy = sum(n_busy.values())
        # A fence whose TOTAL activity is structurally sparse (OC light tasks: the tool runs
        # for seconds, not minutes) cannot satisfy a per-group floor no matter how many
        # retries — that is a precision statement (documented CI), not a capture failure.
        # HARD only when the fence had ample activity yet a group missed it (distribution
        # failure; E9 guards the same from the burst side).
        ample = total_busy >= len(n_busy) * HARD_BUSY
        for grp, nb in sorted(n_busy.items()):
            if nb < HARD_BUSY:
                msg = f"group '{grp}' {role}: only {nb} busy windows (<{HARD_BUSY})"
                if hard_gates and ample:
                    fails.append(msg + " despite ample fence activity — distribution failure")
                else:
                    warns.append(msg + (" — structurally sparse fence, wider CI, document"
                                        if not ample else ""))
            elif nb < WARN_BUSY:
                warns.append(f"group '{grp}' {role}: {nb} busy windows "
                             f"(<{WARN_BUSY} target — wider CI, document)")
        allb = sorted((r for rows_g in gwin.get(role, {}).values() for r in rows_g
                       if r[2] > BUSY_CYC), key=lambda r: r[0])
        # the split-half is only a HARD instrument gate when the fence has AMPLE busy windows
        # (same criterion as the busy floor): a sparse fence's halves hold a handful of
        # windows each and the statistic is small-n noise -> CI warn (measured: web-digest
        # tool 35% "instability" over 2-7 busy windows/group, E2/E3 proofs clean).
        if len(allb) >= 6:
            # split-half CI on busy windows only: the honest version of "character is stable
            # across windows" (a pooled CV over mostly-idle windows would flatter itself)
            h1, h2 = allb[0::2], allb[1::2]
            w1 = sum(r[1] for r in h1) / max(sum(r[2] for r in h1), 1)
            w2 = sum(r[1] for r in h2) / max(sum(r[2] for r in h2), 1)
            d = abs(w1 - w2) / max((w1 + w2) / 2, 1e-9)
            # <=10% = quiet OK; 10-20% = WARN (tool fences drift compositionally over an
            # episode — the harness on the same run is the homogeneity control: if IT is
            # tight, the instrument is fine and the spread is the workload's own CI, which
            # the episode-sum estimator averages; report it); >20% = instrument-level, HARD.
            if d > 0.20:
                # the CPython (GIL) SWE harness is a genuine homogeneity control; the OC
                # gateway is node/V8, whose JIT tier-up gives a REAL early-vs-late IPC drift
                # (measured 21% on two independent episodes with E2/E3 instrument proofs
                # clean) — there it is workload character, reported as CI, not a failure.
                oc_harness = role == "harness" and meta.get("workload") == "oc"
                msg = f"split-half wIPC {role} unstable: {w1:.2f} vs {w2:.2f} ({100*d:.0f}%)"
                if oc_harness:
                    warns.append(msg + " — node/V8 JIT drift; report as CI")
                elif not ample:
                    warns.append(msg + " — sparse fence, small-n statistic; report as CI")
                else:
                    (fails if hard_gates else warns).append(msg)
            elif d > 0.10:
                warns.append(f"split-half wIPC {role}: {w1:.2f} vs {w2:.2f} ({100*d:.0f}%) — "
                             f"compositional drift; report as this episode's CI")
            else:
                warns.append(f"split-half wIPC {role}: {w1:.2f} vs {w2:.2f} (|d| {100*d:.1f}%) OK")
    if wins and rejected / len(wins) > 0.25:
        fails.append(f"{rejected}/{len(wins)} windows rejected")
    elif rejected:
        warns.append(f"{rejected}/{len(wins)} windows rejected total")
    if idle:
        warns.append(f"{idle}/{len(wins)} windows fully idle (model-wait) — timeline signal, not loss")
    rec = f"{rd}/rec_scope1.data"
    if not (os.path.exists(rec) and os.path.getsize(rec) > 50000):
        fails.append("rec_scope1.data missing/small")
    stats = {}
    for role in set(ipc) | set(clocks) | set(kshare):
        stats[role] = {"wipc": round(wsum[role][0] / wsum[role][1], 3) if role in wsum and wsum[role][1] else None,
                       "ipc": round(median(ipc[role]), 3) if role in ipc else None,
                       # task-clock is msec summed over the window: CPUs = msec/1000/winsec
                       "cpus": round(median(clocks[role]) / 1000.0 / winsec, 3) if role in clocks else None,
                       "kernel_pct": round(100 * median(kshare[role]), 1) if role in kshare else None,
                       "cs_per_s": round(median(csps[role]), 0) if role in csps else None}
        m = mix.get(role)
        if m:
            stats[role]["br_ki"] = round(1000 * m["br"] / m["bri"], 2) if m["bri"] else None
            stats[role]["fp_ki"] = round(1000 * m["fp"] / m["fpi"], 3) if m["fpi"] else None
    return fails, warns, stats


def evidence_checks(rd):
    """Independent cross-checks. -> (hard_fails, evidence_lines)"""
    fails, ev = [], []
    meta, roles, winsec = load_meta(rd)

    # E1: window length — perf's own elapsed footer vs configured WINSEC
    els = []
    for w in glob(f"{rd}/group_*_w*.txt"):
        m = ELAPSED_RE.search(open(w).read())
        if m:
            els.append(float(m.group(1)))
    if els:
        med = median(els)
        ok = abs(med - winsec) < 0.6
        ev.append(f"E1 window length: perf elapsed median {med:.2f}s vs WINSEC {winsec}s ({len(els)} windows) -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append("E1 window length mismatch")
    else:
        ev.append("E1 window length: no elapsed footers found -> MISMATCH")
        fails.append("E1 no elapsed footers")

    # E2: our CPUs derivation vs perf's own '# N CPUs utilized' comment
    diffs = []
    for w in glob(f"{rd}/group_core_w*.txt") + glob(f"{rd}/group_priv_w*.txt"):
        txt = open(w).read()
        m2 = ELAPSED_RE.search(txt)
        dur = float(m2.group(1)) if m2 else winsec
        g, _ = parse_group(w)
        for m in CPUS_CMT_RE.finditer(txt):
            cg, cpus_perf = m.group(1), float(m.group(2))
            tc = (g or {}).get(cg, {}).get("task-clock")
            if tc is not None:
                diffs.append(abs(tc / 1000.0 / dur - cpus_perf))
    if diffs:
        ok = median(diffs) < 0.02
        ev.append(f"E2 CPUs formula vs perf's own comment: median |diff| {median(diffs):.4f} CPUs over {len(diffs)} rows -> {'OK' if ok else 'MISMATCH'}")
        if not ok:
            fails.append("E2 CPUs formula disagrees with perf comment")

    # E3: kernel cgroup accounting (cpu.stat deltas) vs PMU task-clock, same window+scope
    try:
        rows = [l.split("\t") for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]]
        # task-clock lives in 'core' (legacy rotation) or 'priv' (2026-07-14 rotation)
        core_wins = [(r[0], r[1], float(r[2]), float(r[3])) for r in rows
                     if len(r) >= 4 and r[1] in ("core", "priv")]
    except OSError:
        core_wins = []
    scope_role = ["harness", "tool", "proxy"]      # CGS order in the chain script
    agree = []
    for i, role in enumerate(scope_role, start=1):
        f = f"{rd}/cpustat_scope{i}.tsv"
        if not os.path.exists(f):
            continue
        samples = []
        for ln in open(f):
            p = ln.split()
            if len(p) >= 3 and p[1] == "usage_usec":
                try:
                    samples.append((float(p[0]), float(p[2])))
                except ValueError:
                    pass
        if len(samples) < 10:
            continue
        ts = [s[0] for s in samples]
        def usage_at(t):
            j = min(max(bisect.bisect_left(ts, t), 0), len(samples) - 1)
            return samples[j][1]
        for wname, gname, t0, t1 in core_wins:
            if t1 <= ts[0] or t0 >= ts[-1] or t1 <= t0:
                continue
            cpus_stat = (usage_at(t1) - usage_at(t0)) / ((t1 - t0) * 1e6)
            g, _ = parse_group(f"{rd}/group_{gname}_w{wname}.txt")
            for cg, evs in (g or {}).items():
                if roles.get(cg) == role and evs.get("task-clock"):
                    agree.append(abs(cpus_stat - evs["task-clock"] / 1000.0 / (t1 - t0)))
    if agree:
        ok = median(agree) < 0.15
        # replays end abruptly at trajectory completion — the last windows straddle teardown
        # and with n of only a handful of window-scopes the median is a boundary artifact.
        # Replays are certified by the anchor section (dso/cost/wIPC); E3 stays informational
        # there and HARD for figure-source episodes.
        is_replay = "replay" in str(meta.get("workload", ""))
        tag3 = "OK" if ok else ("INFO (replay boundary, anchor-certified)" if is_replay else "MISMATCH")
        ev.append(f"E3 cpu.stat vs PMU task-clock (independent subsystems): median |dCPUs| {median(agree):.3f} over {len(agree)} window-scopes -> {tag3}")
        if not ok and not is_replay:
            fails.append("E3 cpu.stat vs PMU disagreement")
    else:
        ev.append("E3 cpu.stat vs PMU: no overlapping samples -> (no proof)")

    # E6: kernel-share cross-check — PMU cycles:k/(u+k) vs the scheduler's independent
    # user_usec/system_usec accounting, same scope, same priv window (pollers log all
    # three cpu.stat fields from 2026-07-08 22:0x on; older episodes skip gracefully)
    try:
        rows6 = [l.split("\t") for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]]
        priv_wins = [(r[0], float(r[2]), float(r[3])) for r in rows6 if len(r) >= 4 and r[1] == "priv"]
    except OSError:
        priv_wins = []
    agree6 = []
    for i, role in enumerate(("harness", "tool", "proxy"), start=1):
        f = f"{rd}/cpustat_scope{i}.tsv"
        if not os.path.exists(f) or not priv_wins:
            continue
        samp = []
        for ln in open(f):
            p = ln.split()
            if len(p) >= 7 and p[1] == "usage_usec" and p[5] == "system_usec":
                try:
                    samp.append((float(p[0]), float(p[4]), float(p[6])))
                except ValueError:
                    pass
        if len(samp) < 10:
            continue
        ts6 = [s[0] for s in samp]
        def at6(t):
            j = min(max(bisect.bisect_left(ts6, t), 0), len(samp) - 1)
            return samp[j]
        for wname, t0, t1 in priv_wins:
            if t1 <= ts6[0] or t0 >= ts6[-1]:
                continue
            _, u0, s0 = at6(t0); _, u1, s1 = at6(t1)
            du, ds = u1 - u0, s1 - s0
            if du + ds < 50000:      # <50ms of activity in the window: ratio unstable
                continue
            g6, _ = parse_group(f"{rd}/group_priv_w{wname}.txt")
            for cg, evs in (g6 or {}).items():
                if roles.get(cg) == role:
                    ck, cu = evs.get("cycles:k", 0), evs.get("cycles:u", 0)
                    if ck + cu > 0:
                        agree6.append(abs(ck / (ck + cu) - ds / (du + ds)))
    if agree6:
        # On shuffled-rotation (nohz_full) runs the scheduler's user/system SPLIT on tickless
        # cores is boundary-vtime accounted and diverges from the PMU by design (measured
        # 26pp on a fast-syscall loop, 2026-07-15) — the PMU is the figure source; keep the
        # cross-check informational there. Tick-accounted legacy runs keep the hard 5pp gate.
        ok = median(agree6) < 0.05
        soft6 = meta.get("rotation") == "shuffled"
        tag6 = "OK" if ok else ("INFO (nohz_full vtime split)" if soft6 else "MISMATCH")
        ev.append(f"E6 kernel-share PMU vs scheduler accounting: median |delta| {100*median(agree6):.1f}pp over {len(agree6)} window-scopes -> {tag6}")
        if not ok and not soft6:
            fails.append("E6 kernel-share cross-subsystem mismatch")

    # E9 (shuffled-rotation runs): heavy-burst census — every group's windows must have had a
    # fair chance at the heavy tool bursts (rare-burst coverage is THE statistical fragility
    # of windowed rotation; the continuous cpu.stat layer sees every burst = ground truth).
    try:
        rows9 = [l.split("\t") for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]]
        wins9 = [(r[1], float(r[2]), float(r[3])) for r in rows9 if len(r) >= 4]
    except OSError:
        wins9 = []
    if wins9 and meta.get("rotation") == "shuffled":
        s9 = []
        try:
            for ln in open(f"{rd}/cpustat_scope2.tsv"):
                p = ln.split()
                if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0:
                    s9.append((float(p[0]), float(p[2])))
        except OSError:
            pass
        hb, cur = [], None
        for (t0, u0), (t1, u1) in zip(s9, s9[1:]):
            r = (u1 - u0) / 1e6 / max(t1 - t0, 1e-9)
            if r > 0.3:
                if cur and t0 - cur[1] < 0.4: cur[1] = t1
                else:
                    if cur: hb.append(cur)
                    cur = [t0, t1]
        if cur: hb.append(cur)
        exposure = {}
        for grp, t0, t1 in wins9:
            n = sum(1 for b0, b1 in hb if b0 < t1 and b1 > t0)
            e = exposure.setdefault(grp, [0, 0]); e[0] += 1; e[1] += n
        # balls-in-bins: with B bursts over G groups, expected empty groups = G(1-1/G)^B —
        # an empty group is only SURPRISING (i.e. evidence of distribution failure) when that
        # expectation is small. G=8: B>=32 gives expected empties ~0.1. Below that, zero
        # exposure is the workload's burst sparsity, reported as CI (measured: scp-crawl
        # missed a DIFFERENT group on each of two rolls at B~<32).
        G9 = max(len(exposure), 1)
        surprise = len(hb) >= 4 * G9
        if exposure and surprise:
            zero = [g for g, (nw, nb) in sorted(exposure.items()) if nb == 0]
            ok = not zero
            ev.append(f"E9 heavy-burst census: {len(hb)} bursts; per-group exposure "
                      f"{ {g: e[1] for g, e in sorted(exposure.items())} } -> "
                      f"{'OK' if ok else 'GROUPS UNEXPOSED: ' + ','.join(zero)}")
            if not ok:
                fails.append(f"E9 groups saw no heavy burst: {','.join(zero)}")
        elif exposure:
            zero = [g for g, (nw, nb) in sorted(exposure.items()) if nb == 0]
            ev.append(f"E9 heavy-burst census: {len(hb)} bursts (<{4*G9} = sparse regime); "
                      f"unexposed groups {zero or 'none'} -> informational CI")

    # E10 (new runs): continuous whole-episode TMA census — zero multiplex, full coverage,
    # L1 sums to slots, L2 children nest under parents. CSV: time,count,,event,cgroup,run,pct
    tc = f"{rd}/tma_cont.csv"
    if os.path.exists(tc):
        S10, badpct, t_first, t_last = {}, 0, None, None
        for ln in open(tc):
            if ln.startswith("#") or not ln.strip():
                continue
            p = [x.strip() for x in ln.split(",")]
            if len(p) < 7:
                continue
            try:
                t = float(p[0])
            except ValueError:
                continue
            t_first = t if t_first is None else t_first; t_last = t
            if p[1].startswith("<"):
                continue                      # idle interval for this cgroup
            try:
                if float(p[6]) < 99.9:
                    badpct += 1
            except ValueError:
                pass
            role = roles.get(p[4])
            if role:
                d10 = S10.setdefault(role, {})
                d10[p[3]] = d10.get(p[3], 0.0) + float(p[1])
        msgs10, bad10 = [], False
        if wins9 and t_last:
            span_w = float(wins9[-1][2]) - float(wins9[0][1])
            cover = t_last / span_w if span_w > 0 else 0
            msgs10.append(f"coverage {100*min(cover,1):.0f}%")
            if cover < 0.90: bad10 = True
        if badpct:
            msgs10.append(f"{badpct} intervals <100% enabled"); bad10 = True
        for role, d10 in sorted(S10.items()):
            l1 = sum(d10.get(k, 0) for k in L1)
            if d10.get("slots") and l1:
                r10 = l1 / d10["slots"]
                if not 0.90 <= r10 <= 1.10:
                    msgs10.append(f"{role} L1/slots {r10:.2f}"); bad10 = True
                for c, par in TMA_NEST.items():
                    if d10.get(c, 0) > 1.05 * max(d10.get(par, 0), 1):
                        msgs10.append(f"{role} {c}>parent"); bad10 = True
        ev.append(f"E10 continuous TMA: {'; '.join(msgs10) if msgs10 else 'no intervals'} -> "
                  f"{'FAIL' if bad10 else 'OK'}")
        if bad10:
            fails.append("E10 continuous-TMA census invalid")

    # E11 (runs with the partition witness, 2026-07-17): unfenced-residual bound.
    # procstat_partition.tsv sees ALL activity on the measured CPUs (kernel threads
    # included); fence cpu.stat sums see only cgroup members. partition_busy - fences =
    # the async-kernel residual (writeback/irq) no cgroup owns. Informational unless the
    # residual exceeds 1% of partition capacity (then the "fences cover the partition"
    # premise of the magnitude figures needs a documented explanation).
    if os.path.exists(f"{rd}/procstat_partition.tsv"):
        try:
            meas = set()
            mspec = json.load(open(f"{rd}/metadata.json")).get("cpus_measured", "")
            for part in str(mspec).split(","):
                if not part: continue
                a_, _, b_ = part.partition("-")
                meas.update(range(int(a_), int(b_ or a_) + 1))
            first, last = {}, {}
            for ln in open(f"{rd}/procstat_partition.tsv"):
                pp = ln.split()
                if len(pp) < 9 or not pp[1].startswith("cpu"): continue
                cpu = int(pp[1][3:])
                if cpu not in meas: continue
                busy = sum(int(x) for x in pp[2:5]) + int(pp[7]) + int(pp[8])  # user+nice+sys+irq+softirq
                t = float(pp[0])
                if cpu not in first: first[cpu] = (t, busy)
                last[cpu] = (t, busy)
            part_busy = sum((last[c][1] - first[c][1]) for c in first) / 100.0  # jiffies -> s
            span = max((last[c][0] - first[c][0]) for c in first) if first else 0.0
            fence = 0.0
            for sc in (1, 2):   # measured-partition fences only (litellm lives on housekeeping)
                rows = [int(pp[2]) for pp in (l.split() for l in open(f"{rd}/cpustat_scope{sc}.tsv"))
                        if len(pp) >= 3 and pp[1] == "usage_usec" and int(pp[2]) >= 0]
                if len(rows) > 1: fence += (rows[-1] - rows[0]) / 1e6
            resid = part_busy - fence
            cap = span * len(first)
            pct = 100.0 * resid / cap if cap else 0.0
            ev.append(f"E11 partition witness: partition busy {part_busy:.1f}cs vs fences {fence:.1f}cs "
                      f"-> residual {resid:.1f}cs = {pct:.2f}% of capacity "
                      f"({'OK' if pct <= 1.0 else 'ABOVE 1% BOUND'})")
            if pct > 1.0:
                fails.append(f"E11 unfenced residual {pct:.2f}% of partition capacity (>1%)")
        except Exception as e:
            ev.append(f"E11 partition witness present but unreadable ({e})")

    # E4 (OC) — two modes:
    #   LINEAGE (lineage.tsv present, rung-2 watcher, accepted 2026-07-12): PID-set purity.
    #   Every record sample carries a pid; the lineage log says which pids are tool-class.
    #   Agent-fence samples from tool-class pids are pre-move residue -> RE-ATTRIBUTED, not
    #   contamination. Name-blind, so it works when spawned tools are node (comm-E4 was
    #   measured blind to that: 87% "pure" vs 15% true on ground truth).
    #   LEGACY (no lineage.tsv): comm-family test, kept verbatim for certified data.
    if meta.get("workload") == "oc" and os.path.exists(f"{rd}/lineage.tsv"):
        tool_pids, agent_pids = set(), set()
        for ln in open(f"{rd}/lineage.tsv"):
            p = ln.rstrip("\n").split("\t")
            if len(p) < 7 or p[1] in ("clockref", "event"): continue
            try: pid = int(p[2])
            except ValueError: continue
            if p[5] == "tool": tool_pids.add(pid)
            elif p[5] == "agent": agent_pids.add(pid)
        agent_pids -= tool_pids            # a pid that ever became tool counts as tool
        def pid_counts(f):
            n = {"agent": 0, "tool": 0, "unknown": 0}
            for ln in open(f):
                q = ln.split()
                if not q: continue
                try: pid = int(q[0])
                except ValueError: continue
                n["tool" if pid in tool_pids else "agent" if pid in agent_pids
                  else "unknown"] += 1
            return n
        f1, f2 = f"{rd}/scope1_pidtime.txt", f"{rd}/scope2_pidtime.txt"
        if os.path.exists(f1) and os.path.exists(f2):
            a, t = pid_counts(f1), pid_counts(f2)
            atot, ttot = max(sum(a.values()), 1), max(sum(t.values()), 1)
            raw = 100 * a["agent"] / atot
            corr_base = atot - a["tool"]   # tool-class samples re-attributed out
            corrected = 100 * a["agent"] / max(corr_base, 1)
            contam = 100 * t["agent"] / ttot
            unk = 100 * (a["unknown"] + t["unknown"]) / (atot + ttot)
            ev.append(f"E4 lineage /agent: purity raw {raw:.1f}% -> corrected {corrected:.1f}% "
                      f"({a['tool']} pre-move samples re-attributed to tool)")
            ev.append(f"E4 lineage /toolexec: agent-pid contamination {contam:.2f}%; "
                      f"unknown pids {unk:.2f}% -> "
                      f"{'OK' if corrected >= 99 and contam <= 0.5 and unk <= 1 else 'FAIL'}")
            if corrected < 99: fails.append(f"E4 lineage agent purity {corrected:.1f}% < 99%")
            if contam > 0.5: fails.append(f"E4 lineage toolexec contamination {contam:.2f}%")
            if unk > 1: fails.append(f"E4 lineage unknown-pid share {unk:.2f}%")
        else:
            ev.append("E4 lineage: lineage.tsv present but pidtime tables missing -> UNJUDGEABLE")
            fails.append("E4 lineage pidtime tables missing")
    # LEGACY comm-mode (certified data; also sanity backstop when no lineage log)
    elif meta.get("workload") == "oc":
        def is_agent_comm(c):
            return c == "node" or c == "bun" or c.startswith("openclaw")
        for i, side, want_agent in ((1, "/agent", True), (2, "/toolexec", False)):
            f = f"{rd}/scope{i}_comm.txt"
            if not os.path.exists(f):
                continue
            tot = fam = 0.0
            for ln in open(f):
                p = ln.split()
                if len(p) >= 2 and p[0].endswith("%"):
                    try:
                        pct = float(p[0].rstrip("%"))
                    except ValueError:
                        continue
                    tot += pct
                    if is_agent_comm(p[-1]):
                        fam += pct
            if tot > 0:
                share = 100 * fam / tot
                if want_agent:
                    # /agent: birth-leakage of exec-ing tool children is physical (linker
                    # startup outruns any poll) — QUANTIFY as a contamination bound, hard-fail
                    # only below the 50% sanity floor. Tool-side purity is the hard gate.
                    ev.append(f"E4 watcher {side}: agent-family = {share:.1f}% (contamination bound {100-share:.1f}% — attach to any OC agent-side claim)")
                    if share < 50:
                        fails.append(f"E4 watcher {side} below sanity floor ({share:.1f}%)")
                else:
                    ok = share <= 10
                    ev.append(f"E4 watcher {side}: agent-family comms = {share:.1f}% of samples -> {'OK' if ok else 'LEAKY'}")
                    if not ok:
                        fails.append(f"E4 watcher separation leaky on {side}")

    # E5 (SWE): the agent demonstrably worked
    if meta.get("workload") == "swe":
        steps = 0
        try:
            steps = open(f"{rd}/agent.log", errors="ignore").read().count("STEP ")
        except OSError:
            pass
        traj = bool(glob(f"{rd}/traj/**/*.traj", recursive=True))
        ok = steps >= 2
        ev.append(f"E5 work: {steps} STEP markers, trajectory {'present' if traj else 'absent'} -> {'OK' if ok else 'NO WORK'}")
        if not ok:
            fails.append("E5 no demonstrated agent work")

    # E7 (SWE): action-uniqueness — a degenerate greedy loop is not agent work. Added
    # 2026-07-11 after a temp-0 episode repeated one grep 554x (certified data has loops too).
    if meta.get("workload") == "swe":
        tfiles = glob(f"{rd}/traj/**/*.traj", recursive=True)
        if tfiles:
            try:
                acts = [(s.get("action") or "").strip()
                        for s in json.load(open(tfiles[0])).get("trajectory", [])]
                longest = cur = 1 if acts else 0
                for i in range(1, len(acts)):
                    cur = cur + 1 if acts[i] == acts[i - 1] else 1
                    longest = max(longest, cur)
                # two failure modes: (1) CONSECUTIVE loop (>=10 identical in a row, e.g.
                # temp-0 grep x554); (2) ALTERNATING/cyclic degeneracy — low overall
                # uniqueness with longest-run=1 (A-B-A-B...), which the consecutive test
                # MISSES. Added 2026-07-12 after the certified django-lite r1 audit: 5%
                # unique but longest-run 1, E7 rated it "clean". Floor: <40% unique over
                # >=20 steps = degenerate.
                uniq = (len(set(acts)) / len(acts)) if acts else 1.0
                consec_bad = longest >= 10
                cyclic_bad = len(acts) >= 20 and uniq < 0.40
                ok = not (consec_bad or cyclic_bad)
                ev.append(f"E7 action-uniqueness: {len(set(acts))}/{len(acts)} "
                          f"({100*uniq:.0f}%) unique, longest identical run {longest} -> "
                          f"{'OK' if ok else 'LOOP'}")
                if consec_bad:
                    fails.append(f"E7 degenerate consecutive loop (x{longest})")
                elif cyclic_bad:
                    fails.append(f"E7 degenerate cyclic loop ({100*uniq:.0f}% unique)")
            except (OSError, ValueError) as e:
                ev.append(f"E7 action-uniqueness: traj unreadable ({e}) -> UNJUDGEABLE")
                fails.append("E7 trajectory unreadable")
    return fails, ev


def behavior_of(rd):
    """Episode-level workload-behavior metrics for cross-run consistency."""
    b = {"steps": 0, "api_calls": 0, "dur_s": 0.0, "tool_cpu_s": 0.0, "outcome": "unknown"}
    try:
        log = open(f"{rd}/agent.log", errors="ignore").read()
        b["steps"] = log.count("STEP ")
        m = re.findall(r"total_api_calls=(\d+)", log)
        b["api_calls"] = int(m[-1]) if m else 0
        if "submitted" in log:
            b["outcome"] = "submitted"
        elif "overall_score" in log:
            b["outcome"] = "scored"
        else:
            b["outcome"] = "capped/died"
    except OSError:
        pass
    try:
        rows = [l.split("\t") for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]]
        if rows:
            b["dur_s"] = float(rows[-1][3]) - float(rows[0][2])
    except (OSError, IndexError, ValueError):
        pass
    try:
        s = [(float(p[0]), float(p[2])) for p in (l.split() for l in open(f"{rd}/cpustat_scope2.tsv"))
             if len(p) >= 3 and float(p[2]) >= 0]
        if len(s) > 1:
            b["tool_cpu_s"] = (s[-1][1] - s[0][1]) / 1e6
    except OSError:
        pass
    return b


def main():
    data, prefix = sys.argv[1], sys.argv[2]
    hard = 0
    configs = {}
    behaviors = {}
    rundirs = sorted(glob(f"{data}/{prefix}_*/run_*"))
    if not rundirs:
        print(f"no {prefix}_*/run_* dirs under {data}")
        sys.exit(1)
    for rd in rundirs:
        cfg = rd.split("/")[-2]
        fails, warns, stats = check_run(rd)
        efails, evidence = evidence_checks(rd)
        fails += efails
        configs.setdefault(cfg, []).append(stats)
        beh = behavior_of(rd)
        behaviors.setdefault(cfg, []).append((rd.split("/")[-1], beh))
        tag = "OK  " if not fails else "FAIL"
        if fails:
            hard += 1
        print(f"[{tag}] {rd}")
        for f in fails:
            print(f"       HARD: {f}")
        for w in warns[:6]:
            print(f"       warn: {w}")
        for e in evidence:
            print(f"       {e}")
        for role in ("harness", "tool", "proxy"):
            if role in stats:
                s = stats[role]
                extra = f"  kernel {s['kernel_pct']}%  cs/s {s['cs_per_s']}" if s.get("kernel_pct") is not None else ""
                print(f"       {role:8s} wIPC {s.get('wipc')}  medIPC {s['ipc']}  CPUs {s['cpus']}{extra}")
        print(f"       behavior: {beh['outcome']}, {beh['steps']} steps, {beh['api_calls']} calls, "
              f"{beh['dur_s']/60:.0f} min, tool {beh['tool_cpu_s']:.1f} CPU-s")
    print("\n== cross-run dispersion per role (layer 3) ==")
    for cfg, runs in sorted(configs.items()):
        for role in ("harness", "tool", "proxy"):
            # activity-weighted IPC (Σinstr/Σcycles) — median-of-windows is idle-diluted
            vals = [s[role]["wipc"] for s in runs if role in s and s[role].get("wipc")]
            if len(vals) >= 2 and median(vals) > 0:
                disp = (max(vals) - min(vals)) / median(vals)
                flag = "  <-- HIGH (consider +2 repeats)" if disp > 0.25 else ""
                print(f"  {cfg:28s} {role:8s} wIPC {[round(v, 2) for v in vals]} disp {disp:.0%}{flag}")
    # ---- live <-> replay determinism anchor (SWE) --------------------------------------------
    # Character (signatures) must reproduce strictly; parallel-payload CPU-COST carries intrinsic
    # runtime-scheduling variance (~8% observed: identical replays, cs/s differing 50%), and a
    # CommandTimeout on one side explains a volume gap (live burns capped work the replay skips).
    def dso_table(rd):
        out = {}
        try:
            for ln in open(f"{rd}/scope2_dso.txt"):
                p = ln.split()
                if len(p) >= 2 and p[0].endswith("%"):
                    out[p[-1]] = float(p[0].rstrip("%"))
        except OSError:
            pass
        return out

    def had_timeout(rd):
        try:
            return "CommandTimeout" in open(f"{rd}/agent.log", errors="ignore").read()
        except OSError:
            return False

    rep_hard = 0
    rep_cfgs = {c.replace(f"{prefix}_replay_swe_", "") for c in configs if c.startswith(f"{prefix}_replay_swe_")}
    if rep_cfgs:
        print("\n== live<->replay anchor (tool side) ==")
    for short in sorted(rep_cfgs):
        live_cfg, rep_cfg = f"{prefix}_swe_{short}", f"{prefix}_replay_swe_{short}"
        by_run = {}
        for rd in glob(f"{data}/{rep_cfg}/run_*") + glob(f"{data}/{live_cfg}/run_*"):
            _, _, st = check_run(rd)
            b = behavior_of(rd)
            try:
                src = json.load(open(f"{rd}/metadata.json")).get("extra", {}).get("source_run")
            except Exception:
                src = None
            by_run[(rd.split("/")[-2], rd.split("/")[-1])] = (st.get("tool", {}), b, src, rd)
        for n in ("run_1", "run_2", "run_3"):
            lv = by_run.get((live_cfg, n)); rp = by_run.get((rep_cfg, n))
            if not (lv and rp):
                continue
            (lt, lb, _, lrd), (rt, rb, _, rrd) = lv, rp
            msgs = []
            # character gate (continuous records — robust at any episode length): HARD
            la, ra = dso_table(lrd), dso_table(rrd)
            if la and ra:
                sim = sum(min(la.get(k, 0), ra.get(k, 0)) for k in set(la) | set(ra))
                msgs.append(f"dso-match {sim:.0f}%{'' if sim >= 80 else '!'}")
                if sim < 80: rep_hard += 1
            # cost gate: HARD unless timeout asymmetry explains it
            if lb["tool_cpu_s"] > 1 and rb["tool_cpu_s"] > 0:
                da = abs(lb["tool_cpu_s"] - rb["tool_cpu_s"])
                d = da / lb["tool_cpu_s"]
                tmo = had_timeout(lrd) != had_timeout(rrd)
                # materiality floor 8 CPU-s: replay counts ~2-4 CPU-s of sandbox-setup the live
                # poller's later start misses (measured constant offset across all pairs)
                bad = d > 0.10 and da > 8 and not tmo
                note = " timeout-asym" if (d > 0.10 and tmo) else ("!" if bad else "")
                msgs.append(f"cpu-s {lb['tool_cpu_s']:.0f}vs{rb['tool_cpu_s']:.0f} ({d:.0%}{note})")
                if bad: rep_hard += 1
            # sampled-mix + cycle rates: informational unless replay long enough to sample well
            solid = rb["dur_s"] >= 240
            for k in ("br_ki", "fp_ki", "wipc"):
                a, b2 = lt.get(k), rt.get(k)
                if a and b2:
                    d = abs(a - b2) / a
                    hardish = solid and k != "wipc" and d > 0.15
                    msgs.append(f"{k} {a}vs{b2} ({d:.0%}{'!' if hardish else ''})")
                    if hardish: rep_hard += 1
            print(f"  {short:14s} {n}: " + "  ".join(msgs))
        # same-traj noise set: signature precision is the gate; cost spread attributed via cs/s
        noise = [(v[0], v[1]) for k, v in by_run.items()
                 if k[0] == rep_cfg and (v[2] == 1 or k[1] == "run_1")]
        wipcs = [t.get("wipc") for t, _ in noise if t.get("wipc")]
        if len(wipcs) >= 2 and median(wipcs) > 0:
            nd = (max(wipcs) - min(wipcs)) / median(wipcs)
            cpus = [round(b["tool_cpu_s"]) for _, b in noise]
            css = [t.get("cs_per_s") for t, _ in noise]
            flag = "  <-- MEASUREMENT UNSTABLE" if nd > 0.10 else "  (signature precision)"
            print(f"  {short:14s} same-traj wIPC {[round(v,3) for v in wipcs]} disp {nd:.0%}{flag}")
            print(f"  {short:14s} same-traj cpu-s {cpus} cs/s {css}  (cost spread = runtime scheduling)")
            if nd > 0.10: rep_hard += 1
    if rep_cfgs:
        print(f"  anchor verdict: {'FAIL — investigate before trusting campaign numbers' if rep_hard else 'PASS — measures stable under pinned behavior'}")
        hard += rep_hard

    print("\n== workload-behavior consistency per config ==")
    for cfg, runs in sorted(behaviors.items()):
        if len(runs) < 2:
            continue
        flags = []
        outs = {b["outcome"] for _, b in runs}
        if len(outs) > 1:
            flags.append(f"MIXED OUTCOMES {sorted(outs)}")
        for key, thr in (("dur_s", 3), ("steps", 3), ("tool_cpu_s", 5)):
            vals = [b[key] for _, b in runs if b[key]]
            if len(vals) >= 2 and min(vals) > 0 and max(vals) / min(vals) > thr:
                flags.append(f"{key} spread {max(vals)/min(vals):.1f}x")
        line = "; ".join(f"{n}: {b['outcome']}/{b['steps']}st/{b['dur_s']/60:.0f}m/{b['tool_cpu_s']:.0f}cpu-s" for n, b in runs)
        mark = f"  <-- CHECK: {', '.join(flags)}" if flags else "  consistent"
        print(f"  {cfg:28s} {line}{mark}")
    sys.exit(1 if hard else 0)


if __name__ == "__main__":
    main()
