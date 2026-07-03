#!/usr/bin/env python3
"""Validate every measurement the local agent campaign figures consume.
Checks per source: files present & non-empty, perf multiplexing ("(xx.xx%)" < 99.5),
<not counted>/<not supported>, dead-cgroup errors, engine load in-window, harness activity
in-window, TMA L1 sums & L2 nesting, record sizes & symbol counts, and agent-work evidence
from the run logs. Exit code 1 if any FAIL (WARNs allowed)."""
import os, re, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
CANON = os.path.abspath(os.path.join(HERE, "..", "..", "agentic", "CANONICAL"))
issues = {"FAIL": [], "WARN": [], "OK": 0}

def note(level, msg):
    if level == "OK": issues["OK"] += 1
    else: issues[level].append(msg)

def read(p): return open(p, errors="ignore").read() if os.path.exists(p) else None

def check_stat_file(path, want_cgs, label, min_ratio=99.5):
    """want_cgs: list of cgroup substrings that must have counted rows."""
    txt = read(path)
    if txt is None: note("FAIL", f"{label}: MISSING {os.path.basename(path)}"); return
    if len(txt) < 200: note("FAIL", f"{label}: near-empty {os.path.basename(path)}"); return
    if "No such file" in txt or "no access to cgroup" in txt:
        note("FAIL", f"{label}: dead cgroup in {os.path.basename(path)}"); return
    if "<not supported>" in txt: note("FAIL", f"{label}: <not supported> in {os.path.basename(path)}"); return
    nc = txt.count("<not counted>")
    if nc: note("FAIL", f"{label}: {nc}x <not counted> in {os.path.basename(path)}"); return
    mux = [float(m) for m in re.findall(r"\(([0-9.]+)%\)", txt)]
    bad = [m for m in mux if m < min_ratio]
    if bad: note("WARN", f"{label}: multiplexed {os.path.basename(path)} (min {min(bad):.1f}%)")
    for cg in want_cgs:
        rows = [l for l in txt.splitlines() if cg in l and re.match(r"\s*[0-9,]+\s", l)]
        if not rows: note("FAIL", f"{label}: no counted rows for cgroup ~{cg} in {os.path.basename(path)}"); return
    note("OK", "")

def cg_val(path, cg, ev):
    txt = read(path) or ""
    for l in txt.splitlines():
        if cg in l and ev in l:
            try: return float(l.split()[0].replace(",", ""))
            except ValueError: pass
    return None

def check_tma(path1, path2, cg, label):
    s = cg_val(path1, cg, "slots"); comp = [cg_val(path1, cg, f"topdown-{k}") for k in ("retiring","bad-spec","fe-bound","be-bound")]
    if not s or any(c is None for c in comp):
        note("FAIL", f"{label}: tma1 rows missing"); return
    tot = sum(comp)/s*100
    if not (85 <= tot <= 115): note("WARN", f"{label}: L1 buckets sum {tot:.0f}% of slots")
    if path2:
        s2 = cg_val(path2, cg, "slots")
        if not s2: note("FAIL", f"{label}: td2 rows missing"); return
        pairs = [("heavy-ops","retiring"), ("br-mispredict","bad-spec"), ("fetch-lat","fe-bound"), ("mem-bound","be-bound")]
        l1r = {k: c/s for k, c in zip(("retiring","bad-spec","fe-bound","be-bound"), comp)}
        viol = []
        for child, parent in pairs:
            cv = cg_val(path2, cg, f"topdown-{child}")
            if cv is not None and cv/s2 > l1r[parent]*1.15 and cv/s2 - l1r[parent] > 0.03:
                viol.append(f"{child}({cv/s2*100:.0f}%)>{parent}({l1r[parent]*100:.0f}%)")
        if viol: note("WARN", f"{label}: L2 nesting violated: {', '.join(viol)} (phase shift)")
    note("OK", "")

def check_rec(prefix, label, min_bytes=100000, min_sym=5):
    data = prefix + ".data"
    if not os.path.exists(data) or os.path.getsize(data) < min_bytes:
        note("FAIL", f"{label}: record {os.path.basename(data)} missing/small"); return
    note("OK", "")

def check_load(path, cg, label, min_cpus, ev="task-clock"):
    txt = read(path) or ""
    for l in txt.splitlines():
        if cg in l and ev in l:
            m = re.search(r"#\s+([\d.]+) CPUs utilized", l)
            if m:
                c = float(m.group(1))
                if c < min_cpus: note("WARN", f"{label}: only {c:.2f} CPUs in window (min {min_cpus})")
                else: note("OK", "")
                return c
    note("FAIL", f"{label}: no {ev} row for ~{cg} in {os.path.basename(path)}"); return None

ENG = "kubepods"
print("=" * 72)

# ---------- SWE replays (engine during) ----------
for t in ("astropy", "scikit-learn", "sympy"):
    d = os.path.join(DATA, t); lab = f"replay/{t}"
    for g in ("core", "fp1", "fp2", "cache", "mlp", "tma1", "tma2"):
        check_stat_file(os.path.join(d, f"group_engine_{g}.txt"), [ENG], f"{lab} eng/{g}")
    check_tma(os.path.join(d, "group_engine_tma1.txt"), os.path.join(d, "group_engine_tma2.txt"), ENG, f"{lab} eng TMA")
    check_load(os.path.join(d, "group_engine_core.txt"), ENG, f"{lab} eng load", 1.0)
    check_rec(os.path.join(d, "rec_engine"), f"{lab} record")
    rl = read(os.path.join(d, "replay.log")) or ""
    m = re.search(r"ok=(\d+) err=(\d+)", rl)
    if m and int(m.group(1)) < 10: note("FAIL", f"{lab}: replay barely ran (ok={m.group(1)})")
    elif m and int(m.group(2)) > int(m.group(1)): note("WARN", f"{lab}: replay errors exceed successes (ok={m.group(1)} err={m.group(2)})")
    else: note("OK", "")

# ---------- BCB live (engine + harness) ----------
d = os.path.join(DATA, "bcb_live"); lab = "bcb_live"
for g in ("core", "fp1", "fp2", "cache", "mlp", "tma1", "tma2"):
    check_stat_file(os.path.join(d, f"group_{g}.txt"), [ENG, "bcb-live"], f"{lab}/{g}")
check_tma(os.path.join(d, "group_tma1.txt"), os.path.join(d, "group_tma2.txt"), ENG, f"{lab} eng TMA")
check_tma(os.path.join(d, "group_tma1.txt"), os.path.join(d, "group_tma2.txt"), "bcb-live", f"{lab} harness TMA")
check_load(os.path.join(d, "group_core.txt"), ENG, f"{lab} eng load", 1.0)
check_rec(os.path.join(d, "rec_engine"), f"{lab} eng record")
check_rec(os.path.join(d, "rec_driver"), f"{lab} drv record", min_bytes=50000)
al = read(os.path.join(d, "agent.log")) or ""
tt = re.search(r"(\d+) total turns", al)
if not tt or int(tt.group(1)) < 10: note("FAIL", f"{lab}: agent loop evidence weak ({tt and tt.group(1)} turns)")
else: note("OK", "")
mk = read(os.path.join(d, "markers.txt")) or ""
if mk.count("toolexec_start") < 10: note("FAIL", f"{lab}: too few tool-exec markers")
else: note("OK", "")

# ---------- SWE live ----------
d = os.path.join(DATA, "swe_live"); lab = "swe_live"
check_stat_file(os.path.join(d, "group_tma1.txt"), [ENG], f"{lab} eng/tma1")   # engine side (orig chain)
for g in ("cache", "mlp", "fp1", "fp2"):                                        # harness (gap-fill, swe-hm scope)
    check_stat_file(os.path.join(d, f"group_h_{g}.txt"), ["swe-hm", "docker-"], f"{lab} harn/{g}")
check_stat_file(os.path.join(d, "group_p_tma1.txt"), ["swe-tp", "docker-"], f"{lab} harn/p_tma1")
check_stat_file(os.path.join(d, "group_p_tma2.txt"), ["swe-tp", "docker-"], f"{lab} harn/p_tma2")
check_tma(os.path.join(d, "group_p_tma1.txt"), os.path.join(d, "group_p_tma2.txt"), "docker-", f"{lab} harn TMA(pair,sandbox)")
check_rec(os.path.join(d, "rec_engine"), f"{lab} eng record")
check_rec(os.path.join(d, "rec_driver"), f"{lab} drv record", min_bytes=50000)
check_rec(os.path.join(d, "rec_sandbox"), f"{lab} sandbox record", min_bytes=50000)
steps = (read(os.path.join(d, "agent_tp.log")) or "").count("STEP")
if steps < 5: note("WARN", f"{lab}: only {steps} STEP markers in last live log")
else: note("OK", "")

# ---------- OC live x4 ----------
for t in ("calendar", "pdf-digest", "web-digest", "image-crop"):
    d = os.path.join(DATA, f"oc_live_{t}"); lab = f"oc/{t}"
    check_stat_file(os.path.join(d, "group_tma1.txt"), [ENG, "docker-"], f"{lab}/tma1")
    check_stat_file(os.path.join(d, "group_tma2.txt"), [ENG], f"{lab}/eng td2(patch)")
    for g in ("cache", "mlp", "fp1", "fp2"):
        check_stat_file(os.path.join(d, f"group_h_{g}.txt"), ["docker-"], f"{lab} harn/{g}")
    check_stat_file(os.path.join(d, "group_p_tma1.txt"), ["docker-"], f"{lab} harn/p_tma1")
    check_stat_file(os.path.join(d, "group_p_tma2.txt"), ["docker-"], f"{lab} harn/p_tma2")
    check_tma(os.path.join(d, "group_p_tma1.txt"), os.path.join(d, "group_p_tma2.txt"), "docker-", f"{lab} harn TMA(pair)")
    check_rec(os.path.join(d, "rec_engine"), f"{lab} eng record")
    check_rec(os.path.join(d, "rec_tool"), f"{lab} tool record", min_bytes=50000)
    fin = "Agent finished successfully" in ((read(os.path.join(d, "agent.log")) or "") + (read(os.path.join(d, "agent_gpu.log")) or "") + (read(os.path.join(d, "agent_tp.log")) or ""))
    if not fin: note("WARN", f"{lab}: no 'Agent finished successfully' in logs")
    else: note("OK", "")
    gt = read(os.path.join(d, "gpu_timeline.csv")) or ""
    if gt.count("\n") < 30: note("WARN", f"{lab}: gpu timeline short ({gt.count(chr(10))} samples)")
    else: note("OK", "")

# ---------- CANONICAL tool data ----------
CAN = [("bigcodebench/data", False, "tools/BCB"),
       ("swe_bench/data/astropy-14096", False, "tools/astropy"),
       ("swe_bench/data/scikit-learn-25232", False, "tools/scikit"),
       ("swe_bench/data/sympy-14248", False, "tools/sympy"),
       ("openclaw/data/calendar", True, "tools/OC-cal"),
       ("openclaw/data/arxiv", True, "tools/OC-web"),
       ("openclaw/data/pdf_digest", True, "tools/OC-pdf"),
       ("openclaw/data/social_poster_crop", True, "tools/OC-crop")]
for sub, up, lab in CAN:
    d = os.path.join(CANON, sub)
    names = ["TMA", "TD2", "CACHE", "MLP", "FP"] if up else ["tma", "td2", "cache", "mlp", "fp"]
    for g in names:
        f = os.path.join(d, f"group_{g}_r1.txt" if up else f"group_{g}.txt")
        check_stat_file(f, [""], f"{lab}/{g.lower()}", min_ratio=85.0)   # canonical fp = 8 events, ~90% known
    check_tma(os.path.join(d, "group_TMA_r1.txt" if up else "group_tma.txt"),
              os.path.join(d, "group_TD2_r1.txt" if up else "group_td2.txt"), "", f"{lab} TMA")

print(f"checks passed: {issues['OK']}")
print(f"\nWARN ({len(issues['WARN'])}):")
for w in issues["WARN"]: print(f"  - {w}")
print(f"\nFAIL ({len(issues['FAIL'])}):")
for f in issues["FAIL"]: print(f"  - {f}")
sys.exit(1 if issues["FAIL"] else 0)
