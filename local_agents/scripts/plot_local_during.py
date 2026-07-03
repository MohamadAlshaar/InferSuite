#!/usr/bin/env python3
"""LOCAL self-served agentic campaign figures (SYSTEM python3): what the CPU does DURING
inference (k3s vLLM 7B engine) vs OUTSIDE (agent loop / tool container), per agent workload.
Sources: local_agents/data/{astropy,scikit-learn,sympy} (SWE trajectory replays, engine side),
bcb_live (Coder-7B live loop, engine+driver cgroups), oc_live_* (Instruct-7B live OpenClaw,
engine+task-container cgroups). Same palettes as the service/thesis TMA and attribution figures."""
import os, re, glob, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6, "axes.axisbelow": True,
})
INSIDE, OUTSIDECOL = "#6a51a3", "#1b9e77"
L1 = [("retiring","Retiring","#009E73"),("fe","Frontend-bound","#0072B2"),
      ("bad","Bad speculation","#D55E00"),("be","Backend-bound","#E69F00")]
TMA2 = [("light_ops","Light operations","#66c2a5"),("heavy_ops","Heavy operations","#1b9e77"),
        ("fetch_lat","Fetch latency","#8da0cb"),("fetch_bw","Fetch bandwidth","#c6dbef"),
        ("br_mispred","Branch mispredict","#fc8d62"),("machine_clears","Machine clears","#fdd0a2"),
        ("mem_bound","Memory bound","#e78ac3"),("core_bound","Core bound","#f4cae4")]
def txtcol(hexcol):
    r, g, b = (int(hexcol[i:i+2], 16) for i in (1, 3, 5))
    return "white" if 0.299*r + 0.587*g + 0.114*b < 150 else "#333333"

EV = ("slots","topdown-retiring","topdown-bad-spec","topdown-fe-bound","topdown-be-bound",
      "topdown-heavy-ops","topdown-br-mispredict","topdown-fetch-lat","topdown-mem-bound")

def parse_cg(path, cg_sub):
    """counter values from a --for-each-cgroup stat file, restricted to one cgroup."""
    d = {}
    if not os.path.exists(path): return d
    for ln in open(path, errors="ignore"):
        if cg_sub not in ln: continue
        parts = ln.split()
        if len(parts) < 2: continue
        try: v = float(parts[0].replace(",", ""))
        except ValueError: continue
        for name in EV:
            if name in ln and name not in d: d[name] = v; break
        m = re.search(r"([\d,.]+)\s+msec task-clock", ln)
        if m and "task_clock_ms" not in d: d["task_clock_ms"] = float(m.group(1).replace(",", ""))
        m = re.search(r"#\s+([\d.]+) CPUs utilized", ln)
        if m and "cpus" not in d: d["cpus"] = float(m.group(1))
    return d

def tma_of(t1, t2=None):
    s1 = sum(t1.get(k, 0) for k in EV[1:5])
    if not s1: return None
    l1 = {"retiring": t1["topdown-retiring"]/s1*100, "bad": t1["topdown-bad-spec"]/s1*100,
          "fe": t1["topdown-fe-bound"]/s1*100, "be": t1["topdown-be-bound"]/s1*100}
    l2 = None
    if t2 and t2.get("slots"):
        s2 = t2["slots"]
        m = {k: t2.get(f"topdown-{n}", 0)/s2*100 for k, n in
             [("heavy_ops","heavy-ops"),("br_mispred","br-mispredict"),("fetch_lat","fetch-lat"),("mem_bound","mem-bound")]}
        l2 = {"heavy_ops": m["heavy_ops"], "light_ops": max(l1["retiring"]-m["heavy_ops"],0),
              "br_mispred": m["br_mispred"], "machine_clears": max(l1["bad"]-m["br_mispred"],0),
              "fetch_lat": m["fetch_lat"], "fetch_bw": max(l1["fe"]-m["fetch_lat"],0),
              "mem_bound": m["mem_bound"], "core_bound": max(l1["be"]-m["mem_bound"],0)}
    return l1, l2

ENG = "kubepods"
# (dir, label, engine-file-prefix, engine cgroup, tool cgroup substring(s) or None, l2_in_window)
WORKLOADS = [
    ("astropy",         "astropy",       "group_engine_", ENG, None,            True),
    ("scikit-learn",    "scikit-learn",  "group_engine_", ENG, None,            True),
    ("sympy",           "sympy",         "group_engine_", ENG, None,            True),
    ("swe_live",        "SWE-agent",     "group_",        ENG, ["swe-live", "docker-"], True),
    ("bcb_live",        "BigCodeBench",  "group_",        ENG, "bcb-live",      True),
    ("oc_live_calendar",   "OC calendar",   "group_",        ENG, "docker-",       True),
    ("oc_live_pdf-digest", "OC pdf-digest", "group_",        ENG, "docker-",       True),
    ("oc_live_web-digest", "OC web-digest", "group_",        ENG, "docker-",       True),
    ("oc_live_image-crop", "OC image-crop", "group_",        ENG, "docker-",       True),
]

rows = []
for d, lab, pref, ecg, tcg, l2ok in WORKLOADS:
    base = os.path.join(DATA, d)
    t1 = parse_cg(os.path.join(base, f"{pref}tma1.txt"), ecg)
    t2 = parse_cg(os.path.join(base, f"{pref}tma2.txt"), ecg) if l2ok else None
    core_e = parse_cg(os.path.join(base, f"{pref}core.txt"), ecg)
    tcgs = tcg if isinstance(tcg, list) else ([tcg] if tcg else [])
    tool_cpus = [parse_cg(os.path.join(base, "group_core.txt"), c).get("cpus") for c in tcgs]
    tool_cpus = sum(v for v in tool_cpus if v is not None) if any(v is not None for v in tool_cpus) else None
    r = tma_of(t1, t2)
    if not r: print(f"skip {d} (no in-window tma1)"); continue
    rows.append((d, lab, r[0], r[1], core_e.get("cpus"), tool_cpus))

# ---- Figs 1+2: TMA during vs outside inference in one chart ----
# Engine is agent-invariant -> ONE purple bar (mean over all workloads). The outside bars carry
# the per-workload identity: agent-side cgroups (harness + tools) from the same windows.
def sum_cgs(path, cgs):
    tot = {}
    for c in cgs:
        for k, v in parse_cg(path, c).items():
            if k in ("task_clock_ms", "cpus"): continue
            tot[k] = tot.get(k, 0) + v
    return tot

SHORT_TMA = {"swe_live": "SWE-agent", "bcb_live": "BigCodeBench", "oc_live_calendar": "OC calendar",
             "oc_live_pdf-digest": "OC pdf-digest", "oc_live_web-digest": "OC web-digest",
             "oc_live_image-crop": "OC image-crop"}
LIVE_T = {"swe_live": ["swe-live", "docker-"], "bcb_live": ["bcb-live"],
          "oc_live_calendar": ["docker-"], "oc_live_pdf-digest": ["docker-"],
          "oc_live_web-digest": ["docker-"], "oc_live_image-crop": ["docker-"]}
# agent-side L2: tma1+td2 must sample the same phase. BCB's stationary loop is consistent by
# construction; SWE/OC use the PAIRED back-to-back windows (group_p_*). Entries whose L2
# components still exceed their L1 parents (sum > 105) are dropped.
outside = []
for d, cgs in LIVE_T.items():
    base = os.path.join(DATA, d)
    if d == "bcb_live":
        t1 = sum_cgs(os.path.join(base, "group_tma1.txt"), cgs)
        t2 = sum_cgs(os.path.join(base, "group_tma2.txt"), cgs)
    else:
        cgs_p = ["swe-tp", "docker-"] if d == "swe_live" else cgs
        t1 = sum_cgs(os.path.join(base, "group_p_tma1.txt"), cgs_p)
        t2 = sum_cgs(os.path.join(base, "group_p_tma2.txt"), cgs_p)
        if not t1.get("slots"):  # pair capture missing -> L1 from the original window, no L2
            t1, t2 = sum_cgs(os.path.join(base, "group_tma1.txt"), cgs), None
    r = tma_of(t1, t2)
    if not r: continue
    l1c, l2c = r
    if l2c and sum(l2c.values()) > 105: l2c = None
    outside.append((SHORT_TMA[d], l1c, l2c))

# tool-execution bars from the CANONICAL captures
CANON_TMA = [(os.path.join("bigcodebench", "data"), False, "BCB tests"),
             (os.path.join("swe_bench", "data", "astropy-14096"), False, "astropy"),
             (os.path.join("swe_bench", "data", "scikit-learn-25232"), False, "scikit-learn"),
             (os.path.join("swe_bench", "data", "sympy-14248"), False, "sympy"),
             (os.path.join("openclaw", "data", "calendar"), True, "OC calendar"),
             (os.path.join("openclaw", "data", "arxiv"), True, "OC web-digest"),
             (os.path.join("openclaw", "data", "pdf_digest"), True, "OC pdf-digest"),
             (os.path.join("openclaw", "data", "social_poster_crop"), True, "OC image-crop")]
CANON_ROOT = os.path.join(HERE, "..", "..", "agentic", "CANONICAL")
tools_tma = []
for sub, up, lab in CANON_TMA:
    d = os.path.join(CANON_ROOT, sub)
    n = (lambda g: os.path.join(d, f"group_{g.upper()}_r1.txt")) if up else (lambda g: os.path.join(d, f"group_{g}.txt"))
    t1 = parse_cg(n("tma"), ""); t2 = parse_cg(n("td2"), "")
    r = tma_of(t1, t2)
    if not r: continue
    l1c, l2c = r
    if l2c and sum(l2c.values()) > 105: l2c = None
    tools_tma.append((lab, l1c, l2c))

def tma_chart(entries, comps, fname, title, wide):
    X = np.arange(len(entries))
    fig, ax = plt.subplots(figsize=(0.62+wide*len(entries), 4.8))
    bot = np.zeros(len(entries))
    for key, lab, col in comps:
        v = np.array([e[1][key] for e in entries])
        ax.bar(X, v, bottom=bot, label=lab, color=col, width=0.62, edgecolor="white", linewidth=0.6)
        for x, (b, vv) in enumerate(zip(bot, v)):
            if vv >= 8: ax.text(x, b+vv/2, f"{vv:.0f}", ha="center", va="center",
                                color=txtcol(col), fontsize=8, fontweight="bold")
        bot += v
    ax.set_xticks(X); ax.set_xticklabels([e[0] for e in entries], fontsize=8.6, rotation=26, ha="right")
    TICKCOL = {"in": INSIDE, "out": OUTSIDECOL, "tool": "#00441b"}
    for t, e in zip(ax.get_xticklabels(), entries): t.set_color(TICKCOL[e[2]])
    prev = None
    for i, e in enumerate(entries):
        if prev and e[2] != prev: ax.axvline(i-0.5, color="#666666", linewidth=1.0, linestyle=(0, (4, 3)))
        prev = e[2]
    ax.set_ylabel("Pipeline slots (%)"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
    ax.set_title(title, fontsize=13, pad=40)
    fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

eng_l1 = {k: float(np.mean([r[2][k] for r in rows])) for k in ("retiring", "bad", "fe", "be")}
eng_l2 = {k: float(np.mean([r[3][k] for r in rows if r[3]]))
          for k in ("heavy_ops", "light_ops", "br_mispred", "machine_clears",
                    "fetch_lat", "fetch_bw", "mem_bound", "core_bound")}
l1_entries = [("vLLM engine\n(during inf.)", eng_l1, "in")] + \
             [(n, l1c, "out") for n, l1c, _ in outside] + \
             [(n, l1c, "tool") for n, l1c, _ in tools_tma]
tma_chart(l1_entries, L1, "local_agents_tma_l1.png",
          "TMA Level 1: engine (during inference), agent harness, tool executions", 1.0)
l2_entries = [("vLLM engine\n(during inf.)", eng_l2, "in")] + \
             [(n, l2c, "out") for n, _l1, l2c in outside if l2c] + \
             [(n, l2c, "tool") for n, _l1, l2c in tools_tma if l2c]
tma_chart(l2_entries, TMA2, "local_agents_tma_l2.png",
          "TMA Level 2: engine (during inference), agent harness, tool executions", 1.15)

# ---- Fig 3: two-view CPU share donuts (live loops only), same style as the timing donuts ----
SHORT = {"swe_live": "SWE-agent", "bcb_live": "BigCodeBench", "oc_live_calendar": "OC calendar",
         "oc_live_pdf-digest": "OC pdf-digest", "oc_live_web-digest": "OC web-digest",
         "oc_live_image-crop": "OC image-crop"}
live = [r for r in rows if r[5] is not None]
if live:
    nc = (len(live) + 1) // 2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.0))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(live):]: ax.axis("off")
    for ax, r in zip(axes, live):
        eng, tool = r[4], r[5]
        shares = [eng/(eng+tool)*100, tool/(eng+tool)*100]
        ax.pie([max(s, 1.0) for s in shares], colors=[INSIDE, OUTSIDECOL], startangle=90,
               counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.13, f"{eng:.2f} CPUs", ha="center", fontsize=10, color=INSIDE)
        ax.text(0, -0.21, f"{tool:.2f} CPUs" if tool >= 0.005 else f"{tool:.3f} CPUs",
                ha="center", fontsize=10, color=OUTSIDECOL)
        ax.text(0, -1.3, SHORT.get(r[0], r[0]), ha="center", fontweight="bold", fontsize=10.5)
        ax.axis("off")
    fig.legend(handles=[Patch(color=INSIDE, label="During inference (vLLM engine)"),
                        Patch(color=OUTSIDECOL, label="Outside inference (agent + tools)")],
               loc="lower center", ncol=2, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUT, "local_agents_two_view_cpu.png")); plt.close(fig)

# ---- Fig 5: GPU vs CPU wall time per live loop (nvidia-smi timeline, guard -> exit) ----
GPU_COL, CPU_COL = "#6a51a3", "#1b9e77"
def gpu_split(path, thresh=10):
    if not os.path.exists(path): return None
    utils = []
    for ln in open(path):
        parts = ln.strip().split(",")
        if len(parts) != 2 or parts[0] == "guard": continue
        try: utils.append(float(parts[1]))
        except ValueError: pass
    if len(utils) < 20: return None
    busy = sum(1 for u in utils if u >= thresh)
    return busy/len(utils)*100

gpu_rows = []
for d in SHORT:
    g = gpu_split(os.path.join(DATA, d, "gpu_timeline.csv"))
    if g is not None: gpu_rows.append((SHORT[d], g))
if gpu_rows:
    nc = (len(gpu_rows) + 1) // 2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.0))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(gpu_rows):]: ax.axis("off")
    for ax, (lab, g) in zip(axes, gpu_rows):
        ax.pie([g, 100-g], colors=[GPU_COL, CPU_COL], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.13, f"GPU {g:.0f}%", ha="center", fontsize=10, color=GPU_COL)
        ax.text(0, -0.21, f"CPU {100-g:.0f}%", ha="center", fontsize=10, color=CPU_COL)
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
        ax.axis("off")
    fig.legend(handles=[Patch(color=GPU_COL, label="GPU generating"),
                        Patch(color=CPU_COL, label="CPU only (tools, agent, orchestration)")],
               loc="lower center", ncol=2, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUT, "local_agents_gpu_time.png")); plt.close(fig)

# ---- Fig 4: software view, engine vs tool per live agent (DSO roles) ----
ROLES = [
    ("GPU busy-wait",          "#6a51a3", re.compile(r"libcuda|libcudart", re.I)),
    ("Node.js / V8 (agent)",   "#56b4e9", re.compile(r"\bnode\b|libnode|/node$|\bv8\b|\[JIT\]", re.I)),
    ("Python interpreter",     "#d94801", re.compile(r"python3|libpython|\.cpython-", re.I)),
    ("BLAS / OpenMP",          "#238b45", re.compile(r"openblas|libgomp|libtorch|libmkl", re.I)),
    ("C library / loader",     "#2171b5", re.compile(r"libc\.so|ld-linux|libm\.so|libstdc|libz\.|libcrypto|libssl", re.I)),
    ("OS kernel",              "#cb181d", re.compile(r"kallsyms|\[kernel|\[vdso\]", re.I)),
]
NEUTRAL = "#b3b3b3"
def parse_dso(path):
    out = {r[0]: 0.0 for r in ROLES}; out["other"] = 0.0
    if not os.path.exists(path): return None
    for ln in open(path, errors="ignore"):
        m = re.match(r"\s*[0-9.]+%\s+([0-9.]+)%\s+(\S+)", ln) or re.match(r"\s*([0-9.]+)%\s+(\S+)", ln)
        if not m: continue
        pct, dso = float(m.group(1)), m.group(2)
        if pct <= 0: continue
        for name, col, rx in ROLES:
            if rx.search(dso): out[name] += pct; break
        else: out["other"] += pct
    tot = sum(out.values())
    return {k: v/tot*100 for k, v in out.items()} if tot else None

panels = []
for d, lab in SHORT.items():
    base = os.path.join(DATA, d)
    e = parse_dso(os.path.join(base, "engine_dso.txt"))
    drv = parse_dso(os.path.join(base, "driver_dso.txt")); tl = parse_dso(os.path.join(base, "tool_dso.txt"))
    if drv and tl:  # sweagent scope + sandbox: weight by their measured CPU load
        w = [parse_cg(os.path.join(base, "group_core.txt"), c).get("cpus", 0) or 0 for c in ("swe-live", "docker-")]
        tot = (w[0] + w[1]) or 1
        t = {k: (drv.get(k, 0)*w[0] + tl.get(k, 0)*w[1])/tot for k in set(drv) | set(tl)}
    else:
        t = drv or tl
    if e and t: panels.append((lab, e, t))
if panels:
    colmap = {n: c for n, c, _ in ROLES}; colmap["other"] = NEUTRAL
    def draw_donut(ax, r, center, ccol, minlab=12):
        parts = [(k, v) for k, v in sorted(r.items(), key=lambda x: -x[1]) if v > 1]
        w, _ = ax.pie([max(v, 1.2) for _, v in parts], colors=[colmap[k] for k, _ in parts],
                      startangle=90, counterclock=False,
                      wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
        for wd, (k, v) in zip(w, parts):
            if v >= minlab:
                a = math.radians((wd.theta1+wd.theta2)/2)
                ax.text(0.79*math.cos(a), 0.79*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=9)
        ax.text(0, 0, center, ha="center", va="center", fontsize=10, fontweight="bold", color=ccol)
        ax.axis("off")
        return {k for k, _ in parts}

    # engine software view is agent-invariant: ONE donut, mean profile over the live loops
    eng_mean = {}
    for _, e, _t in panels:
        for k, v in e.items(): eng_mean[k] = eng_mean.get(k, 0) + v/len(panels)
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    used = draw_donut(ax, eng_mean, "vLLM engine", INSIDE)
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model side: engine CPU during inference", fontsize=12.5, y=1.02)
    fig.savefig(os.path.join(OUT, "local_agents_engine_software.png")); plt.close(fig)

    # outside: one donut per live agent
    nc = (len(panels) + 1) // 2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.4))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(panels):]: ax.axis("off")
    used = set()
    for ax, (lab, _e, t) in zip(axes, panels):
        used |= draw_donut(ax, t, "", OUTSIDECOL)
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Agent machinery: CPU of the harness and its tools during the live loops", fontsize=13, y=1.0)
    fig.savefig(os.path.join(OUT, "local_agents_two_view_software.png")); plt.close(fig)

# ---- Fig 6: engine microarch signature during inference (full-suite workloads only:
# the SWE replays ran under continuous load and BCB's loop outlived all 7 groups; the OC
# episodes died before the cache/fp/mlp windows, so their derived metrics would be idle noise) ----
import collections
def elapsed_of(path):
    if not os.path.exists(path): return 0.0
    for ln in open(path, errors="ignore"):
        m = re.search(r"([0-9.]+)\s+seconds time elapsed", ln)
        if m: return float(m.group(1))
    return 0.0

FPEV = ["fp_arith_inst_retired.scalar_single","fp_arith_inst_retired.scalar_double",
        "fp_arith_inst_retired.128b_packed_single","fp_arith_inst_retired.128b_packed_double",
        "fp_arith_inst_retired.256b_packed_single","fp_arith_inst_retired.256b_packed_double",
        "fp_arith_inst_retired.512b_packed_single","fp_arith_inst_retired.512b_packed_double"]
def parse_all(path, cg):
    d = {}
    if not os.path.exists(path): return d
    for ln in open(path, errors="ignore"):
        if cg not in ln: continue
        parts = ln.split()
        if len(parts) < 2: continue
        try: v = float(parts[0].replace(",", ""))
        except ValueError: continue
        for name in ("cycles","instructions","branches","branch-misses",
                     "mem_load_retired.l1_hit","mem_load_retired.l2_hit","mem_load_retired.l3_hit",
                     "mem_load_retired.l3_miss","l1d_pend_miss.pending","l1d_pend_miss.pending_cycles",
                     "uops_executed.thread", *FPEV):
            if name in ln and name not in d: d[name] = v; break
    return d

ELEM = {"scalar":1, "128b":2, "256b":4, "512b":8}
def sig_of(base, pref, cg):
    g = {grp: parse_all(os.path.join(base, f"{pref}{grp}.txt"), cg) for grp in ("core","fp1","fp2","cache","mlp")}
    if not g["core"].get("cycles"): return None
    ins = g["cache"].get("instructions", 1)
    l1 = g["cache"].get("mem_load_retired.l1_hit", 0); l2 = g["cache"].get("mem_load_retired.l2_hit", 0)
    l3 = g["cache"].get("mem_load_retired.l3_hit", 0); mm = g["cache"].get("mem_load_retired.l3_miss", 0)
    tot = (l1+l2+l3+mm) or 1
    fp = collections.Counter()
    for src in (g["fp1"], g["fp2"]):
        for k, v in src.items():
            if k.startswith("fp_arith_inst_retired."): fp[k] += v
    flops = scalar = packed = 0.0
    for k, v in fp.items():
        w = next((e for t, e in ELEM.items() if t in k), 1)
        w *= 2  # FMA
        flops += v*w
        if "scalar" in k: scalar += v
        else: packed += v
    secs = elapsed_of(os.path.join(base, f"{pref}fp1.txt")) + elapsed_of(os.path.join(base, f"{pref}fp2.txt"))
    return {"IPC": g["core"]["instructions"]/g["core"]["cycles"],
            "L1": l1/tot*100, "L2": l2/tot*100, "L3": l3/tot*100, "MPKI": mm/(ins/1000),
            "AMAT": (l1*5 + l2*15 + l3*50 + mm*250)/tot,
            "MLP": g["mlp"].get("l1d_pend_miss.pending",0)/(g["mlp"].get("l1d_pend_miss.pending_cycles",0) or 1),
            "ILP": g["mlp"].get("uops_executed.thread",0)/(g["mlp"].get("cycles",0) or 1),
            "vec": packed/((scalar+packed) or 1)*100, "GFLOPs": flops/(secs or 1)/1e9}

# three scopes: model side (engine, invariant -> one row), agent harness (live cgroups),
# and the tool executions themselves (CANONICAL local captures, Sonnet-quality workloads).
CANON = os.path.join(HERE, "..", "..", "agentic", "CANONICAL")
def sig_generic(files, cg=None, fp_files=None):
    """files: dict grp->path for cache/mlp (+fp1/fp2 or single fp). cg: cgroup filter or None."""
    def rd(path):
        return parse_all(path, cg) if cg else parse_all(path, "")
    cache = rd(files["cache"]); mlp_g = rd(files["mlp"])
    if not cache.get("cycles"): return None
    fps = [rd(f) for f in fp_files]
    ins = cache.get("instructions", 1)
    l1 = cache.get("mem_load_retired.l1_hit", 0); l2 = cache.get("mem_load_retired.l2_hit", 0)
    l3 = cache.get("mem_load_retired.l3_hit", 0); mm = cache.get("mem_load_retired.l3_miss", 0)
    tot = (l1+l2+l3+mm) or 1
    flops = scalar = packed = 0.0
    for fp in fps:
        for k, v in fp.items():
            if not k.startswith("fp_arith_inst_retired."): continue
            w = next((e for t, e in ELEM.items() if t in k), 1) * 2
            flops += v*w
            if "scalar" in k: scalar += v
            else: packed += v
    secs = sum(elapsed_of(f) for f in fp_files)
    return {"IPC": cache["instructions"]/cache["cycles"],
            "L1": l1/tot*100, "L2": l2/tot*100, "L3": l3/tot*100, "MPKI": mm/(ins/1000),
            "AMAT": (l1*5 + l2*15 + l3*50 + mm*250)/tot,
            "MLP": mlp_g.get("l1d_pend_miss.pending",0)/(mlp_g.get("l1d_pend_miss.pending_cycles",0) or 1),
            "ILP": mlp_g.get("uops_executed.thread",0)/(mlp_g.get("cycles",0) or 1),
            "vec": packed/((scalar+packed) or 1)*100, "GFLOPs": flops/(secs or 1)/1e9}

def sig_local(base, pref, cg):
    return sig_generic({"cache": os.path.join(base, f"{pref}cache.txt"), "mlp": os.path.join(base, f"{pref}mlp.txt")},
                       cg, [os.path.join(base, f"{pref}fp1.txt"), os.path.join(base, f"{pref}fp2.txt")])

def sig_canon(d, upper=False):
    n = (lambda g: os.path.join(d, f"group_{g.upper()}_r1.txt")) if upper else (lambda g: os.path.join(d, f"group_{g}.txt"))
    return sig_generic({"cache": n("cache"), "mlp": n("mlp")}, None, [n("fp")])

COLS = [("IPC","IPC"),("L1","L1 hit %"),("L2","L2 hit %"),("L3","L3 hit %"),("MPKI","LLC-MPKI"),
        ("AMAT","AMAT (cyc)"),("MLP","MLP"),("ILP","ILP"),("vec","vec %FP"),("GFLOPs","GFLOP/s")]
# engine: one invariant row (mean over the 4 full-suite workloads)
eng_sigs = [sig_local(os.path.join(DATA, d), pref, ENG) for d, pref in
            [("astropy","group_engine_"),("scikit-learn","group_engine_"),("sympy","group_engine_"),("bcb_live","group_")]]
eng_sigs = [r for r in eng_sigs if r]
sig_rows = []
if eng_sigs:
    sig_rows.append(("vLLM engine (during inf.)", {k: float(np.mean([r[k] for r in eng_sigs])) for k in eng_sigs[0]}, "eng"))
# harness rows: BCB from the live loop; SWE/OC from the stats-first gap-fill (group_h_*)
HARNESS = [("bcb_live", "group_", ["bcb-live"], "BigCodeBench"),
           ("swe_live", "group_h_", ["swe-hm", "docker-"], "SWE-agent"),
           ("oc_live_calendar", "group_h_", ["docker-"], "OC calendar"),
           ("oc_live_pdf-digest", "group_h_", ["docker-"], "OC pdf-digest"),
           ("oc_live_web-digest", "group_h_", ["docker-"], "OC web-digest"),
           ("oc_live_image-crop", "group_h_", ["docker-"], "OC image-crop")]
for d, pref, cgs, lab in HARNESS:
    base = os.path.join(DATA, d)
    parts = [sig_local(base, pref, c) for c in cgs]
    parts = [r for r in parts if r]
    if not parts: continue
    sig_rows.append((f"harness: {lab}", parts[0] if len(parts) == 1 else
                     {k: float(np.mean([r[k] for r in parts])) for k in parts[0]}, "harn"))
# tool rows: CANONICAL captures
TOOLS = [(os.path.join(CANON, "bigcodebench", "data"), False, "BCB tests"),
         (os.path.join(CANON, "swe_bench", "data", "astropy-14096"), False, "astropy"),
         (os.path.join(CANON, "swe_bench", "data", "scikit-learn-25232"), False, "scikit-learn"),
         (os.path.join(CANON, "swe_bench", "data", "sympy-14248"), False, "sympy"),
         (os.path.join(CANON, "openclaw", "data", "calendar"), True, "OC calendar"),
         (os.path.join(CANON, "openclaw", "data", "arxiv"), True, "OC web-digest"),
         (os.path.join(CANON, "openclaw", "data", "pdf_digest"), True, "OC pdf-digest"),
         (os.path.join(CANON, "openclaw", "data", "social_poster_crop"), True, "OC image-crop")]
for d, up, lab in TOOLS:
    r = sig_canon(d, up)
    if r: sig_rows.append((f"tools: {lab}", r, "tool"))

if len(sig_rows) > 2:
    Mx = np.array([[r[c[0]] for c in COLS] for _, r, _ in sig_rows], float)
    norm = np.zeros_like(Mx)
    for j in range(Mx.shape[1]):
        col = Mx[:, j]; lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi <= lo else (col-lo)/(hi-lo)
    fig, ax = plt.subplots(figsize=(11.5, 0.42*len(sig_rows)+2.2))
    im = ax.imshow(norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    for i, (_, r, _) in enumerate(sig_rows):
        for j, c in enumerate(COLS):
            v = r[c[0]]; txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if norm[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], rotation=25, ha="right")
    ax.set_yticks(range(len(sig_rows))); ax.set_yticklabels([lab for lab, _, _ in sig_rows], fontsize=9)
    CLS_COL = {"eng": INSIDE, "harn": OUTSIDECOL, "tool": "#00441b"}
    for i, (_, _, cls) in enumerate(sig_rows): ax.get_yticklabels()[i].set_color(CLS_COL[cls])
    prev = None
    for i, (_, _, cls) in enumerate(sig_rows):
        if prev and cls != prev: ax.axhline(i-0.5, color="white", linewidth=3)
        prev = cls
    ax.set_title("Microarchitectural signature: engine (during inference), agent harness, and tool executions")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="per-column min\u2013max (relative)")
    fig.savefig(os.path.join(OUT, "local_agents_signature.png")); plt.close(fig)

print("wrote figures to", OUT)
