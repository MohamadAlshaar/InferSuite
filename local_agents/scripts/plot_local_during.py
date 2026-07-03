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
    ("astropy",         "astropy\n(SWE replay)",      "group_engine_", ENG, None,            True),
    ("scikit-learn",    "scikit-learn\n(SWE replay)", "group_engine_", ENG, None,            True),
    ("sympy",           "sympy\n(SWE replay)",        "group_engine_", ENG, None,            True),
    ("swe_live",        "SWE astropy\n(live)",        "group_",        ENG, ["swe-live", "docker-"], True),
    ("bcb_live",        "BigCodeBench\n(live loop)",  "group_",        ENG, "bcb-live",      True),
    ("oc_live_calendar",   "OC calendar\n(live)",     "group_",        ENG, "docker-",       False),
    ("oc_live_pdf-digest", "OC pdf-digest\n(live)",   "group_",        ENG, "docker-",       False),
    ("oc_live_web-digest", "OC web-digest\n(live)",   "group_",        ENG, "docker-",       False),
    ("oc_live_image-crop", "OC image-crop\n(live)",   "group_",        ENG, "docker-",       False),
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

# ---- Fig 1: engine TMA L1 during inference, per agent workload ----
X = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(0.62+1.35*len(rows), 4.4))
bot = np.zeros(len(rows))
for key, lab, col in L1:
    v = np.array([r[2][key] for r in rows])
    ax.bar(X, v, bottom=bot, label=lab, color=col, width=0.62, edgecolor="white", linewidth=0.6)
    for x, (b, vv) in enumerate(zip(bot, v)):
        if vv >= 8: ax.text(x, b+vv/2, f"{vv:.0f}", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
    bot += v
ax.set_xticks(X); ax.set_xticklabels([r[1] for r in rows], fontsize=9)
ax.set_ylabel("Pipeline slots (%)"); ax.set_ylim(0, 100)
ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.13), frameon=False)
fig.savefig(os.path.join(OUT, "local_agents_engine_tma_l1.png")); plt.close(fig)

# ---- Fig 2: engine TMA L2 (workloads with in-window td2) ----
rows2 = [r for r in rows if r[3]]
if rows2:
    X = np.arange(len(rows2))
    fig, ax = plt.subplots(figsize=(0.62+1.5*len(rows2), 4.8))
    bot = np.zeros(len(rows2))
    for key, lab, col in TMA2:
        v = np.array([r[3][key] for r in rows2])
        ax.bar(X, v, bottom=bot, label=lab, color=col, width=0.7, edgecolor="white", linewidth=0.4)
        for x, (b, vv) in enumerate(zip(bot, v)):
            if vv >= 9: ax.text(x, b+vv/2, f"{vv:.0f}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        bot += v
    ax.set_xticks(X); ax.set_xticklabels([r[1] for r in rows2], fontsize=9)
    ax.set_ylabel("Pipeline slots (%)"); ax.set_ylim(0, 100)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, 1.16), frameon=False)
    fig.savefig(os.path.join(OUT, "local_agents_engine_tma_l2.png")); plt.close(fig)

# ---- Fig 3: two-view CPU share donuts (live loops only), same style as the timing donuts ----
SHORT = {"swe_live": "SWE astropy", "bcb_live": "BCB", "oc_live_calendar": "OC calendar",
         "oc_live_pdf-digest": "OC pdf-digest", "oc_live_web-digest": "OC web-digest",
         "oc_live_image-crop": "OC image-crop"}
live = [r for r in rows if r[5] is not None]
if live:
    fig, axes = plt.subplots(1, len(live), figsize=(3.0*len(live), 3.6))
    for ax, r in zip(np.atleast_1d(axes), live):
        eng, tool = r[4], r[5]
        shares = [eng/(eng+tool)*100, tool/(eng+tool)*100]
        ax.pie([max(s, 1.0) for s in shares], colors=[INSIDE, OUTSIDECOL], startangle=90,
               counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.16, SHORT.get(r[0], r[0]), ha="center", fontweight="bold", fontsize=11)
        ax.text(0, -0.14, f"{eng:.2f} CPUs", ha="center", fontsize=9.5, color=INSIDE)
        ax.text(0, -0.38, f"{tool:.2f} CPUs" if tool >= 0.005 else f"{tool:.3f} CPUs",
                ha="center", fontsize=9.5, color=OUTSIDECOL)
        ax.axis("off")
    fig.legend(handles=[Patch(color=INSIDE, label="During inference (vLLM engine)"),
                        Patch(color=OUTSIDECOL, label="Outside inference (agent + tools)")],
               loc="lower center", ncol=2, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.savefig(os.path.join(OUT, "local_agents_two_view_cpu.png")); plt.close(fig)

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
    fig, axes = plt.subplots(2, len(panels), figsize=(3.5*len(panels), 7.4))
    axes = np.atleast_2d(axes)
    used = set()
    for j, (lab, e, t) in enumerate(panels):
        for i, (r, side) in enumerate([(e, "engine"), (t, "tools")]):
            ax = axes[i][j]; ax.axis("off")
            parts = [(k, v) for k, v in sorted(r.items(), key=lambda x: -x[1]) if v > 1]
            for k, _ in parts: used.add(k)
            w, _ = ax.pie([max(v, 1.2) for _, v in parts], colors=[colmap[k] for k, _ in parts],
                          startangle=90, counterclock=False,
                          wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
            for wd, (k, v) in zip(w, parts):
                if v >= 12:
                    a = math.radians((wd.theta1+wd.theta2)/2)
                    ax.text(0.79*math.cos(a), 0.79*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                            color="white", fontweight="bold", fontsize=9)
            ax.text(0, 0, ("During\n" if side == "engine" else "Outside\n") + lab,
                    ha="center", va="center", fontsize=8.2, fontweight="bold",
                    color=INSIDE if side == "engine" else OUTSIDECOL, wrap=False)
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.savefig(os.path.join(OUT, "local_agents_two_view_software.png")); plt.close(fig)

print("wrote figures to", OUT)
