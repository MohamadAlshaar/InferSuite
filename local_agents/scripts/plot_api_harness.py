#!/usr/bin/env python3
"""Frontier-tier (Sonnet-driven) AGENT-SIDE figures from the live API campaign
(local_agents/data/api_*): full TMA L1+L2, portable suite, and software views for the
agent harness at the model tier that actually verifies its work. Same palettes as the
rest of the thesis. SWE rows merge the sweagent scope with its sandbox container
(cpus-weighted for software, counter-summed for TMA/micro), BCB is its driver scope."""
import os, re, math
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
OUTSIDECOL = "#1b9e77"
L1 = [("retiring","Retiring","#009E73"),("fe","Frontend-bound","#0072B2"),
      ("bad","Bad speculation","#D55E00"),("be","Backend-bound","#E69F00")]
TMA2 = [("light_ops","Light operations","#66c2a5"),("heavy_ops","Heavy operations","#1b9e77"),
        ("fetch_lat","Fetch latency","#8da0cb"),("fetch_bw","Fetch bandwidth","#c6dbef"),
        ("br_mispred","Branch mispredict","#fc8d62"),("machine_clears","Machine clears","#fdd0a2"),
        ("mem_bound","Memory bound","#e78ac3"),("core_bound","Core bound","#f4cae4")]
EV = ("slots","topdown-retiring","topdown-bad-spec","topdown-fe-bound","topdown-be-bound",
      "topdown-heavy-ops","topdown-br-mispredict","topdown-fetch-lat","topdown-mem-bound",
      "cycles","instructions","branches","branch-misses",
      "mem_load_retired.l1_hit","mem_load_retired.l2_hit","mem_load_retired.l3_hit",
      "mem_load_retired.l3_miss","l1d_pend_miss.pending","l1d_pend_miss.pending_cycles",
      "uops_executed.thread",
      "fp_arith_inst_retired.scalar_single","fp_arith_inst_retired.scalar_double",
      "fp_arith_inst_retired.128b_packed_single","fp_arith_inst_retired.128b_packed_double",
      "fp_arith_inst_retired.256b_packed_single","fp_arith_inst_retired.256b_packed_double",
      "fp_arith_inst_retired.512b_packed_single","fp_arith_inst_retired.512b_packed_double")

def txtcol(hexcol):
    r, g, b = (int(hexcol[i:i+2], 16) for i in (1, 3, 5))
    return "white" if 0.299*r + 0.587*g + 0.114*b < 150 else "#333333"

def parse_cg(path, cg):
    d = {}
    if not os.path.exists(path): return d
    for ln in open(path, errors="ignore"):
        if cg not in ln: continue
        parts = ln.split()
        if len(parts) < 2: continue
        try: v = float(parts[0].replace(",", ""))
        except ValueError: continue
        for name in EV:
            if name in ln and name not in d: d[name] = v; break
        m = re.search(r"#\s+([\d.]+) CPUs utilized", ln)
        if m and "cpus" not in d: d["cpus"] = float(m.group(1))
    return d

def sum_cgs(path, cgs):
    tot = {}
    for c in cgs:
        for k, v in parse_cg(path, c).items():
            if k != "cpus": tot[k] = tot.get(k, 0) + v
    return tot

def tma_of(t1, t2):
    s1 = sum(t1.get(f"topdown-{k}", 0) for k in ("retiring","bad-spec","fe-bound","be-bound"))
    if not s1: return None
    l1 = {"retiring": t1["topdown-retiring"]/s1*100, "bad": t1["topdown-bad-spec"]/s1*100,
          "fe": t1["topdown-fe-bound"]/s1*100, "be": t1["topdown-be-bound"]/s1*100}
    l2 = None
    if t2.get("slots"):
        s2 = t2["slots"]
        m = {k: t2.get(f"topdown-{n}", 0)/s2*100 for k, n in
             [("heavy_ops","heavy-ops"),("br_mispred","br-mispredict"),("fetch_lat","fetch-lat"),("mem_bound","mem-bound")]}
        l2 = {"heavy_ops": m["heavy_ops"], "light_ops": max(l1["retiring"]-m["heavy_ops"],0),
              "br_mispred": m["br_mispred"], "machine_clears": max(l1["bad"]-m["br_mispred"],0),
              "fetch_lat": m["fetch_lat"], "fetch_bw": max(l1["fe"]-m["fetch_lat"],0),
              "mem_bound": m["mem_bound"], "core_bound": max(l1["be"]-m["mem_bound"],0)}
        if l2 and sum(l2.values()) > 105: l2 = None
    return l1, l2

ELEM = {"scalar": 1, "128b": 2, "256b": 4, "512b": 8}
def elapsed_of(path):
    if not os.path.exists(path): return 0.0
    for ln in open(path, errors="ignore"):
        m = re.search(r"([0-9.]+)\s+seconds time elapsed", ln)
        if m: return float(m.group(1))
    return 0.0

def sig_of(base, cgs):
    g = {grp: sum_cgs(os.path.join(base, f"group_{grp}.txt"), cgs) for grp in ("core","cache","mlp","fp1","fp2")}
    if not g["cache"].get("cycles"): return None
    ins = g["cache"].get("instructions", 1)
    l1 = g["cache"].get("mem_load_retired.l1_hit", 0); l2 = g["cache"].get("mem_load_retired.l2_hit", 0)
    l3 = g["cache"].get("mem_load_retired.l3_hit", 0); mm = g["cache"].get("mem_load_retired.l3_miss", 0)
    tot = (l1+l2+l3+mm) or 1
    flops = scalar = packed = 0.0
    for src in (g["fp1"], g["fp2"]):
        for k, v in src.items():
            if not k.startswith("fp_arith_inst_retired."): continue
            w = next((e for t, e in ELEM.items() if t in k), 1) * 2
            flops += v*w
            if "scalar" in k: scalar += v
            else: packed += v
    secs = elapsed_of(os.path.join(base, "group_fp1.txt")) + elapsed_of(os.path.join(base, "group_fp2.txt"))
    return {"IPC": g["cache"]["instructions"]/g["cache"]["cycles"],
            "L1": l1/tot*100, "L2": l2/tot*100, "L3": l3/tot*100, "MPKI": mm/(ins/1000),
            "AMAT": (l1*5 + l2*15 + l3*50 + mm*250)/tot,
            "MLP": g["mlp"].get("l1d_pend_miss.pending",0)/(g["mlp"].get("l1d_pend_miss.pending_cycles",0) or 1),
            "ILP": g["mlp"].get("uops_executed.thread",0)/(g["mlp"].get("cycles",0) or 1),
            "vec": packed/((scalar+packed) or 1)*100, "GFLOPs": flops/(secs or 1)/1e9}

WL = [("api_astropy",      "SWE astropy",      ["swe-api", "docker-"]),
      ("api_scikit-learn", "SWE scikit-learn", ["swe-api", "docker-"]),
      ("api_sympy",        "SWE sympy",        ["swe-api", "docker-"]),
      ("api_bcb",          "BigCodeBench",     ["bcb-api"]),
      ("api_oc_calendar",  "OC calendar",      ["docker-"]),
      ("api_oc_web-digest","OC web-digest",    ["docker-"]),
      ("api_oc_pdf-digest","OC pdf-digest",    ["docker-"]),
      ("api_oc_image-crop","OC image-crop",    ["docker-"])]

rows = []
for d, lab, cgs in WL:
    base = os.path.join(DATA, d)
    alive = os.path.join(base, "stat_groups_alive.txt")
    if not os.path.exists(alive): continue
    if sum(1 for l in open(alive) if "agent_alive=1" in l) < 6: continue   # skip failed captures
    t1 = sum_cgs(os.path.join(base, "group_tma1.txt"), cgs)
    t2 = sum_cgs(os.path.join(base, "group_tma2.txt"), cgs)
    r = tma_of(t1, t2)
    sig = sig_of(base, cgs)
    if r and sig: rows.append((d, lab, r[0], r[1], sig, base, cgs))

# ---- Fig 1: TMA L1+L2 for the frontier agent side ----
if rows:
    for comps, key, fname, title in [
        (L1, 2, "api_harness_tma_l1.png", "Agent side under Sonnet: TMA Level 1"),
        (TMA2, 3, "api_harness_tma_l2.png", "Agent side under Sonnet: TMA Level 2")]:
        ent = [(lab, r) for _, lab, l1c, l2c, *_ in rows for r in [l1c if key == 2 else l2c] if r]
        X = np.arange(len(ent))
        fig, ax = plt.subplots(figsize=(0.62+1.6*len(ent), 4.6))
        bot = np.zeros(len(ent))
        for k, lab_c, col in comps:
            v = np.array([e[1][k] for e in ent])
            ax.bar(X, v, bottom=bot, label=lab_c, color=col, width=0.62, edgecolor="white", linewidth=0.6)
            for x, (b, vv) in enumerate(zip(bot, v)):
                if vv >= 8: ax.text(x, b+vv/2, f"{vv:.0f}", ha="center", va="center",
                                    color=txtcol(col), fontsize=8.5, fontweight="bold")
            bot += v
        ax.set_xticks(X); ax.set_xticklabels([e[0] for e in ent], fontsize=9, rotation=15, ha="right")
        for t in ax.get_xticklabels(): t.set_color(OUTSIDECOL)
        ax.set_ylabel("Pipeline slots (%)"); ax.set_ylim(0, 100)
        ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, 1.14), frameon=False)
        ax.set_title(title, fontsize=13, pad=42)
        fig.savefig(os.path.join(OUT, fname)); plt.close(fig)

# ---- Fig 2: signature heatmap ----
COLS = [("IPC","IPC"),("L1","L1 hit %"),("L2","L2 hit %"),("L3","L3 hit %"),("MPKI","LLC-MPKI"),
        ("AMAT","AMAT (cyc)"),("MLP","MLP"),("ILP","ILP"),("vec","vec %FP"),("GFLOPs","GFLOP/s")]
if rows:
    Mx = np.array([[r[4][c[0]] for c in COLS] for r in rows], float)
    norm = np.zeros_like(Mx)
    for j in range(Mx.shape[1]):
        col = Mx[:, j]; lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi <= lo else (col-lo)/(hi-lo)
    fig, ax = plt.subplots(figsize=(11.5, 0.5*len(rows)+2.0))
    im = ax.imshow(norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    for i, r in enumerate(rows):
        for j, c in enumerate(COLS):
            v = r[4][c[0]]; txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="black" if norm[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows], color=OUTSIDECOL, fontsize=9.5)
    ax.set_title("Agent side under Sonnet: portable counter suite (harness + tools, same windows)")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="per-column min–max (relative)")
    fig.savefig(os.path.join(OUT, "api_harness_signature.png")); plt.close(fig)

# ---- Fig 3: software donuts (scope + sandbox merged by cpus) ----
ROLES = [
    ("Node.js / V8 (agent)", "#56b4e9", re.compile(r"\bnode\b|libnode|/node$|\bv8\b|\[JIT\]", re.I)),
    ("Python interpreter",   "#d94801", re.compile(r"python3|libpython|\.cpython-", re.I)),
    ("BLAS / OpenMP",        "#238b45", re.compile(r"openblas|libgomp|libtorch|libmkl", re.I)),
    ("C library / loader",   "#2171b5", re.compile(r"libc\.so|ld-linux|libm\.so|libstdc|libz\.|libcrypto|libssl", re.I)),
    ("OS kernel",            "#CC79A7", re.compile(r"kallsyms|\[kernel|\[vdso\]", re.I)),
]
NEUTRAL = "#b3b3b3"
colmap = {n: c for n, c, _ in ROLES}; colmap["other"] = NEUTRAL
def parse_dso(path):
    out = {r[0]: 0.0 for r in ROLES}; out["other"] = 0.0
    if not os.path.exists(path): return None
    for ln in open(path, errors="ignore"):
        m = re.match(r"\s*([0-9.]+)%\s+(\S+)", ln)
        if not m: continue
        pct, dso = float(m.group(1)), m.group(2)
        if pct <= 0: continue
        for name, col, rx in ROLES:
            if rx.search(dso): out[name] += pct; break
        else: out["other"] += pct
    tot = sum(out.values())
    return {k: v/tot*100 for k, v in out.items()} if tot else None

panels = []
for d, lab, l1c, l2c, sig, base, cgs in rows:
    profs, ws = [], []
    core = os.path.join(base, "group_core.txt")
    for i, c in enumerate(cgs, 1):
        pr = parse_dso(os.path.join(base, f"scope{i}_dso.txt"))
        if pr:
            profs.append(pr); ws.append(parse_cg(core, c).get("cpus", 0) or 0.001)
    if not profs: continue
    tot = sum(ws)
    merged = {k: sum(p.get(k, 0)*w for p, w in zip(profs, ws))/tot for k in profs[0]}
    panels.append((lab, merged))
if panels:
    nc = (len(panels)+1)//2 if len(panels) > 3 else len(panels)
    nr = 2 if len(panels) > 3 else 1
    fig, axes = plt.subplots(nr, nc, figsize=(2.7*nc, 3.2*nr))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(panels):]: ax.axis("off")
    used = set()
    for ax, (lab, r) in zip(axes, panels):
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
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
        ax.axis("off")
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Agent machinery under Sonnet: harness and tools, task-clock attribution", fontsize=13, y=1.02)
    fig.savefig(os.path.join(OUT, "api_harness_software.png")); plt.close(fig)

print("wrote", OUT, "-", len(rows), "workloads")
