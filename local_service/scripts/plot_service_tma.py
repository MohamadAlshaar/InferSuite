#!/usr/bin/env python3
"""Service TMA figures for the LOCAL run (SYSTEM python3) — same format as the agents'
02_bottleneck_attribution.png (TMA L2 stacked bars) plus an L1 counterpart, one figure per tier.
Bars = service pods; x-labels coloured by class (purple = during inference, teal = outside).
L2 leaves: 4 measured td2 events + 4 derived siblings (CANONICAL method, clamped at 0)."""
import os, sys, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
INSIDE, OUTSIDECOL = "#6a51a3", "#1b9e77"
TMA2 = [("light_ops","Light operations","#66c2a5"),("heavy_ops","Heavy operations","#1b9e77"),
        ("fetch_lat","Fetch latency","#8da0cb"),("fetch_bw","Fetch bandwidth","#c6dbef"),
        ("br_mispred","Branch mispredict","#fc8d62"),("machine_clears","Machine clears","#fdd0a2"),
        ("mem_bound","Memory bound","#e78ac3"),("core_bound","Core bound","#f4cae4")]
L1 = [("retiring","Retiring","#009E73"),("fe","Frontend-bound","#0072B2"),
      ("bad","Bad speculation","#D55E00"),("be","Backend-bound","#E69F00")]
PODS = [("vllm","vLLM engine\n(during inf.)","in"),("fastapi","FastAPI + BGE","out"),
        ("milvus","Milvus","out"),("mongodb","MongoDB","out"),
        ("seaweed_filer","Seaweed filer","out"),("seaweed_volume","Seaweed volume","out")]
EV = ("slots","topdown-retiring","topdown-bad-spec","topdown-fe-bound","topdown-be-bound",
      "topdown-heavy-ops","topdown-br-mispredict","topdown-fetch-lat","topdown-mem-bound")

def parse(path):
    d = {}
    if not os.path.exists(path): return d
    for ln in open(path, errors="ignore"):
        parts = ln.split()
        if len(parts) < 2: continue
        try: v = float(parts[0].replace(",", ""))
        except ValueError: continue
        for name in EV:
            if name in ln and name not in d: d[name] = v; break
    return d

def tma(tier_dir, pod):
    t1 = parse(os.path.join(tier_dir, f"group_{pod}_tma1.txt"))
    t2 = parse(os.path.join(tier_dir, f"group_{pod}_tma2.txt"))
    s1 = sum(t1.get(k, 0) for k in EV[1:5])
    s2 = t2.get("slots", 0)
    if not s1 or not s2: return None
    l1 = {"retiring": t1["topdown-retiring"]/s1*100, "bad": t1["topdown-bad-spec"]/s1*100,
          "fe": t1["topdown-fe-bound"]/s1*100, "be": t1["topdown-be-bound"]/s1*100}
    m = {"heavy_ops": t2.get("topdown-heavy-ops",0)/s2*100, "br_mispred": t2.get("topdown-br-mispredict",0)/s2*100,
         "fetch_lat": t2.get("topdown-fetch-lat",0)/s2*100, "mem_bound": t2.get("topdown-mem-bound",0)/s2*100}
    l2 = {"heavy_ops": m["heavy_ops"], "light_ops": max(l1["retiring"]-m["heavy_ops"],0),
          "br_mispred": m["br_mispred"], "machine_clears": max(l1["bad"]-m["br_mispred"],0),
          "fetch_lat": m["fetch_lat"], "fetch_bw": max(l1["fe"]-m["fetch_lat"],0),
          "mem_bound": m["mem_bound"], "core_bound": max(l1["be"]-m["mem_bound"],0)}
    return l1, l2

for tier_dir in sorted(glob.glob(os.path.join(DATA, "tok*"))):
    tier = os.path.basename(tier_dir)
    rows = [(lab, cls, tma(tier_dir, key)) for key, lab, cls in PODS]
    rows = [(lab, cls, r) for lab, cls, r in rows if r]
    if len(rows) < 2: print(f"skip {tier}: insufficient data"); continue
    X = np.arange(len(rows))

    # ---- L2 (the agents-style figure) ----
    fig, ax = plt.subplots(figsize=(9.5, 6.2)); bot = np.zeros(len(rows))
    for key, lab, col in TMA2:
        v = np.array([r[1][key] for _, _, r in rows])
        ax.bar(X, v, bottom=bot, label=lab, color=col, width=0.7, edgecolor="white", linewidth=0.4); bot += v
    ax.set_ylabel("Share of pipeline slots (%)"); ax.set_ylim(0, 112); ax.set_xticks(X)
    ax.set_xticklabels([lab for lab, _, _ in rows], rotation=20, ha="right")
    for t, (_, cls, _) in zip(ax.get_xticklabels(), rows): t.set_color(INSIDE if cls == "in" else OUTSIDECOL)
    ax.set_title(f"Micro-architectural bottleneck attribution of the service pods (RAG {tier})")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    d = os.path.join(OUT, tier); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "service_tma_l2.png")); plt.close(fig)

    # ---- L1 counterpart ----
    fig, ax = plt.subplots(figsize=(8.6, 5.6)); bot = np.zeros(len(rows))
    for key, lab, col in L1:
        v = np.array([r[0][key] for _, _, r in rows])
        ax.bar(X, v, bottom=bot, label=lab, color=col, width=0.62, edgecolor="white", linewidth=0.6); bot += v
    ax.set_ylabel("Share of pipeline slots (%)"); ax.set_ylim(0, 112); ax.set_xticks(X)
    ax.set_xticklabels([lab for lab, _, _ in rows], rotation=20, ha="right")
    for t, (_, cls, _) in zip(ax.get_xticklabels(), rows): t.set_color(INSIDE if cls == "in" else OUTSIDECOL)
    ax.set_title(f"Top-down analysis of the service pods, Level 1 (RAG {tier})")
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.38))
    fig.savefig(os.path.join(os.path.join(OUT, tier), "service_tma_l1.png")); plt.close(fig)
    print(f"{tier}: wrote L1+L2 ({len(rows)} pods)")
print("done ->", OUT)
