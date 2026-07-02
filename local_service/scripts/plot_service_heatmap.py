#!/usr/bin/env python3
"""Service microarch signature heatmap for the LOCAL run (SYSTEM python3) — the service analogue of
the agents' 04_signature_heatmap, one per tier. Rows = pods (vLLM during inference + the outside
pods), cols = the portable suite (same set as the H100 service heatmap, so the two are directly
comparable). Derivations via agentic/CANONICAL/microarch.py (single source of truth)."""
import os, sys, re, glob, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "agentic", "CANONICAL"))
import microarch as M
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)
INSIDE, OUTSIDECOL = "#6a51a3", "#1b9e77"

PODS = [("vllm", "vLLM engine (during inf.)", "in"), ("fastapi", "FastAPI + BGE embed", "out"),
        ("milvus", "Milvus (vector search)", "out"), ("mongodb", "MongoDB (cache/history)", "out"),
        ("seaweed_filer", "SeaweedFS filer", "out"), ("seaweed_volume", "SeaweedFS volume", "out")]
GRPS = ["core", "fp1", "fp2", "cache", "mlp"]
COLS = [("IPC","IPC","%.2f"),("L1","L1 hit %","%.2f"),("L2","L2 hit %","%.2f"),("L3","L3 hit %","%.2f"),
        ("MPKI","LLC-MPKI","%.2f"),("AMAT","AMAT (cyc)","%.1f"),("MLP","MLP","%.2f"),("ILP","ILP","%.2f"),
        ("vec","vec %FP","%.2f"),("GFLOPs","GFLOP/s","%.2f")]

def elapsed(path):
    if not os.path.exists(path): return 0.0
    for ln in open(path, errors="ignore"):
        m = re.search(r"([0-9.]+)\s+seconds time elapsed", ln)
        if m: return float(m.group(1))
    return 0.0

def load(tier_dir, pod):
    g = {grp: M.parse(os.path.join(tier_dir, f"group_{pod}_{grp}.txt")) for grp in GRPS}
    if not g["core"]: return None
    ch = M.cache_hits(g["cache"])
    c = g["cache"]
    l1 = c.get("mem_load_retired.l1_hit", 0); l2 = c.get("mem_load_retired.l2_hit", 0)
    l3 = c.get("mem_load_retired.l3_hit", 0); mm = c.get("mem_load_retired.l3_miss", 0)
    tot = (l1 + l2 + l3 + mm) or 1
    amat = (l1*5 + l2*15 + l3*50 + mm*250) / tot
    fp = collections.Counter()
    for src in (g["fp1"], g["fp2"]):
        for k, v in src.items():
            if k.startswith("fp_arith_inst_retired."): fp[k] += v
    f1 = M.flops(g["fp1"]); f2 = M.flops(g["fp2"])
    s1 = elapsed(os.path.join(tier_dir, f"group_{pod}_fp1.txt")); s2 = elapsed(os.path.join(tier_dir, f"group_{pod}_fp2.txt"))
    gflops = (f1/(s1 or 1) + f2/(s2 or 1)) / 1e9
    return {"IPC": M.ipc(g["core"]), "L1": ch["l1"], "L2": ch["l2"], "L3": ch["l3"], "MPKI": ch["mpki"],
            "AMAT": amat, "MLP": M.mlp(g["mlp"]), "ILP": M.ilp(g["mlp"]), "vec": M.avx_pct(fp), "GFLOPs": gflops}


for tier_dir in sorted(glob.glob(os.path.join(DATA, "tok*"))):
    tier = os.path.basename(tier_dir)
    rows, labels, classes = [], [], []
    for key, lab, cls in PODS:
        r = load(tier_dir, key)
        if r: rows.append(r); labels.append(lab); classes.append(cls)
    if len(rows) < 2: print(f"skip {tier}"); continue
    Mx = np.array([[r[c[0]] for c in COLS] for r in rows], float)
    norm = np.zeros_like(Mx)
    for j in range(Mx.shape[1]):
        col = Mx[:, j]; lo, hi = col.min(), col.max()
        norm[:, j] = 0.5 if hi <= lo else (col - lo) / (hi - lo)
    fig, ax = plt.subplots(figsize=(11.5, 5.2))
    im = ax.imshow(norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    for i, r in enumerate(rows):
        for j, c in enumerate(COLS):
            v = r[c[0]]; txt = f"{v:.2f}" if v < 10 else f"{v:.0f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                    color="black" if norm[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], rotation=25, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels)
    for i, cls in enumerate(classes): ax.get_yticklabels()[i].set_color(INSIDE if cls == "in" else OUTSIDECOL)
    ax.set_title(f"Micro-architectural signature across service pods (RAG {tier})")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="per-column min\u2013max (relative)")
    d = os.path.join(OUT, tier); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "service_heatmap.png")); plt.close(fig)
    print(f"{tier}: heatmap with {len(rows)} pods")
print("done ->", OUT)
