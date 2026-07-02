#!/usr/bin/env python3
"""Service DURING-vs-OUTSIDE inference CPU microarchitecture heatmap (SYSTEM python3).

Rows = the service pods measured under a steady RAG tok320 load:
  vllm     INSIDE  — the serving engine (the CPU *during* GPU inference)
  fastapi  OUTSIDE — orchestration + BGE embedding (CPU-side, BGE_DEVICE=cpu)
  milvus   OUTSIDE — vector search
  mongodb  OUTSIDE — cache/metadata store (measured CPU is mostly its FTDC self-telemetry)
(SeaweedFS is omitted: ≈0 CPU on the RAG path — chunk text is served from the inline Milvus field.)
Cols = IPC, L1/L2/L3 hit %, MPKI, AMAT (cyc, est), MLP, ILP, vectorized %FP, GFLOP/s.
No TMA row (this KVM guest lacks the 'slots' PMU event — same limitation as the agentic run).

Input: h100/service/data/group_<pod>_<grp>.txt   (perf stat --for-each-cgroup text, one group per file:
core / fp1 / fp2 / cache / mlp). Derivations come from agentic/CANONICAL/microarch.py (single source of truth).
Output: h100/service/plots/service_microarch.png
"""
import os, sys, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
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

# (key, row label, class). SeaweedFS is omitted — it drew ≈0 CPU on the RAG path (idle object store).
PODS = [
    ("vllm",    "vLLM engine",           "INSIDE"),
    ("fastapi", "FastAPI + BGE embed",   "OUTSIDE"),
    ("milvus",  "Milvus (vector search)", "OUTSIDE"),
    ("mongodb", "MongoDB (cache meta)",  "OUTSIDE"),
]
GRPS = ["core", "fp1", "fp2", "cache", "mlp"]

def elapsed(path):
    if not os.path.exists(path): return 0.0
    for ln in open(path, errors="ignore"):
        m = re.search(r"([0-9.]+)\s+seconds time elapsed", ln)
        if m: return float(m.group(1))
    return 0.0

def merge_fp(fp1, fp2):
    c = collections.Counter()
    for src in (fp1, fp2):
        for k, v in src.items():
            if k.startswith("fp_arith_inst_retired."):
                c[k] += v
    return c

def load(pod):
    g = {grp: M.parse(os.path.join(DATA, f"group_{pod}_{grp}.txt")) for grp in GRPS}
    core, cache, mlp, fp1, fp2 = g["core"], g["cache"], g["mlp"], g["fp1"], g["fp2"]
    ch = M.cache_hits(cache)
    l1 = cache.get("mem_load_retired.l1_hit", 0); l2 = cache.get("mem_load_retired.l2_hit", 0)
    l3 = cache.get("mem_load_retired.l3_hit", 0); mm = cache.get("mem_load_retired.l3_miss", 0)
    tot = (l1 + l2 + l3 + mm) or 1
    amat = (l1*5 + l2*15 + l3*50 + mm*250) / tot
    fp = merge_fp(fp1, fp2)
    f1 = M.flops(M.parse(os.path.join(DATA, f"group_{pod}_fp1.txt")))
    f2 = M.flops(M.parse(os.path.join(DATA, f"group_{pod}_fp2.txt")))
    s1, s2 = elapsed(os.path.join(DATA, f"group_{pod}_fp1.txt")), elapsed(os.path.join(DATA, f"group_{pod}_fp2.txt"))
    gflops = (f1/(s1 or 1) + f2/(s2 or 1)) / 1e9
    return {
        "IPC": M.ipc(core), "L1": ch["l1"], "L2": ch["l2"], "L3": ch["l3"], "MPKI": ch["mpki"],
        "AMAT": amat, "MLP": M.mlp(mlp), "ILP": M.ilp(mlp), "vec": M.avx_pct(fp), "GFLOPs": gflops,
    }

# FP columns are real in this capture: the load is continuous (conc=6), so the 20 s fp windows
# straddle many embedding bursts (FastAPI: 3.35e9 512b-packed-single = the BGE mkl_avx512_sgemm).
COLS = [("IPC","IPC","%.2f"),("L1","L1 hit %","%.2f"),("L2","L2 hit %","%.2f"),("L3","L3 hit %","%.2f"),
        ("MPKI","LLC-MPKI","%.2f"),("AMAT","AMAT (cyc)","%.1f"),("MLP","MLP","%.2f"),("ILP","ILP","%.2f"),
        ("vec","vec %FP","%.2f"),("GFLOPs","GFLOP/s","%.2f")]

rows, labels, classes = [], [], []
for key, lab, cls in PODS:
    if not os.path.exists(os.path.join(DATA, f"group_{key}_core.txt")):
        print(f"skip {key}: no data"); continue
    rows.append(load(key)); labels.append(lab); classes.append(cls)
    print(f"{key:16} " + "  ".join(f"{c[0]}={rows[-1][c[0]]:.2f}" for c in COLS))

if not rows:
    print("NO DATA — run the capture first"); sys.exit(0)

M_ = np.array([[r[c[0]] for c in COLS] for r in rows], float)
# per-column min-max normalization for the color (values still annotated)
norm = np.zeros_like(M_)
for j in range(M_.shape[1]):
    col = M_[:, j]; lo, hi = col.min(), col.max()
    norm[:, j] = 0.5 if hi <= lo else (col - lo) / (hi - lo)

cmap = LinearSegmentedColormap.from_list("teal", ["#f7fcfd", "#66c2a4", "#00441b"])
fig, ax = plt.subplots(figsize=(1.05*len(COLS)+3.0, 0.82*len(rows)+2.2))
ax.imshow(norm, cmap=cmap, aspect="auto", vmin=0, vmax=1)
CLS_COL = {"INSIDE": "#6a51a3", "OUTSIDE": "#1b9e77"}
for i, r in enumerate(rows):
    for j, c in enumerate(COLS):
        v = r[c[0]]
        ax.text(j, i, c[2] % v, ha="center", va="center", fontsize=9.5,
                color="white" if norm[i, j] > 0.6 else "#222")
ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], rotation=35, ha="right", fontsize=9.5)
ax.set_yticks(range(len(rows)))
ax.set_yticklabels([f"{lab}" for lab in labels], fontsize=10.5)
for i, cls in enumerate(classes):
    ax.get_yticklabels()[i].set_color(CLS_COL[cls])
ax.set_yticks(np.arange(-.5, len(rows), 1), minor=True)
ax.set_xticks(np.arange(-.5, len(COLS), 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5); ax.tick_params(which="minor", length=0)
# class bracket labels on the left
ax.text(-0.055, 1.0, "", transform=ax.transAxes)
fig.suptitle("Service CPU microarchitecture — DURING inference (vLLM) vs OUTSIDE (retrieval), under RAG tok320 load",
             fontsize=12.5, y=1.02)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=CLS_COL["INSIDE"], label="INSIDE inference (engine)"),
                   Patch(color=CLS_COL["OUTSIDE"], label="OUTSIDE inference (retrieval/embed/store)")],
          loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False, fontsize=9.5)
fig.text(0.5, -0.02, "Colour = per-column min→max (relative). Measured under a verified continuous RAG tok320 load "
         "(vLLM at 6 concurrent reqs, ~2 CPUs). DURING inference the engine is a high-IPC, L1-resident, ~0-vector-FP "
         "busy-wait spin; OUTSIDE, FastAPI carries the real AVX-512 FP (BGE embedding) at low IPC.",
         ha="center", fontsize=8.2, style="italic", color="#555")
fig.savefig(os.path.join(OUT, "service_microarch.png")); plt.close(fig)
print("fig ->", os.path.join(OUT, "service_microarch.png"))
