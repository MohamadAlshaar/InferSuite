#!/usr/bin/env python3
"""vLLM engine microarch signature on the H100 service node (SYSTEM python3) — drawn in the SAME
format as agentic/inference/plots/make_figures.py fig 03 (03_inference_microarch_signature.png) so the
local (TMA-capable) and H100 (portable-suite) engine signatures are visually comparable in the
thesis. No 'Retiring slots' row: this KVM guest exposes no TMA slots event.

Inputs: h100/service/data/group_vllm_{core,fp1,fp2,cache,mlp}.txt (perf stat, cgroup-scoped)
        h100/service/data/vllm_flat.txt (perf report self%, for the busy-wait share)
Output: h100/service/plots/vllm_signature.png
"""
import os, sys, re, collections
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "agentic", "CANONICAL"))
import microarch as M
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
ACCENT = "#6a51a3"  # inference accent — matches the local during-inference figures
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)

g = {grp: M.parse(os.path.join(DATA, f"group_vllm_{grp}.txt")) for grp in ("core", "fp1", "fp2", "cache", "mlp")}
ch = M.cache_hits(g["cache"])
fp = collections.Counter()
for src in (g["fp1"], g["fp2"]):
    for k, v in src.items():
        if k.startswith("fp_arith_inst_retired."):
            fp[k] += v

# busy-wait share = libcuda + [vdso] self% from the flat attribution (same buckets as
# plot_service_attribution.py: CUDA event-sync spin + the vdso clock-poll around it)
busy = 0.0
FLAT = re.compile(r"^\s*[0-9.]+%\s+([0-9.]+)%\s+\S+\s+(\S+)\s+\[.\]")
for ln in open(os.path.join(DATA, "vllm_flat.txt"), errors="ignore"):
    m = FLAT.match(ln)
    if m and (re.search(r"libcuda", m.group(2), re.I) or m.group(2) == "[vdso]"):
        busy += float(m.group(1))

ipc, ilp, mlp, avx = M.ipc(g["core"]), M.ilp(g["mlp"]), M.mlp(g["mlp"]), M.avx_pct(fp)
METR = [("IPC (of 4.0 retire width)", ipc, 4.0, f"{ipc:.2f}"),
        ("L1 data-cache hit", ch["l1"], 100, f"{ch['l1']:.2f}%"),
        ("ILP (uops/cycle, of 5)", ilp, 5.0, f"{ilp:.2f}"),
        ("MLP (outstanding L1 misses)", mlp, 5.0, f"{mlp:.2f}"),
        ("Vectorized FP (AVX)", avx, 100, f"{avx:.0f}%"),
        ("GPU busy-wait share", busy, 100, f"{busy:.0f}%")]
print("  ".join(f"{m[0]}={m[3]}" for m in METR))

fig, ax = plt.subplots(figsize=(8.2, 4.7))
y = range(len(METR))
ax.barh(list(y), [m[1]/m[2]*100 for m in METR], color=ACCENT, alpha=0.85, edgecolor="white", height=0.6)
for i, m in enumerate(METR):
    ax.text(m[1]/m[2]*100 + 1.5, i, m[3], va="center", fontsize=11, fontweight="bold", color="#222")
ax.set_yticks(list(y)); ax.set_yticklabels([m[0] for m in METR]); ax.invert_yaxis()
ax.set_xlim(0, 118); ax.set_xlabel("fraction of each metric's scale (%)")
ax.set_title("Micro-architectural signature of the engine host CPU during inference")
ax.grid(axis="y", visible=False)
fig.savefig(os.path.join(OUT, "vllm_signature.png")); plt.close(fig)
print("wrote", os.path.join(OUT, "vllm_signature.png"))
