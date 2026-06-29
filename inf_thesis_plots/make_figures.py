#!/usr/bin/env python3
"""Publication figures for the CPU-during-inference result (Phase 1) — INFERENCE ONLY.
Local vLLM Qwen2.5-7B-AWQ on an RTX A2000, sustained agent-prompt load, perf scoped to the whole
engine (API server + VLLM::EngineCore). Non-multiplexed (FP from split passes). No cross-config
comparison. Run with SYSTEM python3 -> writes 01..04 PNGs."""
import os, json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(f"{HERE}/data.json"))["inference"]

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 12, "axes.titlesize": 13.5, "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
ACCENT = "#6a51a3"      # inference accent
RET, FE, BAD, BE = "#009E73", "#0072B2", "#D55E00", "#E69F00"

# ================= 01a. Top-down Level 1 — single column =================
l1d, l2 = D["tma_l1"], D["tma_l2"]
L1 = [("Retiring", l1d["Retiring"], RET), ("Frontend-bound", l1d["Frontend"], FE),
      ("Bad-speculation", l1d["BadSpec"], BAD), ("Backend-bound", l1d["Backend"], BE)]
fig, ax = plt.subplots(figsize=(4.7, 6.2)); bot = 0
for lab, v, col in L1:
    ax.bar(0, v, bottom=bot, color=col, width=0.55, edgecolor="white", linewidth=1.0, label=f"{lab}  ({v:.0f}%)")
    if v >= 3:
        ax.text(0, bot + v/2, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=13)
    bot += v
ax.text(0, bot + 2, f"IPC {D['ipc']:.2f}", ha="center", fontweight="bold", color=ACCENT, fontsize=13)
ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_ylabel("Share of pipeline slots (%)"); ax.set_ylim(0, 114)
ax.set_title("Top-down analysis of the host CPU during inference (Level 1)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=10)
fig.savefig(f"{HERE}/01a_inference_tma_L1.png"); plt.close(fig)

# ================= 01b. Top-down Level 2 (drill-down) — single column =================
LEAVES = [("Retiring · light-ops", l2["light_ops"], "#66c2a5"), ("Retiring · heavy-ops", l2["heavy_ops"], "#1b9e77"),
          ("Front-end · fetch-latency", l2["fetch_lat"], "#9ecae1"), ("Front-end · fetch-bandwidth", l2["fetch_bw"], "#3182bd"),
          ("Bad-spec · branch-mispredict", l2["br_mispred"], "#fc9272"), ("Bad-spec · machine-clears", l2["machine_clears"], "#de2d26"),
          ("Back-end · core-bound", l2["core_bound"], "#fdae6b"), ("Back-end · memory-bound", l2["mem_bound"], "#e6550d")]
fig, ax = plt.subplots(figsize=(5.0, 6.2)); bot = 0
for lab, v, col in LEAVES:
    ax.bar(0, v, bottom=bot, color=col, width=0.55, edgecolor="white", linewidth=0.8, label=f"{lab}  ({v:.0f}%)")
    if v >= 4:
        ax.text(0, bot + v/2, f"{v:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=12)
    bot += v
ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_ylabel("Share of pipeline slots (%)"); ax.set_ylim(0, 114)
ax.set_title("Top-down analysis of the host CPU during inference (Level 2)")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.03), fontsize=9.5)
fig.savefig(f"{HERE}/01b_inference_tma_L2.png"); plt.close(fig)

# ================= 02. CPU time by library/function =================
import math
# library -> (what the CPU is executing, colour). Each a distinct colour.
DESC = {
    "libcuda.so (CUDA driver / sync)": ("CUDA driver — GPU event-synchronization (busy-wait)", "#6a51a3"),
    "[vdso] (clock_gettime)":          ("clock_gettime — timing / polling", "#9e9ac8"),
    "libc.so.6":                       ("C library — memcpy / malloc", "#2171b5"),
    "python3.12":                      ("Python interpreter — scheduling loop", "#d94801"),
    "[kernel]":                        ("OS kernel", "#cb181d"),
    "tokenizers":                      ("(de)tokenization", "#238b45"),
    "libtorch_cpu.so":                 ("PyTorch CPU ops — sampling", "#74c476"),
    "libtorch_python.so":              ("PyTorch Python bindings", "#bdbdbd"),
}
order = sorted(D["dso"].items(), key=lambda kv: -kv[1])
labels = [k for k, _ in order]; vals = [v for _, v in order]
vis = [v ** 0.5 for v in vals]                      # sqrt -> tiny slices stay visible; labels show TRUE %
cols = [DESC[k][1] for k in labels]
fig, ax = plt.subplots(figsize=(10.5, 5.8))
wedges, _ = ax.pie(vis, colors=cols, startangle=90, counterclock=False,
                   wedgeprops=dict(width=0.45, edgecolor="white", linewidth=1.5))
for w, v in zip(wedges, vals):                      # on-wedge true% for the larger slices
    if v >= 5:
        a = math.radians((w.theta1 + w.theta2) / 2)
        ax.text(0.78*math.cos(a), 0.78*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                color="white", fontweight="bold", fontsize=12)
ax.legend(wedges, [f"{DESC[k][0]}   ({v:.2f}%)" for k, v in zip(labels, vals)],
          loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=10, title="Software component",
          title_fontproperties={"weight": "bold"})
ax.set_title("Distribution of host-CPU time during inference, by software component")
fig.text(0.5, 0.01, "Wedge area is √-scaled for legibility; printed values and the legend give the true percentages.",
         ha="center", fontsize=8.5, style="italic", color="#666")
fig.savefig(f"{HERE}/02_cpu_function_donut.png"); plt.close(fig)

# ================= 03. Micro-architectural signature (single subject) =================
# horizontal indicator bars, each metric on its own 0..max scale, value annotated
METR = [("IPC (of 4.0 retire width)", D["ipc"], 4.0, f"{D['ipc']:.2f}"),
        ("Retiring slots", D["tma_l1"]["Retiring"], 100, f"{D['tma_l1']['Retiring']:.0f}%"),
        ("L1 data-cache hit", D["l1_hit"], 100, f"{D['l1_hit']:.1f}%"),
        ("ILP (uops/cycle, of 5)", D["ilp"], 5.0, f"{D['ilp']:.2f}"),
        ("MLP (outstanding L1 misses)", D["mlp"], 5.0, f"{D['mlp']:.2f}"),
        ("Vectorized FP (AVX)", D["avx"], 100, f"{D['avx']:.0f}%"),
        ("GPU busy-wait share", D["sync_path_pct"], 100, f"{D['sync_path_pct']:.0f}%")]
fig, ax = plt.subplots(figsize=(8.2, 5.2))
y = range(len(METR))
ax.barh(list(y), [m[1]/m[2]*100 for m in METR], color=ACCENT, alpha=0.85, edgecolor="white", height=0.6)
for i, m in enumerate(METR):
    ax.text(m[1]/m[2]*100 + 1.5, i, m[3], va="center", fontsize=11, fontweight="bold", color="#222")
ax.set_yticks(list(y)); ax.set_yticklabels([m[0] for m in METR]); ax.invert_yaxis()
ax.set_xlim(0, 118); ax.set_xlabel("fraction of each metric's scale (%)")
ax.set_title("Micro-architectural signature of the host CPU during inference")
ax.grid(axis="y", visible=False)
fig.savefig(f"{HERE}/03_inference_microarch_signature.png"); plt.close(fig)

# ================= 04. Where CPU time goes during inference (donut) =================
busy_pct = D["sync_path_pct"]; work_pct = 100 - busy_pct
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.pie([busy_pct, work_pct], colors=[ACCENT, "#bdbdbd"], startangle=90, counterclock=False,
       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
       autopct=lambda p: f"{p:.0f}%", pctdistance=0.78, textprops=dict(fontsize=14, fontweight="bold", color="white"))
ax.text(0, 0, "CPU time\nduring\ninference", ha="center", va="center", fontsize=11, fontweight="bold")
ax.legend(handles=[Patch(color=ACCENT, label="GPU busy-wait  (CUDA event-sync + clock polling)"),
                   Patch(color="#bdbdbd", label="host-side work  (sampling, detok, scheduling, Python)")],
          loc="lower center", bbox_to_anchor=(0.5, -0.16), fontsize=10)
ax.set_title("Host-CPU time allocation during inference")
fig.savefig(f"{HERE}/04_cpu_time_allocation.png"); plt.close(fig)

print("WROTE ->", HERE)
for f in sorted(os.listdir(HERE)):
    if f.endswith(".png"): print("  ", f)
