#!/usr/bin/env python3
"""EXPERIMENT: TMA Level 2 — each L1 bucket split into its two measured sub-causes.
Retiring = Light + Heavy ops ; Bad-spec = Branch-mispredict + Machine-clears ;
Front-end = Fetch-latency + Fetch-bandwidth ; Back-end = Memory-bound + Core-bound.
Sums topdown counts across all TMA windows per fence (matches load_task). SYSTEM python3."""
import os, glob, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SWE = "/home/mohamad/llm-service-kernel-latest/local_agents/SWE_long"
OUT = os.path.dirname(os.path.abspath(__file__))
TASKS = [("django (Python)", "glm_swe_django"), ("sympy (Python)", "glm_swe_sympy-light"),
         ("babel (JavaScript)", "glm_swe_babel"), ("fmt (C++)", "glm_swe_fmtlib")]
plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight"})

EVENTS = ("slots", "topdown-retiring", "topdown-bad-spec", "topdown-fe-bound", "topdown-be-bound",
          "topdown-heavy-ops", "topdown-br-mispredict", "topdown-fetch-lat", "topdown-mem-bound")


def role_of(cg):
    if "glm-proxy" in cg: return "litellm"
    if "docker-" in cg:   return "tool"
    if "glm-swe" in cg or ".scope" in cg: return "harness"
    return None


def load(task_dir):
    """sum every topdown count per role across all group_tma windows."""
    S = {}
    for w in glob.glob(f"{task_dir}/group_tma_w*.txt"):
        for ln in open(w):
            m = re.match(r"\s*([\d,]+)\s+(\S+)\s+(\S+)", ln)
            if not m: continue
            val, ev, cg = m.group(1), m.group(2), m.group(3)
            if ev not in EVENTS: continue
            role = role_of(cg)
            if not role: continue
            d = S.setdefault(role, {})
            d[ev] = d.get(ev, 0.0) + float(val.replace(",", ""))
    return S


# L2 split: (label, L1 total key, measured-sub key or None for remainder, color)
# each L1 bucket -> [measured sub-cause, remainder]
SPLIT = [
    ("Retiring · light ops",      "topdown-retiring", "REMAIN", "topdown-heavy-ops",  "#66c2a4"),
    ("Retiring · heavy ops",      "topdown-retiring", "SUB",    "topdown-heavy-ops",  "#00695c"),
    ("Front-end · fetch latency", "topdown-fe-bound", "SUB",    "topdown-fetch-lat",  "#74add1"),
    ("Front-end · fetch b/w",     "topdown-fe-bound", "REMAIN", "topdown-fetch-lat",  "#1f5fa8"),
    ("Bad-spec · branch mispred", "topdown-bad-spec", "SUB",    "topdown-br-mispredict", "#f4a582"),
    ("Bad-spec · machine clears", "topdown-bad-spec", "REMAIN", "topdown-br-mispredict", "#b2182b"),
    ("Back-end · memory bound",   "topdown-be-bound", "SUB",    "topdown-mem-bound",  "#fdae61"),
    ("Back-end · core bound",     "topdown-be-bound", "REMAIN", "topdown-mem-bound",  "#d98200"),
]


def txtcol(hexc):
    r, g, b = (int(hexc[i:i+2], 16)/255 for i in (1, 3, 5))
    return "white" if (0.299*r+0.587*g+0.114*b) < 0.6 else "#222222"


rows = []
for disp, cfg in TASKS:
    S = load(f"{SWE}/data/{cfg}/run_1")
    for role in ("tool", "harness"):
        d = S.get(role, {})
        L1sum = sum(d.get(k, 0) for k in
                    ("topdown-retiring", "topdown-bad-spec", "topdown-fe-bound", "topdown-be-bound")) or 1
        seg = []
        for lab, l1key, kind, subkey, col in SPLIT:
            sub = d.get(subkey, 0.0)
            val = sub if kind == "SUB" else max(d.get(l1key, 0.0) - sub, 0.0)
            seg.append(100 * val / L1sum)
        rows.append((f"{disp} — {role}", seg))

fig, ax = plt.subplots(figsize=(13.2, 0.55 * len(rows) + 2.4))
Y = np.arange(len(rows)); left = np.zeros(len(rows))
for si, (lab, _l1, _k, _s, col) in enumerate(SPLIT):
    v = np.array([r[1][si] for r in rows])
    ax.barh(Y, v, left=left, color=col, height=0.62, label=lab, edgecolor="white", linewidth=0.7)
    for y, (l, vv) in enumerate(zip(left, v)):
        if vv >= 4:
            ax.text(l + vv/2, y, f"{vv:.0f}", ha="center", va="center",
                    fontsize=7.6, color=txtcol(col), fontweight="bold")
    left += v
ax.set_yticks(Y); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5); ax.invert_yaxis()
ax.set_xlim(0, 100); ax.set_xlabel("Pipeline slots (%)"); ax.grid(axis="x", alpha=0.4)
ax.legend(ncol=4, fontsize=8.3, loc="upper center", bbox_to_anchor=(0.5, -0.09), frameon=False)
ax.set_title("TMA Level 2 — each L1 bucket split into its measured sub-cause (GLM-5.2 SWE episodes)",
             fontsize=12.5, pad=10)
fig.savefig(f"{OUT}/try_tma_l2.png"); plt.close(fig)
print("wrote try_tma_l2.png")
