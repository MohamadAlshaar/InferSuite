#!/usr/bin/env python3
"""Service CPU attribution donuts (SYSTEM python3): where each pod's CPU time actually goes (perf
'self' %) under a steady RAG tok320 load, split INSIDE inference (vLLM engine) / ROUTING (llm-d envoy)
/ OUTSIDE inference (FastAPI+BGE, Milvus, MongoDB). perf record -e task-clock, per-pod cgroup.

Input:  h100/service/data/<pod>_flat.txt  (perf report, columns: children%  self%  comm  dso  [.] symbol)
Output: h100/service/plots/service_attribution.png
"""
import os, sys, re, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)
NEUTRAL = "#bdbdbd"

PODS = [
    ("vllm",         "vLLM engine",           "INSIDE"),
    ("llmd_gateway", "llm-d gateway", "ROUTING"),
    ("fastapi",      "FastAPI + BGE", "OUTSIDE"),
    ("milvus",       "Milvus", "OUTSIDE"),
    ("mongodb",      "MongoDB", "OUTSIDE"),
]
CLS_COL = {"INSIDE": "#6a51a3", "ROUTING": "#0072B2", "OUTSIDE": "#1b9e77"}

# role -> (color, regex over "dso :: symbol")
# Palette matches the LOCAL during-inference donut (agentic/inference/plots/make_figures.py DESC) so the same
# semantic category has the same colour across the thesis: busy-wait purple #6a51a3, clock-poll #9e9ac8,
# Python #d94801, kernel #cb181d, C library #2171b5.
ROLES = [
    ("CUDA GPU-sync (busy-wait)",  "#6a51a3", re.compile(r"libcuda|libcudart|libc10_cuda|cuEvent|cudaEvent|Synchronize", re.I)),
    ("vDSO clock-poll",            "#9e9ac8", re.compile(r"\[vdso\]")),
    ("MKL/PyTorch GEMM (BGE embed)","#238b45", re.compile(r"libtorch|mkl_|sgemm|dgemm|gemm|libblas|openblas|aten|libc10\b", re.I)),
    ("OpenMP thread-pool (spin)",  "#66c2a4", re.compile(r"libgomp|libiomp|libomp", re.I)),
    ("Python / asyncio",           "#d94801", re.compile(r"libpython|cpython|uvicorn|asyncio|_PyEval|ceval", re.I)),
    ("Envoy routing proxy",        "#6baed6", re.compile(r"envoy", re.I)),
    ("Milvus vector search",       "#E69F00", re.compile(r"milvus|knowhere|faiss", re.I)),
    ("MongoDB engine",             "#41b6c4", re.compile(r"mongod|wiredtiger|WiredTiger", re.I)),
    ("OS kernel (sched/epoll/net)","#cb181d", re.compile(r"\[kernel|finish_task_switch|__schedule|epoll|sys_|softirq|napi|tcp_|sock", re.I)),
    ("C library / allocator",      "#2171b5", re.compile(r"libc\.so|libc-|ld-linux|jemalloc|tcmalloc|\bmalloc|memcpy|memset", re.I)),
]
FLAT = re.compile(r"^\s*[0-9.]+%\s+([0-9.]+)%\s+\S+\s+(\S+)\s+\[.\]\s+(.*)")

def parse_flat(path):
    """return dict role->self%, plus 'unknown'/'other'. Uses the SELF column (2nd %)."""
    out = {r[0]: 0.0 for r in ROLES}; out["unknown"] = 0.0; out["other"] = 0.0
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    for ln in open(path, errors="ignore"):
        m = FLAT.match(ln)
        if not m: continue
        self = float(m.group(1)); dso = m.group(2); sym = m.group(3).strip()
        if self <= 0: continue
        key = f"{dso} :: {sym}"
        if dso == "[unknown]":
            out["unknown"] += self; continue
        for name, col, rx in ROLES:
            if rx.search(key): out[name] += self; break
        else:
            out["other"] += self
    return out

items = []
for key, title, cls in PODS:
    r = parse_flat(os.path.join(DATA, f"{key}_flat.txt"))
    if r is None:
        print(f"skip {key}: empty (≈0 CPU)"); continue
    items.append((key, title, cls, r))
    top = sorted(r.items(), key=lambda x: -x[1])[:3]
    print(f"{key:14} " + " | ".join(f"{k} {v:.0f}%" for k, v in top if v > 1))

if not items:
    print("NO DATA"); sys.exit(0)

colmap = {n: c for n, c, _ in ROLES}; colmap["unknown"] = NEUTRAL; colmap["other"] = "#e0e0e0"
n = len(items); ncol = min(3, n); nrow = math.ceil(n / ncol)
fig = plt.figure(figsize=(4.6*ncol, 4.9*nrow))
gs = fig.add_gridspec(nrow, 2*ncol)
axes = []
for i in range(n):
    r = i // ncol
    n_in_row = min(ncol, n - r*ncol)
    start = (2*ncol - 2*n_in_row) // 2   # center a short last row
    c = i - r*ncol
    axes.append(fig.add_subplot(gs[r, start + 2*c : start + 2*c + 2]))
used = []
for ax, (key, title, cls, r) in zip(axes, items):
    parts = sorted([(k, v) for k, v in r.items() if v > 0.8], key=lambda x: -x[1])
    vals = [v for _, v in parts]; cols = [colmap.get(k, NEUTRAL) for k, _ in parts]
    for k, _ in parts:
        if k not in used and k not in ("unknown", "other"): used.append(k)
    ax.pie(vals, colors=cols, startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.4))
    cum = 0.0; tot = sum(vals) or 1
    for (k, v) in parts:
        if v >= 7:
            a = math.radians(90 - (cum + v/2)/tot*360)
            ax.text(0.78*math.cos(a), 0.78*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=10)
        cum += v
    ax.text(0, 0.08, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=CLS_COL[cls])
    ax.text(0, -0.14, cls, ha="center", va="center", fontsize=7.5, color=CLS_COL[cls])
order = [r[0] for r in ROLES if r[0] in used]
handles = [Patch(color=colmap[k], label=k) for k in order] + [Patch(color=NEUTRAL, label="[unknown] (unresolved leaf)")]
fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=9.0, frameon=False)
fig.suptitle("Service CPU time by software component (RAG tok320)",
             fontsize=13.5, y=1.0)
fig.tight_layout(rect=[0, 0.05, 1, 0.93])
fig.savefig(os.path.join(OUT, "service_attribution.png")); plt.close(fig)
print("fig ->", os.path.join(OUT, "service_attribution.png"))
