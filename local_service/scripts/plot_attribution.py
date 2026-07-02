#!/usr/bin/env python3
"""Software-attribution donuts for the LOCAL service run (SYSTEM python3) — where each pod's CPU
time goes (perf record task-clock self%), per tier + the idle control. Same role buckets and
palette as h100/service/scripts/plot_service_attribution.py (thesis-unified: busy-wait purple,
clock-poll light purple, Python orange-red, kernel red, C library blue, GEMM green)."""
import os, re, glob, math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT  = os.path.join(HERE, "..", "plots"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
NEUTRAL = "#bdbdbd"
PODS = [
    ("vllm",          "vLLM engine",   "INSIDE"),
    ("llmd_gateway",  "llm-d gateway", "ROUTING"),
    ("fastapi",       "FastAPI + BGE", "OUTSIDE"),
    ("milvus",        "Milvus",        "OUTSIDE"),
    ("mongodb",       "MongoDB",       "OUTSIDE"),
    ("seaweed_filer", "Seaweed filer", "OUTSIDE"),
    ("seaweed_volume","Seaweed volume","OUTSIDE"),
]
CLS_COL = {"INSIDE": "#6a51a3", "ROUTING": "#0072B2", "OUTSIDE": "#1b9e77"}
ROLES = [
    ("CUDA GPU-sync (busy-wait)",   "#6a51a3", re.compile(r"libcuda|libcudart|libc10_cuda|cuEvent|cudaEvent|Synchronize", re.I)),
    ("vDSO clock-poll",             "#9e9ac8", re.compile(r"\[vdso\]")),
    ("MKL/PyTorch GEMM (BGE embed)","#238b45", re.compile(r"libtorch|mkl_|sgemm|dgemm|gemm|libblas|openblas|aten|libc10\b", re.I)),
    ("OpenMP thread-pool (spin)",   "#66c2a4", re.compile(r"libgomp|libiomp|libomp", re.I)),
    ("Python / asyncio",            "#d94801", re.compile(r"libpython|cpython|uvicorn|asyncio|_PyEval|ceval", re.I)),
    ("Envoy routing proxy",         "#6baed6", re.compile(r"envoy", re.I)),
    ("Milvus vector search",        "#E69F00", re.compile(r"milvus|knowhere|faiss", re.I)),
    ("MongoDB engine",              "#41b6c4", re.compile(r"mongod|wiredtiger|WiredTiger", re.I)),
    ("SeaweedFS (Go)",              "#8c6bb1", re.compile(r"weed|seaweed", re.I)),
    ("OS kernel (sched/epoll/net)", "#cb181d", re.compile(r"\[kernel|finish_task_switch|__schedule|epoll|sys_|softirq|napi|tcp_|sock", re.I)),
    ("C library / allocator",       "#2171b5", re.compile(r"libc\.so|libc-|ld-linux|jemalloc|tcmalloc|\bmalloc|memcpy|memset", re.I)),
]
FLAT = re.compile(r"^\s*[0-9.]+%\s+([0-9.]+)%\s+\S+\s+(\S+)\s+\[.\]\s+(.*)")

def parse_flat(path):
    out = {r[0]: 0.0 for r in ROLES}; out["unknown"] = 0.0; out["other"] = 0.0
    if not os.path.exists(path) or os.path.getsize(path) == 0: return None, 0
    n = 0
    for ln in open(path, errors="ignore"):
        m = FLAT.match(ln)
        if not m: continue
        self_pct = float(m.group(1)); dso = m.group(2); sym = m.group(3).strip()
        if self_pct <= 0: continue
        n += 1
        key = f"{dso} :: {sym}"
        if dso == "[unknown]": out["unknown"] += self_pct; continue
        for name, col, rx in ROLES:
            if rx.search(key): out[name] += self_pct; break
        else: out["other"] += self_pct
    return out, n

colmap = {n: c for n, c, _ in ROLES}; colmap["unknown"] = NEUTRAL; colmap["other"] = "#e0e0e0"

for tier_dir in sorted(glob.glob(os.path.join(DATA, "tok*")) + [os.path.join(DATA, "idle_control")]):
    tier = os.path.basename(tier_dir)
    items = []
    for key, title, cls in PODS:
        r, n = parse_flat(os.path.join(tier_dir, f"{key}_flat.txt"))
        if r is None or sum(r.values()) < 1: continue
        items.append((key, title, cls, r))
    if not items: print(f"skip {tier}"); continue
    ncol = min(4, len(items)); nrow = math.ceil(len(items) / ncol)
    fig = plt.figure(figsize=(4.3*ncol, 4.6*nrow))
    gs = fig.add_gridspec(nrow, 2*ncol)
    axes = []
    for i in range(len(items)):
        r = i // ncol
        n_in_row = min(ncol, len(items) - r*ncol)
        start = (2*ncol - 2*n_in_row) // 2   # center a short last row
        c = i - r*ncol
        axes.append(fig.add_subplot(gs[r, start + 2*c : start + 2*c + 2]))
    used = set()
    for ax, (key, title, cls, r) in zip(axes, items):
        tot = sum(r.values()) or 1
        parts = [(k, v/tot*100) for k, v in sorted(r.items(), key=lambda x: -x[1]) if v > 0]
        for k, _ in parts: used.add(k)
        vis = [max(v, 0.8) for _, v in parts]
        w, _t = ax.pie(vis, colors=[colmap[k] for k, _ in parts], startangle=90, counterclock=False,
                       wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
        for wd, (k, v) in zip(w, parts):
            if v >= 12:
                a = math.radians((wd.theta1 + wd.theta2) / 2)
                ax.text(0.79*math.cos(a), 0.79*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                        color="white", fontweight="bold", fontsize=8.5)
        ax.text(0, 0.08, title, ha="center", fontweight="bold", fontsize=9, color=CLS_COL[cls])
        ax.text(0, -0.16, cls, ha="center", fontsize=7, color=CLS_COL[cls])
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=colmap["other"], label="other"))
    if "unknown" in used: handles.append(Patch(color=colmap["unknown"], label="[unknown]"))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.04))
    label = "idle control" if tier == "idle_control" else f"RAG {tier}"
    fig.suptitle(f"Service CPU time by software component ({label})", fontsize=13.5, y=1.01)
    d = os.path.join(OUT, tier); os.makedirs(d, exist_ok=True)
    fig.savefig(os.path.join(d, "service_attribution.png")); plt.close(fig)
    print(f"{tier}: attribution donuts ({len(items)} pods)")
print("done ->", OUT)
