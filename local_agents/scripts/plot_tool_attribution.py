#!/usr/bin/env python3
"""Tool-execution CPU by software component, LOCAL run (SYSTEM python3) — the main-text analogue of
the H100 tool-attribution donuts, from perf record task-clock over the replayed Sonnet-run tool
work (deterministic, no model). DSO self%-weighted; same role buckets and palette as
h100/scripts/plot_grand_tool_attribution.py (thesis-unified)."""
import os, re, math
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
SB, BC, OC = "#0072B2", "#D55E00", "#009E73"
ROLES = [
    ("Compiler / build",       "#8c510a", re.compile(r"\bcc1|cc1plus|collect2|/as$|-as$|-ld$|libbfd|libopcodes|gcc|cpp\b", re.I)),
    ("BLAS / OpenMP / Fortran","#238b45", re.compile(r"openblas|libblas|liblapack|libgomp|libmkl|libgfortran|libquadmath|blas", re.I)),
    ("NumPy / SciPy native",   "#e69f00", re.compile(r"_multiarray|umath|cython_|sklearn|scipy|numpy", re.I)),
    ("Python interpreter",     "#d94801", re.compile(r"python3|libpython|/python[0-9]|\.cpython-", re.I)),
    ("Node.js / V8 (agent)",   "#56b4e9", re.compile(r"\bnode\b|libnode|/node$|\bv8\b|\[JIT\]", re.I)),
    ("C library / loader",     "#2171b5", re.compile(r"libc\.so|ld-linux|libm\.so|libstdc|libpthread|libdl|libz\.|libcrypto|libssl", re.I)),
    ("OS kernel",              "#cb181d", re.compile(r"kallsyms|\[kernel|\[vdso\]", re.I)),
]
NEUTRAL = "#b3b3b3"
TASKS = [
    ("bcb_tool",              "code-gen (BCB)",     BC),
    ("swe_tool_astropy",      "astropy (SB)",       SB),
    ("swe_tool_scikit-learn", "scikit-learn (SB)",  SB),
    ("swe_tool_sympy",        "sympy (SB)",         SB),
    ("oc_tool_calendar",      "calendar (OC)",      OC),
    ("oc_tool_web-digest",    "web-digest (OC)",    OC),
    ("oc_tool_pdf-digest",    "pdf-digest (OC)",    OC),
    ("oc_tool_image-crop",    "image-crop (OC)",    OC),
]

def parse_dso(path):
    out = {r[0]: 0.0 for r in ROLES}; out["other"] = 0.0
    for ln in open(path, errors="ignore"):
        m = re.match(r"\s*([0-9.]+)%\s+(\S+)", ln)
        if not m: continue
        pct = float(m.group(1)); dso = m.group(2)
        if pct <= 0: continue
        for name, col, rx in ROLES:
            if rx.search(dso): out[name] += pct; break
        else: out["other"] += pct
    return out

colmap = {n: c for n, c, _ in ROLES}; colmap["other"] = NEUTRAL
ncol = 4; nrow = (len(TASKS) + ncol - 1) // ncol
fig, axes = plt.subplots(nrow, ncol, figsize=(4.3*ncol, 4.7*nrow))
axes = list(axes.flat)
used = set()
for ax, (key, title, tcol) in zip(axes, TASKS):
    r = parse_dso(os.path.join(DATA, key, "perf_dso.txt"))
    tot = sum(r.values()) or 1
    parts = [(k, v/tot*100) for k, v in sorted(r.items(), key=lambda x: -x[1]) if v > 0.4]
    for k, _ in parts: used.add(k)
    vis = [max(v, 0.8) for _, v in parts]
    w, _ = ax.pie(vis, colors=[colmap[k] for k, _ in parts], startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
    for wd, (k, v) in zip(w, parts):
        if v >= 10:
            a = math.radians((wd.theta1 + wd.theta2) / 2)
            ax.text(0.79*math.cos(a), 0.79*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9.5)
    ax.text(0, 0.0, title, ha="center", va="center", fontsize=10, fontweight="bold", color=tcol)
handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Delegated work: CPU of the tool executions themselves", fontsize=13.5, y=1.02)
fig.savefig(os.path.join(OUT, "tool_attribution.png")); plt.close(fig)
print("wrote", os.path.join(OUT, "tool_attribution.png"))
