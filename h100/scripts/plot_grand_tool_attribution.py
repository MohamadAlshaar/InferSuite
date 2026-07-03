#!/usr/bin/env python3
"""Grand OUTSIDE tool-exec CPU attribution (SYSTEM python3): where the recorded (perf record) TOOL CPU
goes, per task, bucketed by DSO into roles. Stacked horizontal bars, one per task, house style.
Reads perf_dso.txt (perf report --sort=dso) for BCB / SWE / OpenClaw. Writes
h100/plots/grand_tool_attribution.png."""
import os, sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.family":"serif", "font.serif":["DejaVu Serif"], "mathtext.fontset":"dejavuserif",
    "font.size":12, "axes.titlesize":13, "axes.labelsize":12, "xtick.labelsize":10.5, "ytick.labelsize":11,
    "axes.spines.top":False, "axes.spines.right":False, "axes.axisbelow":True,
    "legend.frameon":False, "figure.dpi":150, "savefig.dpi":300, "savefig.bbox":"tight",
})
# DSO -> role buckets (order = legend/stack order); colours are distinct, house-ish
TOOL_ROLES=[
    ("Compiler / build",      "#8c510a", re.compile(r"\bcc1|cc1plus|collect2|\bas\b|/as$|-as$|\bld\b|-ld$|libbfd|libopcodes|gcc|cpp\b", re.I)),
    ("BLAS / OpenMP / Fortran","#238b45", re.compile(r"openblas|libblas|liblapack|libgomp|libmkl|libgfortran|libquadmath|blas", re.I)),
    ("NumPy / SciPy native",  "#e69f00", re.compile(r"_multiarray|umath|cython_|sklearn|scipy|numpy", re.I)),
    ("Python interpreter",    "#d94801", re.compile(r"python3|libpython|/python[0-9]|\.cpython-", re.I)),
    ("Node.js / V8 (agent)",  "#56b4e9", re.compile(r"\bnode\b|libnode|/node$|\bv8\b|electron|\[JIT\]", re.I)),
    ("Browser / Chromium",    "#0072b2", re.compile(r"chrom|headless|blink|libskia|cef|nwjs", re.I)),
    ("C library / loader",    "#2171b5", re.compile(r"libc\.so|ld-linux|libm\.so|libstdc|libpthread|libdl|libz\.", re.I)),
    ("OS kernel",             "#cb181d", re.compile(r"kallsyms|\[kernel|\[vdso\]", re.I)),
]
NEUTRAL="#b3b3b3"
def parse_dso(path):
    out=[]
    if not os.path.exists(path): return out
    for ln in open(path):
        m=re.match(r"\s*([0-9.]+)%\s+(.+?)\s*$", ln)
        if not m: continue
        try: out.append((float(m.group(1)), m.group(2)))
        except ValueError: pass
    return out
def roles(pairs):
    d={r[0]:0.0 for r in TOOL_ROLES}; d["other"]=0.0
    for pct,dso in pairs:
        for name,col,rx in TOOL_ROLES:
            if rx.search(dso): d[name]+=pct; break
        else: d["other"]+=pct
    return d

# (row label, bench, perf_dso path)
SB="#0072B2"; BC="#D55E00"; OC="#009E73"
TASKS=[
    ("code-gen (BCB)",    BC, "h100/data/tool_record/perf_dso.txt"),
    ("astropy (SWE)",     SB, "h100/data_swe/toolrec_astropy__astropy-14096/perf_dso.txt"),
    ("scikit-learn (SWE)",SB, "h100/data_swe/toolrec_scikit-learn__scikit-learn-25232/perf_dso.txt"),
    ("sympy (SWE)",       SB, "h100/data_swe/toolrec_sympy__sympy-14248/perf_dso.txt"),
    ("calendar (OC)",     OC, "h100/data_oc/toolrec_calendar/perf_dso.txt"),
    ("pdf-digest (OC)",   OC, "h100/data_oc/toolrec_pdf/perf_dso.txt"),
    ("web-digest (OC)",   OC, "h100/data_oc/toolrec_arxiv/perf_dso.txt"),
    ("image-crop (OC)",   OC, "h100/data_oc/toolrec_social/perf_dso.txt"),
]
rows=[]; labels=[]; lcols=[]
for lab,c,path in TASKS:
    pr=parse_dso(path)
    if not pr: print(f"{lab}: NO DATA ({path})"); continue
    r=roles(pr); tot=sum(r.values()) or 1
    rows.append({k:100*v/tot for k,v in r.items()}); labels.append(lab); lcols.append(c)
    top=sorted(r.items(),key=lambda x:-x[1])[:3]
    print(f"{lab:20}", " | ".join(f"{k.split(' /')[0].split(' (')[0]} {100*v/tot:.0f}%" for k,v in top if v>1))

if rows:
    import math
    colmap={r[0]:r[1] for r in TOOL_ROLES}; colmap["other"]=NEUTRAL
    ncol=4; nrow=math.ceil(len(labels)/ncol)
    fig,axes=plt.subplots(nrow,ncol,figsize=(3.5*ncol, 3.7*nrow+0.7))
    axes=np.array(axes).reshape(-1)
    used=[]
    for ax,r,lab,lc in zip(axes,rows,labels,lcols):
        items=sorted([(k,v) for k,v in r.items() if v>0.6], key=lambda x:-x[1])
        vals=[v for _,v in items]; cols=[colmap[k] for k,_ in items]
        for k,_ in items:
            if k not in used: used.append(k)
        ax.pie(vals, colors=cols, startangle=90, counterclock=False,
               wedgeprops=dict(width=0.44, edgecolor="white", linewidth=1.4))
        cum=0.0; tot=sum(vals) or 1
        for v in vals:
            if v>=10:
                a=math.radians(90-(cum+v/2)/tot*360)
                ax.text(0.78*math.cos(a),0.78*math.sin(a),f"{v:.0f}%",ha="center",va="center",
                        color="white",fontweight="bold",fontsize=10.5)
            cum+=v
        name,pct=items[0]
        ax.text(0,0.07,lab,ha="center",va="center",fontsize=10,fontweight="bold",color=lc)
        ax.text(0,-0.13,f"{name.split(' /')[0].split(' (')[0]} {pct:.0f}%",ha="center",va="center",fontsize=8.4,color="#555")
    for ax in axes[len(labels):]: ax.axis("off")
    handles=[Patch(color=colmap[k],label=k) for k in [r[0] for r in TOOL_ROLES]+["other"] if k in used]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5,-0.01), fontsize=9.3, frameon=False)
    fig.suptitle("Tool-execution CPU by software component", fontsize=14, y=0.99)
    fig.tight_layout(rect=[0,0.05,1,0.885])
    os.makedirs("h100/plots",exist_ok=True)
    fig.savefig("h100/plots/grand_tool_attribution.png"); plt.close(fig)
    print("fig -> h100/plots/grand_tool_attribution.png")
