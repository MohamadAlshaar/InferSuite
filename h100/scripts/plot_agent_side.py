#!/usr/bin/env python3
"""H100 agent-side campaign figures (SYSTEM python3) — the 32B mirror of the local
local_agents figures, from h100/data_agent_side. Three same-window scopes per workload:
engine (vllm-serve scope), agent harness (agent-* scope), tool container (docker-*).
No TMA figures: the KVM guest has no topdown events (portable suite only).
Palettes identical to the local/thesis figures."""
import os, re, math, csv
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data_agent_side")
OUT  = os.path.join(HERE, "..", "plots", "agent_side"); os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
INSIDE, OUTSIDECOL = "#6a51a3", "#1b9e77"
ENG, HARN, TOOL = "vllm-serve", "agent-", "docker-"
WL = [("swe", "SWE astropy"), ("swe-scikit", "SWE scikit-learn"), ("swe-sympy", "SWE sympy"),
      ("bcb", "BigCodeBench"), ("oc-calendar", "OC calendar"),
      ("oc-web", "OC web-digest"), ("oc-pdf", "OC pdf-digest"), ("oc-crop", "OC image-crop")]

EVS = ("cycles","instructions","branches","branch-misses",
       "mem_load_retired.l1_hit","mem_load_retired.l2_hit","mem_load_retired.l3_hit",
       "mem_load_retired.l3_miss","l1d_pend_miss.pending","l1d_pend_miss.pending_cycles",
       "uops_executed.thread",
       "fp_arith_inst_retired.scalar_single","fp_arith_inst_retired.scalar_double",
       "fp_arith_inst_retired.128b_packed_single","fp_arith_inst_retired.128b_packed_double",
       "fp_arith_inst_retired.256b_packed_single","fp_arith_inst_retired.256b_packed_double",
       "fp_arith_inst_retired.512b_packed_single","fp_arith_inst_retired.512b_packed_double")

def parse_cg(path, cg):
    d = {}
    if not os.path.exists(path): return d
    for ln in open(path, errors="ignore"):
        if cg not in ln: continue
        parts = ln.split()
        if len(parts) < 2: continue
        try: v = float(parts[0].replace(",", ""))
        except ValueError: continue
        for name in EVS:
            if name in ln and name not in d: d[name] = v; break
        m = re.search(r"#\s+([\d.]+) CPUs utilized", ln)
        if m and "cpus" not in d: d["cpus"] = float(m.group(1))
    return d

def elapsed_of(path):
    if not os.path.exists(path): return 0.0
    for ln in open(path, errors="ignore"):
        m = re.search(r"([0-9.]+)\s+seconds time elapsed", ln)
        if m: return float(m.group(1))
    return 0.0

ELEM = {"scalar": 1, "128b": 2, "256b": 4, "512b": 8}
def fp_valid(path, cg):
    if not os.path.exists(path): return False
    txt = open(path, errors="ignore").read()
    if "fp_arith" not in txt or "no access to cgroup" in txt: return False
    rows = [l for l in txt.splitlines() if cg in l]
    return bool(rows) and not any("<not counted>" in l for l in rows)

def sig_of(base, cg):
    g = {grp: parse_cg(os.path.join(base, f"group_{grp}.txt"), cg) for grp in ("core","cache","mlp","fp1","fp2")}
    if not g["cache"].get("cycles"): return None
    fp_ok = fp_valid(os.path.join(base, "group_fp1.txt"), cg) and fp_valid(os.path.join(base, "group_fp2.txt"), cg)
    ins = g["cache"].get("instructions", 1)
    l1 = g["cache"].get("mem_load_retired.l1_hit", 0); l2 = g["cache"].get("mem_load_retired.l2_hit", 0)
    l3 = g["cache"].get("mem_load_retired.l3_hit", 0); mm = g["cache"].get("mem_load_retired.l3_miss", 0)
    tot = (l1+l2+l3+mm) or 1
    flops = scalar = packed = 0.0
    for src in (g["fp1"], g["fp2"]):
        for k, v in src.items():
            if not k.startswith("fp_arith_inst_retired."): continue
            w = next((e for t, e in ELEM.items() if t in k), 1) * 2
            flops += v*w
            if "scalar" in k: scalar += v
            else: packed += v
    secs = elapsed_of(os.path.join(base, "group_fp1.txt")) + elapsed_of(os.path.join(base, "group_fp2.txt"))
    return {"IPC": g["cache"]["instructions"]/g["cache"]["cycles"],
            "L1": l1/tot*100, "L2": l2/tot*100, "L3": l3/tot*100, "MPKI": mm/(ins/1000),
            "AMAT": (l1*5 + l2*15 + l3*50 + mm*250)/tot,
            "MLP": g["mlp"].get("l1d_pend_miss.pending",0)/(g["mlp"].get("l1d_pend_miss.pending_cycles",0) or 1),
            "ILP": g["mlp"].get("uops_executed.thread",0)/(g["mlp"].get("cycles",0) or 1),
            "vec": (packed/((scalar+packed) or 1)*100) if fp_ok else float("nan"),
            "GFLOPs": (flops/(secs or 1)/1e9) if fp_ok else float("nan")}

# ---- Fig 1: two-view CPU donuts (engine vs harness+tool cores, same window) ----
rows = []
for d, lab in WL:
    base = os.path.join(DATA, d)
    core = os.path.join(base, "group_core.txt")
    e = parse_cg(core, ENG).get("cpus")
    h = parse_cg(core, HARN).get("cpus") or 0
    t = parse_cg(core, TOOL).get("cpus") or 0
    if e is not None: rows.append((lab, e, h + t))
if rows:
    nc = (len(rows)+1)//2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.0))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(rows):]: ax.axis("off")
    for ax, (lab, e, o) in zip(axes, rows):
        shares = [e/(e+o)*100, o/(e+o)*100]
        ax.pie([max(s, 1.0) for s in shares], colors=[INSIDE, OUTSIDECOL], startangle=90,
               counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.13, f"{e:.2f} CPUs", ha="center", fontsize=10, color=INSIDE)
        ax.text(0, -0.21, f"{o:.2f} CPUs" if o >= 0.005 else f"{o:.3f} CPUs", ha="center", fontsize=10, color=OUTSIDECOL)
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
        ax.axis("off")
    fig.legend(handles=[Patch(color=INSIDE, label="During inference (vLLM engine)"),
                        Patch(color=OUTSIDECOL, label="Outside inference (agent + tools)")],
               loc="lower center", ncol=2, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUT, "h100_agents_two_view_cpu.png")); plt.close(fig)

# ---- Fig 2: GPU vs CPU wall time ----
GPU_COL, CPU_COL = "#6a51a3", "#1b9e77"
gpu_rows = []
for d, lab in WL:
    p = os.path.join(DATA, d, "gpu_timeline.csv")
    if not os.path.exists(p): continue
    utils = [float(r[1]) for r in csv.reader(open(p)) if r and r[0] != "guard"]
    if len(utils) < 20: continue
    # trim the teardown tail: the sampler follows the DRIVER pid, which can outlive the episode
    # (harness scoring/cleanup) -> a trailing all-idle streak is post-episode, not agent CPU time
    end = len(utils)
    while end > 20 and utils[end-1] < 10: end -= 1
    if len(utils) - end < 6: end = len(utils)   # short tail = legitimate in-episode idle
    utils = utils[:end]
    gpu_rows.append((lab, sum(1 for u in utils if u >= 10)/len(utils)*100))
if gpu_rows:
    nc = (len(gpu_rows)+1)//2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.0))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(gpu_rows):]: ax.axis("off")
    for ax, (lab, g) in zip(axes, gpu_rows):
        ax.pie([g, 100-g], colors=[GPU_COL, CPU_COL], startangle=90, counterclock=False,
               wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.5))
        ax.text(0, 0.13, f"GPU {g:.0f}%", ha="center", fontsize=10, color=GPU_COL)
        ax.text(0, -0.21, f"CPU {100-g:.0f}%", ha="center", fontsize=10, color=CPU_COL)
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
        ax.axis("off")
    fig.legend(handles=[Patch(color=GPU_COL, label="GPU generating"),
                        Patch(color=CPU_COL, label="CPU only (tools, agent, orchestration)")],
               loc="lower center", ncol=2, fontsize=10.5, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.savefig(os.path.join(OUT, "h100_agents_gpu_time.png")); plt.close(fig)

# ---- Fig 3: microarch signature heatmap, 3 scopes ----
COLS = [("IPC","IPC"),("L1","L1 hit %"),("L2","L2 hit %"),("L3","L3 hit %"),("MPKI","LLC-MPKI"),
        ("AMAT","AMAT (cyc)"),("MLP","MLP"),("ILP","ILP"),("vec","vec %FP"),("GFLOPs","GFLOP/s")]
sig_rows = []
eng_sigs = [sig_of(os.path.join(DATA, d), ENG) for d, _ in WL]
eng_sigs = [r for r in eng_sigs if r]
if eng_sigs:
    sig_rows.append(("vLLM engine (during inf.)", {k: float(np.nanmean([r[k] for r in eng_sigs])) for k in eng_sigs[0]}, "eng"))
NO_TOOLCALLS = {"oc-web", "oc-pdf"}   # transcripts: zero tool calls, container = gateway runtime only
for d, lab in WL:
    if d.startswith("oc-"): continue  # OC harness scope = idle run_batch orchestrator (noise)
    r = sig_of(os.path.join(DATA, d), HARN)
    if r: sig_rows.append((f"harness: {lab}", r, "harn"))
for d, lab in WL:
    if d == "bcb" or d.startswith("oc-"): continue
    r = sig_of(os.path.join(DATA, d), TOOL)
    if r: sig_rows.append((f"tools: {lab}", r, "tool"))
for d, lab in WL:
    if not d.startswith("oc-"): continue
    r = sig_of(os.path.join(DATA, d), TOOL)
    suffix = " (no tool calls)" if d in NO_TOOLCALLS else ""
    if r: sig_rows.append((f"agent+tools: {lab}{suffix}", r, "tool"))
if len(sig_rows) > 2:
    Mx = np.array([[r[c[0]] for c in COLS] for _, r, _ in sig_rows], float)
    norm = np.zeros_like(Mx)
    for j in range(Mx.shape[1]):
        col = Mx[:, j]; ok = ~np.isnan(col)
        if not ok.any(): continue
        lo, hi = col[ok].min(), col[ok].max()
        norm[:, j] = 0.5 if hi <= lo else (col-lo)/(hi-lo)
    norm = np.nan_to_num(norm, nan=0.0)
    fig, ax = plt.subplots(figsize=(11.5, 0.42*len(sig_rows)+2.2))
    im = ax.imshow(norm, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    for i, (_, r, _) in enumerate(sig_rows):
        for j, c in enumerate(COLS):
            v = r[c[0]]
            txt = "\u2013" if (isinstance(v, float) and v != v) else (f"{v:.2f}" if v < 10 else f"{v:.0f}")
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if norm[i, j] < 0.6 else "white")
    ax.set_xticks(range(len(COLS))); ax.set_xticklabels([c[1] for c in COLS], rotation=25, ha="right")
    ax.set_yticks(range(len(sig_rows))); ax.set_yticklabels([lab for lab, _, _ in sig_rows], fontsize=9)
    CLS = {"eng": INSIDE, "harn": OUTSIDECOL, "tool": "#00441b"}
    for i, (_, _, c) in enumerate(sig_rows): ax.get_yticklabels()[i].set_color(CLS[c])
    prev = None
    for i, (_, _, c) in enumerate(sig_rows):
        if prev and c != prev: ax.axhline(i-0.5, color="white", linewidth=3)
        prev = c
    ax.set_title("Microarchitectural signature at 32B: engine, agent harness, tool executions (H100)")
    ax.grid(False); fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="per-column min–max (relative)")
    fig.savefig(os.path.join(OUT, "h100_agents_signature.png")); plt.close(fig)

# ---- Fig 4+5: software views (engine donut; outside donuts harness+tool merged by cpus) ----
ROLES = [
    ("GPU busy-wait",        "#6a51a3", re.compile(r"libcuda|libcudart|\[vdso\]", re.I)),
    ("Node.js / V8 (agent)", "#56b4e9", re.compile(r"\bnode\b|libnode|/node$|\bv8\b|\[JIT\]", re.I)),
    ("Python interpreter",   "#d94801", re.compile(r"python3|libpython|\.cpython-", re.I)),
    ("BLAS / OpenMP",        "#238b45", re.compile(r"openblas|libgomp|libtorch|libmkl", re.I)),
    ("C library / loader",   "#2171b5", re.compile(r"libc\.so|ld-linux|libm\.so|libstdc|libz\.|libcrypto|libssl", re.I)),
    ("OS kernel",            "#cb181d", re.compile(r"kallsyms|\[kernel", re.I)),
]
NEUTRAL = "#b3b3b3"
def parse_dso(path):
    out = {r[0]: 0.0 for r in ROLES}; out["other"] = 0.0
    if not os.path.exists(path): return None
    for ln in open(path, errors="ignore"):
        m = re.match(r"\s*[0-9.]+%\s+([0-9.]+)%\s+(\S+)", ln) or re.match(r"\s*([0-9.]+)%\s+(\S+)", ln)
        if not m: continue
        pct, dso = float(m.group(1)), m.group(2)
        if pct <= 0: continue
        for name, col, rx in ROLES:
            if rx.search(dso): out[name] += pct; break
        else: out["other"] += pct
    tot = sum(out.values())
    return {k: v/tot*100 for k, v in out.items()} if tot else None

colmap = {n: c for n, c, _ in ROLES}; colmap["other"] = NEUTRAL
def draw_donut(ax, r, center, ccol, minlab=12):
    parts = [(k, v) for k, v in sorted(r.items(), key=lambda x: -x[1]) if v > 1]
    w, _ = ax.pie([max(v, 1.2) for _, v in parts], colors=[colmap[k] for k, _ in parts],
                  startangle=90, counterclock=False,
                  wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.2))
    for wd, (k, v) in zip(w, parts):
        if v >= minlab:
            a = math.radians((wd.theta1+wd.theta2)/2)
            ax.text(0.79*math.cos(a), 0.79*math.sin(a), f"{v:.0f}%", ha="center", va="center",
                    color="white", fontweight="bold", fontsize=9)
    ax.text(0, 0, center, ha="center", va="center", fontsize=10, fontweight="bold", color=ccol)
    ax.axis("off")
    return {k for k, _ in parts}

panels = []
for d, lab in WL:
    base = os.path.join(DATA, d)
    e = parse_dso(os.path.join(base, "engine_dso.txt"))
    drv = parse_dso(os.path.join(base, "driver_dso.txt")); tl = parse_dso(os.path.join(base, "tool_dso.txt"))
    core = os.path.join(base, "group_core.txt")
    if drv and tl:
        w = [parse_cg(core, HARN).get("cpus", 0) or 0, parse_cg(core, TOOL).get("cpus", 0) or 0]
        tot = (w[0] + w[1]) or 1
        t = {k: (drv.get(k, 0)*w[0] + tl.get(k, 0)*w[1])/tot for k in set(drv) | set(tl)}
    else:
        t = drv or tl
    if e and t: panels.append((lab, e, t))
if panels:
    eng_mean = {}
    for _, e, _t in panels:
        for k, v in e.items(): eng_mean[k] = eng_mean.get(k, 0) + v/len(panels)
    fig, ax = plt.subplots(figsize=(3.6, 3.5))
    used = draw_donut(ax, eng_mean, "vLLM engine", INSIDE)
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=2, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Model side: engine CPU during inference (32B)", fontsize=12.5, y=1.02)
    fig.savefig(os.path.join(OUT, "h100_agents_engine_software.png")); plt.close(fig)

    nc = (len(panels)+1)//2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.4))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(panels):]: ax.axis("off")
    used = set()
    for ax, (lab, _e, t) in zip(axes, panels):
        used |= draw_donut(ax, t, "", OUTSIDECOL)
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Agent machinery at 32B: CPU of the harness and its tools", fontsize=13, y=1.0)
    fig.savefig(os.path.join(OUT, "h100_agents_two_view_software.png")); plt.close(fig)

# ---- Fig 6: delegated work — tool-container records (swe + oc; bcb tools run in-harness,
# covered by the July replay-based tool records) ----
tpanels = []
for d, lab in WL:
    if d in ("oc-web", "oc-pdf"): continue  # zero tool calls: container CPU is runtime, not delegation
    r = parse_dso(os.path.join(DATA, d, "tool_dso.txt"))
    if r: tpanels.append((lab, r))
if tpanels:
    nc = (len(tpanels)+1)//2
    fig, axes = plt.subplots(2, nc, figsize=(2.7*nc, 6.4))
    axes = list(np.atleast_1d(axes).flat)
    for ax in axes[len(tpanels):]: ax.axis("off")
    used = set()
    for ax, (lab, r) in zip(axes, tpanels):
        used |= draw_donut(ax, r, "", "#00441b")
        ax.text(0, -1.3, lab, ha="center", fontweight="bold", fontsize=10.5)
    handles = [Patch(color=colmap[k], label=k) for k, _c, _ in ROLES if k in used]
    if "other" in used: handles.append(Patch(color=NEUTRAL, label="other"))
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5), fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Delegated work at 32B: tool-container CPU by software component", fontsize=13, y=1.0)
    fig.savefig(os.path.join(OUT, "h100_agents_tool_software.png")); plt.close(fig)

print("wrote", OUT)
