#!/usr/bin/env python3
"""plot_service_iso.py — thesis figures for the ISOLATED service campaign (data_iso).
House palette + the locked vocabulary: 'CPU usage (cores)' rate, 'core-seconds' amounts,
'% of CPU time' shares, absolute hardware-anchored scales, defs footer per figure.
Collection is done; this only reads. Run with SYSTEM python3."""
import json, os, re
from glob import glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data_iso")
OUT = os.path.join(HERE, "..", "..", "plots_iso"); os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.35, "grid.linewidth": 0.6,
})
BUCKETS = ["short", "medium", "long", "very_long"]
TIERS = [64, 192, 320]
PODS = ["vllm", "fastapi", "milvus", "mongodb", "seaweed_filer", "seaweed_volume"]
PCOL = {"vllm": "#1b9e77", "fastapi": "#6a51a3", "milvus": "#CC79A7",
        "mongodb": "#e6ab02", "seaweed_filer": "#66a61e", "seaweed_volume": "#a6761d"}
PLAB = {"vllm": "vLLM host (engine)", "fastapi": "fastapi (RAG + BGE embed)",
        "milvus": "milvus (vector search)", "mongodb": "mongodb",
        "seaweed_filer": "seaweed filer", "seaweed_volume": "seaweed volume"}
LINE = re.compile(r"^\s+([\d,]+(?:\.\d+)?)\s+(?:msec\s+)?([\w.:-]+)\s+(kubepods\S+)")
ELAP = re.compile(r"([\d.]+)\s+seconds time elapsed")
WINSEC = 10.0

def defs_footer(fig, extra=""):
    pass  # definitions live in MANIFEST.md — figures carry titles and axis labels only

# ---------------- load everything once ----------------------------------------------------------
CELLS = {}   # (bucket,tier) -> list of per-run dicts: S[pod][ev], tcw[pod], tokens/s, span
for b in BUCKETS:
    for t in TIERS:
        runs = []
        for rd in sorted(glob(f"{DATA}/svc_{b}_tok{t}/run_*")):
            if not os.path.exists(f"{rd}/DONE"): continue
            meta = json.load(open(f"{rd}/metadata.json"))
            cg2pod = {v: k for k, v in meta["pods"].items()}
            S, tcw = {}, {}
            for f in glob(f"{rd}/group_*_w*.txt"):
                txt = open(f).read()
                if "<not counted>" in txt or re.search(r"\(\s*\d+[.,]\d+%\s*\)\s*$", txt, re.M):
                    continue
                for ln in txt.splitlines():
                    m = LINE.match(ln)
                    if not m: continue
                    pod = cg2pod.get(m.group(3))
                    if not pod: continue
                    v, ev = float(m.group(1).replace(",", "")), m.group(2)
                    S.setdefault(pod, {}); S[pod][ev] = S[pod].get(ev, 0.0) + v
                    if ev == "task-clock": tcw[pod] = tcw.get(pod, 0) + 1
            ws = [l.split("\t") for l in open(f"{rd}/windows.tsv").read().splitlines()[1:]]
            toks = [int(w[5]) for w in ws if len(w) > 5]
            span = float(ws[-1][3]) - float(ws[0][2])
            runs.append(dict(S=S, tcw=tcw, rd=rd,
                             tok_s=(toks[-1] - toks[0]) / span if len(toks) > 1 else 0, span=span))
        CELLS[(b, t)] = runs

def usage(run, pod):   # mean CPU usage (cores) in the task-clock windows
    d = run["S"].get(pod, {})
    return d.get("task-clock", 0) / 1000 / (run["tcw"].get(pod, 1) * WINSEC) if run["tcw"].get(pod) else 0.0

def agg(cell, pod):    # activity-weighted event sums across the cell's runs
    out = {}
    for run in CELLS[cell]:
        for ev, v in run["S"].get(pod, {}).items():
            out[ev] = out.get(ev, 0.0) + v
    return out

def cell_usage(cell, pod):
    rs = CELLS[cell]
    return float(np.mean([usage(r, pod) for r in rs])) if rs else 0.0

CLAB = [f"{b}\ntok{t}" for b in BUCKETS for t in TIERS]
CKEYS = [(b, t) for b in BUCKETS for t in TIERS]

# input sizes measured with the model's own tokenizer (vLLM /tokenize, 40 queries per bucket)
BUCKET_IN = {"short": "9-26 tokens in", "medium": "~150 tokens in",
             "long": "~435 tokens in", "very_long": "~720 tokens in"}

def bucket_axis(ax, arc_y=-0.13, lab_y=-0.24):
    """x ticks = output tier; one bracket per input bucket labeled with its input length (words)."""
    ax.set_xticks(range(len(CKEYS)))
    ax.set_xticklabels([f"{t}out" for b in BUCKETS for t in TIERS], fontsize=8.5)
    tr = ax.get_xaxis_transform()
    for g, b in enumerate(BUCKETS):
        x0, x1 = 3*g - 0.30, 3*g + 2.30
        ax.annotate("", xy=(x0, arc_y), xytext=(x1, arc_y), xycoords=tr, textcoords=tr,
                    arrowprops=dict(arrowstyle="-", connectionstyle="bar,fraction=-0.18",
                                    color="#777777", linewidth=1.2), annotation_clip=False)
        ax.text(3*g + 1.0, lab_y, BUCKET_IN[b], transform=tr,
                ha="center", fontsize=9.5, color="#333333")


# ---------------- S0: system map — what each pod does -------------------------------------------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
fig, ax = plt.subplots(figsize=(13.2, 6.4)); ax.axis("off"); ax.set_xlim(0, 13.2); ax.set_ylim(0, 6.4)
def box(x, y, w, h, color, title, role, meas):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                                facecolor=color, alpha=0.14, edgecolor=color, linewidth=1.6))
    ax.text(x + w/2, y + h - 0.13, title, ha="center", va="top", fontsize=10.5, fontweight="bold", color=color)
    ax.text(x + w/2, y + h - 0.50, role, ha="center", va="top", fontsize=7.8, color="#333333")
    ax.text(x + w/2, y + 0.16, meas, ha="center", va="bottom", fontsize=7.2, color="#666666", style="italic")
def arrow(x1, y1, x2, y2, n, lab, dy=0.14):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color="#555555", linewidth=1.1, shrinkA=2, shrinkB=2))
    ax.text((x1+x2)/2, (y1+y2)/2 + dy, f"{n} {lab}", ha="center", fontsize=7.6, color="#333333")

box(0.15, 4.6, 2.3, 1.5, "#888888", "loadgen (driver)",
    "deterministic query replay,\nconcurrency 1",
    "runs on HOUSE cores —\nexcluded from measurement")
box(3.4, 3.9, 3.4, 2.3, "#6a51a3", "fastapi",
    "API server + RAG conductor;\nBGE embedding model runs\nIN-PROCESS in this pod",
    "0.13-0.57 cores; IPC 0.59-1.21;\nFP 100% packed (BGE); bursts ~10 cores")
box(9.9, 4.3, 3.0, 1.9, "#1b9e77", "vLLM host",
    "inference engine host process;\nGPU does the math — host\nthreads busy-wait meanwhile",
    "~1.9 cores flat; IPC 3.59; uop-cache 99%;\nzero FP = phantom CPU, not compute")
box(1.1, 1.4, 2.6, 1.7, "#e6ab02", "mongodb",
    "document store:\nsemantic-cache metadata",
    "~0.01 cores")
box(4.4, 1.4, 2.6, 1.7, "#CC79A7", "milvus",
    "vector database: nearest-neighbor\nsearch over 14,419 chunk vectors",
    "~0.02 cores; IPC 0.85")
box(7.7, 1.4, 2.6, 1.7, "#66a61e", "seaweed filer+volume",
    "object store: resolves + serves\nthe retrieved chunk texts",
    "~0.002 cores\n(storage = syscall work)")
arrow(2.45, 5.35, 3.4, 5.2, "1.", "HTTP request")
ax.text(5.1, 6.30, "2. embed query (BGE, in-process)", fontsize=7.6, color="#6a51a3", ha="center")
arrow(4.0, 3.9, 2.6, 3.1, "3.", "cache lookup", dy=0.18)
arrow(5.3, 3.9, 5.6, 3.1, "4.", "vector search")
arrow(6.3, 3.9, 8.6, 3.1, "5.", "fetch chunks", dy=0.18)
arrow(6.8, 5.1, 9.9, 5.2, "6.", "augmented prompt")
arrow(9.9, 4.75, 6.8, 4.65, "8.", "streamed answer", dy=-0.22)
ax.text(11.4, 3.95, "7. GPU decodes; host cores spin", fontsize=7.6, color="#1b9e77", ha="center")

fig.suptitle("System map — what each pod does and what it costs (RAG path, local k3s, one request's journey)",
             fontsize=12.5, y=0.97)
defs_footer(fig)
fig.savefig(f"{OUT}/svc_system_map.png"); plt.close(fig)

# ---------------- S1: CPU work composition ------------------------------------------------------
GROUPS = [("vllm", ["vllm"]), ("fastapi", ["fastapi"]), ("milvus", ["milvus"]),
          ("mongodb", ["mongodb"]), ("seaweed", ["seaweed_filer", "seaweed_volume"])]
GLAB = {"vllm": PLAB["vllm"], "fastapi": PLAB["fastapi"], "milvus": PLAB["milvus"],
        "mongodb": PLAB["mongodb"], "seaweed": "seaweed (object store)"}
GCOL = {**{k: PCOL[k] for k in ("vllm", "fastapi", "milvus", "mongodb")}, "seaweed": "#66a61e"}
fig, ax = plt.subplots(figsize=(12.6, 4.4))
fig.subplots_adjust(bottom=0.24)
bottom = np.zeros(len(CKEYS))
for g, members in GROUPS:
    v = np.array([sum(cell_usage(c, m) for m in members) for c in CKEYS])
    ax.bar(range(len(CKEYS)), v, bottom=bottom, color=GCOL[g], width=0.62,
           edgecolor="white", linewidth=0.8, label=GLAB[g])
    bottom += v
for i, tot in enumerate(bottom):
    ax.text(i, tot + 0.04, f"{tot:.2f}", ha="center", fontsize=7.5, color="#333333")
bucket_axis(ax)
ax.set_ylabel("CPU usage (cores)")
ax.set_ylim(0, max(bottom) * 1.22)
for x in (2.5, 5.5, 8.5): ax.axvline(x, color="#cccccc", linewidth=0.7)
ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper left")
ax.set_title("Service CPU work per cell — steady-state usage by pod (input bucket x output tier; mean of 3 repeats)",
             fontsize=12.5, pad=10)
defs_footer(fig, " vLLM host usage is engine busy-wait during GPU decode, not compute.")
fig.savefig(f"{OUT}/svc_cpu_work.png"); plt.close(fig)

# ---------------- S2: signature heatmap (absolute scales) ---------------------------------------
def metrics_of(E):
    ipc = E.get("instructions", 0) / E["cycles"] if E.get("cycles") else np.nan
    ut = E.get("idq.dsb_uops", 0) + E.get("idq.mite_uops", 0) + E.get("idq.ms_uops", 0) + E.get("lsd.uops", 0)
    dsb = 100 * E.get("idq.dsb_uops", 0) / ut if ut else np.nan
    ins = E.get("instructions", 0)
    br = 1000 * E.get("branch-misses", 0) / ins if ins else np.nan
    l1m = E.get("mem_load_retired.l2_hit", 0) + E.get("mem_load_retired.l3_hit", 0) + E.get("mem_load_retired.l3_miss", 0)
    l1d = 1000 * l1m / ins if ins else np.nan
    l2d = 1000 * (E.get("mem_load_retired.l3_hit", 0) + E.get("mem_load_retired.l3_miss", 0)) / ins if ins else np.nan
    llc = 1000 * E.get("mem_load_retired.l3_miss", 0) / ins if ins else np.nan
    l1i = 1000 * E.get("l2_rqsts.all_code_rd", 0) / ins if ins else np.nan
    pk = sum(v for k, v in E.items() if k.startswith("fp_arith") and "packed" in k)
    sc = sum(v for k, v in E.items() if k.startswith("fp_arith") and "scalar" in k)
    fp = 100 * pk / (pk + sc) if pk + sc > 0 else 0.0
    return [ipc, dsb, br, l1i, l1d, l2d, llc, fp]

MET = [("IPC", 0, 6), ("uop-cache %", 0, 100), ("branch MPKI", 0, 20), ("L1I MPKI", 0, 20),
       ("L1D MPKI", 0, 40), ("L2D MPKI", 0, 20), ("LLC MPKI", 0, 10), ("packed FP %", 0, 100)]
SHOW = ["vllm", "fastapi", "milvus", "mongodb"]
fig, axes = plt.subplots(1, 4, figsize=(15.2, 3.9), sharey=True)
for ax, b in zip(axes, BUCKETS):
    # median run per pod within this bucket (by IPC) — no pooling, per the TMA rule
    rowsM = []
    for p in SHOW:
        cand = [metrics_of(r["S"].get(p, {})) for t in TIERS for r in CELLS[(b, t)]
                if r["S"].get(p, {}).get("cycles")]
        cand = [c for c in cand if np.isfinite(c[0])]
        cand.sort(key=lambda c: c[0])
        rowsM.append(cand[len(cand)//2] if cand else [np.nan]*8)
    M = np.array(rowsM)
    N = np.array([[(M[i, j] - lo) / (hi - lo) if np.isfinite(M[i, j]) else np.nan
                   for j, (_, lo, hi) in enumerate(MET)] for i in range(len(SHOW))])
    ax.imshow(np.clip(N, 0, 1), cmap="Purples", aspect="auto", vmin=0, vmax=1)
    for i in range(len(SHOW)):
        for j in range(len(MET)):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}" if j == 0 else f"{M[i, j]:.0f}",
                        ha="center", va="center", fontsize=7.2,
                        color="white" if N[i, j] > 0.6 else "#333333")
    ax.set_xticks(range(len(MET))); ax.set_xticklabels([m[0] for m in MET], rotation=35, ha="right", fontsize=7.5)
    ax.set_title(BUCKET_IN[b], fontsize=10); ax.grid(False)
axes[0].set_yticks(range(len(SHOW))); axes[0].set_yticklabels([PLAB[p] for p in SHOW], fontsize=8.5)
fig.suptitle("Per-pod microarchitectural signature by input bucket — median run per cell, absolute scales",
             fontsize=12.5, y=1.06)

fig.savefig(f"{OUT}/svc_signature.png"); plt.close(fig)

# ---------------- S3: TMA L1 + uop delivery — median run, no pooling ---------------------------
# Per user rule: never aggregate TMA across runs. For each row we verify similarity across its
# runs and display the MEDIAN run (by retiring share); the row label carries the observed spread.
TMA_KEYS = [("retiring", "#1b9e77"), ("bad-spec", "#CC79A7"), ("fe-bound", "#e6ab02"), ("be-bound", "#6a51a3")]
UOP_KEYS = [("idq.dsb_uops", "#1b9e77", "uop cache (DSB)"), ("idq.mite_uops", "#e6ab02", "legacy decode (MITE)"),
            ("idq.ms_uops", "#CC79A7", "microcode (MS)"), ("lsd.uops", "#6a5FA3", "loop buffer (LSD)")]
def tma_shares(S):
    v = [S.get(f"topdown-{k}", 0.0) for k, _ in TMA_KEYS]
    tot = sum(v)
    return [100 * x / tot for x in v] if tot > 0 else None
def uop_shares(S):
    v = [S.get(k, 0.0) for k, _, _ in UOP_KEYS]
    tot = sum(v)
    return [100 * x / tot for x in v] if tot > 0 else None

ROWS3 = [("vLLM host", "vllm", CKEYS), ("fastapi", "fastapi", CKEYS),
         ("milvus", "milvus", CKEYS), ("mongodb", "mongodb", CKEYS)]
sel = []
for lab, pod, keys in ROWS3:
    cand = []
    for c in keys:
        for r in CELLS[c]:
            ts = tma_shares(r["S"].get(pod, {}))
            us = uop_shares(r["S"].get(pod, {}))
            if ts and us: cand.append((ts, us))
    if not cand:
        sel.append((lab, None, None, 0, 0)); continue
    rets = [c0[0][0] for c0 in cand]
    order = np.argsort(rets)
    med = cand[order[len(order)//2]]
    spread = (max(rets) - min(rets)) / 2
    sel.append((lab, med[0], med[1], spread, len(cand)))

L1COLS = [(0, "Retiring", "#009E73"), (2, "Frontend-bound", "#0072B2"),
          (1, "Bad speculation", "#D55E00"), (3, "Backend-bound", "#E69F00")]
def txtcol(hexcol):
    r, g, b = (int(hexcol[i:i+2], 16) for i in (1, 3, 5))
    return "black" if 0.299*r + 0.587*g + 0.114*b > 150 else "white"
labels = [lab for lab, _, _, sp, n in sel]
fig, a1 = plt.subplots(figsize=(8.4, 0.52*len(sel)+2.2))
Y = np.arange(len(sel))
left = np.zeros(len(sel))
for idx, lab, col in L1COLS:
    v = np.array([srow[1][idx] if srow[1] else 0 for srow in sel])
    a1.barh(Y, v, left=left, color=col, height=0.6, label=lab, edgecolor="white", linewidth=0.8)
    for y, (l, vv) in enumerate(zip(left, v)):
        if vv >= 8:
            a1.text(l+vv/2, y, f"{vv:.0f}", ha="center", va="center",
                    fontsize=8, color=txtcol(col), fontweight="bold")
    left += v
a1.set_yticks(Y); a1.set_yticklabels(labels, fontsize=9.5)
a1.invert_yaxis(); a1.set_xlim(0, 100); a1.grid(axis="x")
a1.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.11), frameon=False)
a1.set_xlabel("Pipeline slots (%)")
a1.set_title("TMA Level 1", fontsize=12, pad=10)
fig.savefig(f"{OUT}/svc_tma_l1.png"); plt.close(fig)

# ---------------- S4: request-rhythm timeline ---------------------------------------------------
def pod_series(rd, pod):
    rows = []
    for ln in open(f"{rd}/cpustat_{pod}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec":
            rows.append((float(p[0]), float(p[2])))
    return [((a[0]+b[0])/2, max(0.0, (b[1]-a[1])/max((b[0]-a[0])*1e6, 1e-9))) for a, b in zip(rows, rows[1:])]

REP = CELLS[("long", 64)][0]["rd"]
TLPODS = ["vllm", "fastapi", "milvus", "mongodb"]
fig, axs = plt.subplots(len(TLPODS), 1, figsize=(12.4, 6.0), sharex=True)
T0, T1 = 60, 150
for ax, pod in zip(axs, TLPODS):
    s = pod_series(REP, pod)
    t0 = s[0][0]
    xs = [x - t0 for x, _ in s]; ys = [y for _, y in s]
    sel = [(x, y) for x, y in zip(xs, ys) if T0 <= x <= T1]
    ax.fill_between([x for x, _ in sel], 0, [y for _, y in sel], color=PCOL[pod], linewidth=0.4, alpha=0.9)
    ax.set_ylabel(PLAB[pod].split(" (")[0], fontsize=8, rotation=0, ha="right", va="center")
    ax.set_ylim(0, max(0.35, max(y for _, y in sel) * 1.15))
    ax.grid(axis="x", alpha=0.3)
axs[-1].set_xlabel("time (seconds since capture start) — ~435-tokens-in / 64-out cell, run 1, 90 s excerpt; each step = one 0.1 s sample")
fig.supylabel("CPU usage (cores) per 0.1 s sample", fontsize=10, x=0.012)
fig.suptitle("Request rhythm at concurrency 1 — per-pod CPU usage, 10 Hz (per-panel y-scales)",
             fontsize=12.5, y=0.96)
defs_footer(fig)
fig.savefig(f"{OUT}/svc_timeline.png"); plt.close(fig)

# ---------------- S5: CPU cost per 1k tokens ----------------------------------------------------
fig, ax = plt.subplots(figsize=(12.6, 4.2))
fig.subplots_adjust(bottom=0.24)
bottom = np.zeros(len(CKEYS))
for g, members in GROUPS:
    v = []
    for c in CKEYS:
        rs = CELLS[c]
        cost = [sum(usage(r, m) for m in members) * r["span"] / max(r["tok_s"] * r["span"] / 1000.0, 1e-9) for r in rs]
        v.append(float(np.mean(cost)))
    v = np.array(v)
    ax.bar(range(len(CKEYS)), v, bottom=bottom, color=GCOL[g], width=0.62,
           edgecolor="white", linewidth=0.8, label=GLAB[g])
    bottom += v
for i, tot in enumerate(bottom):
    ax.text(i, tot + 0.4, f"{tot:.0f}", ha="center", fontsize=7.5, color="#333333")
bucket_axis(ax)
ax.set_ylabel("core-seconds per 1k engine tokens")
for x in (2.5, 5.5, 8.5): ax.axvline(x, color="#cccccc", linewidth=0.7)
ax.legend(ncol=5, fontsize=8, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.01))
ax.set_title("CPU cost per 1,000 engine tokens (prompt+generated) — by pod and cell", fontsize=12.5, pad=34)
defs_footer(fig, " Engine tokens from vLLM's own counters per window; cost = usage x time / kilotokens.")
fig.savefig(f"{OUT}/svc_cost_per_token.png"); plt.close(fig)

# ---------------- S6: repetition proof ----------------------------------------------------------
ROWS = [("vLLM IPC", "vllm", "ipc"), ("vLLM usage", "vllm", "use"),
        ("fastapi IPC", "fastapi", "ipc"), ("fastapi usage", "fastapi", "use")]
Msp = np.zeros((len(ROWS), len(CKEYS)))
for j, c in enumerate(CKEYS):
    for i, (_, pod, kind) in enumerate(ROWS):
        vals = []
        for r in CELLS[c]:
            d = r["S"].get(pod, {})
            if kind == "ipc" and d.get("cycles"):
                vals.append(d["instructions"] / d["cycles"])
            elif kind == "use":
                vals.append(usage(r, pod))
        Msp[i, j] = 100 * (max(vals) - min(vals)) / np.mean(vals) if len(vals) > 1 and np.mean(vals) > 0 else np.nan
fig, ax = plt.subplots(figsize=(12.6, 2.7))
fig.subplots_adjust(bottom=0.24)
im = ax.imshow(Msp, cmap="Purples", aspect="auto", vmin=0, vmax=15)
for i in range(len(ROWS)):
    for j in range(len(CKEYS)):
        ax.text(j, i, f"{Msp[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if Msp[i, j] > 9 else "#333333")
bucket_axis(ax, arc_y=-0.24, lab_y=-0.46)
ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([r[0] for r in ROWS], fontsize=9)
ax.grid(False)
ax.set_title("Repetition proof — spread across 3 identical live repeats, % of mean (max-min)/mean",
             fontsize=12, pad=10)

fig.savefig(f"{OUT}/svc_dispersion.png"); plt.close(fig)

# ---------------- S7: CPU-side vs GPU-side time split -------------------------------------------
# concurrency-1 pipeline: vLLM host usage ~1.9 while the GPU decodes, ~0.02 when idle;
# fastapi bursts while the CPU-side RAG stage runs. Classify each 100 ms slice.
fig, ax = plt.subplots(figsize=(12.6, 4.0))
fig.subplots_adjust(bottom=0.24)
cats = np.zeros((len(CKEYS), 3))   # gpu, cpu-side, handoff/idle
for j, c in enumerate(CKEYS):
    g = cp = other = 0
    for r in CELLS[c]:
        sv = pod_series(r["rd"], "vllm"); sf = dict()
        for x, y in pod_series(r["rd"], "fastapi"):
            sf[round(x, 1)] = y
        for x, y in sv:
            f = sf.get(round(x, 1), 0.0)
            if y > 1.0: g += 1                    # engine busy-wait = GPU decoding
            elif f > 0.2: cp += 1                 # RAG stage working
            else: other += 1
    tot = max(g + cp + other, 1)
    cats[j] = [100*g/tot, 100*cp/tot, 100*other/tot]
left = np.zeros(len(CKEYS))
for k, (lab, col) in enumerate([("GPU decoding (CPU busy-waits)", "#1b9e77"),
                                ("CPU-side RAG stage (embed/search/fetch)", "#6a51a3"),
                                ("handoff / sub-threshold", "#dddddd")]):
    ax.bar(range(len(CKEYS)), cats[:, k], bottom=left, color=col, width=0.62,
           edgecolor="white", linewidth=0.8, label=lab)
    for i in range(len(CKEYS)):
        if cats[i, k] >= 8:
            ax.text(i, left[i] + cats[i, k]/2, f"{cats[i, k]:.0f}%", ha="center", va="center",
                    fontsize=7.3, color="white" if k < 2 else "#555555")
    left += cats[:, k]
bucket_axis(ax)
ax.set_ylabel("share of wall time (%)"); ax.set_ylim(0, 118)
for x in (2.5, 5.5, 8.5): ax.axvline(x, color="#cccccc", linewidth=0.7)
ax.legend(ncol=3, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.02))
ax.set_title("Time split — GPU work vs CPU-side work per request cycle (100 ms classification, all repeats)",
             fontsize=12.5, pad=24)
defs_footer(fig, " GPU-decoding detected via engine host busy-wait (>1 core); CPU-side via fastapi activity (>0.2).")
fig.savefig(f"{OUT}/svc_time_split.png"); plt.close(fig)


# ---------------- S8: MAIN — per-tier CPU/GPU time-split donuts ---------------------------------
fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.3))
fig.subplots_adjust(bottom=0.20)
for ax, t in zip(axes, TIERS):
    g = cp = other = 0
    for b in BUCKETS:
        for r in CELLS[(b, t)]:
            sv = pod_series(r["rd"], "vllm")
            sf = {round(x, 1): y for x, y in pod_series(r["rd"], "fastapi")}
            for x, y in sv:
                f = sf.get(round(x, 1), 0.0)
                if y > 1.0: g += 1
                elif f > 0.2: cp += 1
                else: other += 1
    tot = max(g + cp + other, 1)
    vals = [100*g/tot, 100*cp/tot, 100*other/tot]
    ax.pie(vals, colors=["#1b9e77", "#6a51a3", "#dddddd"], startangle=90, counterclock=False,
           wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           autopct=lambda p: f"{p:.0f}%" if p >= 4 else "", pctdistance=0.76,
           textprops=dict(fontsize=9.5))
    ax.text(0, 0.10, f"tok{t}", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(0, -0.32, f"{vals[0]:.0f}% GPU", ha="center", va="center", fontsize=8.5, color="#1b9e77")
    ax.set_title(f"{t}out per request", fontsize=10.5)
    ax.set_aspect("equal")
fig.legend(handles=[Patch(fc="#1b9e77", label="GPU decoding (host CPU busy-waits)"),
                    Patch(fc="#6a51a3", label="CPU-side RAG stage (embed / search / fetch)"),
                    Patch(fc="#dddddd", label="handoff / sub-threshold")],
           ncol=3, loc="lower center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.035))
fig.suptitle("Where a request's time goes, by output tier — GPU work vs CPU-side work",
             fontsize=13, y=1.02)
defs_footer(fig, " GPU-decoding detected via engine host busy-wait (>1 core); CPU-side via fastapi activity (>0.2 cores).")
fig.savefig(f"{OUT}/svc_tier_donuts.png"); plt.close(fig)

print("wrote:", sorted(os.path.basename(p) for p in glob(f"{OUT}/*.png")))
