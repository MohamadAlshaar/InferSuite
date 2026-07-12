#!/usr/bin/env python3
"""GLM frontier-tier figures — VERIFIED-RESOLVED episodes only (astropy r1+r3,
scikit-learn r1+r3, sympy r1; django excluded: no resolved episode, featured separately
as the non-converging case). Same palette/style family as the thesis. Run with SYSTEM
python3 (matplotlib not in .venv). Output: local_agents/glm_plots/.

Heatmap note: cells are colored by position on ABSOLUTE per-metric reference scales
(stated under each column), not per-column min-max — an IPC of 1.0 is low everywhere,
regardless of what the neighboring workloads scored."""
import os, sys, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from validate_glm_agents import parse_group, load_meta

DATA = os.path.join(HERE, "..", "..", "data")
OUT  = os.path.join(HERE, "..", "..", "glm_plots", "swe"); os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(OUT, "extra"), exist_ok=True)
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 11, "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True,
})
# fixed identity palette (thesis family)
C_HARN, C_TOOL, C_PROXY, C_KERN, C_WAIT = "#6a51a3", "#1b9e77", "#d95f02", "#CC79A7", "#c9c9c9"
L1COLS = [("retiring", "Retiring", "#009E73"), ("fe", "Frontend-bound", "#0072B2"),
          ("bad", "Bad speculation", "#D55E00"), ("be", "Backend-bound", "#E69F00")]
UOPCOLS = [("dsb", "DSB (uop cache)", "#0072B2"), ("mite", "MITE (decode)", "#E69F00"),
           ("ms", "Microcode", "#D55E00"), ("lsd", "LSD (loop)", "#999999")]

# ONE certified episode per task (anchor-documented). Outcomes labeled: the django episode
# never converged within the 40-min budget (5/5 django episodes capped) — featured as the
# honest representative of that task, NOT as a solve.
RESOLVED = [("astropy", "glm_swe_astropy", ["run_1"]),
            ("scikit-learn", "glm_swe_scikit-learn", ["run_1"]),
            ("sympy", "glm_swe_sympy", ["run_1"]),
            ("django", "glm_swe_django-lite", ["run_1"])]
OUTCOME = {"astropy": "resolved", "scikit-learn": "resolved", "sympy": "resolved",
           "django": "capped"}
DISPLAY = {n: (n if OUTCOME[n] == "resolved" else f"{n} ({OUTCOME[n]})") for n, _, _ in RESOLVED}

# Optional side-campaign override (e.g. SWE_long): PLOT_SPEC=<json path> may replace
# data/out/resolved/outcome and stop before the hw-threads/harness sections.
# Without PLOT_SPEC the certified behavior above is untouched.
_SPEC = None
if os.environ.get("PLOT_SPEC"):
    import json as _json
    _SPEC = _json.load(open(os.environ["PLOT_SPEC"]))
    DATA = _SPEC.get("data", DATA)
    OUT = _SPEC.get("out", OUT)
    os.makedirs(OUT, exist_ok=True); os.makedirs(os.path.join(OUT, "extra"), exist_ok=True)
    RESOLVED = [(x[0], x[1], list(x[2])) for x in _SPEC.get("resolved", RESOLVED)]
    OUTCOME = _SPEC.get("outcome", OUTCOME)
    DISPLAY = {n: (n if OUTCOME[n] == "resolved" else f"{n} ({OUTCOME[n]})") for n, _, _ in RESOLVED}

def count_transcript_toolcalls(run_dir):
    """OC episodes carry transcript/chat.jsonl instead of a sweagent traj — count the
    assistant toolCall blocks so call totals stay honest on OC panels."""
    import json as _j, os as _o
    f = f"{run_dir}/transcript/chat.jsonl"
    if not _o.path.exists(f):
        return 0
    n = 0
    for ln in open(f):
        try:
            m = _j.loads(ln)
        except Exception:
            continue
        msg = m.get("message") or {}
        if m.get("type") == "message" and msg.get("role") == "assistant":
            for c in (msg.get("content") or []):
                if "tool" in str(c.get("type", "")).lower():
                    n += 1
    return n

def agent_tool_residency(rd):
    """OC only: fraction of the /agent fence's record samples that belong to tool-class PIDs
    (born pre-move). Used to reallocate the pre-move tool CPU the cgroup billed to /agent, so
    the harness magnitude is the corrected value (not the raw over-count). SWE has no
    lineage.tsv -> returns 0.0 (harness is a clean systemd scope, nothing to correct)."""
    lf = f"{rd}/lineage.tsv"; pf = f"{rd}/scope1_pidtime.txt"
    if not (os.path.exists(lf) and os.path.exists(pf)):
        return 0.0
    tool = set()
    for ln in open(lf):
        p = ln.rstrip("\n").split("\t")
        if len(p) >= 7 and p[5] == "tool":
            try: tool.add(int(p[2]))
            except ValueError: pass
    a_tool = a_all = 0
    for ln in open(pf):
        q = ln.split()
        if not q: continue
        try: pid = int(q[0])
        except ValueError: continue
        a_all += 1
        if pid in tool: a_tool += 1
    return (a_tool / a_all) if a_all else 0.0

def txtcol(hexcol):
    r, g, b = (int(hexcol[i:i+2], 16) for i in (1, 3, 5))
    return "black" if 0.299*r + 0.587*g + 0.114*b > 150 else "white"

STAMP = ("Intel Xeon w5-3425 (Sapphire Rapids, 6-wide) · agents pinned to 20 logical CPUs "
         "(2–11,14–23) @ 3.2 GHz fixed, isolated · model: GLM-5.2 (z.ai API, thinking on)")
def stamp(fig):
    fig.text(0.99, 0.002, STAMP, ha="right", va="bottom", fontsize=7, color="#888888")

# ---------------- data loading -----------------------------------------------------------------
def load_task(cfg, runs):
    """Sum all counted events per role across all windows of the given runs."""
    S = {}   # role -> event -> sum
    for rn in runs:
        rd = f"{DATA}/{cfg}/{rn}"
        meta, roles, _ = load_meta(rd)
        import glob as g
        for w in g.glob(f"{rd}/group_*_w*.txt"):
            gg, why = parse_group(w)
            if why:
                continue
            for cg, ev in gg.items():
                role = roles.get(cg)
                if not role:
                    continue
                d = S.setdefault(role, {})
                for k, v in ev.items():
                    d[k] = d.get(k, 0.0) + v
    return S

def cpustat(rd, i):
    out = []
    try:
        for ln in open(f"{rd}/cpustat_scope{i}.tsv"):
            p = ln.split()
            if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0:
                out.append((float(p[0]), float(p[2])))
    except OSError:
        pass
    return out

def series(rd, i):
    s = cpustat(rd, i)
    t = [(a[0] + b[0]) / 2 for a, b in zip(s, s[1:])]
    v = [max(0.0, (b[1] - a[1]) / max((b[0] - a[0]) * 1e6, 1e-9)) for a, b in zip(s, s[1:])]
    return np.array(t), np.array(v)

def bursts_of(rd):
    t, v = series(rd, 2)
    out, cur = [], None
    for i in range(len(t)):
        if v[i] > 0.3:
            if cur is None: cur = [t[i], t[i], 0.0, 0.0]
            cur[1] = t[i]; cur[2] += v[i] * 0.1; cur[3] = max(cur[3], v[i])
        elif cur and t[i] - cur[1] > 2.0:
            out.append(cur); cur = None
    if cur: out.append(cur)
    return out

TASKS = {}
for name, cfg, runs in RESOLVED:
    TASKS[name] = {"S": load_task(cfg, runs), "cfg": cfg, "runs": runs,
                   "rep": f"{DATA}/{cfg}/run_1"}

# ---------------- derived metrics ---------------------------------------------------------------
def met(S, role):
    d = S.get(role, {})
    I = d.get("instructions", 0) or 1
    cyc = d.get("cycles", 0)
    l1, l2 = d.get("mem_load_retired.l1_hit", 0), d.get("mem_load_retired.l2_hit", 0)
    l3, lm = d.get("mem_load_retired.l3_hit", 0), d.get("mem_load_retired.l3_miss", 0)
    loads = l1 + l2 + l3 + lm
    fp_sc = sum(v for k, v in d.items() if k.startswith("fp_arith") and "scalar" in k)
    fp_pk = sum(v for k, v in d.items() if k.startswith("fp_arith") and "packed" in k)
    dsb, mite = d.get("idq.dsb_uops", 0), d.get("idq.mite_uops", 0)
    ms, lsd = d.get("idq.ms_uops", 0), d.get("lsd.uops", 0)
    ut = (dsb + mite + ms + lsd) or 1
    ck, cu = d.get("cycles:k", 0), d.get("cycles:u", 0)
    tma = {k: d.get(f"topdown-{k}", 0) for k in ("retiring", "bad-spec", "fe-bound", "be-bound")}
    ts = sum(tma.values()) or 1
    pend, pc = d.get("l1d_pend_miss.pending", 0), d.get("l1d_pend_miss.pending_cycles", 0)
    return {
        "IPC": I / cyc if cyc else 0,
        "brMPKI": 1000 * d.get("branch-misses", 0) / I,
        "DSB": 100 * dsb / ut, "MITE": 100 * mite / ut, "MS": 100 * ms / ut, "LSD": 100 * lsd / ut,
        "L1I_MPKI": 1000 * d.get("l2_rqsts.all_code_rd", 0) / I,
        "L1D_MPKI": 1000 * (l2 + l3 + lm) / I,
        "LLC_MPKI": 1000 * lm / I,
        "AMAT": (5*l1 + 15*l2 + 50*l3 + 250*lm) / loads if loads else 0,
        "MLP": pend / pc if pc else 0,
        "kern": 100 * ck / (ck + cu) if ck + cu else 0,
        "vecFP": 100 * fp_pk / (fp_pk + fp_sc) if fp_pk + fp_sc else 0,
        "tma": {"retiring": 100*tma["retiring"]/ts, "bad": 100*tma["bad-spec"]/ts,
                "fe": 100*tma["fe-bound"]/ts, "be": 100*tma["be-bound"]/ts},
    }

def peak_sustained(rd, win=1.0, step=0.25, scope=2):
    """peak occupancy averaged over `win`-second windows (immune to poll-interval jitter,
    which inflates single 0.1 s samples by up to ~5% at saturation)."""
    s = cpustat(rd, scope)
    if len(s) < 3: return 0.0
    T = [r[0] for r in s]; import bisect as _b
    peak, t = 0.0, s[0][0]
    while t + win <= s[-1][0]:
        i = _b.bisect_left(T, t); j = _b.bisect_left(T, t + win)
        if 0 <= i < j < len(s) and s[j][0] > s[i][0]:
            peak = max(peak, (s[j][1]-s[i][1])/1e6/(s[j][0]-s[i][0]))
        t += step
    return min(peak, 20.0)

def defs_footer(fig, extra=""):
    pass  # definitions live in glm_plots/MANIFEST.md — figures carry titles and axis labels only

# ================= Fig 1a/1b: time split + CPU work (split per user review) ====================
D1 = {}
for name, cfg, runs in RESOLVED:
    wall = tool_s = harn_s = 0.0
    cs = np.zeros(3)
    for rn in runs:
        rd = f"{DATA}/{cfg}/{rn}"
        th, vh = series(rd, 1); tt, vt = series(rd, 2)
        if len(tt):
            wall += tt[-1] - tt[0]
            tool_s += float(np.sum(vt > 0.3) * 0.1)
        if len(th):
            harn_s += float(np.sum(vh > 0.05) * 0.1)
        for i in range(3):
            s = cpustat(rd, i + 1)
            if len(s) > 1:
                cs[i] += (s[-1][1] - s[0][1]) / 1e6
        # OC lineage correction: move pre-move tool residency out of harness (cs[0]) into
        # tools (cs[1]); scale harness active-time down likewise. SWE -> frac 0 (no change).
        frac = agent_tool_residency(rd)
        if frac > 0:
            moved = cs[0] * frac
            cs[0] -= moved; cs[1] += moved
            harn_s *= (1 - frac)
    D1[name] = dict(wall=wall, tool_s=tool_s, harn_s=harn_s, cs=cs, resid=agent_tool_residency(f"{DATA}/{cfg}/{runs[0]}"))

# --- Fig 1a: how the episode's wall time passed -------------------------------------------------
fig, axes = plt.subplots(1, len(RESOLVED), figsize=(3.25*len(RESOLVED), 3.6)); axes = np.atleast_1d(axes)
for ax, (name, cfg, runs) in zip(axes, RESOLVED):
    d = D1[name]
    wait_s = max(d["wall"] - d["tool_s"] - d["harn_s"], 0)
    ax.pie([wait_s, d["tool_s"], d["harn_s"]], colors=[C_WAIT, C_TOOL, C_HARN], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           autopct=lambda p: f"{p:.0f}%" if p >= 6 else "", pctdistance=0.78,
           textprops=dict(fontsize=9))
    ax.text(0, 0, f"{d['wall']/60:.0f} min", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_title(f"{name}\n({OUTCOME[name]})", fontsize=11, color="#333333")
    ax.set_aspect("equal")
fig.legend(handles=[Patch(fc=C_WAIT, label="Inference (model round-trip; CPU waits)"),
                    Patch(fc=C_TOOL, label="Tool execution"),
                    Patch(fc=C_HARN, label="Agent harness")],
           ncol=3, loc="lower center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, -0.06))
fig.suptitle("Time split — how each episode's wall-clock passed: agent (harness + tools) vs inference",
             fontsize=13, y=1.04)
fig.savefig(f"{OUT}/glm_time_split.png"); plt.close(fig)

# --- Fig 1b: who did the CPU work, with the measurement table under each donut -------------------
fig, axes = plt.subplots(2, len(RESOLVED), figsize=(3.25*len(RESOLVED), 5.4), height_ratios=[2.0, 1.0], squeeze=False)
for j, (name, cfg, runs) in enumerate(RESOLVED):
    d = D1[name]; cs = d["cs"]
    ax = axes[0, j]
    ax.pie([cs[1], cs[0], cs[2]], colors=[C_TOOL, C_HARN, C_PROXY], startangle=90,
           counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
           autopct=lambda p: f"{p:.0f}%" if p >= 6 else "", pctdistance=0.78,
           textprops=dict(fontsize=9))
    ax.text(0, 0.10, f"{cs.sum():.0f}\ncore-sec", ha="center", va="center", fontsize=10, fontweight="bold")
    ax.text(0, -0.34, f"= {cs.sum()/d['wall']:.2f} cores\navg usage", ha="center", va="center",
            fontsize=7.5, color="#666666")
    ax.set_title(f"{name} ({OUTCOME[name]})", fontsize=10.5, color="#333333")
    ax.set_aspect("equal")
    # the measurement table: activity x (wall, work)
    axT2 = axes[1, j]; axT2.axis("off")
    rows = [("model wait", f"{max(d['wall']-d['tool_s']-d['harn_s'],0)/60:.1f}", "~0"),
            ("tools",      f"{d['tool_s']/60:.1f}", f"{cs[1]:.0f}"),
            ("harness",    f"{d['harn_s']/60:.1f}", f"{cs[0]:.0f}"),
            ("litellm",    "(in waits)", f"{cs[2]:.0f}")]
    tbl = axT2.table(cellText=[[r[0], r[1], r[2]] for r in rows],
                     colLabels=["activity", "wall (min)", "work (core-s)"],
                     cellLoc="center", loc="upper center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.6)
    tbl.scale(1.16, 1.25)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0: cell.set_text_props(color="#555555"); cell.set_facecolor("#f5f5f5")
fig.legend(handles=[Patch(fc=C_TOOL, label="Tool execution"),
                    Patch(fc=C_HARN, label="Agent harness"),
                    Patch(fc=C_PROXY, label="litellm (API proxy to GLM-5.2)")],
           ncol=3, loc="lower center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.075))
fig.suptitle("CPU work — who did the episode's computation (share of core-seconds consumed)",
             fontsize=13, y=1.02)

defs_footer(fig)
fig.savefig(f"{OUT}/glm_cpu_work.png"); plt.close(fig)

# ================= Fig 2: orchestration timelines (HONEST cores-vs-time, no width floor) ========
# Each lane is the raw 10 Hz cpu.stat series drawn as a step-area: x = episode time, height =
# CPU usage (cores) at that instant. NO visibility floor, NO burst-merging, NO threshold — a
# 0.1 s spike renders 0.1 s wide, a 40 s tool burst renders 40 s wide. White = both fences idle
# (model round-trip). Replaces the old burst-bar version whose 3 s floor smeared frequent
# sub-second harness spikes into a misleadingly thick band (measured 2026-07-12).
fig = plt.figure(figsize=(11.8, 2.75*len(RESOLVED)))
outer = fig.add_gridspec(len(RESOLVED), 1, hspace=0.6)
for pnl, (name, _, _) in enumerate(RESOLVED):
    inner = outer[pnl].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.08)
    axT = fig.add_subplot(inner[0]); axH = fig.add_subplot(inner[1], sharex=axT)
    rd = TASKS[name]["rep"]
    tt, vt = series(rd, 2); th, vh = series(rd, 1)
    t0 = min([tt[0]] if len(tt) else [0], [th[0]] if len(th) else [0])[0]
    vt = np.clip(vt, 0, 20.0)   # partition bound: 0.1 s estimator jitter can read ~5% over saturation
    # spike raster: one thin vertical line per 10 Hz sample at its TRUE height. No width floor
    # (a 0.1 s spike stays a spike), but linewidth keeps it visible; dense sustained activity
    # fills into a solid block. This is the honest read of the measured series.
    if len(tt):
        axT.vlines((tt-t0)/60, 0, vt, color=C_TOOL, linewidth=0.6)
    if len(th):
        axH.vlines((th-t0)/60, 0, vh, color=C_HARN, linewidth=0.6)
    tmax = max(2.0, float(vt.max()) if len(vt) else 2.0)
    peak_tool = float(vt.max()) if len(vt) else 0.0
    peak_harn = float(vh.max()) if len(vh) else 0.0
    axT.set_title(f"tool peak {peak_tool:.0f} cores · harness peak {peak_harn:.2f} cores",
                  loc="right", fontsize=8, color="#888888")
    axT.set_ylabel(f"{DISPLAY[name]}\ntool cores", fontsize=9)
    axT.set_ylim(0, tmax*1.1)
    axT.yaxis.set_major_locator(plt.MaxNLocator(4, integer=True))
    axH.set_ylabel("harness\ncores", fontsize=8.5, color=C_HARN)
    # do NOT clip: OC harness is node/V8 (multi-threaded) and bursts past 1 core; SWE harness
    # is Python (GIL) ~1 core. Show the true peak either way.
    axH.set_ylim(0, max(1.25, peak_harn*1.12))
    axH.yaxis.set_major_locator(plt.MaxNLocator(3, integer=True))
    plt.setp(axT.get_xticklabels(), visible=False)
    for a in (axT, axH):
        a.grid(False)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
    if pnl == len(RESOLVED) - 1: axH.set_xlabel("Episode time (minutes)")
fig.legend(handles=[Patch(fc=C_TOOL, label="Tool-fence CPU (cores)"),
                    Patch(fc=C_HARN, label="Harness-fence CPU (cores, own y-scale — note peak per panel)")],
           ncol=2, loc="upper center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, 0.965))
fig.suptitle("Orchestration timeline — measured CPU usage (cores) over time; white = model round-trip (both fences idle)",
             fontsize=12, y=0.995)
fig.supylabel("CPU usage (cores) — instantaneous, from the 10 Hz cpu.stat series", fontsize=9.5, x=0.045)
defs_footer(fig)
fig.savefig(f"{OUT}/glm_timeline.png"); plt.close(fig)

# activity_spans retained for the call-distribution extra figure below
def activity_spans(rd, scope, thr):
    t, v = series(rd, scope)
    spans, cur = [], None
    for i in range(len(t)):
        if v[i] > thr:
            if cur is None: cur = [t[i], t[i], v[i]]
            cur[1] = t[i]; cur[2] = max(cur[2], v[i])
        elif cur and t[i] - cur[1] > 2.0:
            spans.append(cur); cur = None
    if cur: spans.append(cur)
    return t[0] if len(t) else 0, spans

# ================= Fig 3: tool-call structure ===================================================
names = [n for n, _, _ in RESOLVED]
stats = {}
for n, cfg, runs in RESOLVED:
    B = []
    for rn in runs:
        B += bursts_of(f"{DATA}/{cfg}/{rn}")
    durs = [b[1]-b[0] + 0.1 for b in B]; cpus = [b[3] for b in B]   # +0.1: a span covers >=1 sample (10 Hz)
    walls = 0.0; tools = 0.0
    for rn in runs:
        tt, vt = series(f"{DATA}/{cfg}/{rn}", 2)
        if len(tt): walls += tt[-1]-tt[0]; tools += float(np.sum(vt > 0.3) * 0.1)
    import json as _j, glob as _g
    tj = _g.glob(f"{DATA}/{cfg}/{runs[0]}/traj/*/*.traj")
    total = (len(_j.load(open(tj[0])).get("trajectory", [])) if tj
             else count_transcript_toolcalls(f"{DATA}/{cfg}/{runs[0]}"))
    stats[n] = dict(total=total, n=len(B)/len(runs), med=np.median(durs) if durs else 0,
                    mx=max(durs) if durs else 0,
                    peak=min(max(cpus), 20.0) if cpus else 0,
                    sust=max(peak_sustained(f"{DATA}/{cfg}/{rn}") for rn in runs),
                    share=100*tools/walls if walls else 0)
PANELS = [("n", "Tool calls per episode", "{:.0f}"), ("med", "Median call duration (s)", "{:.1f}"),
          ("peak", "Peak CPU usage (cores)", "{:.1f}"), ("share", "Tool-active share of wall (%)", "{:.1f}")]
fig, axes = plt.subplots(1, len(RESOLVED), figsize=(3.1*len(RESOLVED), 2.9)); axes = np.atleast_1d(axes)
for pi, (ax, (k, ttl, fmtv)) in enumerate(zip(axes, PANELS)):
    v = [stats[n][k] for n in names]
    if k == "n":      # light = ALL calls (trajectory); dark = heavy calls (>0.3 cores)
        tv = [stats[n]["total"] for n in names]
        ax.barh(range(len(names)), tv, color=C_TOOL, height=0.55, edgecolor="white", linewidth=0.8)
        ax.barh(range(len(names)), v, color="#0e6b52", height=0.55, edgecolor="white", linewidth=0.8)
        for i, (tot, hv) in enumerate(zip(tv, v)):
            ax.text(tot, i, f" {tot:.0f} ({hv:.0f} heavy)", va="center", fontsize=7.8, color="#333333")
        ax.set_title(ttl + "\nlight = all calls · dark = heavy (>0.3 cores)", fontsize=8.5)
        ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=9.5)
        ax.invert_yaxis(); ax.set_xlim(0, max(tv) * 1.45); ax.grid(axis="x")
        continue
    ax.barh(range(len(names)), v, color=C_TOOL, height=0.55, edgecolor="white", linewidth=0.8)
    if k == "peak":   # two timescales: solid bar = instantaneous (0.1 s); dark tick = sustained (1 s)
        sv = [stats[n]["sust"] for n in names]
        ax.barh(range(len(names)), sv, color="#0e6b52", height=0.55, edgecolor="white", linewidth=0.8)
        for i, (xi, xs) in enumerate(zip(v, sv)):
            ax.text(xi, i, f" {xi:.1f} / {xs:.1f}", va="center", fontsize=8.5, color="#333333")
        ax.set_title(ttl + "\nlight = 0.1 s spike · dark = 1 s sustained", fontsize=8.5)
    else:
        ax.set_title(ttl, fontsize=10)
        for i, x in enumerate(v):
            ax.text(x, i, " " + fmtv.format(x), va="center", fontsize=9, color="#333333")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names if pi == 0 else [], fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlim(0, max(v) * 1.42); ax.grid(axis="x")
fig.suptitle("Tool-call structure of the certified episodes (10 Hz cgroup timelines)", fontsize=12.5, y=1.06)
defs_footer(fig, " Spiky parallelism (astropy: parallel compiler procs ~0.3 s) shows in 0.1 s peaks, not sustained.")
fig.savefig(f"{OUT}/glm_tool_calls.png"); plt.close(fig)

# ================= Fig 4: TMA L1 + uop delivery, per side =======================================
rows = [(f"{DISPLAY[n]} — tool", TASKS[n]["S"], "tool") for n, _, _ in RESOLVED] + \
       [(f"{DISPLAY[n]} — harness", TASKS[n]["S"], "harness") for n, _, _ in RESOLVED]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.6, 0.52*len(rows)+2.2))
Y = np.arange(len(rows))
for ax, which in ((a1, "tma"), (a2, "uop")):
    left = np.zeros(len(rows))
    comps = L1COLS if which == "tma" else UOPCOLS
    for key, lab, col in comps:
        v = []
        for _, S, role in rows:
            m = met(S, role)
            v.append(m["tma"][key] if which == "tma" else m[{"dsb":"DSB","mite":"MITE","ms":"MS","lsd":"LSD"}[key]])
        v = np.array(v)
        ax.barh(Y, v, left=left, color=col, height=0.6, label=lab, edgecolor="white", linewidth=0.8)
        for y, (l, vv) in enumerate(zip(left, v)):
            if vv >= 8:
                ax.text(l+vv/2, y, f"{vv:.0f}", ha="center", va="center",
                        fontsize=8, color=txtcol(col), fontweight="bold")
        left += v
    ax.set_yticks(Y); ax.invert_yaxis(); ax.set_xlim(0, 100); ax.grid(axis="x")
    ax.legend(ncol=2, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), frameon=False)
a1.set_yticklabels([r[0] for r in rows], fontsize=9.5)
a2.set_yticklabels([])
a1.set_xlabel("Pipeline slots (%)"); a2.set_xlabel("uop delivery share (%)")
a1.set_title("TMA Level 1", fontsize=11.5, pad=8); a2.set_title("Frontend uop delivery path", fontsize=11.5, pad=8)
fig.suptitle("Microarchitecture by side — certified GLM-5.2 episodes", fontsize=13, y=1.02)
fig.savefig(f"{OUT}/glm_tma_uop.png"); plt.close(fig)

# ================= Fig 5: signature heatmap on ABSOLUTE reference scales ========================
# (lo, hi) = domain reference range; shade = clamp((v-lo)/(hi-lo)); the printed value is truth.
# Range anchors — hardware ceilings where they exist, else the span the performance
# literature treats as low..severe for the metric (fixed, workload-set-independent):
#   IPC 0..6           six-wide Golden Cove retire width
#   brMPKI 0..20       <1 well-predicted; ~20 severe (15-20 cyc penalty each)
#   L1I MPKI 0..20     >20 = datacenter "instruction-footprint wall" (WSC profiling)
#   L1D MPKI 0..40     >40 = genuinely memory-intensive load streams
#   LLC MPKI 0..10     ~10 = fully memory-bound territory (DRAM traffic)
#   AMAT 5..50         L1-hit latency .. L3-territory
#   MLP 1..16          16 L1D fill buffers
COLS = [("IPC",      "IPC",              0.0, 6.0,  "{:.2f}"),
        ("brMPKI",   "Branch MPKI",      0.0, 20.0, "{:.1f}"),
        ("DSB",      "DSB coverage %",   0.0, 100., "{:.0f}"),
        ("L1I_MPKI", "L1I MPKI",         0.0, 20.0, "{:.1f}"),
        ("L1D_MPKI", "L1D-load MPKI",    0.0, 40.0, "{:.1f}"),
        ("LLC_MPKI", "LLC MPKI",         0.0, 10.0, "{:.2f}"),
        ("AMAT",     "AMAT (cyc)",       5.0, 50.0, "{:.1f}"),
        ("MLP",      "MLP",              1.0, 16.0, "{:.1f}"),
        ("kern",     "OS share %",       0.0, 100., "{:.0f}"),
        ("vecFP",    "Packed %FP",       0.0, 100., "{:.0f}")]
M = np.zeros((len(rows), len(COLS))); TXT = []
for i, (_, S, role) in enumerate(rows):
    m = met(S, role); TXT.append(m)
    for j, (k, *_r) in enumerate(COLS):
        lo, hi = _r[1], _r[2]
        M[i, j] = min(max((m[k]-lo)/(hi-lo), 0), 1)
fig, ax = plt.subplots(figsize=(12.4, 0.52*len(rows)+2.4))
im = ax.imshow(M, aspect="auto", cmap="Purples", vmin=0, vmax=1)
for i in range(len(rows)):
    for j, (k, lab, lo, hi, fv) in enumerate(COLS):
        ax.text(j, i, fv.format(TXT[i][k]), ha="center", va="center", fontsize=8.5,
                color="black" if M[i, j] < 0.55 else "white")
ax.set_xticks(range(len(COLS)))
ax.set_xticklabels([f"{lab}\n[{lo:g}–{hi:g}]" for _, lab, lo, hi, _f in COLS], fontsize=8.2)
ax.tick_params(axis="x", pad=2)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows], fontsize=9.5)
ax.set_title("Per-side signatures on absolute scales — ranges anchored to hardware ceilings where they exist\n"
             "(IPC: 6-wide core; MLP: 16 fill buffers; AMAT: L1-hit 5cyc to L3-territory 50cyc), else fixed empirical references",
             fontsize=11, pad=14)
ax.grid(False)
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
cb.set_label("position on absolute scale (0 = low ref, 1 = high ref)", fontsize=9)
fig.savefig(f"{OUT}/glm_signature.png"); plt.close(fig)

# ================= Fig 6: tool software + kernel share ==========================================
ROLES = [("Python interpreter", "#d94801", re.compile(r"python|\.cpython-|libpython", re.I)),
         ("BLAS / OpenMP",      "#238b45", re.compile(r"openblas|libgomp|libmkl", re.I)),
         ("Search/VCS (grep,git)", "#56b4e9", re.compile(r"^git$|^grep$|^sed$|^find$", re.I)),
         ("C library / loader", "#2171b5", re.compile(r"libc\.so|ld-linux|libm\.so|libstdc|libz\.", re.I)),
         ("OS kernel",          C_KERN,    re.compile(r"kallsyms|\[kernel", re.I))]
def dso_prof(rd, scope):
    out = {r[0]: 0.0 for r in ROLES}; out["other"] = 0.0
    try:
        for ln in open(f"{rd}/scope{scope}_dso.txt"):
            p = ln.split()
            if len(p) < 2 or not p[0].endswith("%"): continue
            pct, dso = float(p[0].rstrip("%")), p[-1]
            for nm, _c, rx in ROLES:
                if rx.search(dso): out[nm] += pct; break
            else: out["other"] += pct
    except OSError: return None
    t = sum(out.values())
    return {k: 100*v/t for k, v in out.items()} if t else None
fig, axes = plt.subplots(2, len(RESOLVED), figsize=(3.4*len(RESOLVED), 7.2), squeeze=False)
cmapd = {n: c for n, c, _ in ROLES}; cmapd["other"] = "#b3b3b3"
for row, (scope, role) in enumerate(((2, "tool"), (1, "harness"))):
    for col, (name, cfg, runs) in enumerate(RESOLVED):
        ax = axes[row, col]
        profs = [p for p in (dso_prof(f"{DATA}/{cfg}/{rn}", scope) for rn in runs) if p]
        mg = {k: np.mean([p[k] for p in profs]) for k in profs[0]} if profs else {}
        ks = [(k, v) for k, v in mg.items() if v > 0.5]
        if ks:
            ax.pie([v for _, v in ks], colors=[cmapd[k] for k, _ in ks], startangle=90,
                   counterclock=False, wedgeprops=dict(width=0.42, edgecolor="white", linewidth=2),
                   autopct=lambda p: f"{p:.0f}%" if p >= 7 else "", pctdistance=0.76,
                   textprops=dict(fontsize=8.5))
        kern_ctr = met(TASKS[name]["S"], role)["kern"]
        ax.text(0, 0, f"kernel\n{kern_ctr:.0f}%", ha="center", va="center", fontsize=9.5, color=C_KERN)
        if row == 0: ax.set_title(f"{name} ({OUTCOME[name]})", fontsize=11)
        ax.set_aspect("equal")
axes[0, 0].set_ylabel("Tool fence", fontsize=11)
axes[1, 0].set_ylabel("Harness fence", fontsize=11)
fig.legend(handles=[Patch(fc=c, label=n) for n, c, _ in ROLES] + [Patch(fc="#b3b3b3", label="other / unresolved")],
           ncol=3, loc="lower center", frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02))
fig.suptitle("What runs inside each fence (record samples; center = kernel share of cycles from counters)",
             fontsize=12, y=0.98)
fig.savefig(f"{OUT}/extra/glm_software_kernel.png"); plt.close(fig)

print("wrote:", sorted(os.listdir(OUT)))

# ================= Fig 7: distribution of tool-action cost (ALL actions, ECDF) ==================
# The timeline shows WHEN; this shows the two populations the bars can't: sub-second reads
# (invisible in the timeline) vs heavy verify runs — from the trajectory's own per-action
# execution times, so every action counts, log axis spans the 3 decades between them.
import json as _json, glob as _g
fig, axes = plt.subplots(1, len(RESOLVED), figsize=(3.65*len(RESOLVED), 3.4), sharey=True); axes = np.atleast_1d(axes)
for ax, (name, cfg, runs) in zip(axes, RESOLVED):
    tj = _g.glob(f"{DATA}/{cfg}/run_1/traj/*/*.traj")
    ets = []
    if tj:
        for s in _json.load(open(tj[0])).get("trajectory", []):
            e = s.get("execution_time", 0)
            if e and e > 0: ets.append(e)
    if not ets: continue
    x = np.sort(np.array(ets)); y = 100.0 * np.arange(1, len(x)+1) / len(x)
    ax.axvspan(0.03, 0.35, color="#eeeeee", zorder=0)
    ax.step(x, y, where="post", color=C_TOOL, lw=2, label="Tool actions (trajectory exec time)")
    # harness processing bursts from the continuous 10 Hz timeline (parse reply, build request)
    _t0, hsp = activity_spans(f"{DATA}/{cfg}/run_1", 1, 0.05)
    hd = np.sort(np.array([max(s[1] - s[0], 0.1) for s in hsp]))
    if len(hd):
        hy = 100.0 * np.arange(1, len(hd)+1) / len(hd)
        ax.step(hd, hy, where="post", color=C_HARN, lw=2, label="Harness bursts (10 Hz timeline)")
    med = np.median(x)
    ax.axvline(med, color="#999999", lw=0.8, ls=":")
    ax.text(med, 4, f" tool median {med:.2f}s", fontsize=8, color="#555555")
    ax.set_xscale("log"); ax.set_xlim(0.03, 300); ax.set_ylim(0, 102)
    ax.set_title(f"{name}\n{len(x)} tool calls · {len(hd)} harness bursts", fontsize=9.5)
    ax.text(0.065, 50, "reads (timeline-invisible)", fontsize=7.5, color="#888888",
            ha="center", va="center", rotation=90)
    ax.set_xlabel("duration (s, log)")
axes[0].set_ylabel("share of units completed (%)")
axes[0].legend(handles=[Patch(fc=C_TOOL, label="Tool calls (traj. exec time)"),
                        Patch(fc=C_HARN, label="Harness bursts (10 Hz)")],
               loc="center right", frameon=False, fontsize=8)
fig.suptitle("Cost distribution of the loop's work units — tool calls vs harness processing bursts (ECDF, certified episodes)",
             fontsize=12, y=1.04)
fig.savefig(f"{OUT}/extra/glm_call_distribution.png"); plt.close(fig)


if _SPEC and _SPEC.get("stop_before_hw"):
    print(f"PLOT_SPEC: stopped before hw-threads/harness sections; figures in {OUT}")
    sys.exit(0)

# ================= Fig 8: hardware-thread occupancy lanes =======================================
# From the full-episode perf records: every sample is tagged with the logical CPU it landed on.
# 20 lanes = the pinned partition, sibling pairs adjacent (cpu N and N+12 share a physical core).
LANES = [c for pair in zip(range(2, 12), range(14, 24)) for c in pair]
fig, axs = plt.subplots(4, 1, figsize=(11.8, 12.2))
for pnl, (name, _, _) in enumerate(RESOLVED):
    ax = axs[pnl]; rd = TASKS[name]["rep"]
    samp = {}
    for sc in (1, 2):
        try:
            arr = np.loadtxt(f"{rd}/scope{sc}_cpulanes.tsv", ndmin=2)
            samp[sc] = arr if arr.size else np.zeros((0, 2))
        except OSError:
            samp[sc] = np.zeros((0, 2))
    tmin = min(a[:, 0].min() for a in samp.values() if len(a))
    tmax = max(a[:, 0].max() for a in samp.values() if len(a))
    nb = 500; edges = np.linspace(0, tmax - tmin, nb + 1); bw = edges[1] - edges[0]
    img = np.ones((len(LANES), nb, 3))
    for sc, col in ((2, (0.11, 0.62, 0.47)), (1, (0.42, 0.32, 0.64))):   # tool green, harness purple
        a = samp[sc]
        for li, cpu in enumerate(LANES):
            ts = a[a[:, 1] == cpu, 0] - tmin
            if not len(ts): continue
            h, _ = np.histogram(ts, bins=edges)
            f = np.clip(h / (99.0 * bw), 0, 1)[:, None]          # busy fraction of that CPU
            img[li] = img[li] * (1 - f) + np.array(col)[None, :] * f
    ax.imshow(img, aspect="auto", interpolation="nearest",
              extent=[0, (tmax - tmin) / 60, len(LANES) - 0.5, -0.5])
    for y in range(1, 10):
        ax.axhline(2 * y - 0.5, color="#dddddd", linewidth=0.5)   # physical-core boundaries
    ax.set_yticks(range(0, 20, 2))
    ax.set_yticklabels([f"{LANES[i]}+{LANES[i+1]}" for i in range(0, 20, 2)], fontsize=7)
    ax.set_ylabel(f"{DISPLAY[name]}\nlogical CPU (SMT pair)", fontsize=9)
    ax.set_title(f"{name} ({OUTCOME[name]})", loc="right", fontsize=8.5, color="#555555")
    if pnl == 3: ax.set_xlabel("Episode time (minutes)")
fig.legend(handles=[Patch(fc="#1b9e77", label="tool fence ran on this logical CPU"),
                    Patch(fc="#6a51a3", label="harness fence ran on this logical CPU")],
           ncol=2, loc="upper center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.905))
fig.suptitle("Which hardware threads ran — per-logical-CPU occupancy (rows = SMT sibling pairs)",
             fontsize=12.5, y=0.94)
defs_footer(fig)
fig.savefig(f"{OUT}/glm_hw_threads.png"); plt.close(fig)
print("wrote glm_hw_threads.png")


# ================= Fig 9: harness anatomy ========================================================
# The tools' behavior is known ground; the HARNESS is the new object. Three questions:
# (a) what does it execute (leaf frames of its fence's full-episode record, categorized)
# (b) how does it perform (microarch card across tasks — invariance = property of the framework)
# (c) how does its cost evolve over an episode (per-burst CPU integral vs time)
import re as _re
CATS = [("interpreter loop", "#6a51a3"), ("Python runtime (other)", "#9e8cc2"),
        ("string building", "#c6b8e0"), ("token counting (tiktoken)", "#1b9e77"),
        ("JSON", "#66c2a5"), ("TLS / hashing", "#CC79A7"), ("OS (syscalls)", "#888888"),
        ("other", "#d9d9d9")]
def classify(sym, dso):
    if _re.match(r"^[0-9a-f]{8,}", sym) or "kallsyms" in dso: return "OS (syscalls)"
    if "tiktoken" in dso: return "token counting (tiktoken)"
    if "_json" in dso: return "JSON"
    if "ssl" in dso or "crypto" in dso or "_hashlib" in dso or "blake" in dso: return "TLS / hashing"
    if "_PyEval_EvalFrameDefault" in sym: return "interpreter loop"
    if _re.search(r"[Uu]nicode|memmove|memcpy|strlen|strcmp|memset", sym): return "string building"
    if "python3" in dso or "lib-dynload" in dso: return "Python runtime (other)"
    return "other"

shares = {}
for name, _, _ in RESOLVED:
    rd = TASKS[name]["rep"]
    agg = {c: 0 for c, _ in CATS}; tot = 0
    for ln in open(f"{rd}/scope1_leaf.txt"):
        m = _re.match(r"\s*(\d+)\s+\t?\s*(.+?)\s+\((.+)\)\s*$", ln)
        if not m: continue
        n, sym, dso = int(m.group(1)), m.group(2), m.group(3)
        agg[classify(sym, dso)] += n; tot += n
    shares[name] = {c: 100.0 * v / tot for c, v in agg.items()}

fig = plt.figure(figsize=(12.6, 10.6))
grid = fig.add_gridspec(3, 1, height_ratios=[1.15, 0.85, 1.25], hspace=0.62, top=0.84, bottom=0.07)
fig.legend(handles=[Patch(fc=col, label=c) for c, col in CATS], ncol=4, fontsize=8,
           frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.935))

# (a) what it executes
axA = fig.add_subplot(grid[0])
names9 = [n for n, _, _ in RESOLVED]
left = np.zeros(len(names9))
for c, col in CATS:
    v = np.array([shares[n][c] for n in names9])
    axA.barh(range(len(names9)), v, left=left, color=col, height=0.62,
             edgecolor="white", linewidth=0.8, label=c)
    for i, (l, x) in enumerate(zip(left, v)):
        if x >= 6: axA.text(l + x/2, i, f"{x:.0f}%", ha="center", va="center", fontsize=8,
                            color="white" if c in ("interpreter loop", "OS (syscalls)") else "#333333")
    left += v
axA.set_yticks(range(len(names9))); axA.set_yticklabels(names9, fontsize=9.5)
axA.invert_yaxis(); axA.set_xlim(0, 100); axA.grid(False)
axA.set_xlabel("share of harness CPU samples (%)", fontsize=9)
axA.set_title("(a) what the harness executes — leaf frames of its full-episode record", loc="left",
              fontsize=10.5, pad=10)

# (b) how it performs — same card across tasks = framework property, not task property
def dsb_share(S):
    ut = S["idq.dsb_uops"] + S["idq.mite_uops"] + S["idq.ms_uops"] + S.get("lsd.uops", 0)
    return 100 * S["idq.dsb_uops"] / ut
CARD = [("IPC", lambda S: S["instructions"]/S["cycles"], 4),
        ("uop-cache (DSB) %", dsb_share, 100),
        ("OS share %", lambda S: 100*S["cycles:k"]/(S["cycles:k"]+S["cycles:u"]), 100),
        ("sustained peak (cores)", None, 1.25)]
axsB = grid[1].subgridspec(1, 4, wspace=0.35)
for k, (ttl, fn, hi) in enumerate(CARD):
    ax = fig.add_subplot(axsB[k])
    v = [peak_sustained(TASKS[n]["rep"], scope=1) if fn is None else fn(TASKS[n]["S"]["harness"]) for n in names9]
    ax.bar(range(len(names9)), v, color=C_HARN, width=0.6, edgecolor="white")
    ax.set_xticks(range(len(names9))); ax.set_xticklabels([DISPLAY[n] for n in names9], fontsize=7.5, rotation=20)
    ax.set_title(ttl, fontsize=9)
    ax.set_ylim(0, hi)
    for i, x in enumerate(v):
        ax.text(i, x, f"{x:.2f}" if hi < 10 else f"{x:.0f}", ha="center", va="bottom", fontsize=7.5)
    if k == 0: ax.set_ylabel("(b) performance card\n(harness fence)", fontsize=9)
    ax.grid(axis="y")

# (c) cost dynamics — CPU consumed per harness burst over the episode
axsC = grid[2].subgridspec(1, 4, wspace=0.3)
for k, name in enumerate(names9):
    ax = fig.add_subplot(axsC[k])
    rd = TASKS[name]["rep"]
    t, v = series(rd, 1)
    bursts9, cur = [], None
    for i in range(len(t)):
        if v[i] > 0.05:
            if cur is None: cur = [t[i], t[i], 0.0]
            cur[1] = t[i]; cur[2] += v[i] * 0.1
        elif cur and t[i] - cur[1] > 2.0:
            bursts9.append(cur); cur = None
    if cur: bursts9.append(cur)
    t0 = t[0] if len(t) else 0
    xs = [(b[0]-t0)/60 for b in bursts9]; ys = [max(b[2], 0.004) for b in bursts9]
    ax.scatter(xs, ys, s=7, color=C_HARN, alpha=0.55, edgecolors="none")
    if len(xs) > 8:
        order = np.argsort(xs); xa = np.array(xs)[order]; ya = np.array(ys)[order]
        w = max(5, len(xa)//8)
        med = [np.median(ya[max(0, i-w):i+w]) for i in range(len(xa))]
        ax.plot(xa, med, color="#3d2a66", linewidth=1.6)
    ax.set_title(f"{DISPLAY[name]} ({len(bursts9)} bursts)", fontsize=9)
    ax.set_yscale("log"); ax.set_ylim(0.003, 30)
    if k == 0: ax.set_ylabel("(c) CPU per harness burst\n(core-seconds, log)", fontsize=9)
    else: ax.set_yticklabels([])
    ax.set_xlabel("episode time (min)", fontsize=8); ax.grid(True, alpha=0.4)

fig.suptitle("Harness anatomy — the agent framework as a workload: what it runs, how it performs, what it costs over time",
             fontsize=12.5, y=0.985)
defs_footer(fig, " OS share = CPU time spent inside the operating system doing system calls for this fence"
            " (network/disk/wakeups). Burst = contiguous harness activity >0.05 cores; cost = its usage integral.")
fig.savefig(f"{OUT}/glm_harness_anatomy.png"); plt.close(fig)
print("wrote glm_harness_anatomy.png")
