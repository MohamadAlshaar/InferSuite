#!/usr/bin/env python3
"""Exploratory figure set (adopted 2026-07-17) — NOT thesis-featured; banked under
<set>/plots/extra/. Driven by PLOT_SPEC like the main plotter; without it, defaults to
SWE_clean. Figures:

  glm_avg_util.png      average cores over the full episode, stacked by fence, % of partition
  glm_turn_latency.png  per-turn end-to-end duration, cumulative turn count per task
  glm_turn_seq.png      every turn as one bar, stacked model-wait / tool / harness
  glm_tool_tail.png     tool-call latency CCDF (SWE: traj execution_time; OC: fence
                        activity spans — no per-call wall log exists, stated on-figure)
  glm_ctx_switches.png  context switches per CPU-second per fence (priv windows)
  glm_os_share.png      kernel share of cycles per fence (values_dump)
  glm_burst_decomp.png  heavy tool bursts: distinct cores x duty (99 Hz lanes; clock
                        aligned to cpu.stat by activity cross-correlation)

Turn boundaries are derived from harness-fence activations (cluster gap 2 s) — the
harness fires once per turn; the derived count is printed next to the logged turn count.
Run with SYSTEM python3."""
import json, glob, os, re
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
if os.environ.get("PLOT_SPEC"):
    SPEC = json.load(open(os.environ["PLOT_SPEC"]))
    DATA, OUT = SPEC["data"], SPEC["out"]
else:
    BASE = os.path.join(HERE, "..", "..", "SWE_clean")
    SPEC = {"resolved": None}
    DATA, OUT = f"{BASE}/data", f"{BASE}/plots"
    SPEC = json.load(open(f"{BASE}/plot_spec.json"))
    DATA, OUT = SPEC["data"], SPEC["out"]
TASKS = [(x[0], f"{DATA}/{x[1]}/{x[2][0]}") for x in SPEC["resolved"]]
VD = json.load(open(f"{OUT}/values_dump.json"))
XOUT = f"{OUT}/extra"
os.makedirs(XOUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "font.size": 11,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.grid": True, "grid.color": "#cccccc", "grid.linewidth": 0.5, "grid.alpha": 0.6,
    "axes.axisbelow": True})
C_TOOL, C_HARN, C_PROXY, C_WAIT = "#1b9e77", "#6a51a3", "#d95f02", "#c9c9c9"
TCOL = dict(zip([t[0] for t in TASKS], ["#6a51a3", "#1b9e77", "#d95f02", "#0072B2"]))
NCORES, HZ = 20, 99.0

def cpustat(rd, sc):
    rows = []
    for ln in open(f"{rd}/cpustat_scope{sc}.tsv"):
        p = ln.split()
        if len(p) >= 3 and p[1] == "usage_usec" and int(p[2]) >= 0:
            rows.append((float(p[0]), int(p[2])))
    return rows

def act_bursts(rows, thr, gap=0.4):
    out, cur = [], None
    for (t0, u0), (t1, u1) in zip(rows, rows[1:]):
        r = (u1 - u0) / 1e6 / max(t1 - t0, 1e-6)
        if r > thr:
            if cur and t0 - cur[1] < gap:
                cur = [cur[0], t1, cur[2] + (u1 - u0) / 1e6, max(cur[3], r)]
            else:
                if cur: out.append(cur)
                cur = [t0, t1, (u1 - u0) / 1e6, r]
    if cur: out.append(cur)
    return out

def turn_starts(rd):
    """SWE: harness-fence activations cluster once per turn (CPython fires per turn;
    validated against the logged step count). OC: the node harness has continuous
    background activity, so activation clustering over-segments — use the transcript's
    per-assistant-message timestamps instead (wall clock, same epoch as cpu.stat)."""
    if glob.glob(f"{rd}/traj/**/*.traj", recursive=True):
        B = act_bursts(cpustat(rd, 1), 0.02)
        if not B: return []
        starts = [B[0][0]]
        for b in B[1:]:
            if b[0] - starts[-1] > 2.0: starts.append(b[0])
        return starts
    f = f"{rd}/transcript/chat.jsonl"
    if not os.path.exists(f): return []
    from datetime import datetime
    starts = []
    for ln in open(f):
        try:
            m = json.loads(ln)
        except Exception:
            continue
        if m.get("type") == "message" and (m.get("message") or {}).get("role") == "assistant":
            ts = m.get("timestamp")
            if ts:
                starts.append(datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    return sorted(starts)

def key_of(lab):
    return [k for k in VD if k == lab][0]

# ---------------- 1: average utilization ----------------
names = [t[0] for t in TASKS]
wall = np.array([VD[key_of(n)]["wall_min"] * 60 for n in names])
harn = np.array([VD[key_of(n)]["cs"][0] for n in names]) / wall
tool = np.array([VD[key_of(n)]["cs"][1] for n in names]) / wall
prox = np.array([VD[key_of(n)]["cs"][2] for n in names]) / wall
tot = harn + tool + prox
fig, ax = plt.subplots(figsize=(8.6, 4.2))
x = np.arange(len(names))
ax.bar(x, tool, color=C_TOOL, width=0.55, label="Tool execution", edgecolor="white")
ax.bar(x, harn, bottom=tool, color=C_HARN, width=0.55, label="Agent harness", edgecolor="white")
ax.bar(x, prox, bottom=tool + harn, color=C_PROXY, width=0.55, label="litellm (API proxy)", edgecolor="white")
for i in range(len(names)):
    ax.text(i, tot[i] + max(tot) * 0.03, f"{tot[i]:.2f} cores\n= {100*tot[i]/NCORES:.1f}% of partition",
            ha="center", fontsize=9, color="#333333")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylabel("average CPU usage (cores)")
ax.set_ylim(0, max(tot) * 1.4)
sec = ax.secondary_yaxis("right", functions=(lambda v: 100 * v / NCORES, lambda p: p * NCORES / 100))
sec.set_ylabel("utilization of the 20-core partition (%)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Average CPU utilization over the full episode", pad=12)
fig.savefig(f"{XOUT}/glm_avg_util.png"); plt.close(fig)
print("wrote glm_avg_util.png")

# ---------------- 2+3: per-turn latency (cumulative) + turn sequence ----------------
TURNS = {}
for lab, rd in TASKS:
    st = turn_starts(rd)
    TURNS[lab] = (st, np.diff(st) if len(st) > 1 else np.array([]))
    logged = VD[key_of(lab)].get("turns")
    print(f"  turn derivation {lab}: derived {max(len(st)-1,0)} vs logged {logged}")

fig, axes = plt.subplots(len(TASKS), 1, figsize=(9.2, 7.6), sharex=True)
for ax, (lab, rd) in zip(np.atleast_1d(axes), TASKS):
    d = TURNS[lab][1]
    if not len(d): ax.set_visible(False); continue
    xs = np.sort(d); ys = np.arange(1, len(xs) + 1)
    ax.step(xs, ys, where="post", color="#6a51a3", lw=2)
    ax.fill_between(xs, ys, step="post", color="#6a51a3", alpha=0.12)
    ax.set_ylim(0, len(xs) * 1.3)
    med, p95 = np.median(d), np.percentile(d, 95)
    ax.axvline(med, color="#222222", lw=1.6)
    ax.axvline(p95, color="#222222", lw=1.1, linestyle=(0, (4, 3)))
    ax.text(0.99, 0.30, f"{lab} — {len(d)} turns\nmedian {med:.1f}s (solid) · p95 {p95:.0f}s (dashed)",
            transform=ax.transAxes, ha="right", fontsize=9, color="#333333")
    ax.set_ylabel("turns completed")
    ax.set_xscale("log")
    ax.set_xticks([2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["2", "5", "10", "20", "50", "100"])
np.atleast_1d(axes)[-1].set_xlabel("turn end-to-end duration (seconds, log scale)")
fig.suptitle("Per-turn end-to-end latency — cumulative", y=0.995, fontsize=13)
fig.subplots_adjust(hspace=0.14)
fig.savefig(f"{XOUT}/glm_turn_latency.png"); plt.close(fig)
print("wrote glm_turn_latency.png")

def overlap(spans, a, b):
    s = 0.0
    for x0, x1 in spans:
        if x1 <= a or x0 >= b: continue
        s += min(x1, b) - max(x0, a)
    return s

fig, axes = plt.subplots(len(TASKS), 1, figsize=(12.4, 9.0))
for ax, (lab, rd) in zip(np.atleast_1d(axes), TASKS):
    st, d = TURNS[lab]
    if not len(d): ax.set_visible(False); continue
    tool_sp = [(b[0], b[1]) for b in act_bursts(cpustat(rd, 2), 0.005)]
    harn_sp = [(b[0], b[1]) for b in act_bursts(cpustat(rd, 1), 0.02)]
    n = len(d)
    tl = np.array([overlap(tool_sp, st[i], st[i + 1]) for i in range(n)])
    hn = np.array([overlap(harn_sp, st[i], st[i + 1]) for i in range(n)])
    wait = np.clip(d - tl - hn, 0, None)
    xs = np.arange(1, n + 1)
    ax.bar(xs, wait, width=1.0, color=C_WAIT, label="model round-trip")
    ax.bar(xs, tl, bottom=wait, width=1.0, color=C_TOOL, label="tool execution")
    ax.bar(xs, hn, bottom=wait + tl, width=1.0, color=C_HARN, label="harness")
    k = max(n // 12, 3)
    roll = np.convolve(d, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(1, len(roll) + 1) + k // 2, roll, color="#222222", lw=1.6, label="rolling mean")
    ax.set_ylabel("turn duration (s)")
    ax.text(0.995, 0.93, f"{lab} — {n} turns · median {np.median(d):.1f}s",
            transform=ax.transAxes, ha="right", va="top", fontsize=9.5, color="#333333")
    ax.set_xlim(0, n + 1)
np.atleast_1d(axes)[-1].set_xlabel("turn number")
np.atleast_1d(axes)[0].legend(ncol=4, fontsize=8.6, frameon=False, loc="upper left")
fig.suptitle("Every turn: end-to-end duration and where it went", y=0.995, fontsize=13)
fig.subplots_adjust(hspace=0.28)
fig.savefig(f"{XOUT}/glm_turn_seq.png"); plt.close(fig)
print("wrote glm_turn_seq.png")

# ---------------- 4: tool latency long tail ----------------
fig, ax = plt.subplots(figsize=(8.8, 4.8))
used_spans = False
for lab, rd in TASKS:
    tj = glob.glob(f"{rd}/traj/**/*.traj", recursive=True)
    if tj:
        et = np.array([float(s.get("execution_time") or 0)
                       for s in json.load(open(tj[0]))["trajectory"]])
        et = et[et > 0]; src = "calls"
    else:
        et = np.array([b[1] - b[0] for b in act_bursts(cpustat(rd, 2), 0.005)])
        et = et[et > 0]; src = "activity spans"; used_spans = True
    if not len(et): continue
    xs = np.sort(et); cc = 100 * (1 - np.arange(1, len(xs) + 1) / len(xs))
    top5 = 100 * xs[-max(len(xs) // 20, 1):].sum() / xs.sum()
    ax.plot(xs, np.maximum(cc, 100 / len(xs)), color=TCOL[lab], lw=2,
            label=f"{lab}: {len(xs)} {src} · med {np.median(xs):.2f}s · max {xs[-1]:.0f}s · top 5% = {top5:.0f}% of tool time")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30, 100]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30", "100"])
ax.set_yticks([1, 3, 10, 30, 100]); ax.set_yticklabels(["1", "3", "10", "30", "100"])
ax.set_xlabel("tool execution time (seconds, log)")
ax.set_ylabel("share at least this long (%, log)")
ax.legend(fontsize=7.8, frameon=True, facecolor="white", edgecolor="#dddddd", loc="lower left")
ttl = "Tool latency — the long tail"
if used_spans:
    ttl += "  (no per-call log in this set: fence activity spans)"
ax.set_title(ttl, pad=10, fontsize=12)
fig.savefig(f"{XOUT}/glm_tool_tail.png"); plt.close(fig)
print("wrote glm_tool_tail.png")

# ---------------- 5: context switches per fence ----------------
def fence_of(cg):
    if "glm-proxy" in cg: return "litellm"
    if cg.endswith("/toolexec") or ("docker-" in cg and "/agent" not in cg and "/toolexec" not in cg
                                    and "measured.slice" in cg and "glm-" not in cg.split("/")[-1]):
        return "tool"
    if cg.endswith("/agent") or "glm-swe" in cg or "glm-oc" in cg: return "harness"
    return None
CS = {}
for lab, rd in TASKS:
    acc = {f: [0.0, 0.0] for f in ("tool", "harness", "litellm")}
    for f in glob.glob(f"{rd}/group_priv_w*.txt"):
        for ln in open(f):
            m = re.match(r"\s*([\d,.]+)\s+(msec task-clock|context-switches)\s+(\S+)", ln)
            if not m: continue
            val = float(m.group(1).replace(",", "")); cg = m.group(3)
            fe = fence_of(cg)
            if not fe: continue
            if "task-clock" in m.group(2): acc[fe][1] += val / 1000.0
            else: acc[fe][0] += val
    CS[lab] = {f: (a[0] / a[1] if a[1] > 0.05 else 0) for f, a in acc.items()}
fig, ax = plt.subplots(figsize=(8.8, 4.4))
x = np.arange(len(TASKS)); w = 0.26
for i, (fe, c) in enumerate((("tool", C_TOOL), ("harness", C_HARN), ("litellm", C_PROXY))):
    v = [CS[lab][fe] for lab, _ in TASKS]
    ax.bar(x + (i - 1) * w, v, w, color=c, label=fe, edgecolor="white")
    for xi, vi in zip(x + (i - 1) * w, v):
        if vi: ax.text(xi, vi * 1.02, f"{vi:,.0f}", ha="center", fontsize=7.6, color="#333333")
ax.set_xticks(x); ax.set_xticklabels([lab for lab, _ in TASKS], fontsize=9)
ax.set_ylabel("context switches per CPU-second")
ax.legend(frameon=False, fontsize=9)
ax.set_title("Context-switch rate per fence", pad=10)
fig.savefig(f"{XOUT}/glm_ctx_switches.png"); plt.close(fig)
print("wrote glm_ctx_switches.png")

# ---------------- 6: OS share per fence ----------------
fig, ax = plt.subplots(figsize=(8.2, 4.2))
x = np.arange(len(TASKS)); w = 0.32
for i, (side, c) in enumerate((("tool", C_TOOL), ("harness", C_HARN))):
    v = [VD[key_of(lab)][side]["kern"] for lab, _ in TASKS]
    ax.bar(x + (i - 0.5) * w, v, w, color=c, label=side, edgecolor="white")
    for xi, vi in zip(x + (i - 0.5) * w, v):
        ax.text(xi, vi + 0.5, f"{vi:.0f}%", ha="center", fontsize=8.6, color="#333333")
ax.set_xticks(x); ax.set_xticklabels([lab for lab, _ in TASKS], fontsize=9)
ax.set_ylabel("share of cycles spent in the OS (%)")
ax.legend(frameon=False, fontsize=9)
ax.set_title("OS share per fence", pad=10)
fig.savefig(f"{XOUT}/glm_os_share.png"); plt.close(fig)
print("wrote glm_os_share.png")

# ---------------- 7: heavy-burst cores x duty decomposition ----------------
def align_offset(rows, lt):
    e0 = rows[0][0]; dur = rows[-1][0] - e0; nb = int(dur) + 2
    rate = np.zeros(nb)
    for (t0, u0), (t1, u1) in zip(rows, rows[1:]):
        rate[int(t0 - e0)] += (u1 - u0) / 1e6
    l0 = lt[0]; base = np.histogram(lt - l0, bins=np.arange(0, nb + 1))[0].astype(float)
    best = (0, -1.0)
    for off in range(-60, 61):
        h = np.roll(base, off); h[:max(off, 0)] = 0
        c = float(np.dot(h[:nb], rate[:nb]))
        if c > best[1]: best = (off, c)
    return e0 + best[0] - l0

fig, axes = plt.subplots(1, len(TASKS), figsize=(3.4 * len(TASKS), 3.8), sharey=True, sharex=True)
for ax, (lab, rd) in zip(np.atleast_1d(axes), TASKS):
    rows = cpustat(rd, 2)
    try:
        L = [(float(p[0]), int(p[1])) for p in (l.split() for l in open(f"{rd}/scope2_cpulanes.tsv"))
             if len(p) == 2]
    except OSError:
        L = []
    heavy = [b for b in act_bursts(rows, 0.005) if b[3] > 0.3 and b[2] > 0.001]
    X, Y, Sz, skip = [], [], [], 0
    if L and heavy:
        lt = np.array([t for t, _ in L])
        off = align_offset(rows, lt)
        ts = lt + off; cpus = np.array([c for _, c in L])
        for t0, t1, coresec, peak in heavy:
            m = (ts >= t0 - 0.05) & (ts <= t1 + 0.05); nsm = int(m.sum())
            if nsm < 8: skip += 1; continue
            nc = len(set(cpus[m].tolist()))
            duty = min(100 * nsm / (HZ * max(t1 - t0, 1 / HZ) * nc), 100)
            X.append(nc); Y.append(duty); Sz.append(coresec)
    ax.scatter(X, Y, s=np.sqrt(np.array(Sz)) * 26 + 8 if Sz else 10, c=C_TOOL,
               alpha=0.55, edgecolors="white", linewidths=0.6)
    ax.set_title(lab, fontsize=10, pad=16)
    ax.text(0.5, 1.01, f"{len(X)} of {len(X)+skip} heavy bursts decomposable",
            transform=ax.transAxes, ha="center", fontsize=7.4, color="#666666")
    ax.set_xlim(0, 21); ax.set_ylim(0, 105); ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("distinct cores touched")
np.atleast_1d(axes)[0].set_ylabel("average duty per touched core (%)")
fig.suptitle("Heavy tool bursts: cores touched × duty (dot area = core-seconds)", fontsize=12.5, y=1.06)
fig.savefig(f"{XOUT}/glm_burst_decomp.png"); plt.close(fig)
print("wrote glm_burst_decomp.png")
print(f"exploratory set -> {XOUT}")
