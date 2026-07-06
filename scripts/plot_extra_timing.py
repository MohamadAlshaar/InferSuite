#!/usr/bin/env python3
"""Loop-timing analysis from EXISTING captures — no new runs.

Per tier (extra/local, extra/h100, extra/local_api):
  gpu_duty_timelines.png  GPU-busy vs idle strips per workload (local + h100)
  window_cdf.png          CDFs of GPU-busy (generation) and GPU-idle (tool+harness) window durations
  exact_bursts.png        exact tool-exec bursts: BCB markers.txt + SWE trajectory execution_time
  bcb_loop_timeline.png   (api) exec bursts vs model round-trip gaps from markers.txt
  summary.csv             per-workload window stats

Sources: gpu_timeline.csv (epoch,util ~1.9Hz), markers.txt (toolexec_start/end epochs),
*.traj (per-step execution_time). GPU segmentation: busy = util>=15%%, leading/trailing
idle clipped (setup/teardown).
"""
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DUR = "#6a51a3"   # during inference / generation window
OUT = "#1b9e77"   # outside inference / tool+harness window
plt.rcParams.update({"font.family": "DejaVu Sans", "savefig.dpi": 200,
                     "axes.spines.top": False, "axes.spines.right": False})

LOCAL = {  # label -> capture dir
    "BCB": "local_agents/data/bcb_live",
    "SWE astropy": "local_agents/data/swe_live",
    "SWE scikit": "local_agents/data/swe_live_scikit",
    "SWE sympy": "local_agents/data/swe_live_sympy",
    "OC calendar": "local_agents/data/oc_live_calendar",
    "OC image-crop": "local_agents/data/oc_live_image-crop",
    "OC pdf-digest": "local_agents/data/oc_live_pdf-digest",
    "OC web-digest": "local_agents/data/oc_live_web-digest",
}
H100 = {
    "BCB": "h100/data_agent_side/bcb",
    "SWE astropy": "h100/data_agent_side/swe",
    "SWE scikit": "h100/data_agent_side/swe-scikit",
    "SWE sympy": "h100/data_agent_side/swe-sympy",
    "OC calendar": "h100/data_agent_side/oc-calendar",
    "OC image-crop": "h100/data_agent_side/oc-crop",
    "OC pdf-digest": "h100/data_agent_side/oc-pdf",
    "OC web-digest": "h100/data_agent_side/oc-web",
}
SWE_TRAJ = {
    "local": ["agentic/swe_agent/runs/live_local_7b/*/*.traj"],
    "h100": ["h100/data_agent_side/swe_trajectories/live_32b/*/*.traj"],
    "local_api": ["agentic/swe_agent/runs/api_live/*/*/*.traj"],
}
BCB_MARKERS = {"local": "local_agents/data/bcb_live/markers.txt",
               "local_api": "local_agents/data/api_bcb/markers.txt"}

BUSY_THR = 15.0


def read_timeline(d):
    f = os.path.join(d, "gpu_timeline.csv")
    if not os.path.exists(f):
        return None, None
    t, u = [], []
    for row in csv.reader(open(f)):
        if len(row) < 2:
            continue
        try:
            t.append(float(row[0])); u.append(float(row[1]))
        except ValueError:
            continue
    return np.array(t), np.array(u)


def segments(t, u):
    """(busy_durations, idle_durations, spans) with leading/trailing idle clipped."""
    busy = u >= BUSY_THR
    if not busy.any():
        return [], [], []
    i0, i1 = np.argmax(busy), len(busy) - np.argmax(busy[::-1])
    t, busy = t[i0:i1], busy[i0:i1]
    spans, cur = [], busy[0]
    start = t[0]
    for k in range(1, len(busy)):
        if busy[k] != cur:
            spans.append((cur, start, t[k]))
            cur, start = busy[k], t[k]
    spans.append((cur, start, t[-1] + np.median(np.diff(t))))
    b = [e - s for c, s, e in spans if c]
    i = [e - s for c, s, e in spans if not c]
    return b, i, spans


def read_markers(f):
    """-> (exec_durations, gap_durations) from toolexec_start/end epochs."""
    if not f or not os.path.exists(f):
        return [], []
    ev = []
    for ln in open(f):
        p = ln.split()
        if len(p) >= 2 and p[1] in ("toolexec_start", "toolexec_end"):
            ev.append((float(p[0]), p[1]))
    execs, gaps, prev_end, t0 = [], [], None, None
    for ts, kind in ev:
        if kind == "toolexec_start":
            if prev_end is not None:
                gaps.append(ts - prev_end)
            t0 = ts
        elif t0 is not None:
            execs.append(ts - t0)
            prev_end, t0 = ts, None
    return execs, gaps


def read_traj_exec(patterns):
    out = {}
    for pat in patterns:
        for f in glob.glob(pat):
            inst = os.path.basename(f).replace(".traj", "")
            try:
                tr = json.load(open(f)).get("trajectory", [])
            except Exception:
                continue
            xs = [s["execution_time"] for s in tr if s.get("execution_time") is not None]
            if xs:
                out[inst] = xs
    return out


def cdf(ax, xs, label, color, ls="-"):
    xs = np.sort(np.asarray(xs))
    if not len(xs):
        return
    ax.step(xs, np.arange(1, len(xs) + 1) / len(xs), where="post",
            label=f"{label} (n={len(xs)})", color=color, ls=ls, lw=1.6)


def duty_fig(workloads, path, tier_name):
    rows = [(lab, *read_timeline(d)) for lab, d in workloads.items()]
    missing = [lab for lab, t, u in rows if t is None or not len(t)]
    rows = [(lab, t, u) for lab, t, u in rows if t is not None and len(t)]
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 1.05 * len(rows) + 1.2), sharex=False)
    for ax, (lab, t, u) in zip(np.atleast_1d(axes), rows):
        b, i, spans = segments(t, u)
        tz = spans[0][1] if spans else t[0]
        for c, s, e in spans:
            ax.axvspan((s - tz) / 60, (e - tz) / 60, color=DUR if c else OUT, alpha=0.85 if c else 0.45, lw=0)
        ax.set_yticks([]); ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9)
        ax.margins(x=0)
        dur = sum(b); tot = sum(b) + sum(i)
        ax.text(1.002, 0.5, f"{100*dur/tot:.0f}% gen" if tot else "", transform=ax.transAxes,
                fontsize=8, va="center", color="#333")
    np.atleast_1d(axes)[-1].set_xlabel("episode time [min]")
    if missing:
        np.atleast_1d(axes)[-1].annotate("no GPU timeline captured for: " + ", ".join(missing),
                                         xy=(0, -0.9), xycoords="axes fraction", fontsize=7.5, color="#777")
    fig.suptitle(f"GPU duty cycle per episode — {tier_name}\n"
                 f"purple = generation window (GPU busy), green = tool + harness window (GPU idle)\n"
                 f"tool bursts shorter than the ~0.5 s sampling period are invisible (see exact_bursts)",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


def window_fig(workloads, path, tier_name, summary_rows):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab10")
    for k, (lab, d) in enumerate(workloads.items()):
        t, u = read_timeline(d)
        if t is None or not len(t):
            continue
        b, i, _ = segments(t, u)
        cdf(a1, b, lab, cmap(k % 10))
        cdf(a2, i, lab, cmap(k % 10))
        for name, xs in (("gen_window", b), ("idle_window", i)):
            if xs:
                summary_rows.append([lab, name, len(xs), round(float(np.median(xs)), 2),
                                     round(float(np.percentile(xs, 90)), 2), round(float(np.sum(xs)), 1)])
    for ax, ttl in ((a1, "generation windows (GPU busy)"), (a2, "tool + harness windows (GPU idle)")):
        ax.set_xscale("log"); ax.set_xlabel("window length [s]"); ax.set_ylabel("CDF")
        ax.set_title(ttl, fontsize=10); ax.grid(alpha=0.25, lw=0.5)
    a2.legend(fontsize=7, loc="lower right", frameon=False)
    fig.suptitle(f"Loop window lengths — {tier_name}  (GPU-timeline segmentation, ~0.5 s resolution)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


def exact_fig(tier, path, summary_rows):
    execs, gaps = read_markers(BCB_MARKERS.get(tier))
    trajs = read_traj_exec(SWE_TRAJ.get(tier, []))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    if execs:
        cdf(a1, execs, "BCB test-run burst", OUT)
        summary_rows.append(["BCB exact", "tool_burst", len(execs), round(float(np.median(execs)), 3),
                             round(float(np.percentile(execs, 90)), 3), round(float(np.sum(execs)), 1)])
    if gaps:
        cdf(a1, gaps, "BCB between-exec gap (model turn)", DUR)
        summary_rows.append(["BCB exact", "model_gap", len(gaps), round(float(np.median(gaps)), 3),
                             round(float(np.percentile(gaps, 90)), 3), round(float(np.sum(gaps)), 1)])
    cmap = plt.get_cmap("Dark2")
    for k, (inst, xs) in enumerate(sorted(trajs.items())):
        cdf(a2, xs, inst.split("__")[0], cmap(k % 8))
        summary_rows.append([f"SWE {inst.split('__')[0]}", "step_exec", len(xs),
                             round(float(np.median(xs)), 3), round(float(np.percentile(xs, 90)), 3),
                             round(float(np.sum(xs)), 1)])
    a1.set_title("BCB loop events (markers.txt, exact)", fontsize=10)
    a2.set_title("SWE per-step tool execution (trajectory, exact)", fontsize=10)
    for ax in (a1, a2):
        ax.set_xscale("log"); ax.set_xlabel("duration [s]"); ax.set_ylabel("CDF")
        ax.grid(alpha=0.25, lw=0.5); ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle(f"Exact tool-side event durations — {tier.replace('_', '/')}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


def bcb_loop_timeline(path):
    f = BCB_MARKERS["local_api"]
    ev = [(float(l.split()[0]), l.split()[1]) for l in open(f) if "toolexec" in l]
    t0 = ev[0][0]
    fig, ax = plt.subplots(figsize=(11, 2.6))
    start = None
    prev_end = 0.0
    for ts, kind in ev:
        if kind == "toolexec_start":
            ax.axvspan(prev_end, (ts - t0) / 60, color=DUR, alpha=0.55, lw=0)
            start = (ts - t0) / 60
        elif start is not None:
            ax.axvspan(start, (ts - t0) / 60, color=OUT, alpha=0.95, lw=0)
            prev_end = (ts - t0) / 60
            start = None
    ax.set_yticks([]); ax.set_xlabel("run time [min]"); ax.margins(x=0)
    ax.set_title("BCB under the frontier API — the loop as measured (markers.txt)\n"
                 "purple = model round-trip (local CPU idle), green = local test execution", fontsize=10.5)
    fig.tight_layout(); fig.savefig(path, bbox_inches="tight"); plt.close(fig)
    print("wrote", path)


def write_summary(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["workload", "window_type", "n", "median_s", "p90_s", "total_s"])
        w.writerows(rows)
    print("wrote", path)


for tier, workloads in (("local", LOCAL), ("h100", H100)):
    rows = []
    duty_fig(workloads, f"extra/{tier}/gpu_duty_timelines.png",
             "self-served 7B (workstation)" if tier == "local" else "self-served 32B (H100)")
    window_fig(workloads, f"extra/{tier}/window_cdf.png",
               "self-served 7B (workstation)" if tier == "local" else "self-served 32B (H100)", rows)
    exact_fig(tier, f"extra/{tier}/exact_bursts.png", rows)
    write_summary(rows, f"extra/{tier}/summary.csv")

rows = []
exact_fig("local_api", "extra/local_api/burst_shadow_cdf.png", rows)
bcb_loop_timeline("extra/local_api/bcb_loop_timeline.png")
write_summary(rows, "extra/local_api/summary.csv")
