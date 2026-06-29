#!/usr/bin/env python3
"""Produce the 2 key plots for a cloud agentic run from the metal-side perf data.
  TIME donut: waiting-on-inference / tool-exec / agent-controller   (2-box: inference = metal idle)
  TMA bar:    Retiring / Frontend / Bad-spec / Backend  from the cgroup topdown events
Usage: cloud_plots.py <markers.txt> <tool_timeline.csv(cgroup)> <ctrl_timeline.csv|-> <label> <outdir>
"""
import sys, csv
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

markers, tool_tl, ctrl_tl, label, outdir = sys.argv[1:6]
t0 = t1 = None
for l in open(markers):
    p = l.split()
    if len(p) >= 2 and p[1] == "perf_start": t0 = float(p[0])
    if len(p) >= 2 and p[1] == "agent_done": t1 = float(p[0])

def cyc_per_sec(path):
    d = {}
    try:
        for r in csv.reader(open(path)):
            if len(r) < 4 or not r[0] or r[0].startswith("#"): continue
            if r[3].strip() == "cycles":
                try: d[int(float(r[0]))] = d.get(int(float(r[0])), 0) + float(r[1])
                except ValueError: pass
    except FileNotFoundError: pass
    return d

def tma_agg(path):
    agg = {}
    for r in csv.reader(open(path)):
        if len(r) < 4 or not r[0] or r[0].startswith("#"): continue
        ev = r[3].strip()
        if ev in ("cycles","instructions","slots","topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"):
            try: agg[ev] = agg.get(ev, 0) + float(r[1])
            except ValueError: pass
    return agg

tool = cyc_per_sec(tool_tl)
ctrl = cyc_per_sec(ctrl_tl) if ctrl_tl not in ("-", "") else {}
W = int(t1 - t0) if (t0 and t1) else (max(list(tool) + list(ctrl), default=0) + 1)
TH = 0.3e9   # >~7% of a core = active
infS = toolS = ctrlS = 0
for s in range(W):
    if tool.get(s, 0) > TH: toolS += 1
    elif ctrl.get(s, 0) > TH: ctrlS += 1
    else: infS += 1
tot = max(infS + toolS + ctrlS, 1)

# ---- plot 1: TIME donut ----
fig, ax = plt.subplots(figsize=(6.2, 6))
v = [infS, toolS, ctrlS]; cols = ["#d62728", "#1f77b4", "#bdbdbd"]
ax.pie(v, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.40, edgecolor="white"))
ax.text(0, 0, f"{infS/tot*100:.0f}% infer\n{toolS/tot*100:.0f}% tool\n{ctrlS/tot*100:.0f}% agent", ha="center", va="center", fontsize=13, fontweight="bold")
ax.set_title(f"{label} — where wall-clock TIME goes\n(window {W}s; 2-box: infer = waiting on remote GPU)", fontsize=11)
ax.legend([plt.Rectangle((0,0),1,1,color=c) for c in cols], ["inference (GPU, remote)", "CPU tool-exec", "agent-controller"], loc="lower center", bbox_to_anchor=(0.5,-0.08), ncol=3, frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(f"{outdir}/time_donut.png", dpi=130); plt.close(fig)

# ---- plot 2: TMA bar (tool/code-exec cgroup) ----
a = tma_agg(tool_tl); sl = a.get("slots", 0)
ipc = a.get("instructions", 0) / max(a.get("cycles", 1), 1)
ret, fe, bd, be = (a.get(k, 0) for k in ("topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"))
parts = [("Retiring", ret/sl*100, "#2ca02c"), ("Frontend-bound", fe/sl*100, "#1f77b4"),
         ("Bad-spec", bd/sl*100, "#7f7f7f"), ("Backend-bound", be/sl*100, "#ff7f0e")] if sl else []
fig, ax = plt.subplots(figsize=(4.6, 6)); bottom = 0
for name, val, col in parts:
    ax.bar(0, val, 0.6, bottom=bottom, label=name, color=col)
    if val >= 4: ax.text(0, bottom+val/2, f"{val:.0f}", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    bottom += val
ax.set_xticks([0]); ax.set_xticklabels([f"{label}\ntool-exec\nIPC {ipc:.2f}"], fontsize=9)
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0, 105)
ax.set_title("TMA — tool/code-exec CPU", fontsize=11)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{outdir}/tma_bar.png", dpi=130, bbox_inches="tight"); plt.close(fig)
print(f"{label}: TIME infer {infS/tot*100:.0f}% / tool {toolS/tot*100:.0f}% / agent {ctrlS/tot*100:.0f}% (window {W}s)")
if sl: print(f"{label}: TMA IPC {ipc:.2f} | Retiring {ret/sl*100:.0f} / FE {fe/sl*100:.0f} / BadSpec {bd/sl*100:.0f} / BE {be/sl*100:.0f}")
print(f"saved {outdir}/time_donut.png + {outdir}/tma_bar.png")
