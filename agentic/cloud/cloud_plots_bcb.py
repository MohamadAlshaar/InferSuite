#!/usr/bin/env python3
"""2 plots + CPU time for agentic BigCodeBench (cloud, 2-box).
  TIME donut: inference (remote GPU, code-gen) vs tool-exec (test subprocess, markers)
  TMA bar:    from system-wide perf during ACTIVE seconds (= the numpy/BLAS test-exec)
  + actual CPU core-seconds.
Usage: cloud_plots_bcb.py <markers.txt> <perf.csv> <label> <outdir> [freq_hz]
"""
import sys, csv, collections
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

markers, perfcsv, label, outdir = sys.argv[1:5]
FHZ = float(sys.argv[5]) if len(sys.argv) > 5 else 3.0e9
M = [l.split() for l in open(markers) if l.strip()]
t0 = [float(p[0]) for p in M if len(p) >= 2 and p[1] == "RUN_START"][0]
t1 = [float(p[0]) for p in M if len(p) >= 2 and p[1] == "RUN_END"][0]
stack = {}; tool = 0.0; nph = 0
for p in M:
    if len(p) >= 3 and p[1] == "toolexec_start": stack[p[2]] = float(p[0])
    elif len(p) >= 3 and p[1] == "toolexec_end" and p[2] in stack:
        tool += float(p[0]) - stack.pop(p[2]); nph += 1
W = t1 - t0
inf = max(W - tool, 0)

# perf -I -x, : relative_ts, value, unit, event, ...
persec = collections.defaultdict(dict)
for r in csv.reader(open(perfcsv)):
    if len(r) < 4 or not r[0] or r[0].startswith("#"): continue
    try: s = int(float(r[0])); v = float(r[1])
    except ValueError: continue
    persec[s][r[3].strip()] = persec[s].get(r[3].strip(), 0) + v
TH = 0.3e9
agg = collections.defaultdict(float); coresec = 0.0; act = 0
for s, d in persec.items():
    c = d.get("cycles", 0); coresec += c / FHZ
    if c > TH:
        act += 1
        for k in ("cycles","instructions","slots","topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"):
            agg[k] += d.get(k, 0)

# ---- plot 1: TIME donut ----
fig, ax = plt.subplots(figsize=(6.2, 6))
v = [inf, tool]; cols = ["#d62728", "#1f77b4"]
ax.pie(v, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(width=0.40, edgecolor="white"))
ax.text(0, 0, f"{inf/W*100:.0f}% infer\n{tool/W*100:.0f}% tool", ha="center", va="center", fontsize=14, fontweight="bold")
ax.set_title(f"{label} — where wall-clock TIME goes\n(window {W:.0f}s; {nph} test-exec runs; tool wall {tool:.0f}s)", fontsize=11)
ax.legend([plt.Rectangle((0,0),1,1,color=c) for c in cols], ["inference (remote GPU, code-gen)", "CPU tool-exec (numpy/BLAS test)"], loc="lower center", bbox_to_anchor=(0.5,-0.06), ncol=1, frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(f"{outdir}/time_donut.png", dpi=130); plt.close(fig)

# ---- plot 2: TMA bar ----
sl = agg["slots"]; ipc = agg["instructions"] / max(agg["cycles"], 1)
ret, fe, bd, be = (agg[k] for k in ("topdown-retiring","topdown-fe-bound","topdown-bad-spec","topdown-be-bound"))
parts = [("Retiring", ret/sl*100, "#2ca02c"), ("Frontend-bound", fe/sl*100, "#1f77b4"),
         ("Bad-spec", bd/sl*100, "#7f7f7f"), ("Backend-bound", be/sl*100, "#ff7f0e")] if sl else []
fig, ax = plt.subplots(figsize=(4.6, 6)); bottom = 0
for name, val, col in parts:
    ax.bar(0, val, 0.6, bottom=bottom, label=name, color=col)
    if val >= 4: ax.text(0, bottom+val/2, f"{val:.0f}", ha="center", va="center", color="white", fontweight="bold", fontsize=11)
    bottom += val
ax.set_xticks([0]); ax.set_xticklabels([f"{label}\ncode-exec\nIPC {ipc:.2f}"], fontsize=9)
ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0, 105)
ax.set_title("TMA — code-exec CPU (numpy/BLAS test)", fontsize=10)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig(f"{outdir}/tma_bar.png", dpi=130, bbox_inches="tight"); plt.close(fig)

print(f"{label}: window {W:.0f}s | inference {inf/W*100:.0f}% | tool {tool/W*100:.0f}% ({nph} test runs, {tool:.0f}s wall)")
print(f"{label}: CPU core-seconds (whole metal box) = {coresec:.0f}  | active seconds = {act}")
if sl: print(f"{label}: code-exec TMA IPC {ipc:.2f} | Retiring {ret/sl*100:.0f} / FE {fe/sl*100:.0f} / BadSpec {bd/sl*100:.0f} / BE {be/sl*100:.0f}")
