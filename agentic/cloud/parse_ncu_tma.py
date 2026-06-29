#!/usr/bin/env python3
"""Parse ncu --csv output into a GPU 'TMA'-style picture:
  - Warp-stall-reason breakdown (the GPU analog of CPU top-down) as a stacked bar
  - Speed-of-Light: compute% vs memory% of peak  ->  compute-bound vs memory-bound
Averages metrics across the captured decode kernels. Usage: parse_ncu_tma.py ncu_metrics.csv
"""
import sys, csv, collections

path = sys.argv[1] if len(sys.argv) > 1 else "ncu_out/ncu_metrics.csv"
vals = collections.defaultdict(list)
# ncu --csv: a header row then rows incl columns "Metric Name","Metric Value" (+ "Kernel Name")
with open(path, newline="") as f:
    rows = list(csv.reader(f))
hdr = None
for r in rows:
    if "Metric Name" in r and "Metric Value" in r:
        hdr = {c: i for i, c in enumerate(r)}; continue
    if not hdr or len(r) <= max(hdr["Metric Name"], hdr["Metric Value"]): continue
    name = r[hdr["Metric Name"]].strip()
    try: v = float(r[hdr["Metric Value"]].replace(",", ""))
    except ValueError: continue
    vals[name].append(v)

def avg(metric):
    xs = vals.get(metric, [])
    return sum(xs) / len(xs) if xs else 0.0

sol_c = avg("sm__throughput.avg.pct_of_peak_sustained_elapsed")
sol_m = avg("gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed")
occ   = avg("sm__warps_active.avg.pct_of_peak_sustained_active")

STALLS = ["long_scoreboard","short_scoreboard","wait","barrier","membar","lg_throttle",
          "mio_throttle","math_pipe_throttle","tex_throttle","not_selected","no_instructions",
          "imc_miss","drain","dispatch_stall"]
stall = {s: avg(f"smsp__average_warps_issue_stalled_{s}_per_issue_active.ratio") for s in STALLS}
tot = sum(stall.values()) or 1.0
pct = {s: 100*stall[s]/tot for s in STALLS}

print("="*64)
print("GPU 'TMA' — decode kernels (avg over captured kernels), TP=1")
print("="*64)
print(f"Speed-of-Light:  compute {sol_c:.0f}% of peak | memory {sol_m:.0f}% of peak | occupancy {occ:.0f}%")
verdict = "MEMORY-bound" if sol_m > sol_c else "COMPUTE-bound"
print(f"  => {verdict}  (LLM decode is typically memory-bandwidth-bound)")
print("Warp stall-reason breakdown (where issue cycles go):")
for s in sorted(pct, key=pct.get, reverse=True):
    if pct[s] >= 1: print(f"  {s:18} {pct[s]:5.1f}%")

# stacked bar (only if matplotlib present; mirrors the CPU TMA style)
try:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    top = [s for s in sorted(pct, key=pct.get, reverse=True) if pct[s] >= 1]
    fig, ax = plt.subplots(figsize=(4.5, 6))
    bottom = 0
    for s in top:
        ax.bar(0, pct[s], 0.6, bottom=bottom, label=s)
        if pct[s] >= 4: ax.text(0, bottom+pct[s]/2, f"{pct[s]:.0f}", ha="center", va="center", color="white", fontsize=9)
        bottom += pct[s]
    ax.set_xticks([0]); ax.set_xticklabels([f"L40S decode\nSoL: c{sol_c:.0f}%/m{sol_m:.0f}%\n{verdict}"], fontsize=9)
    ax.set_ylabel("% of warp issue-stall cycles"); ax.set_ylim(0, 105)
    ax.set_title("GPU warp-stall ('TMA') — 32B decode, TP=1", fontsize=11)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8, frameon=False)
    fig.tight_layout(); fig.savefig("/opt/agentic/ncu_out/gpu_tma.png", dpi=130, bbox_inches="tight")
    print("figure -> /opt/agentic/ncu_out/gpu_tma.png")
except Exception as e:
    print(f"(plot skipped: {e})")
