#!/usr/bin/env python3
"""Plot SWE-agent perf timelines for the agentic LLM<->tool loop.

Three probes, all local:
  - sandbox cgroup perf  -> tool-exec CPU            (perf_timeline.csv)
  - sweagent host proc   -> agent-controller CPU     (host_agent_timeline.csv)
  - nvidia-smi util       -> vLLM inference (GPU)      (gpu_timeline.csv)
markers.txt carries perf_start / host_perf_start epochs for alignment.
Figures -> figures/.
"""
import csv, collections, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "runs", "perf")
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)

def parse_perf(path):
    """perf -I -x, CSV -> {time_float: {event: value}} (works for -G and -p)."""
    rows = collections.defaultdict(dict)
    if not os.path.exists(path): return rows
    for line in open(path):
        p = line.rstrip("\n").split(",")
        if len(p) < 4 or not p[0][:1].isdigit(): continue
        try:
            rows[float(p[0])][p[3]] = float(p[1].replace(",", "")) if p[1] not in ("", "<not counted>") else 0.0
        except ValueError:
            pass
    return rows

rows = parse_perf(os.path.join(R, "perf_timeline.csv"))
ts = sorted(rows)
if not ts:
    print("no sandbox perf data"); sys.exit(1)
rel = [t - ts[0] for t in ts]
def col(ev): return [rows[t].get(ev, 0.0) for t in ts]
cycles, instr, slots = col("cycles"), col("instructions"), col("slots")
ret, fe, bad, be = col("topdown-retiring"), col("topdown-fe-bound"), col("topdown-bad-spec"), col("topdown-be-bound")
# LEGACY TMA (Cascade Lake / g4dn.metal): no slots/topdown-* counter -> compute from classic events.
# SLOTS = 4*cpu_clk_unhalted.thread; downstream pct(x)=x/slots and the aggregate work unchanged.
if not any(slots):
    clk, ui, urs = col("cpu_clk_unhalted.thread"), col("uops_issued.any"), col("uops_retired.retire_slots")
    idq, rec = col("idq_uops_not_delivered.core"), col("int_misc.recovery_cycles")
    slots = [4*clk[i] for i in range(len(ts))]
    ret, fe = urs, idq
    bad = [max(ui[i]-urs[i]+4*rec[i], 0) for i in range(len(ts))]
    be  = [max(slots[i]-ret[i]-fe[i]-bad[i], 0) for i in range(len(ts))]
ipc = [(instr[i]/cycles[i]) if cycles[i] else 0 for i in range(len(ts))]
def pct(x): return [(x[i]/slots[i]*100) if slots[i] else 0 for i in range(len(ts))]
retp, fep, badp, bep = pct(ret), pct(fe), pct(bad), pct(be)
dur = rel[-1] if rel else 0
print(f"sandbox intervals={len(ts)} duration={dur:.0f}s")

# ---- Fig 1: sandbox activity + IPC ----
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
a1.fill_between(rel, [c/1e9 for c in cycles], color="#1f77b4", alpha=0.3); a1.plot(rel, [c/1e9 for c in cycles], color="#1f77b4", lw=1)
a1.set_ylabel("sandbox activity\n(Gcycles/s)"); a1.set_title("SWE-agent sandbox CPU over a SWE-bench run")
a2.plot(rel, ipc, color="#d62728", lw=1); a2.set_ylabel("IPC"); a2.set_xlabel("time (s)"); a2.set_ylim(0, max(2.0, (max(ipc) if ipc else 2)*1.1))
for a in (a1, a2): a.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/01_activity_ipc_timeline.png", dpi=130); plt.close(fig)

# ---- Fig 2: TMA over time ----
fig, ax = plt.subplots(figsize=(11, 4))
ax.stackplot(rel, retp, fep, badp, bep, labels=["Retiring", "Frontend-bound", "Bad-spec", "Backend-bound"],
             colors=["#2ca02c", "#1f77b4", "#7f7f7f", "#ff7f0e"], alpha=0.85)
ax.set_ylim(0, 100); ax.set_xlim(0, dur); ax.set_ylabel("% of pipeline slots"); ax.set_xlabel("time (s)")
ax.set_title("TMA top-down over time — SWE-agent sandbox")
ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/02_tma_timeline.png", dpi=130); plt.close(fig)

# ---- Fig 3: overall sandbox TMA ----
S = sum(slots) or 1
agg = [sum(ret)/S*100, sum(fe)/S*100, sum(bad)/S*100, sum(be)/S*100]
oipc = (sum(instr)/sum(cycles)) if sum(cycles) else 0
fig, ax = plt.subplots(figsize=(6, 4)); bottom = 0
for v, lab, cI in zip(agg, ["Retiring", "Frontend-bound", "Bad-spec", "Backend-bound"], ["#2ca02c", "#1f77b4", "#7f7f7f", "#ff7f0e"]):
    ax.bar(0, v, bottom=bottom, color=cI, label=f"{lab} {v:.0f}%"); bottom += v
ax.set_xticks([0]); ax.set_xticklabels([f"sandbox\n(IPC={oipc:.2f})"]); ax.set_ylabel("% of pipeline slots"); ax.set_ylim(0, 100)
ax.set_title("Overall TMA — SWE-agent sandbox"); ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.02, 0.5))
fig.tight_layout(); fig.savefig(f"{OUT}/03_tma_overall.png", dpi=130); plt.close(fig)

# ---- Fig 4: activity histogram ----
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist([c/1e9 for c in cycles], bins=40, color="#9467bd", alpha=0.8)
ax.set_xlabel("sandbox activity (Gcycles/s)"); ax.set_ylabel("# 1s intervals")
ax.set_title("Sandbox activity distribution"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/04_activity_hist.png", dpi=130); plt.close(fig)

# ============ markers + 3 timelines aligned to perf_start ============
mk = {}
MARK = os.path.join(R, "markers.txt")
if os.path.exists(MARK):
    for line in open(MARK):
        pp = line.split()
        if len(pp) >= 2:
            try: mk[pp[1]] = float(pp[0])
            except ValueError: pass
perf_start = mk.get("perf_start"); host_start = mk.get("host_perf_start")

sandbox_s = {int(round(rel[i])): cycles[i] for i in range(len(ts))}
# host-agent: perf -p timestamps are relative to host-perf start -> shift onto perf_start origin
host_rows = parse_perf(os.path.join(R, "host_agent_timeline.csv"))
shift = (host_start - perf_start) if (host_start and perf_start) else 0.0
host_s = {int(round((t - min(host_rows)) + shift)): host_rows[t].get("cycles", 0.0) for t in host_rows} if host_rows else {}
# gpu util: epoch -> rel
gpu_s = {}
GCSV = os.path.join(R, "gpu_timeline.csv")
if os.path.exists(GCSV) and perf_start:
    for line in open(GCSV):
        p = line.strip().split(",")
        if len(p) < 2: continue
        try: gpu_s[int(round(float(p[0]) - perf_start))] = float(p[1])
        except ValueError: pass

if gpu_s or host_s:
    # ---- Fig 5: 3-way overlay ----
    fig, ax = plt.subplots(figsize=(11, 4)); axb = ax.twinx()
    sx = sorted(sandbox_s); hx = sorted(host_s); gx = sorted(gpu_s)
    ax.fill_between(sx, [sandbox_s[s]/1e9 for s in sx], color="#1f77b4", alpha=0.35, label="sandbox tool CPU")
    if hx: ax.plot(hx, [host_s[s]/1e9 for s in hx], color="#2ca02c", lw=1.2, label="host-agent CPU")
    ax.set_ylabel("CPU (Gcycles/s)"); ax.set_xlabel("time (s)"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    if gx:
        axb.plot(gx, [gpu_s[s] for s in gx], color="#d62728", lw=1.2)
        axb.set_ylabel("GPU util %", color="#d62728"); axb.set_ylim(0, 105); axb.tick_params(axis="y", colors="#d62728")
    ax.set_title("Agentic loop — GPU inference (red) vs sandbox tool (blue) vs host-agent CPU (green)")
    fig.tight_layout(); fig.savefig(f"{OUT}/05_cpu_vs_gpu.png", dpi=130); plt.close(fig)

    # ---- 4-way per-second classification ----
    BUSY = 1.5e8  # >0.15 Gcyc/s => that CPU component was doing real work this second
    allsec = sorted(set(gpu_s) | set(sandbox_s) | set(host_s))
    # The GPU sampler ("while sleep 1; nvidia-smi") runs at ~1.03s/iter -> drifts, so ~1
    # integer-second bucket every ~31s has NO sample. Treating those as util=0 mislabels
    # mid-inference seconds as "idle". Forward-fill missing seconds between real samples.
    gpu_ff = {}
    if gpu_s:
        lo, hi, last = min(gpu_s), max(gpu_s), 0.0
        for s in range(lo, hi + 1):
            if s in gpu_s: last = gpu_s[s]
            gpu_ff[s] = last  # carry last known util across a dropped tick
    inf = tool = host = idle = 0
    for s in allsec:
        if gpu_ff.get(s, gpu_s.get(s, 0)) > 50: inf += 1
        elif sandbox_s.get(s, 0) > BUSY: tool += 1
        elif host_s.get(s, 0) > BUSY: host += 1
        else: idle += 1
    n = len(allsec) or 1
    print(f"=== 4-way time split over {n}s ===")
    for lab, v in [("GPU inference", inf), ("sandbox tool-exec", tool), ("host-agent CPU", host), ("inter-turn latency", idle)]:
        print(f"  {lab:18s} {v:3d}s ({v/n*100:.0f}%)")

    # ---- Fig 6: 4-way donut ----
    vals = [inf, tool, host, idle]
    labs = [f"GPU inference  {inf}s ({inf/n*100:.0f}%)",
            f"sandbox tool-exec  {tool}s ({tool/n*100:.0f}%)",
            f"host-agent CPU  {host}s ({host/n*100:.0f}%)",
            f"inter-turn latency  {idle}s ({idle/n*100:.0f}%)"]
    cols = ["#d62728", "#1f77b4", "#2ca02c", "#cccccc"]
    keep = [i for i in range(4) if vals[i] > 0]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = ax.pie([vals[i] for i in keep], colors=[cols[i] for i in keep], startangle=90,
               wedgeprops=dict(width=0.42, edgecolor="white"))[0]
    ax.legend(w, [labs[i] for i in keep], loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=False)
    ax.text(0, 0, f"{n}s\nwall-clock", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("SWE-bench agentic run — where the time goes")
    fig.tight_layout(); fig.savefig(f"{OUT}/06_cpu_vs_gpu_donut.png", dpi=130); plt.close(fig)
else:
    print("(no gpu/host timelines -> skipping overlay + donut)")

# ============ 4th probe: vLLM SERVER CPU + CPU core-seconds breakdown ============
# Answers "where is the CPU": the serving stack (vLLM tokenize/schedule/sample/detok +
# enforce-eager engine loop) runs CONCURRENTLY with the GPU and is the dominant CPU user,
# which the GPU/tool/idle wall-clock donut cannot show (those are mutually exclusive time).
vllm_c = {}
VCSV = os.path.join(R, "vllm_timeline.csv")
if os.path.exists(VCSV) and perf_start:
    for line in open(VCSV):
        p = line.strip().split(",")
        if len(p) < 2: continue
        try: vllm_c[int(round(float(p[0]) - perf_start))] = float(p[1])
        except ValueError: pass

if vllm_c:
    # CPU frequency to convert perf cycles -> cores (approx; tool/agent are tiny either way)
    FREQ = 3.5e9
    try: FREQ = int(open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq").read()) * 1000
    except Exception: pass
    sb_cores = {s: sandbox_s.get(s, 0)/FREQ for s in sandbox_s}
    ha_cores = {s: host_s.get(s, 0)/FREQ for s in host_s}
    # core-seconds (resource usage = integral of cores over time; 1s samples)
    vllm_cs = sum(vllm_c.values())
    tool_cs = sum(sb_cores.values())
    host_cs = sum(ha_cores.values())
    total_cs = vllm_cs + tool_cs + host_cs or 1
    print(f"\n=== CPU core-seconds breakdown (resource usage, concurrent with GPU) ===")
    print(f"  vLLM inference-server : {vllm_cs:7.1f} core-s ({vllm_cs/total_cs*100:.0f}%)  peak {max(vllm_c.values()):.1f} cores")
    print(f"  sandbox tool-exec     : {tool_cs:7.1f} core-s ({tool_cs/total_cs*100:.0f}%)")
    print(f"  host agent-controller : {host_cs:7.1f} core-s ({host_cs/total_cs*100:.0f}%)")
    print(f"  (freq~{FREQ/1e9:.1f}GHz for cycles->cores; vLLM measured directly in cores)")

    # ---- Fig 8: CPU(cores) + GPU(util) over time ----
    clip = lambda v: max(0.0, min(v, 8.0))  # guard against counter-reset spikes
    fig, ax = plt.subplots(figsize=(11, 4)); axb = ax.twinx()
    vx = sorted(vllm_c); sx = sorted(sb_cores); hx = sorted(ha_cores)
    ax.fill_between(vx, [clip(vllm_c[s]) for s in vx], color="#9467bd", alpha=0.45, label="vLLM server CPU")
    ax.plot(sx, [clip(sb_cores[s]) for s in sx], color="#1f77b4", lw=1, label="sandbox tool CPU")
    ax.plot(hx, [clip(ha_cores[s]) for s in hx], color="#2ca02c", lw=1, label="host-agent CPU")
    peak = max([clip(v) for v in list(vllm_c.values())+list(sb_cores.values())+list(ha_cores.values())] or [2])
    ax.set_ylim(0, max(2.0, peak*1.2))
    ax.set_ylabel("CPU (cores)"); ax.set_xlabel("time (s)"); ax.grid(alpha=0.3); ax.legend(loc="upper right", fontsize=8)
    if gpu_s:
        gx = sorted(gpu_s); axb.plot(gx, [gpu_s[s] for s in gx], color="#d62728", lw=1, alpha=0.6)
        axb.set_ylabel("GPU util %", color="#d62728"); axb.set_ylim(0, 105); axb.tick_params(axis="y", colors="#d62728")
    ax.set_title("CPU cores (vLLM server / tool / agent) vs GPU util — concurrent")
    fig.tight_layout(); fig.savefig(f"{OUT}/08_cpu_cores_timeline.png", dpi=130); plt.close(fig)

    # ---- Fig 9: CPU core-seconds donut ----
    vals = [vllm_cs, tool_cs, host_cs]
    labs = [f"vLLM server  {vllm_cs:.0f} core-s ({vllm_cs/total_cs*100:.0f}%)",
            f"sandbox tool  {tool_cs:.1f} core-s ({tool_cs/total_cs*100:.0f}%)",
            f"host agent  {host_cs:.1f} core-s ({host_cs/total_cs*100:.0f}%)"]
    cols = ["#9467bd", "#1f77b4", "#2ca02c"]
    keep = [i for i in range(3) if vals[i] > 0.05]
    fig, ax = plt.subplots(figsize=(8, 5))
    w = ax.pie([vals[i] for i in keep], colors=[cols[i] for i in keep], startangle=90,
               wedgeprops=dict(width=0.42, edgecolor="white"))[0]
    ax.legend(w, [labs[i] for i in keep], loc="center left", bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=False)
    ax.text(0, 0, f"{total_cs:.0f}\ncore-s", ha="center", va="center", fontsize=12, fontweight="bold")
    ax.set_title("Where the CPU goes — agentic run (inference-server dominates)")
    fig.tight_layout(); fig.savefig(f"{OUT}/09_cpu_breakdown_donut.png", dpi=130); plt.close(fig)
else:
    print("(no vllm_timeline.csv -> skipping CPU breakdown; re-run with the 4th probe)")

print("wrote:")
for fn in sorted(os.listdir(OUT)):
    print("  ", os.path.join(OUT, fn))
