#!/usr/bin/env python3
"""SWE-bench LOCAL 7B validation — the 4 deliverables.
Provenance: local workstation (Xeon w5-3425 Sapphire Rapids + RTX A2000), Qwen2.5-Coder-7B-AWQ
served by local vLLM, instance scikit-learn__scikit-learn-10297, 2026-06-24. NOT cloud/32B data.
Reads: runs/perf/*.csv (live run) + runs/replay/group_*.txt (deep microarch).
Run with SYSTEM python3 (matplotlib not in .venv)."""
import os, collections, json, glob
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PERF = os.path.join(HERE, "runs", "perf")
REPLAY = os.path.join(HERE, "runs", "replay")
OUT = os.path.join(HERE, "results_local7b"); os.makedirs(OUT, exist_ok=True)
TITLE = "SWE-agent · Qwen2.5-Coder-7B (local) · scikit-learn-10297"

def parse_csv_timeline(path):
    agg = collections.Counter()
    if not os.path.exists(path): return agg
    for line in open(path):
        q = line.rstrip("\n").split(",")
        if len(q) < 4 or not q[0][:1].isdigit(): continue
        try: agg[q[3]] += float(q[1]) if q[1] not in ("", "<not counted>") else 0.0
        except ValueError: pass
    return agg

def parse_perf_txt(path):
    agg = collections.Counter()
    if not os.path.exists(path): return agg
    for line in open(path):
        p = line.split()
        if len(p) < 2: continue
        try: v = float(p[0].replace(",", ""))
        except ValueError: continue
        agg[p[1]] = v
    return agg

def tma_of(agg):
    sl = agg.get("slots", 0) or 1; cyc = agg.get("cycles", 0) or 1; ins = agg.get("instructions", 0)
    ret, fe, bad, be = (agg.get("topdown-"+k, 0)/sl*100 for k in ("retiring","fe-bound","bad-spec","be-bound"))
    return dict(ipc=ins/cyc, cyc=cyc, ret=ret, fe=fe, bad=bad, be=be)

# ---- data ----
agent = parse_csv_timeline(os.path.join(PERF, "perf_timeline.csv"))
infer = parse_csv_timeline(os.path.join(PERF, "vllm_perf_timeline.csv"))
at, it = tma_of(agent), tma_of(infer)

# GPU util timeline (inference activity over wall-clock)
gpu = []
gp = os.path.join(PERF, "gpu_timeline.csv")
if os.path.exists(gp):
    for l in open(gp):
        x = l.split(",")
        if len(x) > 1 and x[1].strip().replace(".", "").isdigit(): gpu.append(float(x[1]))
gpu_busy_frac = (sum(1 for u in gpu if u > 5)/len(gpu)) if gpu else 0

# vLLM-server CPU cores timeline -> mean cores during inference
vcores = []
vp = os.path.join(PERF, "vllm_timeline.csv")
if os.path.exists(vp):
    for l in open(vp):
        x = l.strip().split(",")
        if len(x) > 1:
            try: vcores.append(float(x[1]))
            except ValueError: pass

# wall-clock duration + average cores (cycles / freq / duration) for the absolute framing
FREQ = 4.0e9
def timeline_dur(path):
    ts = []
    if os.path.exists(path):
        for l in open(path):
            q = l.split(",")
            if q and q[0][:1].isdigit():
                try: ts.append(float(q[0]))
                except ValueError: pass
    return (max(ts) if ts else 0) or 1
dur = max(timeline_dur(os.path.join(PERF, "perf_timeline.csv")),
          timeline_dur(os.path.join(PERF, "vllm_perf_timeline.csv")))
agent_cores = at["cyc"]/FREQ/dur
infer_cores = it["cyc"]/FREQ/dur
NCPU = os.cpu_count() or 24
total_cores = agent_cores + infer_cores

# loop count from trajectory
traj_files = glob.glob(os.path.join(HERE, "runs", "heavy", "scikit-learn__scikit-learn-10297", "*.traj"))
acts = collections.Counter(); n_turns = 0
if traj_files:
    d = json.load(open(traj_files[0])); traj = d.get("trajectory", d) if isinstance(d, dict) else d
    n_turns = len(traj)
    for t in traj:
        if isinstance(t, dict) and t.get("action"):
            acts[t["action"].split()[0] if t["action"].split() else "?"] += 1

# deep microarch from replay
rep = {g: parse_perf_txt(os.path.join(REPLAY, f"group_{g}.txt")) for g in ("tma","cache","fp","mlp")}

# =========================================================================
# Deliverable 1: CPU-vs-GPU donut (two views: wall-clock time + CPU core-seconds)
# =========================================================================
fig, ax = plt.subplots(1, 2, figsize=(11, 5))
# wall-clock: GPU-inference-active vs not (the run is dominated by waiting on inference)
infq = gpu_busy_frac*100
ax[0].pie([infq, 100-infq], labels=[f"GPU inference\nactive\n{infq:.0f}%", f"tool-exec /\nidle\n{100-infq:.0f}%"],
          colors=["#4C72B0", "#DD8452"], autopct="", startangle=90, wedgeprops=dict(width=0.42))
ax[0].set_title("Wall-clock time\n(GPU busy = inference)")
# CPU core-seconds: inference-CPU (vLLM engine) vs tool-exec-CPU (sandbox). cycles ~ core-seconds.
ic, ac = it["cyc"], at["cyc"]; tot = ic+ac or 1
ax[1].pie([ic/tot*100, ac/tot*100],
          labels=[f"Inference CPU\n(vLLM engine)\n{ic/tot*100:.0f}%", f"Tool-exec CPU\n(sandbox)\n{ac/tot*100:.0f}%"],
          colors=["#55A868", "#C44E52"], autopct="", startangle=90, wedgeprops=dict(width=0.42))
ax[1].set_title("CPU core-seconds\n(cycles: during vs outside inference)")
fig.suptitle("① CPU vs GPU — "+TITLE, fontsize=11, weight="bold")
fig.text(0.5, 0.055, f"vLLM-engine {infer_cores:.2f} cores  vs  tool-exec {agent_cores:.2f} cores  "
         f"({ic/ac:.0f}× more CPU during inference)", ha="center", fontsize=9, weight="bold")
fig.text(0.5, 0.005, f"absolute: only ~{total_cores:.2f} of {NCPU} cores busy → the box is ~{100-total_cores/NCPU*100:.0f}% idle; "
         f"the split above is of the ACTIVE CPU only (agent tool-exec ≈ idle at {agent_cores:.2f} cores)",
         ha="center", fontsize=8, style="italic", color="#555")
plt.tight_layout(rect=[0,0.03,1,0.95]); plt.savefig(os.path.join(OUT, "1_cpu_vs_gpu_donut.png"), dpi=130); plt.close()

# =========================================================================
# Deliverable 2: TMA — Agent-CPU (outside inference) vs Inference-CPU (during)
# =========================================================================
cats = ["Retiring", "Frontend", "Bad-spec", "Backend"]
tcol = ["#55A868", "#DD8452", "#C44E52", "#4C72B0"]  # TMA convention: retiring=green, FE=orange, bad-spec=red, BE=blue
panels = [("Agent tool-exec CPU\n(outside inference)", at, agent_cores),
          ("Inference CPU — vLLM engine\n(during inference)", it, infer_cores)]
fig, axes = plt.subplots(1, 2, figsize=(9, 5.5), sharey=True)
for ax, (lbl, t, cores) in zip(axes, panels):
    vals = [t["ret"], t["fe"], t["bad"], t["be"]]
    bottom = 0
    for v, c, name in zip(vals, tcol, cats):
        ax.bar(0, v, bottom=bottom, color=c, width=0.5, label=name, edgecolor="white", linewidth=1)
        if v > 3:  # label each segment with its % in the middle
            ax.text(0, bottom+v/2, f"{name}\n{v:.0f}%", ha="center", va="center",
                    fontsize=9.5, weight="bold", color="white")
        bottom += v
    ax.set_title(f"{lbl}\nIPC {t['ipc']:.2f}  ·  {cores:.2f} cores", fontsize=10.5, weight="bold")
    ax.set_ylim(0, 100); ax.set_xlim(-0.6, 0.6); ax.set_xticks([])
axes[0].set_ylabel("% of pipeline slots")
fig.suptitle("② Top-down (TMA): outside-inference vs during-inference CPU\n"+TITLE, fontsize=11, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93]); plt.savefig(os.path.join(OUT, "2_tma_agent_vs_inference.png"), dpi=130); plt.close()

# =========================================================================
# Deliverable 3: agent loop count (turns + action breakdown)
# =========================================================================
fig, ax = plt.subplots(figsize=(8, 5))
if acts:
    items = acts.most_common()
    ax.barh([k for k,_ in items][::-1], [v for _,v in items][::-1], color="#4C72B0")
    ax.set_xlabel("count")
ax.set_title(f"③ Agent loop: {n_turns} turns, {sum(acts.values())} tool actions\n"+TITLE, fontsize=11, weight="bold")
for i,(k,v) in enumerate(acts.most_common()[::-1]): ax.text(v+0.1, i, str(v), va="center", fontsize=9)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "3_agent_loop_count.png"), dpi=130); plt.close()

# =========================================================================
# Deliverable 4: CPU microarch table (deep, from replay)
# =========================================================================
def fp_share(a):
    g = lambda k: a.get("fp_arith_inst_retired."+k, 0)
    s1,s128,s256,s512 = g("scalar_single"),g("128b_packed_single"),g("256b_packed_single"),g("512b_packed_single")
    sd,d128,d256,d512 = g("scalar_double"),g("128b_packed_double"),g("256b_packed_double"),g("512b_packed_double")
    flops = s1+s128*4+s256*8+s512*16+sd+d128*2+d256*4+d512*8
    avx = (s512*16+d512*8)/flops*100 if flops else 0
    return flops, avx
def cache_amat(a):
    l1,l2,l3,m = (a.get("mem_load_retired."+k,0) for k in ("l1_hit","l2_hit","l3_hit","l3_miss"))
    t = l1+l2+l3+m
    if not t: return None
    return (l1*4+l2*12+l3*40+m*200)/t, l1/t*100, m/t*100

rows = []
src = "replay" if rep["tma"] else "live cgroup"
tm = tma_of(rep["tma"]) if rep["tma"] else at
rows.append((f"IPC (tool-exec, {src})", f"{tm['ipc']:.2f}"))
rows.append(("TMA Retiring / FE / BadSpec / BE", f"{tm['ret']:.0f}% / {tm['fe']:.0f}% / {tm['bad']:.0f}% / {tm['be']:.0f}%"))
ca = cache_amat(rep["cache"])
rows.append(("AMAT (cyc) · L1 hit% · miss%", f"{ca[0]:.1f} · {ca[1]:.0f}% · {ca[2]:.1f}%" if ca else "N/A — deep replay failed on this traj"))
if rep["fp"]:
    fl, avx = fp_share(rep["fp"]); rows.append(("FP lane-weighted FLOPs · AVX-512 share", f"{fl/1e6:.1f}M · {avx:.0f}%"))
else:
    fl = avx = 0; rows.append(("FP / AVX-512 (deep replay)", "N/A — run-replay incompatible w/ traj"))
mlp_a = rep["mlp"]
pend, pendc = mlp_a.get("l1d_pend_miss.pending",0), mlp_a.get("l1d_pend_miss.pending_cycles",0)
uops = mlp_a.get("uops_executed.thread",0); cyc = mlp_a.get("cycles",0) or 1
if pendc: rows.append(("MLP · ILP", f"{pend/pendc:.2f} · {uops/cyc:.2f}"))
rows.append(("Inference-CPU IPC (vLLM engine)", f"{it['ipc']:.2f}"))
rows.append(("vLLM-engine vs tool-exec cycles", f"{it['cyc']/1e9:.0f}B vs {at['cyc']/1e9:.1f}B  ({it['cyc']/(at['cyc'] or 1):.0f}×)"))

fig, ax = plt.subplots(figsize=(9, 0.6+0.5*len(rows))); ax.axis("off")
tbl = ax.table(cellText=[[k,v] for k,v in rows], colLabels=["CPU microarch metric (tool-exec)", "value"],
               cellLoc="left", colLoc="left", loc="center", colWidths=[0.62,0.38])
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 1.5)
for j in range(2): tbl[0,j].set_facecolor("#4C72B0"); tbl[0,j].set_text_props(color="w", weight="bold")
ax.set_title("④ CPU microarchitecture — tool-exec phase (deep replay)\n"+TITLE, fontsize=11, weight="bold", pad=14)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "4_microarch_table.png"), dpi=130, bbox_inches="tight"); plt.close()

print("=== wrote 4 deliverables to", OUT, "===")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"): print("  ", f)
print(f"\nsummary: {n_turns} turns/{sum(acts.values())} actions | GPU-inference {infq:.0f}% wall | "
      f"infer-CPU {it['cyc']/1e9:.0f}B cyc (IPC {it['ipc']:.2f}) vs tool-exec {at['cyc']/1e9:.1f}B (IPC {at['ipc']:.2f})")
if rep['fp']: print(f"replay FP: {fl/1e6:.1f}M FLOPs, AVX-512 {avx:.0f}%  (light edits -> expect ~0 AVX)")
