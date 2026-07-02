#!/usr/bin/env python3
"""H100 version of build_gpu_tma.py — same warp-state -> GPU-topdown construction, but kclass() also
recognises the bf16 cuBLAS/cutlass GEMM (unquantized) instead of only Marlin. Output beside this script.
Usage: build_gpu_tma_h100.py <regime> <report.ncu-rep>"""
import sys, csv, io, subprocess, json, os

regime, report = sys.argv[1], sys.argv[2]
HERE = os.path.dirname(os.path.abspath(__file__))
DUR = "gpu__time_duration.sum"
PRE = "smsp__average_warps_issue_stalled_"
SUF = "_per_issue_active.ratio"
REASONS = ["selected","long_scoreboard","short_scoreboard","lg_throttle","tex_throttle","imc_miss",
           "mio_throttle","math_pipe_throttle","no_instruction","branch_resolving","barrier","membar",
           "wait","not_selected","drain","dispatch_stall","misc","sleeping"]
CAT = {
 "selected":"Issued",
 "long_scoreboard":"Mem · global/L2", "short_scoreboard":"Mem · L1/shared",
 "lg_throttle":"Mem · LSU throttle", "tex_throttle":"Mem · LSU throttle", "mio_throttle":"Mem · LSU throttle",
 "imc_miss":"Mem · const-cache",
 "math_pipe_throttle":"Compute · math pipe",
 "wait":"Latency (fixed deps)",
 "no_instruction":"Front-end (fetch)",
 "branch_resolving":"Branch resolve",
 "barrier":"Synchronization", "membar":"Synchronization",
 "not_selected":"Scheduler-covered",
 "drain":"Other","dispatch_stall":"Other","misc":"Other","sleeping":"Other",
}

def num(s):
    s = (s or "").replace(",", "").strip()
    try: return float(s)
    except: return 0.0

raw = subprocess.run(["ncu","-i",report,"--csv","--page","raw"], capture_output=True, text=True).stdout
rows = list(csv.DictReader(io.StringIO(raw)))
def col(frag):
    for k in rows[0].keys():
        if frag in k: return k
    return None
durc = col(DUR) or col("gpu__time_duration.avg")
rcols = {r: col(PRE + r + SUF) for r in REASONS}
smc = col("sm__throughput.avg.pct_of_peak_sustained_elapsed")
memc = col("gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed")
issc = col("smsp__issue_active.avg.per_cycle_active")
knc = col("Kernel Name") or col("Demangled Name") or col("Function Name")

UARCH = {
    "IPC": "sm__inst_issued.avg.per_cycle_active",
    "Occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
    "Eligible warps": "smsp__warps_eligible.avg.per_cycle_active",
    "SIMT eff": "smsp__thread_inst_executed_per_inst_executed.ratio",
    "Tensor pipe": "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
    "FMA pipe": "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active",
    "ALU pipe": "sm__pipe_alu_cycles_active.avg.pct_of_peak_sustained_active",
    "L1 hit": "l1tex__t_sector_hit_rate.pct",
    "L2 hit": "lts__t_sector_hit_rate.pct",
    "DRAM BW": "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "Registers": "launch__registers_per_thread",
}
uac = {k: col(v) for k, v in UARCH.items()}
ua_g = {}

def kclass(name):                                        # group raw kernels into algorithmic classes
    n = (name or "").lower()
    if "arlin" in n: return "AWQ GEMM (Marlin)"
    if "reshape_and_cache" in n: return "KV-cache write"
    if "flash" in n or "attention" in n or "fmha" in n: return "Attention (flash)"
    if "rms_norm" in n or "rmsnorm" in n: return "RMSNorm"
    if "rotary" in n: return "RoPE"
    if "act_and_mul" in n or "silu" in n or "gelu" in n or "swiglu" in n: return "Activation (SwiGLU)"
    # unquantized bf16 matmul (cuBLAS/cutlass on Hopper): the FP16-run analogue of the Marlin GEMM
    if any(t in n for t in ("nvjet","xmma","cutlass","sm90","s16816","wgmma","cublas","gemm","gemv","cgemm","hgemm")):
        return "GEMM (bf16)"
    if "elementwise" in n or "vectorized" in n or "indexselect" in n or "index_" in n or "slot_mapping" in n:
        return "Elementwise/index"
    return "Other"

agg = {}; tot_dur = 0.0; nk = 0; sm_sol = 0.0; mem_sol = 0.0; iss = 0.0
kcls = {}
for row in rows:
    d = num(row.get(durc))
    if d <= 0: continue
    states = {r: num(row.get(rcols[r])) for r in REASONS if rcols[r]}
    s = sum(states.values())
    if s <= 0: continue
    nk += 1; tot_dur += d
    sm_sol += num(row.get(smc)) * d; mem_sol += num(row.get(memc)) * d
    iss += num(row.get(issc)) * d
    issued_frac = states.get("selected", 0.0) / s
    kc = kcls.setdefault(kclass(row.get(knc)), {"dur": 0.0, "sm": 0.0, "mem": 0.0, "issued": 0.0, "iss": 0.0, "cat": {}, "ua": {}})
    kc["dur"] += d; kc["sm"] += num(row.get(smc)) * d; kc["mem"] += num(row.get(memc)) * d
    kc["issued"] += issued_frac * d; kc["iss"] += num(row.get(issc)) * d
    for r, v in states.items():
        agg[r] = agg.get(r, 0.0) + (v / s) * d
        kc["cat"][CAT[r]] = kc["cat"].get(CAT[r], 0.0) + (v / s) * d
    for nm, c in uac.items():
        if c:
            val = num(row.get(c)); kc["ua"][nm] = kc["ua"].get(nm, 0.0) + val * d
            ua_g[nm] = ua_g.get(nm, 0.0) + val * d
brk_reason = {r: agg.get(r, 0.0) / tot_dur * 100 for r in REASONS}
cat = {}
for r, pct in brk_reason.items():
    cat[CAT[r]] = cat.get(CAT[r], 0.0) + pct

out_path = os.path.join(HERE, "runs/ncu/gpu_tma.json")
data = json.load(open(out_path)) if os.path.exists(out_path) else {}
kcls_out = {k: {"time_pct": v["dur"] / tot_dur * 100, "sol_compute": v["sm"] / v["dur"],
                "sol_memory": v["mem"] / v["dur"], "issued_pct": v["issued"] / v["dur"] * 100,
                "issue_eff": v["iss"] / v["dur"],
                "by_category": {c: v["cat"].get(c, 0.0) / v["dur"] * 100 for c in v["cat"]},
                "uarch": {nm: v["ua"].get(nm, 0.0) / v["dur"] for nm in UARCH if uac[nm]}}
            for k, v in kcls.items()}
lane_eff_pct = (ua_g.get("SIMT eff", 0.0) / tot_dur) / 32 * 100
data[regime] = {"kernels": nk, "by_reason": brk_reason, "by_category": cat,
                "sol_compute_pct": sm_sol / tot_dur, "sol_memory_pct": mem_sol / tot_dur,
                "issue_eff_per_cycle": iss / tot_dur, "by_kernel_class": kcls_out,
                "uarch": {nm: ua_g.get(nm, 0.0) / tot_dur for nm in UARCH if uac[nm]},
                "lane_eff_pct": lane_eff_pct,
                "lwr_pct": cat.get("Issued", 0.0) * lane_eff_pct / 100}
json.dump(data, open(out_path, "w"), indent=2)

print(f"=== {regime}: {nk} kernels | Speed-of-Light: compute {sm_sol/tot_dur:.0f}% / memory {mem_sol/tot_dur:.0f}% of peak | issue {iss/tot_dur:.2f}/cycle ===")
for c in sorted(cat, key=lambda k: -cat[k]):
    if cat[c] >= 0.4: print(f"  {c:24s} {cat[c]:5.1f}%")
print("  kernel classes (time% | compute% / mem% of peak):")
for k in sorted(kcls_out, key=lambda x: -kcls_out[x]["time_pct"]):
    v = kcls_out[k]
    if v["time_pct"] >= 1: print(f"    {k:22s} {v['time_pct']:5.1f}%  | {v['sol_compute']:4.0f}% / {v['sol_memory']:4.0f}%")
