#!/usr/bin/env python3
"""Consolidate the VERIFIED Phase-1 (CPU-during-inference) results + the tool-exec comparison
into inf_thesis_plots/data.json. All non-multiplexed; FP from the 2 clean split passes."""
import sys, json, os, collections
sys.path.insert(0, "../common"); import microarch as M
HERE = "/home/mohamad/llm-service-kernel-latest"
P = "runs/phase1"

t = M.parse(f"{P}/group_TMA.txt"); c = M.parse(f"{P}/group_CACHE.txt"); m = M.parse(f"{P}/group_MLP.txt")
fa = M.parse(f"{P}/group_FP_A.txt"); fb = M.parse(f"{P}/group_FP_B.txt")
fp = collections.Counter(); fp.update(fa);
for k, v in fb.items():
    if k.startswith("fp_arith"): fp[k] = v   # merge the 256/512 events from pass B

l2 = M.tma_l2(t); ch = M.cache_hits(c)
infer = dict(
    ipc=M.ipc(t),
    tma_l1={"Retiring": l2["retiring"], "Frontend": l2["fe_bound"], "BadSpec": l2["bad_spec"], "Backend": l2["be_bound"]},
    tma_l2={k: l2[k] for k in ("light_ops","heavy_ops","fetch_lat","fetch_bw","br_mispred","machine_clears","mem_bound","core_bound")},
    l1_hit=ch["l1"], l2_hit=ch["l2"], l3_hit=ch["l3"], miss=ch["miss"], mpki=ch["mpki"],
    avx=M.avx_pct(fp), mflop=M.flops(fp)/1e6, mlp=M.mlp(m), ilp=M.ilp(m),
    cycles_per_60s=t.get("cycles",0),
    cross_pass_ipc={"TMA": M.ipc(t), "CACHE": M.ipc(c), "FP_A": M.ipc(fa), "FP_B": M.ipc(fb), "MLP": M.ipc(m)},
)
# function/library breakdown during inference (verified from perf report --sort=dso)
infer["dso"] = {"libcuda.so (CUDA driver / sync)": 85.31, "[vdso] (clock_gettime)": 10.86,
                "libc.so.6": 1.30, "python3.12": 1.13, "[kernel]": 0.42,
                "tokenizers": 0.29, "libtorch_cpu.so": 0.18, "libtorch_python.so": 0.17}
infer["sync_path_pct"] = 96.2   # libcuda + vdso (busy-wait + timing)
infer["top_symbol"] = "cuEventSynchronize"
infer["dominant_process"] = "VLLM::EngineCore (99.2% of samples)"

# tool-exec comparison: the 8 agent workloads (CANONICAL) + aggregate
SPEC = [("astropy","SWE-bench","swe_bench/data/astropy-14096",False),
        ("scikit-learn","SWE-bench","swe_bench/data/scikit-learn-25232",False),
        ("sympy","SWE-bench","swe_bench/data/sympy-14248",False),
        ("code-gen","BigCodeBench","bigcodebench/data",False),
        ("calendar","OpenClaw","openclaw/data/calendar",True),
        ("image-crop","OpenClaw","openclaw/data/social_poster_crop",True),
        ("web-digest","OpenClaw","openclaw/data/arxiv",True),
        ("pdf-digest","OpenClaw","openclaw/data/pdf_digest",True)]
tool = []
for task, bench, d, up in SPEC:
    g = lambda x: M.parse(f"{HERE}/agentic/CANONICAL/{d}/group_{x.upper() if up else x}{'_r1' if up else ''}.txt")
    tm = g("tma")
    L = M.tma_l1(tm)
    tool.append({"task": task, "bench": bench, "ipc": M.ipc(tm),
                 "tma_l1": {"Retiring": L["retiring"], "Frontend": L["fe-bound"], "BadSpec": L["bad-spec"], "Backend": L["be-bound"]}})
# aggregate tool-exec TMA (mean)
agg = {k: sum(w["tma_l1"][k] for w in tool)/len(tool) for k in ("Retiring","Frontend","BadSpec","Backend")}
agg_ipc = sum(w["ipc"] for w in tool)/len(tool)

out = {"inference": infer, "tool_exec_workloads": tool,
       "tool_exec_aggregate": {"ipc": agg_ipc, "tma_l1": agg},
       "_provenance": "Phase-1 CPU-during-inference: local vLLM Qwen2.5-7B-AWQ on RTX A2000, sustained agent-prompt load, perf scoped to whole engine (API server + VLLM::EngineCore, 232 threads). Non-multiplexed (FP from 2 split passes). Tool-exec = CANONICAL 8-workload TMA."}
json.dump(out, open(f"{HERE}/inf_thesis_plots/data.json", "w"), indent=2)

print("INFERENCE CPU:")
print(f"  IPC {infer['ipc']:.2f}  cross-pass {min(infer['cross_pass_ipc'].values()):.2f}-{max(infer['cross_pass_ipc'].values()):.2f}")
print(f"  TMA L1: ret {infer['tma_l1']['Retiring']:.0f} fe {infer['tma_l1']['Frontend']:.0f} bad {infer['tma_l1']['BadSpec']:.0f} be {infer['tma_l1']['Backend']:.0f}")
print(f"  L1-hit {infer['l1_hit']:.1f}%  MLP {infer['mlp']:.2f} ILP {infer['ilp']:.2f}  AVX {infer['avx']:.0f}% MFLOP {infer['mflop']:.0f} (CLEAN, split passes)")
print(f"  sync-path {infer['sync_path_pct']:.0f}% (libcuda+vdso)  top symbol {infer['top_symbol']}")
print(f"TOOL-EXEC aggregate: IPC {agg_ipc:.2f}  TMA ret {agg['Retiring']:.0f} fe {agg['Frontend']:.0f} bad {agg['BadSpec']:.0f} be {agg['Backend']:.0f}")
print("-> wrote inf_thesis_plots/data.json")
