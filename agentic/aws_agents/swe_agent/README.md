# SWE-agent on SWE-bench (Qwen2.5-32B, our perf methodology)

Code-REPAIR agent on a real SWE-bench instance (scikit-learn__scikit-learn-10297, Ridge
store_cv_values bug). fc_local.yaml = function-calling mode (bash + edit_anthropic + submit).
Runs in the SWE-bench Docker container; perf = container cgroup (TMA + cache/FP/MLP).

## Config breaks fixed this session
1. Duplicate --agent.model.completion_kwargs -> 2nd overrode 1st, silently dropping
   parallel_tool_calls:false (the exit_format breakage). FIX: merged into one.
2. Missing external/SWE-agent/config/fc_local.yaml (editable install never staged external/). FIX: scp'd it.
3. SWE-agent caches solved instances -> perf re-run skipped. FIX: rm runs/heavy before perf run.

## Result (104-step trajectory, submitted a patch)
- window 985s | tool-exec 33s (3%) / inference 952s (97%) | **29 CPU core-seconds**
- TMA: IPC 1.66, **Retiring 44 / FE 33 / Bad-spec 11 / Backend 14** | **AVX 0%** | L1 99% | MLP 1.7
- Most core-seconds of any LIVE agentic run (104 steps of real Python), but RETIRING+FRONTEND-bound
  Python INTERPRETER (imports, reproduce scripts, edit/bash tools, light Ridge instantiation) —
  NOT heavy numerics. The bug is an API-signature fix, so the vectorized numpy/BLAS test suite
  never got triggered (AVX 0%).

## CROSS-WORKLOAD CPU MAP (the thesis deliverable)
All agentic workloads are inference/GPU-dominated by wall-clock (90-100%). The CPU character differs:
| workload | core-s | TMA bound | AVX | what the CPU does |
|---|---|---|---|---|
| BigCodeBench GT-exec | 290 | Backend | 26% AVX-512 | heavy numpy/BLAS (compute payload) |
| SWE-agent (104 steps) | 29 | Retiring+FE | 0% | Python interpreter (light scikit-learn) |
| OpenClaw (tools) | 4 | FE+Backend | 0% | node.js/V8 + curl/python |
| claw-eval multimodal | 1.2 | Backend | 50% | chromium/graphics render |
| claw-eval general/multi_turn | ~0 | - | - | file-I/O / text consultation |
=> The CPU lights up only on the TOOL PAYLOAD: vectorized compute (numpy=AVX-512, render=AVX) is
backend-bound; agent runtimes (python interpreter, node.js) are retiring/frontend-bound, no AVX.
