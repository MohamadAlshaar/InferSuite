#!/usr/bin/env python3
"""H100 / FP16(bf16, UNQUANTIZED) 32B version of run_regime.py — identical methodology.
enforce_eager=True (profileable), enable_prefix_caching=False + distinct warmup (real prefill),
FlashAttention-2 backend, NVTX 'GEN' fence around only the measured pass.
Usage: run_regime_h100.py {prefill|decode|normal}"""
import sys, os, json, random
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # engine in-process -> kernels visible to ncu
os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"
os.environ["VLLM_FLASH_ATTN_VERSION"] = "2"   # FA2 kernel is ncu-profileable; FA3 (cutlass persistent) is not
import torch
from vllm import LLM, SamplingParams

regime = sys.argv[1] if len(sys.argv) > 1 else "prefill"
HERE = os.path.dirname(os.path.abspath(__file__))
P = json.load(open(f"{HERE}/prompts.json")); random.seed(0)

llm = LLM(model="Qwen/Qwen2.5-32B-Instruct", max_model_len=18000,
          gpu_memory_utilization=0.95, enforce_eager=True, disable_log_stats=True,
          enable_prefix_caching=False, max_num_seqs=1)   # bf16, unquantized

if regime == "prefill":            # large context in, 1 token out -> pure prefill GEMM/attention
    prompt = max((p for p in P if p["n_tokens"] <= 16000), key=lambda p: p["n_tokens"])["text"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
elif regime == "decode":           # 1-token prompt + long generation -> every step is M=1 (pure decode)
    prompt = "Hello"
    sp = SamplingParams(max_tokens=256, temperature=0.7, ignore_eos=True, min_tokens=256)
else:                              # normal: a real agent prompt, realistic output
    prompt = random.choice([p for p in P if 8000 <= p["n_tokens"] <= 16000])["text"]
    sp = SamplingParams(max_tokens=64, temperature=0.7, ignore_eos=True, min_tokens=64)

# ---- warmup (NOT profiled): distinct throwaway prompt so the measured prompt is never pre-cached ----
llm.generate(["Warm up the engine and the allocator before profiling."],
             SamplingParams(max_tokens=8, temperature=0.0))
torch.cuda.synchronize()
# ---- measured pass inside NVTX 'GEN' ----
torch.cuda.nvtx.range_push("GEN")
llm.generate([prompt], sp)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print(f"REGIME {regime} done")
