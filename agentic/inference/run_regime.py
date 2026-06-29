#!/usr/bin/env python3
"""Offline vLLM forward pass for one inference regime, NVTX-fenced for ncu.
enforce_eager=True so kernels launch individually (profileable; CUDA graphs would hide them).
Attention backend = FlashAttention-2 (vLLM default on this Ampere GPU).
CRITICAL: enable_prefix_caching=False AND a distinct throwaway warmup prompt, so the measured prompt
is computed from scratch (a real prefill) and NOT served from the KV prefix cache. Warmup runs OUTSIDE
the 'GEN' NVTX range; only the measured pass is inside it.
Usage: run_regime.py {prefill|decode|normal}"""
import sys, os, json, random
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # engine in-process -> kernels visible to ncu
import torch
from vllm import LLM, SamplingParams

regime = sys.argv[1] if len(sys.argv) > 1 else "prefill"
HERE = "/home/mohamad/llm-service-kernel-latest/agentic/inference"
P = json.load(open(f"{HERE}/prompts.json")); random.seed(0)

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=32768,
          gpu_memory_utilization=0.90, enforce_eager=True, disable_log_stats=True,
          enable_prefix_caching=False)   # <- so the measured prefill is NOT a prefix-cache hit

if regime == "prefill":            # large context in, 1 token out -> pure prefill GEMM/attention
    prompt = max((p for p in P if p["n_tokens"] <= 16000), key=lambda p: p["n_tokens"])["text"]
    sp = SamplingParams(max_tokens=1, temperature=0.0)
elif regime == "decode":           # 1-token prompt + long generation -> EVERY step is M=1 (pure decode);
    prompt = "Hello"                # no prefill confound, no launch-skip needed. KV grows 1..256 (shallow caveat)
    sp = SamplingParams(max_tokens=256, temperature=0.7, ignore_eos=True, min_tokens=256)
else:                              # normal: a real agent prompt, realistic output
    prompt = random.choice([p for p in P if 8000 <= p["n_tokens"] <= 16000])["text"]
    sp = SamplingParams(max_tokens=64, temperature=0.7, ignore_eos=True, min_tokens=64)

# ---- warmup (NOT profiled): a DISTINCT throwaway prompt so the measured prompt is never pre-cached ----
llm.generate(["Warm up the engine and the allocator before profiling."],
             SamplingParams(max_tokens=8, temperature=0.0))
torch.cuda.synchronize()
# ---- measured pass inside NVTX 'GEN' (first time this prompt is seen -> real prefill) ----
torch.cuda.nvtx.range_push("GEN")
llm.generate([prompt], sp)
torch.cuda.nvtx.range_pop()
torch.cuda.synchronize()
print(f"REGIME {regime} done")
