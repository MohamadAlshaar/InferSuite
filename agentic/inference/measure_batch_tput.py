#!/usr/bin/env python3
"""Ground the architecture model: does GPU decode FREE-BATCH? Sweep B concurrent sequences that SHARE a
prefix (tree-search shape; prefix-caching reuses the parent KV) and measure aggregate decode tok/s vs B.
Free-batching => aggregate tok/s rises ~linearly with B (more candidates ~= same time) up to a crossover."""
import os, time, json
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams

HERE = "/home/mohamad/llm-service-kernel-latest/agentic/inference"
P = json.load(open(f"{HERE}/prompts.json"))
prompt = min((p for p in P if p["n_tokens"] >= 800), key=lambda p: p["n_tokens"])["text"]  # ~1K shared prefix

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=4096,
          gpu_memory_utilization=0.90, enforce_eager=True, enable_prefix_caching=True)
sp = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True, min_tokens=128)
llm.generate([prompt], sp)   # warmup + prime the shared prefix into the cache

print(f"{'B':>4} {'agg tok/s':>10} {'per-seq':>8} {'vs B=1':>7}")
base = None
for B in (1, 2, 4, 8, 16, 32, 64):
    prompts = [prompt] * B
    t = time.time(); out = llm.generate(prompts, sp); dt = time.time() - t
    ntok = sum(len(o.outputs[0].token_ids) for o in out)
    agg = ntok / dt; per = agg / B
    if base is None: base = agg
    print(f"{B:>4} {agg:>10.0f} {per:>8.1f} {agg/base:>6.2f}x", flush=True)
print("\nfree-batching => 'vs B=1' rises ~linearly with B (the GPU's efficient regime the architecture needs)")
