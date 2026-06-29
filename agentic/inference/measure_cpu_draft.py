#!/usr/bin/env python3
"""De-risk the ACTUAL mechanism: run the 0.5B draft model ON THE CPU (bf16/AMX) and measure its
autoregressive decode rate. For CPU-draft/GPU-verify to work, the CPU must draft k tokens faster than
the GPU verifies one block (~22 ms at the measured 45 tok/s base), else the CPU is the bottleneck."""
import time, torch
torch.set_num_threads(12)
from transformers import AutoModelForCausalLM, AutoTokenizer

M = "Qwen/Qwen2.5-0.5B-Instruct"
tok = AutoTokenizer.from_pretrained(M)
model = AutoModelForCausalLM.from_pretrained(M, torch_dtype=torch.bfloat16).eval()
print(f"loaded {M} on CPU bf16 | threads {torch.get_num_threads()} | AMX {torch.cpu._is_amx_tile_supported() if hasattr(torch.cpu,'_is_amx_tile_supported') else '?'}")

ids = tok("Write a long, detailed technical essay about computer architecture and memory systems.", return_tensors="pt").input_ids
with torch.no_grad():
    model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=True)   # warmup
for N in (64, 200):
    t = time.time()
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=N, do_sample=False, use_cache=True)
    dt = time.time() - t; n = out.shape[1] - ids.shape[1]
    print(f"CPU 0.5B decode: {n} tok / {dt:.2f}s = {n/dt:.0f} tok/s  ({dt/n*1e3:.1f} ms/token)")

# feasibility vs GPU verify window
rate = n / dt
print(f"\nGPU verify window ~ 22 ms/block (45 tok/s base). For the CPU to stay off the critical path:")
for k in (3, 5, 8):
    draft_ms = k / rate * 1e3
    print(f"  k={k}: CPU drafts {k} tok in {draft_ms:.0f} ms  -> {'OK (< 22ms)' if draft_ms < 22 else 'CPU IS THE BOTTLENECK (> 22ms)'}")
