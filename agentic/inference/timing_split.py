#!/usr/bin/env python3
"""Measure the prefill-vs-decode GPU-time split for a representative agent turn, so the 'Prompts'
regime can be presented as the real time-weighted blend of the prefill and decode operating points."""
import os, time, json, random
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"; os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
import torch
from vllm import LLM, SamplingParams
HERE = "/home/mohamad/llm-service-kernel-latest/agentic/inference"
P = json.load(open(f"{HERE}/prompts.json")); random.seed(0)
llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=32768, gpu_memory_utilization=0.90,
          enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False)
prompt = random.choice([p for p in P if 8000 <= p["n_tokens"] <= 16000])

def timed(o):
    llm.generate(["warm"], SamplingParams(max_tokens=4))  # distinct warmup so prompt not cached-effect
    torch.cuda.synchronize(); t = time.time()
    llm.generate([prompt["text"]], SamplingParams(max_tokens=o, temperature=0.7, ignore_eos=True, min_tokens=o))
    torch.cuda.synchronize(); return time.time() - t

t1 = timed(1); t129 = timed(129)
pf = t1                       # ~ prefill (+1 decode step)
dec_per_tok = (t129 - t1) / 128
out = {"prompt_tokens": prompt["n_tokens"], "prefill_s": pf, "decode_per_tok_s": dec_per_tok}
for O in (64, 128, 256):
    td = dec_per_tok * O
    out[f"prefill_frac_O{O}"] = pf / (pf + td)
print(json.dumps(out, indent=2))
json.dump(out, open(f"{HERE}/runs/ncu/timing_split.json", "w"), indent=2)
