#!/usr/bin/env python3
"""De-risk speculation: measure decode tok/s for 7B alone (base) vs 7B + 0.5B speculative draft at k tokens.
The speedup IS the de-risk number (speedup ~= mean accepted length). Same tokenizer (Qwen2.5) so no aligner.
Usage: measure_spec.py {base|<k>}   (run once per mode in its own process for clean GPU)."""
import sys, os, time, json
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"   # engine in-process -> clean teardown on exit
import torch
from vllm import LLM, SamplingParams

HERE = "/home/mohamad/llm-service-kernel-latest/agentic/inference"
P = json.load(open(f"{HERE}/prompts.json"))
prompts = [p["text"] for p in P if 2000 <= p["n_tokens"] <= 8000][:1]    # SINGLE-STREAM (batch=1) = the regime
mode = sys.argv[1]   # base | ngram<k> | <k>(draft, vocab-blocked)

kw = dict(model="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=10000,
          gpu_memory_utilization=0.90, disable_log_stats=False, enforce_eager=True)
if mode.startswith("ngram"):
    k = int(mode[5:] or 5)
    kw["speculative_config"] = {"method": "ngram", "num_speculative_tokens": k, "prompt_lookup_max": k, "prompt_lookup_min": 2}
elif mode != "base":
    kw["speculative_config"] = {"model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": int(mode)}
llm = LLM(**kw)

sp = SamplingParams(max_tokens=200, temperature=0.0, ignore_eos=True, min_tokens=200)  # greedy target -> clean acceptance
llm.generate(prompts[:1], sp); torch.cuda.synchronize()      # warmup
t = time.time(); out = llm.generate(prompts, sp); dt = time.time() - t
ntok = sum(len(o.outputs[0].token_ids) for o in out)
print(f"RESULT mode={mode}: {ntok} tok / {dt:.2f}s = {ntok/dt:.1f} tok/s  over {len(prompts)} seqs", flush=True)
