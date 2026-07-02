"""torch.profiler (CUPTI) variant of run_regime_h100.py — same model/prompts/regimes, but wraps the
measured GEN pass in a profiler and exports a chrome trace with REAL kernel names + durations, so we
can get the FA3 attention time-share that ncu/nsys cannot. Usage: run_regime_prof.py {prefill|decode|normal}"""
import sys, os, json, random
os.environ["VLLM_USE_FLASHINFER_SAMPLER"]="0"; os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"]="0"
import torch
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams
regime=sys.argv[1] if len(sys.argv)>1 else "prefill"
HERE="/home/ubuntu/gpu_h100"; P=json.load(open(f"{HERE}/prompts.json")); random.seed(0)
llm=LLM(model="Qwen/Qwen2.5-32B-Instruct", max_model_len=18000, gpu_memory_utilization=0.90,
        enforce_eager=True, disable_log_stats=True, enable_prefix_caching=False)
if regime=="prefill":
    prompt=max((p for p in P if p["n_tokens"]<=16000),key=lambda p:p["n_tokens"])["text"]; sp=SamplingParams(max_tokens=1,temperature=0.0)
elif regime=="decode":
    prompt="Hello"; sp=SamplingParams(max_tokens=256,temperature=0.7,ignore_eos=True,min_tokens=256)
else:
    prompt=random.choice([p for p in P if 8000<=p["n_tokens"]<=16000])["text"]; sp=SamplingParams(max_tokens=64,temperature=0.7,ignore_eos=True,min_tokens=64)
llm.generate(["Warm up the engine before profiling."], SamplingParams(max_tokens=8,temperature=0.0)); torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    llm.generate([prompt], sp); torch.cuda.synchronize()
out=f"{HERE}/runs/nsys/{regime}_trace.json"; prof.export_chrome_trace(out)
print(f"PROF {regime} done -> {out}")
