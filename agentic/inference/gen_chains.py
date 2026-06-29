#!/usr/bin/env python3
"""COLLECT-ONLY (plot/analyze separately). Test bed for entropy-gated adaptive compute (ClaudesLens UQ -> LLM
agent loop). For N GSM8K questions, sample 32 reasoning chains each; for every chain record: extracted answer,
MEAN TOKEN SHANNON ENTROPY (computed CPU-side from the top-k logprobs vLLM already returns -> the 'free in the
decode shadow' signal), mean chosen-token logprob (confidence), and token count. Dump JSON for offline policy eval."""
import os, re, json, math
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
from datasets import load_dataset

N = 80
ds = load_dataset("openai/gsm8k", "main", split="test")
qs = [(ds[i]["question"], ds[i]["answer"].split("####")[-1].strip().replace(",", "")) for i in range(N)]

llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-AWQ", max_model_len=2048,
          gpu_memory_utilization=0.90, enforce_eager=True, enable_prefix_caching=True)
SYS = "Solve the math problem step by step. End your answer with a line: 'The answer is N' where N is the final number."

def msgs(q): return [{"role": "system", "content": SYS}, {"role": "user", "content": q}]
def extract(t):
    m = re.search(r"answer is\s*\$?(-?[\d,]*\.?\d+)", t, re.I)
    if not m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", t)
        return nums[-1].replace(",", "") if nums else None
    return m.group(1).replace(",", "")

def chain_stats(o):
    """mean per-token Shannon entropy over the returned top-k distribution (CPU-side UQ), + mean chosen-token logprob."""
    ents, chosen = [], []
    for tok_id, lpdict in zip(o.token_ids, o.logprobs or []):
        ps = [math.exp(lp.logprob) for lp in lpdict.values()]
        s = sum(ps) or 1.0
        ps = [p / s for p in ps]                      # renorm over the top-k (proxy for full entropy)
        ents.append(-sum(p * math.log(p + 1e-12) for p in ps))
        if tok_id in lpdict: chosen.append(lpdict[tok_id].logprob)
    return (sum(ents) / len(ents) if ents else 0.0,
            sum(chosen) / len(chosen) if chosen else 0.0)

sp = SamplingParams(n=32, temperature=0.7, top_p=0.95, max_tokens=256, logprobs=20)
data = []
for i, (q, gt) in enumerate(qs):
    out = llm.chat([msgs(q)], sp, use_tqdm=False)[0]
    chains = []
    for o in out.outputs:
        ent, conf = chain_stats(o)
        chains.append({"ans": extract(o.text), "ent": ent, "conf": conf, "ntok": len(o.token_ids)})
    data.append({"gt": gt, "chains": chains})
    if (i + 1) % 10 == 0: print(f"...{i+1}/{N}", flush=True)

json.dump(data, open("/home/mohamad/llm-service-kernel-latest/agentic/inference/runs/sync/chains.json", "w"))
print(f"GEN_DONE N={len(data)} chains/q={len(data[0]['chains'])}")
