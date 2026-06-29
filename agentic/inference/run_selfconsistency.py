#!/usr/bin/env python3
"""MAKE-OR-BREAK test of the thesis core: does the HARDWARE-PREDICTED search width B* (=16, from f(B) alone)
match the task-measured quality-per-wall-second optimum? GSM8K self-consistency: per question sample B reasoning
chains (single-agent regime: ONE question at a time so the only GPU parallelism is the B candidates => f(B)
applies), majority-vote the answer. Sweep B, measure accuracy and wall-time, find the acc/time peak."""
import os, re, time, collections
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
from vllm import LLM, SamplingParams
from datasets import load_dataset

N = 50
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
def eq(a, b):
    try: return abs(float(a) - float(b)) < 1e-3
    except: return False

print(f"PREDICTION (hardware f(B) only): B* ~= 16, peak in {{8,16}}, must beat B=1 and B=32. N={N} GSM8K q.\n")
print(f"{'B':>3} {'acc':>6} {'time_s':>8} {'acc/time(x100)':>14}")
results = []
for B in (1, 4, 8, 16, 32):
    sp = SamplingParams(n=B, temperature=0.7, top_p=0.95, max_tokens=256)
    correct = 0; t0 = time.time()
    for q, gt in qs:
        out = llm.chat([msgs(q)], sp, use_tqdm=False)[0]
        votes = [extract(o.text) for o in out.outputs]; votes = [v for v in votes if v]
        if votes:
            ans = collections.Counter(votes).most_common(1)[0][0]
            if eq(ans, gt): correct += 1
    dt = time.time() - t0; acc = correct / N
    results.append((B, acc, dt, acc / dt))
    print(f"{B:>3} {acc:>6.3f} {dt:>8.1f} {acc/dt*100:>14.4f}", flush=True)

peak = max(results, key=lambda r: r[3])
print(f"\nEMPIRICAL acc/time PEAK at B={peak[0]} (acc {peak[1]:.3f}, {peak[2]:.0f}s)")
print(f"  B=1 acc/time {results[0][3]*100:.4f} | B=32 {results[-1][3]*100:.4f}")
hit = peak[0] in (8, 16) and peak[3] > results[0][3] and peak[3] > results[-1][3]
print(f"VERDICT: predicted B*~16 {'== ' if peak[0]==16 else ('~ ' if peak[0]==8 else '!= ')}empirical B={peak[0]}"
      f"  ->  {'WIN (device-curve prediction hits task optimum)' if hit else 'MISS (prediction does not hold)'}")
print("SC_DONE")
