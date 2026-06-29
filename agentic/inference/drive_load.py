#!/usr/bin/env python3
"""Sustained concurrent generation load on the local vLLM, to keep the engine busy
(steady state) while perf samples the CPU. Replays the real agent prompts.
Usage: drive_load.py <duration_s> <concurrency> <max_tokens>"""
import sys, json, time, random, threading, urllib.request

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 420
CONC = int(sys.argv[2]) if len(sys.argv) > 2 else 16
OUT = int(sys.argv[3]) if len(sys.argv) > 3 else 256
URL = "http://localhost:8000/v1/completions"
P = json.load(open("/home/mohamad/llm-service-kernel-latest/agentic/inference/prompts.json"))
random.seed(1)

stop = time.time() + DUR
done = [0]; toks = [0]; lock = threading.Lock()

def worker():
    while time.time() < stop:
        pr = random.choice(P)
        body = json.dumps({"model": "qwen7b", "prompt": pr["text"], "max_tokens": OUT,
                           "temperature": 0.7, "ignore_eos": True}).encode()
        try:
            req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=600)
            d = json.loads(r.read())
            with lock:
                done[0] += 1; toks[0] += d.get("usage", {}).get("completion_tokens", 0)
        except Exception:
            pass

ts = [threading.Thread(target=worker, daemon=True) for _ in range(CONC)]
[t.start() for t in ts]
t0 = time.time()
while time.time() < stop:
    time.sleep(5)
    el = time.time() - t0
    print(f"[load] {el:5.0f}s  reqs={done[0]:4d}  decode_tok/s={toks[0]/el:6.0f}", flush=True)
[t.join(timeout=2) for t in ts]
print(f"[load] DONE reqs={done[0]} total_decode_tok={toks[0]}", flush=True)
