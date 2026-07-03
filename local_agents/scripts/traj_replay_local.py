#!/usr/bin/env python3
"""Replay a recorded (Claude-Sonnet-driven) SWE-agent trajectory's request stream against the LOCAL
k3s engine, with forced decode = the recorded reply's token count. Adapted from
h100/scripts/traj_replay_engine.py with keep-recent context truncation for --max-model-len 8192.
Loops forever over the given trajectory (sustained load for capture windows); parent kills it."""
import os, sys, json, time, urllib.request

BASE = os.environ.get("VLLM", "http://10.43.21.159:8000/v1")
MODEL = os.environ.get("MODEL", "qwen2.5-7b-instruct-awq")
CTX_BUDGET = int(os.environ.get("CTX_BUDGET", "6500"))   # est. tokens (chars/4) for the message context
DECODE_CAP = int(os.environ.get("DECODE_CAP", "1024"))

def toks(s): return max(1, len(s) // 4)
def flatten(m):
    r = m.get("role", "user"); c = m.get("content", "")
    if isinstance(c, list): c = " ".join((x.get("text", "") if isinstance(x, dict) else str(x)) for x in c)
    if r not in ("system", "user", "assistant"): r = "user"
    return {"role": r, "content": str(c)}

def truncate(msgs):
    """keep system + most recent messages within CTX_BUDGET est. tokens"""
    sys_msgs = [m for m in msgs if m["role"] == "system"]
    rest = [m for m in msgs if m["role"] != "system"]
    budget = CTX_BUDGET - sum(toks(m["content"]) for m in sys_msgs)
    kept = []
    for m in reversed(rest):
        t = toks(m["content"])
        if budget - t < 0 and kept: break
        if t > budget:  # single huge message: clip its tail-end content
            m = {"role": m["role"], "content": m["content"][-budget*4:]}
            t = budget
        kept.append(m); budget -= t
    return sys_msgs + list(reversed(kept))

def one(msgs, n):
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": n, "min_tokens": n,
                       "ignore_eos": True, "temperature": 0.4}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=600) as r: json.load(r)

def build(traj):
    h = json.load(open(traj)).get("history", []); msgs = []; reqs = []
    for m in h:
        if m.get("role") == "assistant":
            reqs.append((truncate(list(msgs)), min(toks(flatten(m)["content"]), DECODE_CAP)))
        msgs.append(flatten(m))
    return reqs

def main():
    traj = sys.argv[1]
    reqs = build(traj)
    print(f"[replay] {os.path.basename(traj)}: {len(reqs)} requests/loop, ctx<= {CTX_BUDGET} est-tok", flush=True)
    loop = 0; ok = err = 0
    while True:
        for i, (ctx, n) in enumerate(reqs):
            if not ctx: continue
            try: one(ctx, n); ok += 1
            except Exception as e:
                err += 1
                if err <= 3: print(f"[replay] req{i} err: {e}", flush=True)
        loop += 1
        print(f"[replay] loop {loop} done (ok={ok} err={err})", flush=True)

if __name__ == "__main__": main()
