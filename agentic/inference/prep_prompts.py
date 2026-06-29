#!/usr/bin/env python3
"""Extract real agent prompts from the transcripts, tokenize with the Qwen tokenizer,
keep <=32K, and sample ~40 spanning the size range (stratified). -> prompts.json
Run with the bigcodebench venv python (has transformers)."""
import json, glob, ast, random, os
from transformers import AutoTokenizer

AG = "/home/mohamad/llm-service-kernel-latest/agentic"
CAP = 32000           # leave headroom under 32768
random.seed(7)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ")

pool = []   # (text, agent)

# ---- OpenClaw: concatenate message contents per conversation, take prefixes ----
for cj in glob.glob(f"{AG}/openclaw/external/WildClawBench/output/openclaw/**/chat.jsonl", recursive=True):
    msgs = []
    for l in open(cj, errors="ignore"):
        try: d = json.loads(l)
        except: continue
        if d.get("type") != "message": continue
        m = d.get("message", {})
        if not isinstance(m, dict): continue
        c = m.get("content")
        if isinstance(c, list):
            c = " ".join(str(x.get("text","")) if isinstance(x, dict) else str(x) for x in c)
        if isinstance(c, str) and c.strip():
            msgs.append(c.strip())
    # build a few prefixes of growing size
    for frac in (0.25, 0.5, 1.0):
        t = "\n".join(msgs[:max(1, int(len(msgs)*frac))])
        if t: pool.append((t, "OpenClaw"))

# ---- SWE: the per-step 'query' (message list) text ----
for tj in glob.glob(f"{AG}/swe_agent/runs/api/**/*.traj", recursive=True):
    try: d = json.load(open(tj))
    except: continue
    for s in d.get("trajectory", []):
        q = s.get("query")
        if isinstance(q, list):
            t = "\n".join(str(mm.get("content","")) for mm in q if isinstance(mm, dict))
        elif isinstance(q, str): t = q
        else: t = ""
        if t.strip(): pool.append((t.strip(), "SWE-bench"))

# ---- BCB: the recorded prompt/instruction text ----
for l in open(f"{AG}/bigcodebench/runs/agentic_claude/executed.jsonl", errors="ignore"):
    try: d = json.loads(l)
    except: continue
    t = ""
    for k in ("prompt","instruction","code"):
        v = d.get(k)
        if isinstance(v, str): t += v + "\n"
    if t.strip(): pool.append((t.strip(), "BigCodeBench"))

# ---- tokenize, keep <=CAP, dedup-ish ----
sized = []
seen = set()
for text, ag in pool:
    n = len(tok(text, add_special_tokens=False).input_ids)
    if 32 <= n <= CAP:
        key = (ag, n // 256)
        if key in seen and len([x for x in sized if x[2]==ag])>40: continue
        sized.append((text, ag, n)); seen.add(key)

# ---- stratified sample ~40 spanning sizes (buckets, balanced across agents) ----
def bucket(n):
    for b in (1000,4000,8000,16000,32000):
        if n<=b: return b
    return 32000
by = {}
for text, ag, n in sized: by.setdefault((ag,bucket(n)), []).append((text,ag,n))
sample = []
# aim: ~8 BCB(small), ~16 SWE, ~16 OpenClaw, spanning buckets
targets = {"BigCodeBench":8, "SWE-bench":16, "OpenClaw":16}
for ag, k in targets.items():
    buckets = [b for (a,b) in by if a==ag]
    per = max(1, k//max(1,len(set(buckets))))
    picked = []
    for b in sorted(set(buckets)):
        cand = by.get((ag,b), [])
        random.shuffle(cand)
        picked += cand[:per]
    random.shuffle(picked)
    sample += picked[:k]

random.shuffle(sample)
out = [{"agent": ag, "n_tokens": n, "text": text} for (text, ag, n) in sample]
json.dump(out, open(f"{AG}/inference/prompts.json", "w"))
import statistics
ns = [o["n_tokens"] for o in out]
print(f"sampled {len(out)} prompts | tokens min {min(ns)} median {statistics.median(ns):.0f} max {max(ns)}")
from collections import Counter
print("by agent:", dict(Counter(o["agent"] for o in out)))
print("by bucket:", dict(Counter(bucket(o["n_tokens"]) for o in out)))
