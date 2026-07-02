#!/usr/bin/env python3
"""REPLAY the recorded LLM request stream to vLLM with FORCED decode length, so the engine does the
IDENTICAL prefill+decode work under every perf group (deterministic across counter groups). The
engine is measured externally by capture_orchestration.sh (scoped to the vLLM PIDs) while this feeds
the recorded requests. Reads requests.jsonl ({messages, completion_tokens} per line).
Usage: replay_engine.py <requests.jsonl>
"""
import os, sys, json, urllib.request
BASE=os.environ.get("VLLM","http://localhost:8000/v1"); MODEL=os.environ.get("MODEL","coder-32b")
REQF=sys.argv[1] if len(sys.argv)>1 else os.environ.get("CORPUS","/home/ubuntu/bcb/corpus")+"/requests.jsonl"
MARK=open("/tmp/bcb_agentic_markers.txt","a")
def mark(t):
    import time; MARK.write(f"{time.time():.3f} {t}\n"); MARK.flush()
def one(messages, ct):
    n=max(1,int(ct))
    body=json.dumps({"model":MODEL,"messages":messages,"max_tokens":n,"min_tokens":n,
                     "ignore_eos":True,"temperature":0.4,"frequency_penalty":0.3}).encode()
    req=urllib.request.Request(BASE+"/chat/completions",body,
                               {"Content-Type":"application/json","Authorization":"Bearer dummy"})
    with urllib.request.urlopen(req,timeout=300) as r: json.load(r)
def main():
    reqs=[json.loads(l) for l in open(REQF) if l.strip()]
    mark("RUN_START")
    for i,e in enumerate(reqs):
        try: one(e["messages"], e.get("completion_tokens",1))
        except Exception as ex: print(f"  req{i} err {ex}")
    mark("RUN_END")
    print(f"=== REPLAY_ENGINE: {len(reqs)} recorded requests replayed with forced decode ===")
if __name__=="__main__": main()
