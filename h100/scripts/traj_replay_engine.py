#!/usr/bin/env python3
"""Replay a recorded SWE-agent trajectory's REQUEST STREAM to vLLM with forced decode -> the
DURING-inference engine orchestration for the SWE-agent workload shape (long, prefill-heavy contexts,
~70 turns). Reconstructs each request from the .traj `history` (the messages BEFORE each assistant
turn); forced decode length = the recorded assistant message's ~token count (chars/4, capped). The
engine is measured externally by capture_orchestration.sh (scoped to the vLLM PIDs).
Usage: traj_replay_engine.py <trajectory.traj>
"""
import os, sys, json, time, urllib.request
BASE=os.environ.get("VLLM","http://localhost:8000/v1"); MODEL=os.environ.get("MODEL","coder-32b")
TRAJS=sys.argv[1:]
MARK=open("/tmp/swe_markers.txt","a")
def mark(t): MARK.write(f"{time.time():.3f} {t}\n"); MARK.flush()
def toks(s): return max(1, len(s)//4)
def flatten(m):
    r=m.get("role","user"); c=m.get("content","")
    if isinstance(c,list): c=" ".join((x.get("text","") if isinstance(x,dict) else str(x)) for x in c)
    if r not in ("system","user","assistant"): r="user"   # fold 'tool' observations as user turns (prefill shape)
    return {"role":r,"content":str(c)}
def one(msgs,n):
    body=json.dumps({"model":MODEL,"messages":msgs,"max_tokens":n,"min_tokens":n,"ignore_eos":True,
                     "temperature":0.4}).encode()
    req=urllib.request.Request(BASE+"/chat/completions",body,{"Content-Type":"application/json","Authorization":"Bearer dummy"})
    with urllib.request.urlopen(req,timeout=600) as r: json.load(r)
def build(traj):
    h=json.load(open(traj)).get("history",[]); msgs=[]; reqs=[]
    for m in h:
        if m.get("role")=="assistant":
            reqs.append((list(msgs), toks(flatten(m)["content"])))   # request = context before this completion
        msgs.append(flatten(m))
    return reqs
def main():
    mark("RUN_START"); total=0
    for traj in TRAJS:
        reqs=build(traj); total+=len(reqs)
        for i,(ctx,n) in enumerate(reqs):
            if not ctx: continue
            try: one(ctx, min(n,2048))
            except Exception as e: print(f"  {os.path.basename(traj)} req{i} err {e}")
        print(f"  replayed {len(reqs)} reqs from {os.path.basename(traj)}", flush=True)
    mark("RUN_END")
    print(f"=== SWE traj replay: {total} requests from {len(TRAJS)} trajectories ===")
if __name__=="__main__": main()
