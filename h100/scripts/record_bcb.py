#!/usr/bin/env python3
"""RECORD run: the live agentic BCB loop, but it SAVES the work so the other perf groups can be
measured by REPLAY (deterministic across counter groups) instead of re-generating live:
  - every executed tool program -> $CORPUS/prog_<tid>_<turn>.py   (replayed by replay_tool.sh)
  - every LLM request (messages + completion_tokens) -> $CORPUS/requests.jsonl  (replayed by replay_engine.py)
The engine (during-inference) CORE group is measured LIVE around this by capture_orchestration.sh.
Usage: record_bcb.py <N_tasks> <MAXTURNS>   (CORPUS via env)
"""
import os, re, sys, ast, json, time, subprocess, tempfile, urllib.request
from bigcodebench.data import get_bigcodebench
BASE=os.environ.get("VLLM","http://localhost:8000/v1"); MODEL=os.environ.get("MODEL","coder-32b")
N=int(sys.argv[1]) if len(sys.argv)>1 else 12; MAXTURNS=int(sys.argv[2]) if len(sys.argv)>2 else 6
CORPUS=os.environ.get("CORPUS","/home/ubuntu/bcb/corpus"); os.makedirs(CORPUS,exist_ok=True)
REQ=open(f"{CORPUS}/requests.jsonl","w")
MARK=open("/tmp/bcb_agentic_markers.txt","a")
def mark(t): MARK.write(f"{time.time():.3f} {t}\n"); MARK.flush()
def chat(messages):
    body=json.dumps({"model":MODEL,"messages":messages,"temperature":0.4,"max_tokens":1400,"frequency_penalty":0.3}).encode()
    req=urllib.request.Request(BASE+"/chat/completions",body,{"Content-Type":"application/json","Authorization":"Bearer dummy"})
    with urllib.request.urlopen(req,timeout=180) as r: j=json.load(r)
    ct=j.get("usage",{}).get("completion_tokens",0)
    REQ.write(json.dumps({"messages":messages,"completion_tokens":ct})+"\n"); REQ.flush()
    return j["choices"][0]["message"]["content"]
def extract_code(txt):
    m=re.findall(r"```(?:python)?\n(.*?)```",txt,re.S); return m[-1] if m else txt
def run_test(code,test,tid,turn):
    prog=code+"\n\n"+test+"\n\nimport unittest\nif __name__=='__main__':\n    unittest.main()\n"
    open(f"{CORPUS}/prog_{tid.replace('/','_')}_{turn}.py","w").write(prog)  # save for tool replay
    with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f: f.write(prog); path=f.name
    mark(f"toolexec_start {tid}")
    try:
        p=subprocess.run([sys.executable,path],capture_output=True,text=True,timeout=40)
        passed=(p.returncode==0); out=(p.stderr or p.stdout)[-1500:]
    except subprocess.TimeoutExpired: passed,out=False,"TIMEOUT"
    finally: mark(f"toolexec_end {tid}"); os.unlink(path)
    return passed,out
NET_LIBS={"ftplib","requests","urllib","socket","smtplib","http","ssl","telnetlib","imaplib"}
def _libs(e):
    v=e.get("libs",[])
    if isinstance(v,str):
        try: v=ast.literal_eval(v)
        except Exception: v=[]
    return set(v)
def main():
    d=get_bigcodebench(subset="hard")
    tids=[t for t in d if not (NET_LIBS & _libs(d[t]))][:N]
    SYS=("You are a Python coding agent. Implement the requested function exactly (keep the name "
         "`task_func`). Reply with ONE ```python``` code block only.")
    solved=0; turns=0; mark("RUN_START")
    for tid in tids:
        e=d[tid]; msgs=[{"role":"system","content":SYS},{"role":"user","content":e["instruct_prompt"]}]
        for turn in range(MAXTURNS):
            turns+=1
            try: reply=chat(msgs)
            except Exception as ex: print(f"  {tid} chat-err {ex}"); break
            code=extract_code(reply); passed,out=run_test(code,e["test"],tid,turn)
            if passed: solved+=1; print(f"  {tid} SOLVED turn {turn+1}"); break
            msgs.append({"role":"assistant","content":reply})
            msgs.append({"role":"user","content":f"The tests failed:\n{out}\nFix the function (task_func)."})
        else: print(f"  {tid} unsolved after {MAXTURNS} turns")
    mark("RUN_END")
    print(f"=== RECORD BigCodeBench: {solved}/{len(tids)} solved, {turns} turns; corpus={CORPUS} "
          f"({len(os.listdir(CORPUS))} files) ===")
if __name__=="__main__": main()
