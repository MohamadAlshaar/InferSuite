#!/usr/bin/env python3
"""Tool-exec profiling variant of the agentic BCB loop: identical think->act->observe loop, but each
test subprocess (the TOOL, running the generated code) is wrapped in `perf stat` so we capture the
OUTSIDE-inference (tool-exec) microarch on the SAME box as the during-inference orchestration.
The tool runs while the vLLM engine is idle (queue-wait) -> the per-subprocess counters are clean.
Per-run counter CSVs land in $TOOLPERF_DIR; aggregate them for IPC/cache-MPKI/branch-MPKI/FP.
Events via $TOOLPERF_EVENTS (default = core group; run again with the FP group).
Usage: agentic_bcb_toolperf.py <N_tasks> <MAXTURNS>
"""
import os, re, sys, ast, json, time, subprocess, tempfile, urllib.request
from bigcodebench.data import get_bigcodebench

BASE=os.environ.get("VLLM","http://localhost:8000/v1"); MODEL=os.environ.get("MODEL","coder-32b")
N=int(sys.argv[1]) if len(sys.argv)>1 else 12; MAXTURNS=int(sys.argv[2]) if len(sys.argv)>2 else 3
PERF=os.environ.get("PERF","perf")
EVENTS=os.environ.get("TOOLPERF_EVENTS","cycles,instructions,cache-references,cache-misses,branch-instructions,branch-misses")
TP=os.environ.get("TOOLPERF_DIR","/tmp/toolperf"); os.makedirs(TP,exist_ok=True)
MARK=open("/tmp/bcb_agentic_markers.txt","a")
def mark(t): MARK.write(f"{time.time():.3f} {t}\n"); MARK.flush()

def chat(messages):
    body=json.dumps({"model":MODEL,"messages":messages,"temperature":0.4,"max_tokens":1400,"frequency_penalty":0.3}).encode()
    req=urllib.request.Request(BASE+"/chat/completions",body,{"Content-Type":"application/json","Authorization":"Bearer dummy"})
    with urllib.request.urlopen(req,timeout=180) as r: return json.load(r)["choices"][0]["message"]["content"]
def extract_code(txt):
    m=re.findall(r"```(?:python)?\n(.*?)```",txt,re.S); return m[-1] if m else txt

def run_test(code,test,tid,turn):
    prog=code+"\n\n"+test+"\n\nimport unittest\nif __name__=='__main__':\n    unittest.main()\n"
    with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f: f.write(prog); path=f.name
    stat=f"{TP}/{tid.replace('/','_')}_{turn}.csv"
    mark(f"toolexec_start {tid}")
    try:
        cmd=[PERF,"stat","-x,","-o",stat,"-e",EVENTS,"--",sys.executable,path]
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=40)
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
    print(f"=== toolperf BigCodeBench: {solved}/{len(tids)} solved, {turns} tool-exec runs (events={EVENTS.split(',')[0]}...) ===")
if __name__=="__main__": main()
