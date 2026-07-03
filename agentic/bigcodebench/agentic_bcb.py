#!/usr/bin/env python3
"""Agentic BigCodeBench: a real think->act->observe loop (vs the one-shot GT probe).
Each task: model writes code -> harness RUNS the test (TOOL, real numpy/pandas CPU) ->
feeds the error back -> model fixes -> repeat. The test runs EVERY turn regardless of
whether the weak model emits tool-calls, so tool-exec CPU genuinely fires (unlike OpenClaw).

Writes markers around each test-exec (the tool phase) so the perf wrapper can separate
GPU-generation time from CPU-tool time. Usage:
    agentic_bcb.py <N_tasks> <max_turns>
"""
import os, re, sys, ast, json, time, subprocess, tempfile, urllib.request
from bigcodebench.data import get_bigcodebench

BASE = os.environ.get("VLLM", "http://localhost:8000/v1")
MODEL = os.environ.get("MODEL", "qwen2.5-32b")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
MAXTURNS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
MARK = open("/tmp/bcb_agentic_markers.txt", "a")
def mark(tag): MARK.write(f"{time.time():.3f} {tag}\n"); MARK.flush()

def chat(messages):
    body = json.dumps({"model": MODEL, "messages": messages, "temperature": 0.4,
                       "max_tokens": 1400, "frequency_penalty": 0.3}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)["choices"][0]["message"]["content"]

def extract_code(txt):
    m = re.findall(r"```(?:python)?\n(.*?)```", txt, re.S)
    return m[-1] if m else txt   # fall back to raw

def run_test(code, test, tid):
    """Write model code + test, run it (the TOOL). Returns (passed, output_tail)."""
    prog = code + "\n\n" + test + "\n\nimport unittest\nif __name__=='__main__':\n    unittest.main()\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(prog); path = f.name
    mark(f"toolexec_start {tid}")
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=20)
        passed = (p.returncode == 0)
        out = (p.stderr or p.stdout)[-1500:]
    except subprocess.TimeoutExpired:
        passed, out = False, "TIMEOUT (60s)"
    finally:
        mark(f"toolexec_end {tid}"); os.unlink(path)
    return passed, out

NET_LIBS = {"ftplib", "requests", "urllib", "socket", "smtplib", "http", "ssl", "telnetlib", "imaplib"}
def _libs(e):
    """libs is stored as a STRING repr of a list (e.g. "['ftplib','os']") -> parse it."""
    v = e.get("libs", [])
    if isinstance(v, str):
        try: v = ast.literal_eval(v)
        except Exception: v = []
    return set(v)
def main():
    d = get_bigcodebench(subset="hard")
    # SKIP network/IO-blocking tasks: their tests dial real servers (e.g. ftp.dlptest.com)
    # and hang on the sandbox's blocked network -> timeouts that are NOT compute.
    tids = [t for t in d if not (NET_LIBS & _libs(d[t]))]
    if os.environ.get("HEAVY_LIBS"):   # compute-heavy subset: tasks whose tests exercise numeric libs
        want = {"sklearn", "pandas", "scipy", "numpy", "matplotlib"}
        tids = [t for t in tids if want & _libs(d[t])]
    tids = tids[:N]
    SYS = ("You are a Python coding agent. Implement the requested function exactly "
           "(keep the name `task_func`). Reply with ONE ```python``` code block only.")
    solved = 0; turns_used = 0
    mark("RUN_START")
    for tid in tids:
        e = d[tid]
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": e["instruct_prompt"]}]
        for turn in range(MAXTURNS):
            turns_used += 1
            try: reply = chat(msgs)
            except Exception as ex: print(f"  {tid} chat-err {ex}"); break
            code = extract_code(reply)
            passed, out = run_test(code, e["test"], tid)
            if passed:
                solved += 1; print(f"  {tid} SOLVED turn {turn+1}"); break
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": f"The tests failed:\n{out}\nFix the function (task_func)."})
        else:
            print(f"  {tid} unsolved after {MAXTURNS} turns")
    mark("RUN_END")
    print(f"=== agentic BigCodeBench: {solved}/{len(tids)} solved, {turns_used} total turns (= {turns_used} tool-exec runs) ===")

if __name__ == "__main__":
    main()
