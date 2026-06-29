#!/usr/bin/env python3
"""Agentic BigCodeBench-Hard with a HOSTED Claude model (Anthropic API).
Faithful to BCB data prep: uses the official `instruct_prompt` + the official `test` (run as the
real tool) from the Hard subset via get_bigcodebench(subset='hard'). Agentic loop (deviation from
BCB's one-shot design, as chosen): generate -> RUN the real test (tool) -> feed errors back -> fix
-> repeat up to MAXTURNS. The test runs every turn (real numpy/scipy/etc CPU = the tool-exec we
measure). Inference is remote (Claude), so local CPU during a chat() call is ~idle (API wait).

Captures: per-task solved/turns(loops) + per-turn test execution_time; markers around each tool-exec
(for the time split); and an executed.jsonl of every (code,test) actually run (for a clean,
deterministic per-counter-group microarch replay afterwards).
Usage: agentic_bcb_claude.py <N_tasks|all> <max_turns>
"""
import os, re, sys, ast, json, time, subprocess, tempfile, shutil
import anthropic
from bigcodebench.data import get_bigcodebench

MODEL   = os.environ.get("MODEL", "claude-sonnet-4-6")
N_ARG   = sys.argv[1] if len(sys.argv) > 1 else "all"
MAXTURNS= int(sys.argv[2]) if len(sys.argv) > 2 else 4
OUTDIR  = os.environ.get("OUTDIR", "runs/agentic_claude"); os.makedirs(OUTDIR, exist_ok=True)
MARK    = open("/tmp/bcb_agentic_markers.txt", "a")
EXEC_LOG= open(os.path.join(OUTDIR, "executed.jsonl"), "w")
def mark(tag): MARK.write(f"{time.time():.3f} {tag}\n"); MARK.flush()

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
SYS = ("You are a Python coding agent. Implement the requested function exactly "
       "(keep the name `task_func`). Reply with ONE ```python``` code block only — "
       "include all imports the function needs.")

def chat(messages):
    # Claude: system is a separate param; send ONLY temperature (Claude 4.x rejects temperature+top_p together)
    r = client.messages.create(model=MODEL, system=SYS, messages=messages,
                               max_tokens=4096, temperature=0.0)
    return "".join(b.text for b in r.content if b.type == "text")

def extract_code(txt):
    m = re.findall(r"```(?:python)?\n(.*?)```", txt, re.S)
    return m[-1] if m else txt

def run_test(code, test, tid, turn):
    prog = code + "\n\n" + test + "\n\nimport unittest\nif __name__=='__main__':\n    unittest.main()\n"
    # Isolated CLEAN dir so the script's sys.path[0] has NO stdlib-shadowing files (e.g. a stray /tmp/re.py).
    d = tempfile.mkdtemp(prefix="bcb_exec_")
    path = os.path.join(d, "solution_test.py")
    open(path, "w").write(prog)
    EXEC_LOG.write(json.dumps({"task_id": tid, "turn": turn, "code": code, "test": test}) + "\n"); EXEC_LOG.flush()
    mark(f"toolexec_start {tid}"); t0 = time.time()
    try:
        p = subprocess.run([sys.executable, path], capture_output=True, text=True, timeout=30, cwd=d)
        passed = (p.returncode == 0); out = (p.stderr or p.stdout)[-1800:]
    except subprocess.TimeoutExpired:
        passed, out = False, "TIMEOUT (30s)"
    dt = time.time() - t0
    mark(f"toolexec_end {tid}"); shutil.rmtree(d, ignore_errors=True)
    return passed, out, dt

NET_LIBS = {"ftplib","requests","urllib","socket","smtplib","http","ssl","telnetlib","imaplib"}
def _libs(e):
    v = e.get("libs", [])
    if isinstance(v, str):
        try: v = ast.literal_eval(v)
        except Exception: v = []
    return set(v)

def main():
    d = get_bigcodebench(subset="hard")
    tids = [t for t in d if not (NET_LIBS & _libs(d[t]))]          # skip network tasks (hang on blocked net)
    if N_ARG != "all": tids = tids[:int(N_ARG)]
    results = []; solved = 0; turns_total = 0
    mark("RUN_START")
    for tid in tids:
        e = d[tid]
        msgs = [{"role": "user", "content": e["instruct_prompt"]}]
        rec = {"task_id": tid, "solved": False, "turns": 0, "exec_times": []}
        for turn in range(MAXTURNS):
            turns_total += 1; rec["turns"] = turn + 1
            try: reply = chat(msgs)
            except Exception as ex: rec["error"] = str(ex)[:200]; print(f"  {tid} chat-err {ex}"); break
            passed, out, dt = run_test(extract_code(reply), e["test"], tid, turn)
            rec["exec_times"].append(round(dt, 3))
            if passed:
                rec["solved"] = True; solved += 1; print(f"  {tid} SOLVED turn {turn+1}"); break
            msgs.append({"role": "assistant", "content": reply})
            msgs.append({"role": "user", "content": f"The tests failed:\n{out}\nFix the function (task_func)."})
        else:
            print(f"  {tid} unsolved after {MAXTURNS} turns")
        results.append(rec)
    mark("RUN_END")
    json.dump({"model": MODEL, "n_tasks": len(tids), "solved": solved, "turns_total": turns_total,
               "results": results}, open(os.path.join(OUTDIR, "results.json"), "w"), indent=1)
    print(f"=== agentic BCB-Hard ({MODEL}): {solved}/{len(tids)} solved, {turns_total} total turns/tool-runs ===")

if __name__ == "__main__":
    main()
