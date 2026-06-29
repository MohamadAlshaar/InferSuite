#!/usr/bin/env python3
"""Deterministically re-run the BCB tool-exec (code+test) recorded during the agentic loop, so a
single perf pass can wrap the whole workload -> clean un-multiplexed per-counter-group microarch.
By default replays the LAST execution per task (the final solution attempt) — representative of the
tool-exec, bounded to ~one run/task. Runs each in an isolated clean dir (no /tmp stdlib shadowing).
Usage: replay_executions.py <executed.jsonl> [all|last]
"""
import os, sys, json, time, subprocess, tempfile, shutil, collections
EXEC = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "last"
recs = [json.loads(l) for l in open(EXEC)]
if MODE == "last":
    by = collections.OrderedDict()
    for r in recs: by[r["task_id"]] = r        # keep last per task
    recs = list(by.values())
npass = nfail = 0; t0 = time.time()
for r in recs:
    prog = r["code"] + "\n\n" + r["test"] + "\n\nimport unittest\nif __name__=='__main__':\n    unittest.main()\n"
    d = tempfile.mkdtemp(prefix="bcb_replay_"); p = os.path.join(d, "s.py"); open(p, "w").write(prog)
    try:
        rc = subprocess.run([sys.executable, p], capture_output=True, text=True, timeout=30, cwd=d).returncode
        npass += (rc == 0); nfail += (rc != 0)
    except subprocess.TimeoutExpired:
        nfail += 1
    shutil.rmtree(d, ignore_errors=True)
print(f"replayed {len(recs)} executions ({MODE}): {npass} pass / {nfail} fail in {time.time()-t0:.1f}s")
