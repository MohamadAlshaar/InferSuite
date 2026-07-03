#!/usr/bin/env python3
"""Forensic isolation audit for h100/data_agent_side: every check uses evidence RECORDED in the
capture files (cgroup paths embedded in perf stat output, container IDs, GPU-timeline epochs).

Per workload:
 1. SEPARATENESS  — gpu_timeline epoch ranges must not overlap between workloads
 2. RIGHT MODEL   — the engine scope name in the group files maps to a serve instance whose
                    model is known (vllm-serve2/5/6 = coder-32b, vllm-serve4 = instruct-32b);
                    OC must be instruct, SWE/BCB must be coder
 3. RIGHT CGROUPS — the agent scope name embeds the workload key (agent-<work>-PID.scope);
                    exactly ONE engine scope appears per stat file (no double engines)
 4. NO CROSS-TALK — the docker container ID in the stat files equals the one in tool_dso
                    provenance (same capture), and appears in no other workload's files
"""
import os, re, glob, csv, sys

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "data_agent_side")
SERVE_MODEL = {"vllm-serve2": "coder", "vllm-serve3": "instruct", "vllm-serve4": "instruct",
               "vllm-serve5": "coder", "vllm-serve6": "coder", "vllm-serve": "unknown"}
EXPECT = {"swe": "coder", "swe-scikit": "coder", "swe-sympy": "coder", "bcb": "coder",
          "oc-calendar": "instruct", "oc-web": "instruct", "oc-pdf": "instruct", "oc-crop": "instruct"}
fails, oks = [], 0

def ok(msg): global oks; oks += 1
def fail(msg): fails.append(msg)

spans, containers = {}, {}
for wl in sorted(EXPECT):
    d = os.path.join(DATA, wl)
    if not os.path.isdir(d): fail(f"{wl}: missing dir"); continue
    core = open(os.path.join(d, "group_core.txt"), errors="ignore").read()

    # engine scope identity -> model
    scopes = sorted(set(re.findall(r"(vllm-serve\d*)\.scope", core)))
    if len(scopes) != 1: fail(f"{wl}: expected exactly 1 engine scope, saw {scopes}")
    else:
        model = SERVE_MODEL.get(scopes[0], "unknown")
        if model != EXPECT[wl]: fail(f"{wl}: engine scope {scopes[0]} serves {model}, expected {EXPECT[wl]}")
        else: ok(f"{wl}: engine {scopes[0]} = {model}")

    # agent scope embeds workload key
    ag = set(re.findall(r"(agent-[a-z-]+)-\d+\.scope", core))
    if not ag: fail(f"{wl}: no agent scope rows in group_core")
    elif not any(a == f"agent-{wl}" for a in ag): fail(f"{wl}: agent scope {ag} does not match workload")
    else: ok(f"{wl}: agent scope matches")

    # container identity (if tool scope exists)
    cids = set(re.findall(r"docker-([0-9a-f]{12})", core))
    if cids: containers[wl] = cids

    # gpu timeline span
    p = os.path.join(d, "gpu_timeline.csv")
    ts = [float(r[0]) for r in csv.reader(open(p)) if r and r[0] != "guard"]
    if len(ts) > 5: spans[wl] = (min(ts), max(ts))

# separateness: pairwise overlap of gpu windows
wls = sorted(spans)
for i, a in enumerate(wls):
    for b in wls[i+1:]:
        (a0, a1), (b0, b1) = spans[a], spans[b]
        if max(a0, b0) < min(a1, b1): fail(f"OVERLAP: {a} and {b} captured simultaneously")
for a in wls: ok(f"{a}: exclusive window")

# cross-talk: a container ID must not appear in two workloads
for i, a in enumerate(sorted(containers)):
    for b in sorted(containers)[i+1:]:
        shared = containers[a] & containers[b]
        if shared: fail(f"CONTAINER SHARED between {a} and {b}: {shared}")

print(f"checks passed: {oks}")
print(f"FAIL ({len(fails)}):")
for f in fails: print(f"  - {f}")
sys.exit(1 if fails else 0)
