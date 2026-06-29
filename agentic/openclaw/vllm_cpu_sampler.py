#!/usr/bin/env python3
"""4th probe: vLLM inference-server CPU usage (cores) over time -> vllm_timeline.csv.
The serving-side CPU (tokenize/schedule/sample/detok, enforce-eager engine loop) that the
sandbox-cgroup and host-agent probes do NOT capture. Samples /proc deltas of the vLLM
processes (host-visible via minikube's PID namespace). Records epoch,cores (epoch-aligned
like the GPU sampler, so downstream alignment doesn't assume a fixed interval)."""
import os, glob, time, sys
CLK = os.sysconf("SC_CLK_TCK")
out = sys.argv[1]

def vllm_pids():
    r = []
    for d in glob.glob("/proc/[0-9]*"):
        try:
            cl = open(d + "/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except Exception:
            continue
        if ("vllm" in cl.lower() or "EngineCore" in cl) and "sampler" not in cl and "grep" not in cl:
            r.append(d.split("/")[-1])
    return r

pids = vllm_pids()

def ticks():
    t = 0
    for p in pids:
        try:
            rem = open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()  # robust to comm with spaces/parens
            t += int(rem[11]) + int(rem[12])  # utime + stime (fields 14,15 -> idx 11,12 after ')')
        except Exception:
            pass
    return t

last, lt = ticks(), time.time()
with open(out, "w") as f:
    while True:
        time.sleep(1)
        now, cur = time.time(), ticks()
        cores = (cur - last) / CLK / (now - lt) if now > lt else 0.0
        f.write(f"{now:.3f},{cores:.3f}\n"); f.flush()
        last, lt = cur, now
