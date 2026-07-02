"""Decompose OpenClaw agent-loop wall time into inference (GPU) vs tool (CPU) from chat.jsonl
timestamps: gap ending at an assistant message = LLM generation; gap ending at a user message
= tool-result/exec. Average over all completed runs (>=8 messages) per task."""
import json, glob, os
from datetime import datetime

BASE = "/home/ubuntu/oc/WildClawBench/output/openclaw"
TASKS = {
    "calendar":  "01_Productivity_Flow/01_Productivity_Flow_task_6_calendar_scheduling",
    "pdf-digest":"01_Productivity_Flow/01_Productivity_Flow_task_10_pdf_digest",
    "web-digest":"01_Productivity_Flow/01_Productivity_Flow_task_1_arxiv_digest",
    "image-crop":"05_Creative_Synthesis/05_Creative_Synthesis_task_10_social_poster_multi_crop",
}
def ts(s):
    return datetime.strptime(s.replace("Z","+0000"), "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
def decompose(path):
    msgs = []
    for ln in open(path):
        try: d = json.loads(ln)
        except: continue
        if d.get("type") != "message": continue
        role = (d.get("message") or {}).get("role", "")
        t = d.get("timestamp")
        if t: msgs.append((ts(t), role))
    if len(msgs) < 3: return None
    inf = tool = 0.0
    for i in range(1, len(msgs)):
        dt = msgs[i][0] - msgs[i-1][0]
        if dt < 0: continue
        if msgs[i][1] == "assistant": inf += dt
        elif msgs[i][1] == "toolResult": tool += dt
    wall = msgs[-1][0] - msgs[0][0]
    return inf, tool, wall
out = {}
for name, sub in TASKS.items():
    runs = []
    for d in glob.glob(os.path.join(BASE, sub, "*")):
        cj = os.path.join(d, "chat.jsonl")
        if not os.path.exists(cj): continue
        r = decompose(cj)
        if r and (r[0] + r[1]) > 5:   # meaningful loop (>5s of accounted work)
            runs.append(r)
    if not runs:
        print(name, "-> no usable runs"); continue
    infs = sum(r[0] for r in runs); tools = sum(r[1] for r in runs)
    tot = infs + tools
    toolpct = 100 * tools / tot if tot else 0
    mean_wall = sum(r[2] for r in runs) / len(runs)
    out[name] = {"tool_pct": round(toolpct, 1), "gpu_pct": round(100-toolpct,1),
                 "n_runs": len(runs), "mean_wall_s": round(mean_wall,1),
                 "inf_s": round(infs/len(runs),1), "tool_s": round(tools/len(runs),1)}
    print(f"{name:12} runs={len(runs)} tool%={toolpct:5.1f} gpu%={100-toolpct:5.1f} mean_wall={mean_wall:.0f}s")
json.dump(out, open("/home/ubuntu/oc/oc_timesplit.json","w"), indent=2)
print("WROTE /home/ubuntu/oc/oc_timesplit.json")
