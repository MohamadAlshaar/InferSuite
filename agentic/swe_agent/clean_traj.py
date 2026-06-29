#!/usr/bin/env python3
"""Clean a SWE-agent .traj for function_calling replay: truncate the history at the first
assistant turn that has no tool_calls (the trailing 'Exit'/submit turn that run-replay
rejects with 'trajectory item N is missing a tool call'). Keeps the longest valid prefix.

Usage: clean_traj.py <in.traj> <out.traj>
"""
import json, sys
inp, out = sys.argv[1], sys.argv[2]
t = json.load(open(inp))
h = t.get("history", [])
cut = len(h)
for i, m in enumerate(h):
    if m.get("role") == "assistant" and not m.get("tool_calls"):
        cut = i; break
dropped = len(h) - cut
t["history"] = h[:cut]
# keep trajectory consistent if present (it is replayed step-wise); truncate proportionally
tr = t.get("trajectory")
if isinstance(tr, list) and len(tr) > 0 and len(h) > 0:
    keep = max(0, round(len(tr) * cut / len(h)))
    t["trajectory"] = tr[:keep]
json.dump(t, open(out, "w"))
print(f"cleaned: dropped {dropped} trailing history item(s) from index {cut}; "
      f"history {len(h)}->{cut}")
