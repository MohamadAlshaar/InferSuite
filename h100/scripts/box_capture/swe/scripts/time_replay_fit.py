"""Whole-loop inference timing with context truncated to fit max_model_len (mimics SWE-agent
history processor: keep system + most-recent messages). Forced-decode = recorded output tokens.
Sums per-turn GPU time over ALL turns -> honest per-task inference wall-time."""
import sys, os, time, json, urllib.request, urllib.error
sys.path.insert(0, "/home/ubuntu/swe/scripts")
import traj_replay_engine as R
BASE = os.environ.get("VLLM", "http://localhost:8000/v1"); MODEL = os.environ.get("MODEL", "coder-32b")
CAP = 16384
CPT = 2.6  # chars/token (conservative for code/JSON)
def tok(s): return int(len(s) / CPT) + 1
def fit(msgs, n):
    budget = 14800 - n   # big margin below max_model_len to absorb token-estimate error
    sysm = [m for m in msgs if m["role"] == "system"]; rest = [m for m in msgs if m["role"] != "system"]
    base = sum(tok(m["content"]) for m in sysm)
    kept = []; total = base
    for m in reversed(rest):
        c = tok(m["content"])
        if total + c > budget and kept: break
        kept.append(m); total += c
    kept.reverse()
    out = sysm + kept
    # guaranteed fit: repeatedly trim the largest message tail until total <= budget
    for _ in range(64):
        over = sum(tok(m["content"]) for m in out) - budget
        if over <= 0: break
        big = max(range(len(out)), key=lambda i: len(out[i]["content"]))
        keepchars = max(120, len(out[big]["content"]) - int((over + 16) * CPT))
        out[big] = dict(out[big]); out[big]["content"] = out[big]["content"][-keepchars:]
    return out
def one(msgs, n):
    body = json.dumps({"model": MODEL, "messages": msgs, "max_tokens": n, "min_tokens": n,
                       "ignore_eos": True, "temperature": 0.4}).encode()
    req = urllib.request.Request(BASE + "/chat/completions", body,
                                 {"Content-Type": "application/json", "Authorization": "Bearer dummy"})
    with urllib.request.urlopen(req, timeout=600) as r: json.load(r)
def run(argv):
    out = {}
    for traj in argv:
        name = os.path.basename(traj).split("__")[0]; reqs = R.build(traj)
        t0 = time.time(); ok = 0; fail = 0
        for ctx, n in reqs:
            if not ctx: continue
            nn = min(n, 2048)
            try: one(fit(ctx, nn), nn); ok += 1
            except urllib.error.HTTPError as e:
                fail += 1; print("  ", name, "HTTP", e.code, e.read().decode()[:120], flush=True)
            except Exception as e: fail += 1; print("  ", name, "err", str(e)[:80], flush=True)
        dt = time.time() - t0
        out[name] = {"infer_s": round(dt, 2), "ok": ok, "fail": fail, "turns": len(reqs)}
        print(f"INFER {name}: {dt:.2f}s  ok={ok} fail={fail}/{len(reqs)}", flush=True)
    json.dump(out, open("/home/ubuntu/swe/infer_times.json", "w"), indent=2)
    print("INFER_ALL_DONE")
if __name__ == "__main__":
    run(sys.argv[1:])
