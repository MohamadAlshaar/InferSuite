#!/usr/bin/env python3
"""OFFLINE policy eval (SYSTEM python3). chains.json = 32 chains/q with CPU-side entropy + confidence.
Two questions:
  (Q1 informativeness) does the CPU entropy signal beat RANDOM gating at matched cost? (ClaudesLens premise transfer)
  (Q2 usefulness)      does ANY entropy-driven adaptive policy beat UNIFORM self-consistency on acc-vs-cost?
Policies: uniform-B; warmup-m then escalate-to-K if uncertain, gated by {chain-0 entropy, disagreement of first m}."""
import json, collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
np.random.seed(0)
HERE = "/home/mohamad/llm-service-kernel-latest/agentic/inference"
data = json.load(open(f"{HERE}/runs/sync/chains.json"))
N = len(data)

def ok(ans, gt):
    try: return abs(float(ans) - float(gt)) < 1e-3
    except: return False
def vote(chains):
    c = collections.Counter(ch["ans"] for ch in chains if ch["ans"])
    return c.most_common(1)[0][0] if c else None
def disagree(chains):                       # fraction NOT in the plurality among given chains
    c = collections.Counter(ch["ans"] for ch in chains if ch["ans"])
    if not c: return 1.0
    return 1 - c.most_common(1)[0][1] / len(chains)

UNI = [(B, float(np.mean([ok(vote(q["chains"][:B]), q["gt"]) for q in data]))) for B in (1,2,3,4,6,8,12,16,24,32)]

def warmup_gate_curve(signal_of, taus, m, K, higher_uncertain=True):
    """generate m chains always (cost m); if signal>tau escalate to K (vote over K) else keep vote-of-m."""
    out=[]
    for tau in taus:
        a,c=[],[]
        for q in data:
            s = signal_of(q, m)
            unc = (s > tau) if higher_uncertain else (s < tau)
            if unc: a.append(ok(vote(q["chains"][:K]), q["gt"])); c.append(K)
            else:   a.append(ok(vote(q["chains"][:m]),  q["gt"])); c.append(m)
        out.append((float(np.mean(c)), float(np.mean(a))))
    return out

ent0  = lambda q,m: q["chains"][0]["ent"]
disM  = lambda q,m: disagree(q["chains"][:m])
qs = np.linspace(0,100,26)
POL = {
  "entropy gate m1 K8":   warmup_gate_curve(ent0, np.percentile([ent0(q,1) for q in data], qs), 1, 8),
  "entropy gate m1 K16":  warmup_gate_curve(ent0, np.percentile([ent0(q,1) for q in data], qs), 1, 16),
  "entropy gate m1 K32":  warmup_gate_curve(ent0, np.percentile([ent0(q,1) for q in data], qs), 1, 32),
  "disagree gate m4 K32": warmup_gate_curve(disM, np.linspace(-0.01,1.0,26), 4, 32),
}
# RANDOM null (escalate random fraction to 32)
RAND=[]
for p in np.linspace(0,1,26):
    accs=[np.mean([ok(vote(q["chains"][:32]),q["gt"]) if np.random.RandomState(s+int(p*97)).random()<p
                   else ok(q["chains"][0]["ans"],q["gt"]) for q in data]) for s in range(15)]
    RAND.append((1+p*31, float(np.mean(accs))))

tgt = UNI[-1][1] - 0.005
def cost_to(curve, t):
    ok_=[c for c,a in curve if a>=t]; return min(ok_) if ok_ else float('inf')
print(f"N={N} | uniform: B1={UNI[0][1]:.3f} B4={UNI[3][1]:.3f} B8={UNI[5][1]:.3f} B16={UNI[7][1]:.3f} B32={UNI[-1][1]:.3f}")
print(f"\nQ2 cost to reach acc {tgt:.3f}:  UNIFORM={cost_to(UNI,tgt):.1f}")
for k,v in POL.items(): print(f"    {k:22s} {cost_to(v,tgt):5.1f}  {'<= beats uniform' if cost_to(v,tgt)<cost_to(UNI,tgt) else ''}")
print(f"    {'random null':22s} {cost_to(RAND,tgt):5.1f}")

def acc_at(curve,cost): return min(((abs(c-cost),a) for c,a in curve))[1]
print("\nQ1 informativeness (acc at matched avg-cost): entropy-gate-K32 vs RANDOM vs UNIFORM")
for cost in (4,8,16):
    print(f"  cost {cost:2d}:  entropy {acc_at(POL['entropy gate m1 K32'],cost):.3f} | random {acc_at(RAND,cost):.3f} | uniform {acc_at(UNI,cost):.3f}")

plt.figure(figsize=(7.2,5))
plt.plot([c for c,_ in UNI],[a for _,a in UNI],"ks-",label="uniform self-consistency",ms=5,lw=2)
plt.plot([c for c,_ in RAND],[a for _,a in RAND],"--",color="gray",label="random gate (null)")
for k,st in [("entropy gate m1 K32","o-"),("disagree gate m4 K32","^-")]:
    plt.plot([c for c,_ in POL[k]],[a for _,a in POL[k]],st,label=k,ms=4)
plt.xlabel("avg chains / question  (GPU cost proxy)"); plt.ylabel("accuracy")
plt.title("Entropy-gated adaptive compute vs uniform self-consistency (GSM8K, Qwen2.5-7B, N=80)")
plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(f"{HERE}/runs/sync/adaptive_pareto.png", dpi=130)
print(f"\nsaved adaptive_pareto.png")
