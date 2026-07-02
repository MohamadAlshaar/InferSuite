#!/usr/bin/env python3
"""Parse torch.profiler chrome traces -> per-kernel-class GPU time-share (incl FA3 Attention, a
CUTLASS sm90 kernel ncu cannot replay) -> inject into gpu_tma.json by_kernel_class[cls][time_pct].
ncu keeps the microarch (uarch/SoL) for profileable kernels; this supplies the time% for ALL classes."""
import json, os
def kclass(name):
    n=(name or "").lower()
    if "arlin" in n: return "AWQ GEMM (Marlin)"
    if "reshape_and_cache" in n: return "KV-cache write"
    if "flash" in n or "attention" in n or "fmha" in n: return "Attention (flash)"
    if "rms_norm" in n or "rmsnorm" in n: return "RMSNorm"
    if "rotary" in n: return "RoPE"
    if "act_and_mul" in n or "silu" in n or "gelu" in n or "swiglu" in n: return "Activation (SwiGLU)"
    if any(t in n for t in ("nvjet","xmma","cutlass","sm90","s16816","wgmma","cublas","gemm","gemv","cgemm","hgemm")): return "GEMM (bf16)"
    if "elementwise" in n or "vectorized" in n or "index" in n or "slot_mapping" in n or "cat" in n: return "Elementwise/index"
    return "Other"
GT=os.path.expanduser("~/gpu_h100/runs/ncu/gpu_tma.json"); G=json.load(open(GT))
def cls_share(r):
    p=os.path.expanduser(f"~/gpu_h100/runs/nsys/{r}_trace.json")
    if not os.path.exists(p): return {}
    ev=json.load(open(p)).get("traceEvents",[]); ct={}
    for e in ev:
        if isinstance(e,dict) and e.get("cat")=="kernel" and e.get("dur",0)>0:
            ct[kclass(e.get("name",""))]=ct.get(kclass(e.get("name","")),0.0)+e["dur"]
    tot=sum(ct.values()) or 1
    return {c:100*v/tot for c,v in ct.items()}
for r in ["prefill","decode","normal"]:
    if r not in G: continue
    sh=cls_share(r)
    if not sh: print(f"{r}: no trace"); continue
    bkc=G[r].setdefault("by_kernel_class",{})
    for c,pct in sh.items(): bkc.setdefault(c,{})["time_pct"]=round(pct,2)
    print(f"{r}: "+" | ".join(f"{c} {pct:.0f}%" for c,pct in sorted(sh.items(),key=lambda x:-x[1]) if pct>=1))
json.dump(G,open(GT,"w"),indent=1); print("injected torch-profiler time-share (incl FA3 Attention) ->",GT)
