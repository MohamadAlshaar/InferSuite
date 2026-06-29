#!/usr/bin/env python3
"""Diagnose the split: is the missing speedup poor OVERLAP (fixable) or dispatch overhead (fundamental)?
Measure each engine's slice alone, then naive vs threaded overlap. overlap_eff = max(slice_G,slice_C)/(split-exch)."""
import time, threading, torch
import torch.nn.functional as F
torch.set_num_threads(12)
dev = "cuda"; d_model, d_int = 3584, 18944; DT = torch.bfloat16

def make(B, fg):
    di_g = int(d_int * fg)
    g = torch.randn(d_int, d_model, dtype=DT, device=dev)*0.02
    u = torch.randn(d_int, d_model, dtype=DT, device=dev)*0.02
    d = torch.randn(d_model, d_int, dtype=DT, device=dev)*0.02
    W = dict(g=g,u=u,d=d, gg=g[:di_g].contiguous(), ug=u[:di_g].contiguous(), dg=d[:,:di_g].contiguous(),
             gc=g[di_g:].cpu().contiguous(), uc=u[di_g:].cpu().contiguous(), dc=d[:,di_g:].cpu().contiguous())
    return W, torch.randn(d_model, B, dtype=DT, device=dev)*0.5

def t_(fn, it=50, wm=10):
    for _ in range(wm): fn()
    torch.cuda.synchronize(); s=time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter()-s)/it

for B in (1, 8):
    print(f"\n===== B={B}, frac_gpu=0.62 =====")
    W, x = make(B, 0.62); x_c = x.to("cpu")
    full = lambda: W["d"] @ (F.silu(W["g"]@x)*(W["u"]@x))
    gpu_slice = lambda: W["dg"] @ (F.silu(W["gg"]@x)*(W["ug"]@x))
    cpu_slice = lambda: W["dc"] @ (F.silu(W["gc"]@x_c)*(W["uc"]@x_c))
    tF = t_(full); tG = t_(gpu_slice); tC = t_(lambda:(cpu_slice(),None))
    def naive():
        xc = x.to("cpu")
        yg = W["dg"] @ (F.silu(W["gg"]@x)*(W["ug"]@x))
        yc = W["dc"] @ (F.silu(W["gc"]@xc)*(W["uc"]@xc))
        torch.cuda.synchronize(); return yg + yc.to(dev)
    def threaded():
        xc = x.to("cpu"); r={}
        def job(): r["y"] = W["dc"] @ (F.silu(W["gc"]@xc)*(W["uc"]@xc))
        th = threading.Thread(target=job); th.start()
        yg = W["dg"] @ (F.silu(W["gg"]@x)*(W["ug"]@x))
        torch.cuda.synchronize(); th.join(); return yg + r["y"].to(dev)
    tN = t_(naive); tT = t_(threaded)
    print(f"GPU full        : {tF*1e3:6.3f} ms")
    print(f"GPU slice (0.62): {tG*1e3:6.3f} ms   CPU slice (0.38): {tC*1e3:6.3f} ms   max={max(tG,tC)*1e3:.3f}")
    print(f"split naive     : {tN*1e3:6.3f} ms  -> speedup {tF/tN:.3f}x  overlap_eff {max(tG,tC)/tN:.2f}")
    print(f"split threaded  : {tT*1e3:6.3f} ms  -> speedup {tF/tT:.3f}x  overlap_eff {max(tG,tC)/tT:.2f}")
    print(f"ideal (perfect overlap) = max(slice)+exch ~= {max(tG,tC)*1e3:.3f} ms -> ceiling speedup {tF/max(tG,tC):.3f}x")
