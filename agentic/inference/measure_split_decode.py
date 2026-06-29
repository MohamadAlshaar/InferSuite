#!/usr/bin/env python3
"""Minimal split-decode prototype: one SwiGLU FFN layer (Qwen2.5-7B dims), bf16, split by intermediate
dim across GPU+CPU. GPU computes a partial down-proj output, CPU computes the rest IN PARALLEL, sum the
two d_model-sized partials (the only exchange). Tests the model's predicted ~1.6x over GPU-only at decode.
Run with the vLLM venv python."""
import time, torch
import torch.nn.functional as F
torch.set_num_threads(12)
dev = "cuda"
d_model, d_int = 3584, 18944
DT = torch.bfloat16

def make(B, frac_gpu):
    di_g = int(d_int * frac_gpu); di_c = d_int - di_g
    # full weights on GPU (baseline); slices are exact partitions of the same weights
    gate = torch.randn(d_int, d_model, dtype=DT, device=dev) * 0.02
    up   = torch.randn(d_int, d_model, dtype=DT, device=dev) * 0.02
    down = torch.randn(d_model, d_int, dtype=DT, device=dev) * 0.02
    W = dict(gate=gate, up=up, down=down,
             gate_g=gate[:di_g].contiguous(), up_g=up[:di_g].contiguous(), down_g=down[:, :di_g].contiguous(),
             gate_c=gate[di_g:].cpu().contiguous(), up_c=up[di_g:].cpu().contiguous(), down_c=down[:, di_g:].cpu().contiguous())
    x = torch.randn(d_model, B, dtype=DT, device=dev) * 0.5
    return W, x, di_g, di_c

def ffn_gpu(W, x):
    h = F.silu(W["gate"] @ x) * (W["up"] @ x)
    return W["down"] @ h

def ffn_cpu(W, x_c):
    h = F.silu(W["gate_c"] @ x_c) * (W["up_c"] @ x_c)
    return W["down_c"] @ h

def ffn_split(W, x):
    x_c = x.to("cpu")                       # exchange in: small d_model activation (~tau_P)
    h_g = F.silu(W["gate_g"] @ x) * (W["up_g"] @ x)   # GPU slice, ASYNC (kernels queued)
    y_g = W["down_g"] @ h_g
    h_c = F.silu(W["gate_c"] @ x_c) * (W["up_c"] @ x_c)  # CPU slice runs WHILE GPU kernels execute
    y_c = W["down_c"] @ h_c
    y_cg = y_c.to(dev)                       # exchange out: small d_model partial
    torch.cuda.synchronize()                 # wait for GPU slice
    return y_g + y_cg

def timed(fn, W, x, iters=50, warm=10):
    for _ in range(warm): fn(W, x)
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(iters): fn(W, x)
    torch.cuda.synchronize(); return (time.perf_counter() - t) / iters

for B in (1, 4, 8):
    print(f"\n===== batch B={B} =====")
    W, x, di_g, di_c = make(B, 0.62)
    # correctness
    err = (ffn_split(W, x).float() - ffn_gpu(W, x).float()).abs().max().item()
    rng = ffn_gpu(W, x).float().abs().max().item()
    print(f"correctness: max abs err {err:.4f} (range {rng:.2f}, rel {err/max(rng,1e-9):.1e})")
    t_gpu = timed(ffn_gpu, W, x); t_cpu = timed(lambda W,x: ffn_cpu(W, x.to('cpu')), W, x)
    print(f"GPU-only  : {t_gpu*1e3:6.3f} ms   ({1e3/t_gpu*B:.0f} ffn-tok/s)")
    print(f"CPU-only  : {t_cpu*1e3:6.3f} ms")
    # sweep the split fraction to find the real balance
    print("  frac_gpu | split ms | speedup vs GPU-only")
    best = (0, 1e9)
    for fg in (0.5, 0.6, 0.62, 0.7, 0.8, 0.9):
        W2, x2, _, _ = make(B, fg)
        t = timed(ffn_split, W2, x2)
        sp = t_gpu / t
        print(f"    {fg:.2f}   | {t*1e3:7.3f}  | {sp:.3f}x" + ("  <-- best" if t < best[1] else ""))
        if t < best[1]: best = (fg, t)
    print(f"  BEST: frac_gpu={best[0]:.2f}, split {best[1]*1e3:.3f} ms, speedup {t_gpu/best[1]:.3f}x  (model predicted ~1.6x)")
