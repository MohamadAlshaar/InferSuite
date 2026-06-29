#!/usr/bin/env python3
"""Ground the co-compute optimization model on the real box.
(1) r_C(B): CPU decode-matmul rate vs batch B (does AMX engage; what usable bandwidth/tok-s).
(2) tau_P : CPU<->GPU small round-trip + sync latency, vs the 303 us/layer feasibility threshold.
Run with the vLLM venv python (torch + cuda)."""
import time, os
import torch

NTHREADS = 12
torch.set_num_threads(NTHREADS)
K, N = 3584, 18944                  # Qwen2.5-7B hidden -> FFN-intermediate: the dominant decode matmul
Wbytes = N * K * 2                  # bf16 weight bytes for this matmul

print(f"torch {torch.__version__} | threads {NTHREADS} | AMX bf16 supported: {torch.cpu._is_amx_tile_supported() if hasattr(torch.cpu,'_is_amx_tile_supported') else 'n/a'}")
print(f"matmul W[{N},{K}] x X[{K},B]  (weight {Wbytes/1e6:.0f} MB bf16)\n")

def bench(W, X, iters=30, warm=5):
    for _ in range(warm): _ = W @ X
    t = time.perf_counter()
    for _ in range(iters): _ = W @ X
    return (time.perf_counter() - t) / iters

print("=== (1) r_C(B): CPU matmul vs batch ===")
print(f"{'B':>4} | {'bf16 ms':>8} {'tok/s':>7} {'GB/s':>6} {'GFLOP/s':>8} | {'fp32 ms':>8} {'AMX speedup':>11}")
Wb = torch.randn(N, K, dtype=torch.bfloat16); Wf = Wb.float()
r_at = {}
for B in (1, 2, 4, 8, 16, 32, 64, 128):
    Xb = torch.randn(K, B, dtype=torch.bfloat16); Xf = Xb.float()
    dtb = bench(Wb, Xb); dtf = bench(Wf, Xf)
    toks = B / dtb; gbs = Wbytes / dtb / 1e9; gf = 2 * N * K * B / dtb / 1e9
    r_at[B] = toks
    print(f"{B:>4} | {dtb*1e3:8.2f} {toks:7.0f} {gbs:6.0f} {gf:8.0f} | {dtf*1e3:8.2f} {dtf/dtb:10.2f}x")
print(f"-> CPU tok/s at B=1: {r_at[1]:.0f} ; at B=16: {r_at[16]:.0f} ; ratio (batching unlock) {r_at[16]/r_at[1]:.1f}x")

print("\n=== (2) tau_P: CPU<->GPU round-trip + sync ===")
if torch.cuda.is_available():
    d = 3584
    xg = torch.randn(d, dtype=torch.float16, device="cuda")
    pin = torch.empty(d, dtype=torch.float16, pin_memory=True)
    torch.cuda.synchronize()
    def probe(fn, iters=2000, warm=200):
        for _ in range(warm): fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(iters): fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t) / iters
    # tiny kernel + sync (the per-step sync cost, spin-mode default)
    t_sync = probe(lambda: (xg.add_(1.0), torch.cuda.synchronize()))
    # d2h + h2d round trip (pageable)
    def rt_pageable():
        y = xg.to("cpu"); z = y.to("cuda"); torch.cuda.synchronize()
    t_rt = probe(rt_pageable, iters=1000)
    # d2h + h2d round trip (pinned)
    def rt_pinned():
        pin.copy_(xg, non_blocking=True); xg.copy_(pin, non_blocking=True); torch.cuda.synchronize()
    t_rtp = probe(rt_pinned, iters=1000)
    print(f"tiny-kernel + sync       = {t_sync*1e6:7.1f} us")
    print(f"round-trip d2h+h2d page  = {t_rt*1e6:7.1f} us")
    print(f"round-trip d2h+h2d pinned= {t_rtp*1e6:7.1f} us")
    tauP = t_rtp * 2     # ~2 exchanges per layer
    print(f"-> tau_P (2 exch/layer, pinned) = {tauP*1e6:.0f} us/layer  vs threshold 303 us/layer -> "
          + ("FEASIBLE (co-compute profitable)" if tauP < 303e-6 else "INFEASIBLE (sync eats the gain)"))
else:
    print("no CUDA")
