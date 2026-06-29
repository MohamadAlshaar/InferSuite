"""Runs at interpreter startup in EVERY python process that has this dir on PYTHONPATH
(the vLLM API server AND the spawned EngineCore worker), BEFORE torch/CUDA is imported.

If CUDA_BLOCKING_SYNC=1, set the CUDA context schedule policy to cudaDeviceScheduleBlockingSync (0x04)
so cuEventSynchronize / cuStreamSynchronize BLOCK (CPU sleeps) instead of spin-polling. Default (unset/0)
leaves the policy at cudaDeviceScheduleAuto (spin) — the phantom busy-wait. Must run before any CUDA
context exists (cudaSetDeviceFlags errors on an active context), which interpreter-startup guarantees."""
import os, sys, glob, ctypes

if os.environ.get("CUDA_BLOCKING_SYNC") == "1":
    cands = sorted(glob.glob("/home/mohamad/llm-service-kernel-latest/agentic/bigcodebench/.venv/"
                             "lib/python*/site-packages/nvidia/*/lib/libcudart.so*"))
    cands += ["libcudart.so.13", "libcudart.so.12", "libcudart.so"]
    CUDA_DEVICE_SCHEDULE_BLOCKING_SYNC = 0x04
    ok = False
    for c in cands:
        try:
            lib = ctypes.CDLL(c)
            rc = lib.cudaSetDeviceFlags(ctypes.c_uint(CUDA_DEVICE_SCHEDULE_BLOCKING_SYNC))
            print(f"[cudasync pid={os.getpid()}] cudaSetDeviceFlags(BlockingSync) via {os.path.basename(c)} rc={rc}",
                  file=sys.stderr, flush=True)
            ok = (rc == 0)
            if ok:
                break
        except Exception:
            continue
    if not ok:
        print(f"[cudasync pid={os.getpid()}] FAILED to set blocking sync (rc!=0 or no libcudart)", file=sys.stderr, flush=True)
else:
    print(f"[cudasync pid={os.getpid()}] spin mode (default cudaDeviceScheduleAuto)", file=sys.stderr, flush=True)
