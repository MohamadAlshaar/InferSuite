"""Announce-only. The blocking lever is evblock.so (LD_PRELOAD), which forces the BLOCKING_SYNC
flag on every CUDA event -- that is what governs cudaEventSynchronize (the phantom busy-wait).

This sitecustomize intentionally does NOT call cudaSetDeviceFlags: on this box the ctypes glob
loaded a mismatched libcudart (so.13 vs torch's 12.x) and set the schedule flag on a foreign
context, which crashed EngineCore init at request_memory. evblock.so alone is sufficient."""
import os, sys
mode = "BLOCK (evblock.so per-event BLOCKING_SYNC)" if os.environ.get("CUDA_BLOCKING_SYNC") == "1" else "spin (default)"
print(f"[cudasync pid={os.getpid()}] {mode}", file=sys.stderr, flush=True)
