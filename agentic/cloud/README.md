# us-west-2 agentic 2-box setup (PREPARED — not launched)

Two plain EC2 instances in one VPC (no EKS). Resolves the GPU+PMU split:
- **GPU box** `g6e.2xlarge` (1× **L40S 48 GB**, Nitro) → vLLM 32B + **ncu** (TP=1 fits 32B + long ctx)
- **CPU box** `c7i.metal-24xl` (**bare metal → PMU**) → agents + Docker + **perf/TMA**

Region **us-west-2 / us-west-2a** (only region where both are currently available).
Why us-west-2 and not London: L40S isn't offered in eu-west-2; agentic workloads
fetch their own data so they don't need the London cluster/volumes.

## Cost
- **Prepared, not launched (now): $0.** Only free scaffolding (VPC/SG/key) exists.
- Running: L40S ~$2.2/hr + c7i.metal ~$5.4/hr (bill only while up).
- No EKS control-plane fee (no cluster).

## Files
- `env.sh` — free scaffolding IDs (VPC/subnet/SG/key/AMI) — already created.
- `userdata_gpu.sh` / `userdata_metal.sh` — auto-bootstrap at boot (vLLM+ncu / perf+docker).
- `launch.sh` — **the only paid step**; launches both, writes `state.env` (IDs/IPs).
- `stage_harnesses.sh` — scp the `agentic/` harnesses + perf scripts to the metal box.
- `ncu_roofline.sh` + `parse_ncu_tma.py` — GPU roofline + warp-stall ("GPU TMA"); staged on the GPU box.
- `teardown.sh [--all]` — terminate (and optionally delete SG/key).

## Run order (when you say go)
```bash
cd agentic/cloud
bash launch.sh                 # launch both (billing starts); waits for running; prints IPs
# wait for bootstrap: ssh ... 'test -f /opt/agentic/READY'  (both boxes)
bash stage_harnesses.sh        # copy harnesses to metal box

# --- GPU box: model + GPU TMA ---
ssh -i ~/.ssh/agentic-uw2.pem ubuntu@$GPU_IP
  bash /opt/agentic/download_models.sh        # ~20GB 32B-AWQ
  # GPU 'TMA' (roofline + warp-stall), TP=1 enforce_eager:
  bash /opt/agentic/ncu_roofline.sh           # -> /opt/agentic/ncu_out/gpu_tma.png
  # or serve for behavior runs:
  bash /opt/agentic/serve_behavior.sh         # 32B @ 32k ctx on the L40S

# --- CPU box: agents + tool-exec TMA ---
ssh -i ~/.ssh/agentic-uw2.pem ubuntu@$CPU_IP
  # point harnesses at the GPU box (in-VPC): http://$GPU_PRIV:8000/v1, model qwen2.5-32b
  # run SWE-agent / BigCodeBench / OpenClaw under perf (bare-metal PMU works here)

bash teardown.sh               # when done — terminate (no lingering billing)
```

## What each box can/can't measure (the split)
- GPU box (Nitro): ✅ ncu GPU roofline/warp-stall; ❌ no CPU PMU.
- CPU metal box: ✅ perf/TMA of tool-exec; runs the agents.
- "CPU during inference" (vLLM serving spin) needs GPU+PMU on one box → only the LOCAL
  workstation can do that; its character (retiring-bound spin) is already captured.
