#!/usr/bin/env bash
# user-data for the L40S (g6e) GPU box. Runs once at boot as root on the DLAMI
# (which already has NVIDIA driver + CUDA + docker). Installs vLLM + Nsight tools,
# enables GPU profiling for ncu, and stages the serve/ncu helper scripts.
# Heavy/long steps (model download, serving, profiling) are written as scripts to
# /opt/agentic and triggered later over SSH so they can be monitored.
set -uxo pipefail
exec > /var/log/agentic_setup.log 2>&1
mkdir -p /opt/agentic

# --- enable GPU performance counters for ncu (otherwise ncu = ERR_NVGPUCTRPERM) ---
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' > /etc/modprobe.d/nvidia-prof.conf
# (takes effect after driver reload/reboot; until then run ncu with sudo)

# --- vLLM in a venv ---
apt-get update -y
python3 -m venv /opt/agentic/venv
/opt/agentic/venv/bin/pip install -U pip wheel
/opt/agentic/venv/bin/pip install "vllm" huggingface_hub

# --- Nsight Compute (ncu) + Nsight Systems (nsys) ---
if ! command -v ncu >/dev/null 2>&1; then
  apt-get install -y nsight-compute nsight-systems 2>/dev/null || \
  apt-get install -y cuda-nsight-compute-12-4 cuda-nsight-systems-12-4 2>/dev/null || true
fi
# fall back to whatever ships with the CUDA toolkit
ln -sf "$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1)" /usr/local/bin/ncu 2>/dev/null || true

# --- helper: download the models (32B AWQ) — triggered manually, monitorable ---
cat > /opt/agentic/download_models.sh <<'DL'
#!/usr/bin/env bash
set -e
. /opt/agentic/venv/bin/activate
# Coder-32B for SWE/BigCodeBench; general-32B for OpenClaw (uncomment if needed)
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
# huggingface-cli download Qwen/Qwen2.5-32B-Instruct-AWQ
echo "models downloaded"
DL
chmod +x /opt/agentic/download_models.sh

# --- helper: serve the 32B at TP=1, enforce_eager (REQUIRED for clean ncu) ---
cat > /opt/agentic/serve_ncu.sh <<'SV'
#!/usr/bin/env bash
# TP=1 single L40S, enforce_eager so each kernel is a clean launch for ncu.
. /opt/agentic/venv/bin/activate
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
  --served-model-name qwen2.5-32b --host 0.0.0.0 --port 8000 \
  --quantization awq_marlin --max-model-len "${MAXLEN:-16384}" \
  --gpu-memory-utilization 0.92 --max-num-seqs 1 \
  --tensor-parallel-size 1 --enforce-eager \
  --enable-auto-tool-choice --tool-call-parser hermes
SV
chmod +x /opt/agentic/serve_ncu.sh

# --- helper: serve the 32B for long-context BEHAVIOR (L40S 48GB fits TP=1 + long ctx) ---
cat > /opt/agentic/serve_behavior.sh <<'SB'
#!/usr/bin/env bash
. /opt/agentic/venv/bin/activate
export VLLM_ATTENTION_BACKEND=FLASH_ATTN VLLM_USE_FLASHINFER_SAMPLER=0
exec vllm serve Qwen/Qwen2.5-Coder-32B-Instruct-AWQ \
  --served-model-name qwen2.5-32b --host 0.0.0.0 --port 8000 \
  --quantization awq_marlin --max-model-len "${MAXLEN:-32768}" \
  --gpu-memory-utilization 0.95 --max-num-seqs 4 \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice --tool-call-parser hermes
SB
chmod +x /opt/agentic/serve_behavior.sh

chown -R ubuntu:ubuntu /opt/agentic
touch /opt/agentic/READY
echo "GPU box bootstrap complete"
