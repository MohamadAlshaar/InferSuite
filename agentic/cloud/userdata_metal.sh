#!/usr/bin/env bash
# user-data for the c7i.metal-24xl CPU box (bare metal -> has the CPU PMU).
# Runs the agents + Docker + perf/TMA. The DLAMI already has docker + python.
set -uxo pipefail
exec > /var/log/agentic_setup.log 2>&1
mkdir -p /opt/agentic

# --- enable perf / TMA on the bare-metal PMU ---
echo 'kernel.perf_event_paranoid=-1' > /etc/sysctl.d/99-perf.conf
echo 'kernel.kptr_restrict=0'       >> /etc/sysctl.d/99-perf.conf
sysctl --system

# --- perf tool matching the running kernel + build basics ---
apt-get update -y
apt-get install -y "linux-tools-$(uname -r)" linux-tools-generic \
  python3-venv python3-pip git tmux jq 2>/dev/null || \
  apt-get install -y linux-tools-generic python3-venv python3-pip git tmux jq

# docker present on DLAMI; make ubuntu able to use it
usermod -aG docker ubuntu 2>/dev/null || true

# --- sanity probe: does the PMU work here (should, it's bare metal) ---
perf stat -e cycles,instructions -- sleep 0.2 > /opt/agentic/pmu_check.txt 2>&1 || true

chown -R ubuntu:ubuntu /opt/agentic
touch /opt/agentic/READY
echo "metal box bootstrap complete"
