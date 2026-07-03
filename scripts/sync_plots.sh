#!/usr/bin/env bash
# sync_plots.sh — build/refresh the curated figure gallery at plots/ from all generator output
# locations. plots/ is a VIEW: never edit it directly; regenerate figures at their source and
# re-run this script. Structure: plots/<domain>/<setup>/... where <setup> encodes where it ran:
#   local      = fully local workstation (Xeon w5-3425 + RTX A2000, self-served 7B)
#   local_api  = local workstation CPU; model = Claude Sonnet via API (no local serving)
#   h100       = rented H100 node (KVM guest, self-hosted 32B)
#   eks        = EKS cluster (deployed service; H100 GPU node)
#   l40s       = L40S cloud box (ncu study)
set -euo pipefail
cd "$(dirname "$0")/.."
R() { mkdir -p "plots/$2" && rsync -a --delete-after --exclude='*.json' "$1" "plots/$2/"; }

# ---------- service ----------
R "local_service/plots/tok64/"        service/local/tok64
R "local_service/plots/tok192/"       service/local/tok192
R "local_service/plots/tok320/"       service/local/tok320
R "local_service/plots/idle_control/" service/local/idle_control
mkdir -p plots/service/local && cp -a local_service/plots/timing_donuts.png local_service/plots/timing_cpu_stages.png plots/service/local/
R "h100/service/plots/"               service/h100
R "thesis_plots/figures/full_benchmark/cross_tier/" service/eks/cross_tier
R "thesis_plots/figures/full_benchmark/tok64/"      service/eks/tok64
R "thesis_plots/figures/full_benchmark/tok192/"     service/eks/tok192
R "thesis_plots/figures/full_benchmark/tok320/"     service/eks/tok320

# ---------- agents ----------
mkdir -p plots/agents/local_api
cp -a agentic/thesis_figures/0*.png plots/agents/local_api/            # cross-workload (Sonnet driver)
cp -a local_agents/plots/tool_attribution.png plots/agents/local_api/  # tool software view (replays + live Sonnet OC)
R "h100/plots/bcb/" agents/h100/bcb
R "h100/plots/swe/" agents/h100/swe
R "h100/plots/oc/"  agents/h100/oc
mkdir -p plots/agents/h100 && cp -a h100/plots/grand_*.png plots/agents/h100/
# local self-served during-inference per-agent figures land here once plotted:
mkdir -p plots/agents/local

# ---------- during-inference engine studies ----------
mkdir -p plots/engine/local
cp -a inf_thesis_plots/0*.png plots/engine/local/
cp -a agentic/thesis_figures/phantom_cpu.png plots/engine/local/

# ---------- gpu ----------
R "inf_thesis_plots/gpu/"  gpu/local_a2000
R "h100/plots/gpu/"        gpu/h100
R "agentic/aws_agents/gpu/" gpu/l40s

echo "synced -> plots/ ($(find plots -name '*.png' | wc -l) figures)"
