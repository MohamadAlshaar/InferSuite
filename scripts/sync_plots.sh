#!/usr/bin/env bash
# sync_plots.sh — refresh the curated figure gallery at plots/ from the CURRENT generator
# output locations. plots/ is a VIEW: never edit it directly; regenerate figures at their
# source and re-run this script.
#
# THESIS SCOPE (locked 2026-07-12): the actively regenerated sets are the two isolated
# campaigns + the isolated service run. Everything else under plots/ (h100/, eks/,
# local_api/, service/local tok-trees, gpu/h100, gpu/l40s) is a FROZEN legacy snapshot —
# its sources moved to archive/ and are deliberately NOT resynced or deleted here.
set -euo pipefail
cd "$(dirname "$0")/.."
R() { mkdir -p "plots/$2" && rsync -a --delete-after --exclude='*.json' "$1" "plots/$2/"; }

# ---------- thesis sets (live) ----------
R "local_agents/SWE_clean/plots/" agents/swe_clean       # SWE-agent x GLM-5.2 hardened campaign
R "local_agents/OC_clean/plots/"  agents/oc_clean        # OpenClaw x GLM-5.2 hardened campaign
R "local_service/plots_iso/"      service/iso            # isolated k3s service campaign

# ---------- still-generated engine/GPU figures (sources remain in tree) ----------
[ -d agentic/inference/plots/gpu ] && R "agentic/inference/plots/gpu/" gpu/local_a2000
if ls agentic/inference/plots/0*.png >/dev/null 2>&1; then
  mkdir -p plots/engine/local && cp -a agentic/inference/plots/0*.png plots/engine/local/
fi

echo "synced -> plots/ ($(find plots -name '*.png' | wc -l) figures; legacy snapshots untouched)"
