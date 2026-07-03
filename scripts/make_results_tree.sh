#!/usr/bin/env bash
# make_results_tree.sh — build the curated DATA/results view at results/ as SYMLINKS into the
# real data locations (no copies: some sources are large and/or gitignored). Same structure and
# setup names as plots/ (see plots/README.md): domain -> setup -> (tier | bench).
# Idempotent; re-run any time.
set -euo pipefail
cd "$(dirname "$0")/.."
L() { mkdir -p "results/$(dirname "$2")"; ln -sfnT "$(realpath --relative-to="results/$(dirname "$2")" "$1")" "results/$2"; }

# ---------- service ----------
L local_service/data                      service/local
L h100/service/data                       service/h100
L benchmark_results/run_20260609_140052   service/eks

# ---------- agents ----------
L local_agents/data                       agents/local        # during (7B replays) + tool software views
L agentic/CANONICAL/swe_bench/data        agents/local_api/swe
L agentic/CANONICAL/bigcodebench/data     agents/local_api/bcb
L agentic/CANONICAL/openclaw/data         agents/local_api/oc
L h100/data                               agents/h100/bcb
L h100/data_swe                           agents/h100/swe
L h100/data_oc                            agents/h100/oc

# ---------- during-inference engine studies ----------
L inf_thesis_plots/data.json              engine/local/data.json
L agentic/inference/runs/sync             engine/local/phantom

# ---------- gpu ----------
L agentic/inference/runs/ncu              gpu/local_a2000
L h100/data_gpu                           gpu/h100
L agentic/aws_agents/gpu                  gpu/l40s

echo "results/ view built ($(find results -type l | wc -l) links)"
