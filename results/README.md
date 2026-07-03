# Results / data gallery (curated view)

The mirror of `plots/` for the RAW DATA: one tree, **domain -> setup -> (tier | bench)**, built
entirely from **symlinks** into the real data locations (no copies) by `scripts/make_results_tree.sh`.
Setup names are the same as in `plots/README.md`:

| setup | meaning |
|---|---|
| `local` | fully local workstation (self-served 7B) |
| `local_api` | local CPU; model = Claude Sonnet via API |
| `h100` | rented H100 node (self-hosted 32B) |
| `eks` | EKS deployed cluster |
| `l40s` | L40S ncu study |

```
service/  local -> local_service/data       (counters+TMA per tier, idle control, 12-cell timing grid)
          h100  -> h100/service/data        (attribution + microarch + idle control + PROVENANCE.md)
          eks   -> benchmark_results/run_20260609_140052
agents/   local     -> local_agents/data    (during-inference 7B replays; tool software views)
          local_api -> agentic/CANONICAL/{swe_bench,bigcodebench,openclaw}/data  (canonical tool TMA)
          h100/{bcb,swe,oc} -> h100/data*   (32B campaign captures)
engine/   local/data.json -> agentic/inference/plots/data.json   local/phantom -> agentic/inference/runs/sync
gpu/      local_a2000 -> agentic/inference/runs/ncu   h100 -> h100/data_gpu   l40s -> agentic/aws_agents/gpu
```

Note: some targets are **local-only** (gitignored raw captures: `local_service/data`,
`local_agents/data`) — those links resolve on the measurement machine but not in a fresh clone.
