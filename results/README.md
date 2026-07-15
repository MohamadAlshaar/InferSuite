# Results / data gallery (curated view)

The mirror of `plots/` for the RAW DATA: one tree, **domain -> setup -> (tier | bench)**, built
entirely from **symlinks** into the real data locations (no copies) by `scripts/make_results_tree.sh`.
Setup names are the same as in `plots/README.md`:

| setup | meaning |
|---|---|
| `local` | fully local workstation (self-served 7B) |
| `local_api` | local CPU; model = commercial frontier LLM via API |
| `h100` | rented H100 node (self-hosted 32B) |
| `eks` | EKS deployed cluster |
| `l40s` | L40S ncu study |

```
service/  local -> local_service/data_iso  (ISOLATED 36-cell: counters+TMA per tier, idle control)
          h100  -> h100/service/data        (attribution + microarch + idle control + PROVENANCE.md)
          eks   -> benchmark_results/run_20260609_140052
agents/   swe_long  -> local_agents/SWE_long/data   (ISOLATED long-horizon SWE-agent x GLM-5.2)
          oc_long   -> local_agents/OC_long/data    (ISOLATED long-horizon OpenClaw x GLM-5.2, lineage-fenced)
          certified_40min -> archive/certified_glm_40min  (superseded 40-min GLM campaign, archived)
engine/   local/data.json -> agentic/inference/plots/data.json   local/phantom -> agentic/inference/runs/sync
gpu/      local_a2000 -> agentic/inference/runs/ncu   h100 -> h100/data_gpu   l40s -> agentic/aws_agents/gpu
```

Updated 2026-07-12: the live isolated campaigns (`swe_long`, `oc_long`, service `data_iso`) are the
current sets; the earlier 40-min GLM campaign and exploratory runs were moved to `archive/`.

Note: some targets are **local-only** (gitignored raw captures) — those links resolve on the
measurement machine but not in a fresh clone.
