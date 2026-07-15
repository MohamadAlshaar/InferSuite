# Figure gallery (curated view)

All figures in one tree, organized **domain → setup → tier/bench**. This directory is a *view*:
never edit here — regenerate at the source and re-run `scripts/sync_plots.sh`.

**THESIS SETS (live, resynced):** `agents/swe_long/` (← `local_agents/SWE_long/plots/`),
`agents/oc_long/` (← `local_agents/OC_long/plots/`), `service/iso/` (← `local_service/plots_iso/`),
plus `engine/local/` + `gpu/local_a2000/` (← `agentic/inference/plots/`). Each live set carries
its own MANIFEST at the source.

**Everything else below is a FROZEN legacy snapshot** (h100/, eks/, local_api/, service/local
tok-trees, gpu/h100, gpu/l40s): sources archived 2026-07-12 under `archive/`; kept for browsing,
no longer resynced.

## Setups (what ran where)

| setup | machine | CPU measurement | model |
|---|---|---|---|
| `local` | workstation (Xeon w5-3425 bare metal + RTX A2000) | full TMA + perf record | self-served Qwen2.5-7B-AWQ (vLLM/k3s) |
| `local_api` | workstation CPU only | full TMA + perf record | Claude Sonnet 4.6 via API (no local serving) |
| `h100` | rented H100 PCIe node (KVM guest) | portable suite + perf record (no TMA) | self-hosted Qwen2.5 32B (Coder/Instruct) |
| `eks` | EKS cluster (c7i.metal CPU node + p5 H100 GPU node) | TMA on CPU pods; GPU node has no PMU | Qwen2.5-Instruct behind llm-d |
| `l40s` | L40S cloud box | — (GPU ncu study) | Coder-32B-AWQ |

## Layout

```
service/  local/{tok64,tok192,tok320,idle_control}  + timing_*   (12-cell grid, TMA L1+L2, attribution)
          h100/                                                   (single-node k3s @32B, attribution + signature)
          eks/{cross_tier,tok64,tok192,tok320}                    (deployed-cluster benchmark)
agents/   local_api/                                              (cross-workload figures + tool software view)
          local/                                                  (self-served 7B during-inference, per agent)
engine/   local/                                                  (during-inference TMA/donut/signature + phantom)
gpu/      local_a2000/  h100/  l40s/                              (ncu top-downs per GPU)
```
