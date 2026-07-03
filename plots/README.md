# Figure gallery (curated view)

All figures in one tree, organized **domain → setup → tier/bench**. This directory is a *view*:
figures are generated at their source locations (`local_service/`, `local_agents/`, `h100/`,
`inf_thesis_plots/`, `agentic/thesis_figures/`, `thesis_plots/`) and copied here by
`scripts/sync_plots.sh`. Never edit here — regenerate at the source and re-sync.

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
          h100/{bcb,swe,oc} + grand_*                             (self-hosted 32B campaign)
          local/                                                  (self-served 7B during-inference, per agent)
engine/   local/                                                  (during-inference TMA/donut/signature + phantom)
gpu/      local_a2000/  h100/  l40s/                              (ncu top-downs per GPU)
```
