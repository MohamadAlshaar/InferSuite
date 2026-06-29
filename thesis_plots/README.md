# thesis_plots

Publication-quality figures for the thesis, generated from the benchmark result
data. Each figure is saved as **`.pdf`** (vector, for LaTeX `\includegraphics`)
and **`.png`** (quick preview). Style is shared and colourblind-safe (Okabe–Ito).

## Layout

```
thesis_plots/
├── style.py            shared matplotlib style (fonts, palette, save helper)
├── plot_gpu_sweep.py   figures for a GPU prefill/decode sweep
├── plot_benchmark.py   figures for the full CPU/stack benchmark (mirrors the HTML report)
└── figures/
    ├── gpu_sweep/
    └── full_benchmark/
```

## Generate

```bash
pip install matplotlib            # one-time

# GPU sweep (defaults to the latest GPU_benchmark/results/run_*)
python3 thesis_plots/plot_gpu_sweep.py [RUN_DIR]

# Full benchmark (defaults to benchmark_results/run_20260609_140052)
python3 thesis_plots/plot_benchmark.py [RUN_DIR] --tier tok320 --path rag
```

## Figures

**GPU sweep** (`figures/gpu_sweep/`)
| file | shows |
|------|-------|
| `prefill_ttft` | TTFT vs input length — the compute-bound regime (super-linear bend at long context) |
| `decode_tpot`  | TPOT vs output length — memory-bound; the gentle rise is the growing-KV-cache effect |
| `regime`       | compute-engine vs HBM utilisation, prefill vs decode — the headline contrast |

**Full benchmark** (`figures/full_benchmark/`) — detailed, **per token tier**
(`tok64/`, `tok192/`, `tok320/`), plus a `cross_tier/` folder.

Per tier:
| file | shows |
|------|-------|
| `cpu_gpu_donut` | **donut**: overall CPU vs GPU share of request time (GPU ≈99.5%) |
| `cpu_vs_gpu`    | CPU vs GPU time per cell — log scale, both visible |
| `pipeline`      | CPU pipeline only (embed/Milvus/SeaweedFS/Mongo) — ms scale, small times visible |
| `decomposition` | full stacked request time incl. vLLM |
| `latency`       | e2e per cell — **mean + min–max whiskers** (n=20, not percentiles) |
| `streaming`     | TTFT / TPOT per cell (mean + min–max) *(tok64 & tok320 only)* |
| `tma`           | top-down slot breakdown per pod — the cross-pod microarchitecture takeaway |
| `ipc`           | instructions-per-cycle per pod |

Cross-tier (`cross_tier/`):
| file | shows |
|------|-------|
| `vllm_pod_cpu` | vLLM pod **CPU-utilised %** + **context-switches** across tiers (it is GPU-bound) |
| `tpot_compare` | TPOT per cell across tiers (tok64 ≈58 ms inflated → tok320 ≈32 ms true) |
| `ttft_compare` | TTFT per cell across tiers |

Run: `python3 thesis_plots/plot_benchmark.py [RUN_DIR] --path rag`

## Notes
- The figures regenerate deterministically from the result dirs, so they're
  git-ignored (the data and scripts are the source of truth).
- `plot_benchmark.py` reads the per-cell CSVs (latency/decomposition), the
  `tma/tma_slots_*` files (TMA), and `perf_pass1_*` (IPC).
