# Agents handover — file map and reading order

For someone taking over the **agent** side of InferSuite (SWE-agent + OpenClaw campaigns).
The durable rules and gotchas live in the repo-root `CLAUDE.md`; this is the guided tour.

## Reading order

1. `CLAUDE.md` — conventions and the "issues we ran into" list. Read before touching anything.
2. `./measure.sh help` — the only entry point you need at first.
3. One run directory end-to-end (fmt is the richest): `local_agents/SWE_clean/data/glm_swe_fmtlib/run_1/`.
4. `local_agents/SWE_clean/plots/MANIFEST.md` — what every figure means.
5. `local_agents/scripts/glm/run_glm_campaign.sh` top to bottom — the capture design, commented.
6. `local_agents/scripts/glm/validate_glm_agents.py` — what "valid" means (gates E1–E11).

## Entry point

| file | contents |
|---|---|
| `measure.sh` | One command for every campaign: `agents-swe`/`agents-oc` × `preflight`/`dryrun`/`smoke`/`campaign`/`validate`, plus `plots` and `validate` over banked data. Header documents the env knobs (never run agents at temperature 0). |

## Campaign kit — `local_agents/scripts/glm/`

| file | contents |
|---|---|
| `run_glm_campaign.sh` | The heart. Isolation shield + ISO-PROOF gate (refuses to run unless the partition is proven quiet); 8 zero-multiplex counter groups with shuffled 5 s rotation; continuous TMA; capture stack (10 Hz per-fence CPU pollers, partition-wide residual witness, 99 Hz records); SWE / OC / replay episode drivers; loop guard; teardown. |
| `oc_lineage_watcher.py` | OpenClaw fencing: kernel process events; fork by the gateway stays agent-side, fork+exec becomes a tool root; writes the lineage log the validator treats as ground truth. |
| `validate_glm_agents.py` | Proof-based validator, gates E1–E11 (window integrity, isolation, cpu.stat-vs-PMU agreement, OC lineage purity, action uniqueness, burst census, continuous-TMA census, partition residual). A run is not data until this passes. |
| `plot_glm_results.py` | Main figure set (time split, CPU work, timelines, tool calls, signature, TMA L1/tree, …) and `values_dump.json` — every number any figure displays. |
| `audit_plots.py` | Recomputes every dumped value independently from raw files; must report ALL MATCH before figures are trusted. |
| `plot_harness_scaling.py` | Cross-campaign harness cost vs conversation length (turns^~2.7 power law). |
| `plot_exploratory.py` | Non-featured extra set (utilization, per-turn latency, turn composition, long tail, ctx switches, OS share, burst decomposition) → `plots/extra/`. |
| `plot_internal_tools.py`, `plot_calls_vs_bursts.py`, `plot_call_structure.py` | Supporting figure scripts. |
| `gen_lanes_leaf.sh` | Derives per-CPU lane tables and leaf-symbol tables from the records. |
| `gen_manifest.py` | Regenerates the plots MANIFEST. |
| `litellm_glm.yaml`, `my_api_glm.json` | Proxy + model config. The API key lives at `~/.glm_key` (never print it). |

## Banked campaigns — `local_agents/SWE_clean/`, `local_agents/OC_clean/`

Per task, `data/<config>/run_N/` contains:

| artifact | contents |
|---|---|
| `cpustat_scope{1,2,3}.tsv` | 10 Hz cumulative CPU time per fence (1=harness, 2=tool, 3=litellm proxy). The exact source of every core-second/timeline/burst number. |
| `procstat_partition.tsv` | (new runs) partition-wide per-CPU 10 Hz series — the unfenced-residual witness for gate E11. |
| `group_<g>_wNNN.txt` + `windows.tsv` + `n_windows` | The zero-mux counter windows and the realized shuffled order (provenance). |
| `tma_cont.csv` | Continuous whole-episode TMA L1/L2 intervals. |
| `rec_scope*.data` + `scope*_{comm,dso,ksym,leaf,pidtime}.txt` + `scope*_cpulanes.tsv` | 99 Hz records and their derived what-ran/which-CPU tables. |
| `traj/` (SWE) / `transcript/chat.jsonl` + lineage log (OC) | The agent's own record: actions + per-tool execution time (SWE); timestamped messages + tool calls (OC). |
| `agent.log`, `metadata.json`, `DONE` | Harness log (STEP markers = turns for SWE), capture metadata (cpus, winsec, group order, pods/cgroups), completion marker. |

Beside the data: `plots/` (figures + `MANIFEST.md` + `values_dump.json`), `plots/extra/`
(exploratory), `plot_spec.json` (which labels/configs/runs feed the figures).

## The harnesses and history

| location | contents |
|---|---|
| `agentic/swe_agent/`, `agentic/openclaw/` | The two agent frameworks the campaigns drive — unmodified; all measurement is external. |
| `archive/glm_softiso_long_campaigns/` | Superseded soft-isolation campaigns (SWE_long/OC_long); still read by the harness-scaling figure. |
| `agentic/inference/` | GPU-side studies (ncu GPU-TMA, spin-vs-block busy-wait experiment). |
| `plots/` (repo root) | Curated read-only gallery synced by `scripts/sync_plots.sh` — never edit figures there. |

## Data availability

The repo IS self-sufficient for figures: all figure-critical raw data (fence CPU series,
counter windows, continuous TMA, derived record tables, trajectories/transcripts, metadata)
is tracked for SWE_clean, OC_clean, and data_iso, plus the harness-scaling inputs from the
archived campaigns. Two exceptions:

- the heavy perf record binaries (`rec_*.data`) stay out of git — their derived
  comm/dso/ksym/leaf/lane tables ARE tracked, and no plotter reads the raw records
  (only `gen_lanes_leaf.sh` re-derivation would need them; an offline tarball
  `infersuite_perf_records.tar.*` was made at handover time);
- one oversized trajectory ships compressed — after cloning run:
  `gunzip -k local_agents/SWE_clean/data/glm_swe_sympy/run_1/traj/sympy__sympy-14248/sympy__sympy-14248.traj.gz`

## The narrative

The thesis repo (`~/thesis/InferSuite_thesis`) is the write-up: agent results in the Results
chapter, OpenClaw in Appendix B, the harness discussion in the Discussion chapter. Its own
CLAUDE.md carries the writing rules (code-free prose, figure approval flow).
