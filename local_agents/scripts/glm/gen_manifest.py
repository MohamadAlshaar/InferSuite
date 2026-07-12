#!/usr/bin/env python3
"""gen_manifest.py — regenerate local_agents/glm_plots/MANIFEST.md: the structured index
of every GLM-campaign artifact (figures, episode data, replays, kit, logs). Rerunnable."""
import json, os, time
from glob import glob

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "glm_plots")

def behavior(rd):
    b = {"steps": 0, "outcome": "?", "wall": 0.0, "cpu_s": 0.0}
    try:
        meta = json.load(open(f"{rd}/metadata.json"))
    except OSError:
        meta = {}
    try:
        log = open(f"{rd}/agent.log", errors="ignore").read()
        b["steps"] = log.count("STEP ")
        if str(meta.get("workload", "")).endswith("replay"):
            b["outcome"] = f"replay of run_{meta.get('extra', {}).get('source_run', '?')}"
        elif "Insufficient balance" in log:
            b["outcome"] = "DIED: API credit exhausted"
        elif "submitted" in log:
            b["outcome"] = "submitted"
        elif "overall_score" in log:
            b["outcome"] = "completed (OC scored)"
        else:
            b["outcome"] = "capped (drain limit)"
    except OSError:
        pass
    for i in (1, 2, 3):
        try:
            s = [(float(p[0]), float(p[2])) for p in (l.split() for l in open(f"{rd}/cpustat_scope{i}.tsv"))
                 if len(p) >= 3 and p[1] == "usage_usec" and float(p[2]) >= 0]
            if len(s) > 1:
                b["cpu_s"] += (s[-1][1] - s[0][1]) / 1e6
                b["wall"] = max(b["wall"], (s[-1][0] - s[0][0]) / 60)
        except OSError:
            pass
    return b

L = []
L.append(f"# GLM-5.2 campaign — artifact manifest (generated {time.strftime('%Y-%m-%d %H:%M')})")
L.append("\nRegenerate: `python3 local_agents/scripts/glm/gen_manifest.py` · "
         "Figures: `python3 local_agents/scripts/glm/plot_glm_results.py` (system python3) · "
         "Audit: `python3 local_agents/scripts/glm/audit_plots.py`\n")

L.append("## Figures — `local_agents/glm_plots/` (from ONE certified episode per task; "
         "astropy/scikit-learn/sympy resolved, django capped)\n")
FIGS = [   # main set, aligned to the five wanted results + the harness deep-dive
 ("swe/glm_time_split.png", "1: time split — how each episode's wall-clock passed: agent (harness+tools) vs inference"),
 ("swe/glm_cpu_work.png", "1b: CPU work — who did the computation (core-second donuts + wall-vs-work table per task)"),
 ("swe/glm_tool_calls.png", "2+3: tool-call count, duration, peak parallelism (spike/sustained), wall share"),
 ("swe/glm_hw_threads.png", "3: parallelism ground truth — per-core occupancy lanes (SMT pairs adjacent)"),
 ("swe/glm_timeline.png", "4: orchestration timeline — every tool call as a bar + harness sub-panel"),
 ("swe/glm_tma_uop.png", "5: TMA Level 1 + frontend uop delivery, per side"),
 ("swe/glm_signature.png", "5: per-side signature heatmap on absolute hardware-anchored scales"),
 ("swe/glm_harness_anatomy.png", "NEW: the harness as a workload — what it executes (leaf frames), performance card (invariant across tasks), per-burst cost growth over the episode"),
]
EXTRA = [
 ("swe/extra/glm_software_kernel.png", "software composition of both fences + kernel share (demoted: tool-side story already established)"),
 ("swe/extra/glm_call_distribution.png", "ECDF of call/burst durations (demoted)"),
]
for f, desc in FIGS:
    L.append(f"- **{f}** — {desc}")
L.append("\nDemoted to `glm_plots/extra/` (kept, not part of the main narrative):")
for f, desc in EXTRA:
    L.append(f"- {f} — {desc}")

L.append("""
## Definitions (units and boundaries used by every figure)

Units follow the established literature conventions (Borg/K8s, HPC accounting, ISCA WSC):
- **core** — one schedulable slot; here core = logical CPU (SMT-2 on). Machine has 24,
  the pinned partition 20 (10 physical cores x 2). The lanes figure is the exact per-core
  record; all other numbers are summaries of it.
- **CPU usage (cores)** — rate: core-seconds consumed per second, the Borg/Kubernetes
  measure ("a task using two cores all the time = usage 2.0"; k8s 500m = 0.5). Fractional
  usage is standard: 0.5 = half a core's worth of execution in the window, however many
  actual cores carried it. Ceiling = 20. (grid view: green share of a column)
- **core-seconds** — amount: total CPU time summed over threads. 1 wall-second at usage N
  adds N core-s, so totals legitimately exceed episode duration (scikit: 1771 core-s in a
  6.8-min episode = parallel bursts). (grid: total green cells)
- **shares (%)** — component breakdowns are presented as % of CPU time consumed, the
  warehouse-scale-profiling convention (Kanev et al., ISCA'15: "datacenter tax = ~30% of
  cycles").
- **Wall time** — episode duration, first to last sample of the continuous 10 Hz timeline
  (harness launch to agent exit; django = our 40 min drain cap).
- **Two-view donuts** — top row splits wall time; bottom row splits CPU·s. They differ
  because a tool-second can employ ~20 CPUs while a harness-second employs <1.
- **Harness vs tool boundary** — kernel cgroup accounting. Harness = the sweagent process's
  own scope on the host. Tool = the Docker sandbox cgroup; every action arrives there via
  docker-exec RPC and all its children inherit the cgroup (incl. compiler grandchildren).
  Verified per episode by per-fence DSO tables (cc1/OpenBLAS only ever in the tool fence).
- **API proxy** — host litellm scope (:8100) bridging agent to z.ai GLM-5.2; third fence.
- **Heavy tool call** — burst of tool-fence CPU > 0.3 CPUs sustained (10 Hz timeline, gaps
  <2 s merged). Sub-threshold activity is kept: ECDF uses all trajectory calls; the timeline
  shows a presence lane. Threshold is display-side only.
- **Peak parallelism** — reported at two timescales: instantaneous (0.1 s samples, capped at
  the 20-CPU partition bound; poll jitter can read ~5% above saturation) and sustained (1 s
  average). Spiky parallelism (astropy's ~0.3 s parallel compiler procs: 16.6 spike / 2.7
  sustained) is distinguished from saturation (scikit: 20.0 / 20.0).
- **Topology** — Xeon w5-3425, SMT-2 on: measured partition = 10 physical cores = 20 logical
  CPUs (2-11 + siblings 14-23); housekeeping on 0-1,12-13.
""")
L.append("\n## Featured episodes (sources of every figure)\n")
L.append("| task | instance | data dir | outcome | resolve-verified | anchor |")
L.append("|---|---|---|---|---|---|")
FEAT = [("astropy", "astropy__astropy-14096", "glm_swe_astropy/run_1", "resolved", "yes (swebench harness)", "PASS (pair+3 noise)"),
        ("scikit-learn", "scikit-learn__scikit-learn-25232", "glm_swe_scikit-learn/run_1", "resolved", "yes", "PASS (pair+3 noise)"),
        ("sympy", "sympy__sympy-14248", "glm_swe_sympy/run_1", "resolved", "yes", "live-certified (E1-E6)"),
        ("django", "django__django-11133", "glm_swe_django-lite/run_1", "capped (40 min)", "n/a — no patch", "PASS (pair+2 noise)")]
for r in FEAT:
    L.append("| " + " | ".join(r) + " |")

L.append("\n## All campaign episodes — `local_agents/data/`\n")
L.append("| data dir | steps | wall (min) | CPU-s | outcome |")
L.append("|---|---|---|---|---|")
for rd in sorted(glob(f"{DATA}/glm_*/run_*")):
    if not os.path.exists(f"{rd}/DONE"):
        continue
    b = behavior(rd)
    rel = "/".join(rd.split("/")[-2:])
    L.append(f"| {rel} | {b['steps']} | {b['wall']:.0f} | {b['cpu_s']:.0f} | {b['outcome']} |")

L.append("\n### Per-run contents (identical layout in every run dir)")
L.append("""
- `group_<g>_w<NNN>.txt` — one 10 s zero-multiplexing counter window (groups: tma, core,
  cache, fp1, fp2, mlp, fe, icache, priv), all scopes same window via `--for-each-cgroup`
- `windows.tsv` — epoch bracket + post-window aliveness per window
- `cpustat_scope{1,2,3}.tsv` — 10 Hz cgroup timelines (1=harness, 2=tool, 3=proxy;
  usage/user/system from 2026-07-09 on)
- `rec_scope{N}.data` + `scope{N}_{dso,comm,ksym}.txt` — full-episode 99 Hz records + tables
- `agent.log`, `traj/` (SWE trajectories incl. preds.json), `metadata.json` (provenance:
  model, cgroups, cpusets, git rev), `n_windows`, `DONE` (resume marker)
""")

L.append("## Validation & certification chain\n")
L.append("""- Per-episode proofs: `validate_glm_agents.py <data> glm` — E1 window length, E2 CPUs
  formula vs perf's comment, E3 cpu.stat vs PMU, E4 OC watcher purity, E5 work, E6 kernel
  share PMU vs scheduler; plus behavior consistency and the live<->replay anchor section
- Replay anchor data: `glm_replay_swe_{astropy,scikit-learn,django-lite}/run_{1..5}`
  (run_N pairs live run_N; runs 4-5 = same-trajectory noise repeats). Verdict: PASS
- Figures-vs-data audit: `audit_plots.py` — 96/96 checks incl. sustained peaks + hw-lane
  pinned-only proof (2026-07-09; was 84 before the definitions/relabel pass)
- Resolve verification: swebench harness reports under `logs/run_evaluation/` (repo root),
  5/5 submitted patches resolved
- Kit: `local_agents/scripts/glm/` — run_glm_campaign.sh (stages: preflight, dryrun incl.
  kernel calibration, isolation-test, smoke, campaign swe|oc, replay-anchor, validate),
  oc_cgroup_watcher.sh, litellm_glm.yaml, my_api_glm.json, campaign.conf, plot_glm_results.py,
  validate_glm_agents.py, audit_plots.py, gen_manifest.py; logs: campaign_swe.out,
  django_lite*.out, anchor_*.out, smoke_*.out
- Superseded/backup: `data/_bak_astropy_8grp/` (pre-priv-group astropy run, evidence only)
""")

open(f"{OUT}/MANIFEST.md", "w").write("\n".join(L) + "\n")
print(f"wrote {OUT}/MANIFEST.md ({len(L)} lines)")
