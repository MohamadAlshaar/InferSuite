# GLM frontier-tier campaign kit

One-command, rerunnable measurement campaign for the frontier agent tier: **SWE-agent ×4 +
OpenClaw ×4, driven live by GLM-5.2** (z.ai, thinking enabled), harness and tool CPU fenced
into separate cgroups, full microarch suite, repeats with dispersion checks, runtime isolation.

## Quick start

```bash
./run_glm_campaign.sh preflight        # fail-fast checks, no spend
./run_glm_campaign.sh dryrun           # zero-multiplexing gate (8 groups vs dummy load)
./run_glm_campaign.sh isolation-test   # apply all isolation knobs, verify, revert, verify
./run_glm_campaign.sh smoke            # proxy path: chat + responses API (~2 requests)
./run_glm_campaign.sh campaign swe     # SWE phase: 4 instances x 3 repeats  (~2.5-4 h)
#   -> review validator + sanity output in chat, then:
./run_glm_campaign.sh campaign oc      # OC phase: 4 tasks x 3 repeats       (~1.5-2 h)
./run_glm_campaign.sh validate         # 3-layer validation over all glm_* data
```

Single-episode smokes (`smoke-swe`, `smoke-django`, `smoke-oc`) run the FULL campaign path
(isolation + capture) for one episode; thanks to resume markers the episode counts toward the
campaign — nothing is wasted. Ctrl-C at any point is safe: INT/TERM are routed into the EXIT
trap, so isolation is always restored.

Future rerun with another model: `MODEL_ID=<id> GLM_ENDPOINT=<url> KEYFILE=~/.other_key
TIER_PREFIX=<name> ./run_glm_campaign.sh all` — nothing else to edit.

## What is measured, where

| Scope | Cgroup | Contains |
|---|---|---|
| SWE harness | `measured.slice/glm-swe-*.scope` | sweagent python (model calls, parsing, orchestration) |
| SWE tool | `measured.slice/docker-<sweb>.scope` | the SWE-bench sandbox: every executed command/test |
| OC harness | `<container>.scope/agent` | node/V8 (openclaw agent + gateway), watcher-sorted |
| OC tool | `<container>.scope/toolexec` | chromium, python, converters — everything not node |
| proxy | litellm scope (housekeeping CPUs) | model round-trip bookkeeping |

Per episode (`local_agents/data/glm_<wl>_<cfg>/run_<n>/`): `group_<g>_w<NNN>.txt` (continuously
cycled 10 s windows: tma core cache fp1 fp2 mlp fe icache), `windows.tsv` (epoch bracket +
alive per window), `rec_scopeN.data` + `scopeN_{dso,comm}.txt` (full-episode task-clock
records), `cpustat_scopeN.tsv` (10 Hz cpu.stat timeline), `agent.log`, `metadata.json`
(provenance), `traj/` (SWE), `DONE` (resume marker).

## Isolation (runtime-only, restored by trap on ANY exit)

- CPUs 2-11,14-23 = measured partition (`measured.slice`, docker cgroup-parent switched to it);
  CPUs 0-1,12-13 = housekeeping (system.slice + user.slice shielded there, IRQs steered there,
  proxy/pollers/perf writers pinned there).
- performance governor + no_turbo=1 (fixed ~base clock), THP never, NMI watchdog off
  (frees a GP counter), k3s stopped, stale perf killed.
- Hard guarantee about kernel per-cpu threads requires isolcpus (reboot) — deliberately NOT
  used; residual is <1% and outside the measured cgroups anyway (documented in Methods).

## Measurement rules

- **cgroups, never PIDs** (tool processes are children of containerd, invisible to PID-attach).
- Same-window decomposition: one `perf stat --for-each-cgroup=<all scopes>` per window.
- **Zero multiplexing**: 8 small groups, each fits the 8 GP counters; merged TMA group
  (slots + 8 topdown) costs 0 GP counters -> L1+L2 nest in the same window. Validator rejects
  any window with a scaling annotation or `<not counted>`.
- Windows cycle for the WHOLE episode -> N windows/group/run; medians + dispersion, never
  single-window numbers.
- OC harness/tool split is comm-based (agent family = node/openclaw*/bun vs rest), convergent
  3-way sweep at 20 ms. /toolexec purity is a HARD gate (measured 0.0%). /agent carries
  physical birth-leakage (exec/linker CPU of tool children outruns any poll) — E4 reports it
  per episode as a contamination bound; attach that bound to any OC agent-side claim.

## Gotchas (inherited from the repo's history)

- perf binary: `/usr/lib/linux-tools-6.8*/perf` (`/usr/bin/perf` is broken on this box).
- sweagent run-batch silently skips instances with existing trajectories -> output dirs are
  per-run (`runs/glm_live/<inst>_r<N>`) and wiped before launch.
- Kill stale root `perf -a` before capturing (PMU-holding orphans -> `<not counted>`).
- litellm cost tracking does not know glm-5.2 -> sweagent cost limits set to 0 (unlimited);
  episode wall is bounded by SWE_DRAIN_S/OC_DRAIN_S instead.
- The GLM key was pasted in a chat transcript -> rotate after the campaign.
