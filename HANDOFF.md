# HANDOFF — 2026-07-11 (end of session)

Master's thesis (InferSuite): characterize what the CPU does DURING vs OUTSIDE LLM inference.
This session: finished + certified the SWE-GLM figure set, ran + plotted the ISOLATED service
campaign, verified the OpenClaw methodology with 3 live GLM episodes, scouted the JS pilot.

## 1. State of the three studies

### A. SWE agents x GLM-5.2 (COMPLETE, certified)
- Data: `local_agents/data/glm_swe_*` (+ replays `glm_replay_swe_*`); all E1-E6 OK; 5/5 resolves.
- Figures: `local_agents/glm_plots/swe/` (10 figs) + `swe/extra/` (2 demoted).
  Main set maps to the 5 wanted results: time_split, cpu_work (donut+table), tool_calls
  (all-vs-heavy two-tone), hw_threads (per-core lanes), timeline, tma_uop, signature,
  harness_anatomy (leaf categories / perf card / burst-growth), thread_lanes.
- Audit: `scripts/glm/audit_plots.py` — ALL MATCH (~100 checks). MANIFEST with DEFINITIONS
  section (answers the 7-point critique: cores, core-seconds, boundaries, heavy=0.3, SMT).
- Key findings: harness invariance (IPC 2.45-2.89, uop-cache 69-75%, 1-core ceiling);
  harness = ~75% CPython machinery + tiktoken/JSON/strings; per-burst cost grows ~50x with
  context (django); tool species: cc1 (astropy) / OpenBLAS 20-core saturation (scikit) /
  CPython (sympy) / treadmill (django).

### B. ISOLATED local service campaign (COMPLETE, certified)
- Data: `local_service/data_iso/svc_<bucket>_tok<tier>/run_<1..3>` — 36/36 cells,
  validator `scripts/iso/validate_service.py` = 366 OK / 0 FAIL (report: full_validation.txt).
- Figures: `local_service/plots_iso/` (9) + MANIFEST. MAIN = svc_tier_donuts (GPU decode owns
  95/98/99% of wall per tier); svc_system_map; composition / signature / tma (median-run, no
  pooling) / timeline / cost-per-token / dispersion / time-split.
- Cell labels: `64out/192out/320out` ticks + bracket `9-26 / ~150 / ~435 / ~720 tokens in`
  (measured via vLLM /tokenize).
- Findings: vLLM invariant spin (IPC 3.59, uop-cache 99%, ~1.9 cores); fastapi input-ladder
  (IPC 0.59->1.21, FP 100% packed, embed bursts ~10 cores); storage = OS-share workloads;
  cost 4->11 core-s per 1k tokens with tier; repeats reproduce vLLM to 0.0-0.5%.

### C. OpenClaw x GLM-5.2 (methodology VERIFIED; campaign not yet run)
- 3 certified episodes in `local_agents/data/glm_oc_*`:
  * calendar run_1 — solved 1.00, ZERO tool exec (pure remote reasoning), 21 core-s total,
    proxy owns 78% of local CPU (streaming relay of thinking).
  * linkapix run_1 — 0.41 (image 0 / description 0.83), first real code loop, 2 spawned PIDs.
  * jigsaw-med run_1 — 0.85, 18.4 min, 54 spawned PIDs (GLM apt/pip-installed tesseract+opencv
    toolchain!), tools own 113 of 144 core-s. E4 FAILED agent-side (21.5% purity — spawn-storm
    birth leakage); tool-side clean (0.0% contamination). Tool-side numbers valid; agent-side
    numbers carry the bound.
- Three-way join VALIDATED on all 3 (transcript chat.jsonl events x fence PIDs x records):
  framework-helper share = 0 in every episode; built-ins (read/write/edit/image) execute
  IN the gateway process (agent fence) — only spawned execs reach /toolexec.
- Figures: `local_agents/glm_plots/oc/` — oc_spectrum (MAIN: task flips regime 78% proxy ->
  78% tools), oc_two_view (3 rows), oc_timeline_{linkapix,jigsaw} (transcript arrows joined
  onto fences), oc_system_map, oc_what_ran. Plotters: plot_oc_episodes.py, plot_oc_system_map.py.
- OC architecture notes: gateway = node/V8 server; built-in vs spawned tool distinction;
  streaming = chat-UX legacy, pure overhead headless (litellm pays ~10x gateway per fragment).

## 2. LOCKED conventions (memory: feedback-figure-vocabulary)
- CPU usage (cores) = core-seconds per second; amounts in core-seconds; shares % of CPU time;
  core = logical CPU; partition = 20 (10 phys x SMT-2). NO bare "CPUs"/"CPU-s" labels.
- Peaks: spike (0.1 s, capped at 20) / sustained (1 s) — both shown.
- TMA/signature: NEVER pool runs — median run per row/cell, spread documented.
- Figures: good titles + axis labels ONLY (no footers/subtitle essays; definitions in MANIFESTs).
- "OS share" not "kernel". "heavy call" = >0.3 cores burst (display-only; count sensitivity
  documented: 0.2/0.3/0.4 -> e.g. sympy 163/111/71). Plot with SYSTEM python3.
- Figures enter the thesis only after user approval in chat.

## 3. Open items (priority order)
1. **JS pilot (scouted, ready)**: SWE-bench Multimodal dev split (102 instances, public tests,
   local grading works — swebench 4.1.0 ships constants/javascript.py; the local SWE-agent
   checkout has MM support). These are CODING tasks (patch + tests); images only decorate the
   issues. Plan: smoke whether images survive the litellm path; then 1 instance —
   `markedjs/marked` if text-only (fair), chartjs if vision works. Pre-registered hypothesis:
   harness signature invariant; tool fence = V8/JIT species (L1I/MITE pressure, higher
   bad-spec, zero packed FP).
2. **OC campaign proper**: 4 tasks x 3 repeats (calendar, web-digest, pdf-digest, image-crop;
   linkapix + jigsaw-med are also in the kit's OCT map now). Consider (3) first:
3. **eBPF exec-time sorting (watcher rung 2)** — justified by jigsaw's E4 failure; makes the
   /agent fence exact under spawn-storms while OpenClaw stays stock.
4. Service supplementary captures (optional): per-request latency anatomy (loadgen rerun +
   kubectl cp before teardown); record-enabled cells for fastapi leaf anatomy + hw lanes.
5. Hygiene: rotate GLM key (it was pasted in chat!) + old Anthropic key; commit the kits
   (much is untracked); OC figure titles still carry single-episode caveats (by design).

## 4. Environment state at handoff
- GLM balance: recharged & working; key at ~/.glm_key (never echo).
- k3s RUNNING, service stack up; vLLM serves qwen2.5-7b-instruct-awq (base model was swapped
  back from the SWE-era coder).
- KUBECONFIG=$HOME/.kube/k3s-local.yaml (the default kubeconfig is STALE — ignore it).
- Isolation fully restored (governor/turbo/THP/IRQ/slices); no stale perf; monitors stopped.
- perf binary: /usr/lib/linux-tools-6.8.0-134/perf (report/script need -f as root on user data).

## 5. One-command re-entry points
- Agent figs:   cd local_agents/scripts/glm && python3 plot_glm_results.py && python3 audit_plots.py
- OC episode:   env REPEATS=1 OC_TASKS="<task>" ./run_glm_campaign.sh campaign oc
- OC figs:      python3 plot_oc_episodes.py ; python3 plot_oc_system_map.py
- Service figs: cd local_service/scripts/iso && python3 plot_service_iso.py
- Validators:   python3 validate_glm_agents.py ../../data glm ; python3 validate_service.py
