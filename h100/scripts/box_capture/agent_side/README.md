# H100 agent-side campaign (the scope the 32B runs missed)

The July H100 campaign measured the engine (during inference) and the tool executions, but never
the **agent harness** as its own scope, and never all three scopes in the same windows. This kit
ports the validated local chains (`local_agents/scripts/*_live_two_view.sh`) to the bare-vLLM box.

What it adds at 32B: harness portable counters (IPC/cache/MLP/FP) + perf records + same-window
CPU two-view + GPU wall-time timelines, for SWE, BCB, and the four OC tasks — at real agent
quality (the 32B actually runs tests and browses, unlike the local 7B).
What it cannot add: TMA (the box is a KVM guest, no topdown events). Harness TMA L1/L2 stays a
local-only result.

## Run order (single H100, ~2-3 h total)

1. Standard box setup (as in the July campaign): vllmenv + both 32B models, `~/bcb` (agentic_bcb.py
   + venv), `~/swe` (sweagent + fc_local.yaml + swebench astropy image), `~/WildClawBench`
   (wildclawbench image, `my_api.json` -> `http://localhost:8000/v1`, model id `instruct-32b`,
   api `openai-completions`). Copy this dir + `serve_h100.sh` to the box.
2. Coder phase:
   `systemd-run --user --scope --unit=vllm-serve -- env MODEL=Qwen/Qwen2.5-Coder-32B-Instruct NAME=coder-32b ./serve_h100.sh`
   then `./agent_side_chain.sh swe` and `./agent_side_chain.sh bcb`.
3. Instruct phase: restart serve with `MODEL=Qwen/Qwen2.5-32B-Instruct NAME=instruct-32b`,
   then `./agent_side_chain.sh oc-calendar oc-web oc-pdf oc-crop` (one at a time).
4. `rsync -a ~/agent_side_data/ <repo>/h100/data_agent_side/` (new tree, strict provenance —
   do not mix with the July `h100/data*`).

## Design notes (carried over from the local campaign)

- Work guard polls `/metrics` `vllm:num_requests_running` (serve uses `--disable-log-stats`,
  so there are no `Running:` log lines to grep).
- Stats run FIRST and in parallel with the records: 32B episodes are longer than the local 7B
  ones, but the guard-to-capture pattern is kept identical for comparability.
- Engine cgroup is resolved from the `VLLM::EngineCore` worker (the API server PID misses the
  CPU burner — the historical undercount bug).
- Per-group aliveness flags land in `stat_groups_alive.txt`; validation requires >=3/5 groups
  in-window plus both records.
