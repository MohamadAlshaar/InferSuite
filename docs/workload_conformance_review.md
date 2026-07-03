# Workload conformance & literature review (2026-07-03)

Independent agent review: benchmark papers vs our implementations vs our findings.
Full deviations table and citations below; headline outcomes:

- No deviation invalidates the CPU-characterization claims.
- PRESENTATION RULES: (1) never cite our BCB 3-turn loop-solve rates as BigCodeBench pass@1;
  (2) OC 0.00 scores = task non-completion corroborated by zero tool calls, not calibrated
  hybrid grades (judge-based metrics need an external judge model); (3) SWE behavioral stats
  (turns/actions) carry the temp-0.4 + tool_choice=required caveat — but tool_choice does not
  force TEST execution, so the 0-pytest finding is genuinely behavioral.
- SWE autosubmit-on-context-exit is protocol-conformant (upstream handle_error_with_autosubmission).
- fc_local.yaml is byte-identical to upstream anthropic_filemap.yaml (official config).
- WildClawBench custom endpoints + single-task runs are officially sanctioned.
- Literature: inference dominance 71-98% (2605.26297) brackets our 84-100%; weak-model
  tool-under-use and verification-skipping are supported (2605.10912 curation, 2604.02547);
  NOTHING in the literature measures FP/microarch character of tests or tool exec
  (2511.00739 = nearest neighbor, stops at energy/latency) -> our differentiator confirmed novel.
- SWE-bench Verified possibly deprecated Feb 2026 - re-verify before citing.
- Fixed here: agentic_bcb.py timeout feedback string said 60s, actual 20s.
