# claw-eval (Qwen2.5-32B-Instruct, claw-eval's OWN agent) — 3 splits, our perf methodology

claw-eval = self-contained benchmark + its OWN thin python agent (NOT OpenClaw). --sandbox mode
(agent on host, container runs /exec). vLLM-native (OpenAI-compatible -> our 32B). judge off.

## ROOT-CAUSE BUG FOUND (user was right "something is wrong")
claw-eval's faithful temperature 0.0 + its strict "use native tool calls, be brief" system prompt
made the 32B DEGENERATE: it emitted a truncated narration then EOS *before* the tool call (exactly
97 tokens), so claw-eval saw "no tool call" and ended the turn -> agent quit after 1-2 turns.
NOT a vLLM cap / max_tokens (verified: vLLM gave 184-594 tokens in isolation). FIX = temperature 0.6
(necessary deviation; the 32B is unstable at greedy temp 0, frontier models aren't). This likely also
explains the OpenClaw plan-only stalls (BigCodeBench worked because it used temp 0.4).

## Results @ temp 0.6 (agent genuinely works now)
| split | task | window | tool-exec | CPU | microarch |
|---|---|---|---|---|---|
| general | T100_reverse_decoder | 47s | ~0s | 0.1 | agent did 4 tool calls, but fast file-I/O (<1s) |
| multimodal | M001_clock (webpage) | 56s | 3s | 1.2 core-s | IPC 1.84, **AVX 50%**, BE 33 (chromium render) |
| multi_turn | C18 (consultation) | 250s | ~0s | 0 | text advice, no tools; 2 models on GPU |
Plots: clawneval_3splits_time_donuts.png, clawneval_3splits_tma.png.

## Finding
With a WORKING agent (temp fixed), agentic CPU is still task-payload-dependent: it lights up only
when a task carries real compute (multimodal webpage/chromium render = AVX/backend-bound). File-I/O
tasks and text consultations stay ~0 CPU. So the CPU story is about the TOOL PAYLOAD, not the agent.
Cross-workload: BigCodeBench numpy (290 core-s, 26% AVX-512) >> claw-eval render (1.2 core-s, 50% AVX)
>> OpenClaw node.js tools (4 core-s, 0% AVX) >> I/O/consultation (~0).
