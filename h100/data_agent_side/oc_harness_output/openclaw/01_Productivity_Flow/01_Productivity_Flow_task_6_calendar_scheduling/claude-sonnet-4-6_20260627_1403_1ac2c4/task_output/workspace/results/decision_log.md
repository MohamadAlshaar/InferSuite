### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Scheduled Meetings

- `req_004` **Vision Demo Rehearsal** (P0) — 2026-04-14 11:00–12:00 CST — required: alice@example.com, grace@example.com
- `req_006` **Offline Eval Deep Dive** (P0) — 2026-04-15 10:00–11:30 CST — required: bob@example.com, carol@example.com, frank@example.com
- `req_002` **Dataset Labeling Escalation** (P1) — 2026-04-13 13:00–13:30 CST — required: carol@example.com, frank@example.com
- `req_005` **Red Team Security Review** (P1) — 2026-04-14 13:30–14:30 CST — required: frank@example.com, heidi@example.com
- `req_009` **Infra Cost Reduction Workshop** (P1) — 2026-04-16 13:00–14:30 CST — required: bob@example.com, frank@example.com
- `req_012` **Compliance Follow-up** (P1) — 2026-04-17 13:30–14:00 CST — required: dave@example.com, grace@example.com
- `req_003` **Q2 Budget Alignment** (P2) — 2026-04-13 09:00–09:45 CST — required: dave@example.com, heidi@example.com
- `req_007` **Hiring Panel Calibration** (P2) — 2026-04-15 14:30–15:15 CST — required: dave@example.com, heidi@example.com
- `req_010` **Design Partner Feedback Review** (P2) — 2026-04-16 15:00–16:00 CST — required: grace@example.com, heidi@example.com
- `req_015` **Ops Overflow Triage** (P2) — 2026-04-14 17:30–18:00 CST — required: frank@example.com, heidi@example.com

### High Priority Decisions

**req_001 (P0 — CapRL Launch Readiness) — NOT scheduled:** Requires Alice, Bob, and Carol simultaneously for 60 min. Window 1 (10:30–12:00 CST Mon 2026-04-13): Alice's existing `Product Planning Sync` (10:30–11:30) consumes the first 60-min slot; the only remaining gap (11:30–12:00) is only 30 min and the next slot (11:30–12:30) crosses the hard lunch constraint. Window 2 (15:30–17:30 CST): Bob's existing `Infrastructure Operations Check` (16:30–17:30) blocks the last available slot. No lower-priority meeting was dropped — the conflict is with immovable original events.

**req_004 (P0 — Vision Demo Rehearsal) — Scheduled Tue 11:00–12:00 CST:** Processed first among P0 requests (after req_001 failed). Alice and Grace are both free in window 1 (11:00–12:00 CST on 2026-04-14). Scheduled immediately to secure the slot before any lower-priority request could claim it.

**req_006 (P0 — Offline Eval Deep Dive) — Scheduled Wed 10:00–11:30 CST:** Requires Bob, Carol, Frank (90 min). Alice is optional but unavailable Wednesdays — since unavailability only blocks *required* attendees, this does not prevent scheduling. Bob, Carol, and Frank are all free 10:00–11:30 on 2026-04-15 (Wednesday). Alice is omitted from the event's attendee list since she has a conflicting meeting.

**req_011 (P0 — Friday Research Sync) — NOT scheduled:** Only window is 10:30–12:00 CST on Fri 2026-04-17. Alice's existing `Hiring Debrief` (10:30–11:30) blocks the first 60-min slot, and the 11:30–12:30 slot crosses the lunch break hard constraint. Original events cannot be moved, so no valid slot exists for all three required attendees (Alice, Bob, Carol).

**req_008 (P1) and req_014 (P1) — Both blocked by Alice's Wednesday unavailability:** Alice is a required attendee in both and all their preferred windows fall on Wednesday (req_014) or are otherwise blocked (req_008's Thursday window is too narrow: only 30 min free after Alice's 09:30 standup before the 10:30 window boundary).

**req_013 (P2 — Late Friday Launch Retro) — Double hard block:** Frank is unavailable Fridays from 17:00 onward; the only window is Fri 17:00–18:00. Alice also has `Weekly Wrap-up` (16:30–17:30) conflicting. Either constraint alone would prevent scheduling.

**Priority trade-off (req_005 vs req_003 window overlap):** req_005 (P1) was processed before req_003 (P2) and claimed the 13:30–14:30 slot on 2026-04-14. req_003 (Q2 Budget Alignment, P2) was successfully placed in the Mon 09:00–09:45 window instead — no conflict arose.

### Unscheduled Requests

- `req_001` — alice@example.com has an existing meeting that conflicts with 10:30–11:30
- `req_011` — alice@example.com has an existing meeting that conflicts with 10:30–11:30
- `req_008` — alice@example.com is unavailable Wednesday during this window
- `req_014` — alice@example.com is unavailable Wednesday during this window
- `req_013` — alice@example.com has an existing meeting that conflicts with 17:00–18:00

### Original Calendar Conflict Analysis

No time conflicts were detected among the 30 original events. Five original events (orig-003, orig-009, orig-015, orig-021, orig-027) fall during 12:00–13:00 CST (lunch hour) — these are preserved unmodified per the `preserve_original_events` hard constraint. The lunch-break constraint applies only to newly scheduled meetings. All 30 original events appear unmodified in `scheduled.ics`.
