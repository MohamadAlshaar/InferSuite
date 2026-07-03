### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Scheduled Meetings

| ID | Title | Priority | Slot (Asia/Shanghai) | Attendees |
|----|-------|----------|----------------------|-----------|
| req_002 | Dataset Labeling Escalation | P1 | Mon 2026-04-13 13:00–13:30 | carol, frank |
| req_003 | Q2 Budget Alignment | P2 | Mon 2026-04-13 09:00–09:45 | dave, heidi |
| req_004 | Vision Demo Rehearsal | P0 | Tue 2026-04-14 11:00–12:00 | alice, grace |
| req_005 | Red Team Security Review | P1 | Tue 2026-04-14 13:30–14:30 | frank, heidi |
| req_006 | Offline Eval Deep Dive | P0 | Wed 2026-04-15 10:00–11:30 | bob, carol, frank |
| req_007 | Hiring Panel Calibration | P2 | Wed 2026-04-15 14:30–15:15 | dave, heidi |
| req_009 | Infra Cost Reduction Workshop | P1 | Thu 2026-04-16 13:00–14:30 | bob, frank |
| req_010 | Design Partner Feedback Review | P2 | Thu 2026-04-16 15:00–16:00 | grace, heidi |
| req_012 | Compliance Follow-up | P1 | Fri 2026-04-17 13:30–14:00 | dave, grace |
| req_015 | Ops Overflow Triage | P2 | Tue 2026-04-14 17:30–18:00 | frank, heidi |

### High Priority Decisions

**Scheduling order:** Requests were processed in strict priority order (P0 → P1 → P2). Within each tier, requests were processed by ID to ensure determinism. Original calendar events were never modified or removed.

**P0 meetings scheduled (weight=5 each):**
- `req_004` **Vision Demo Rehearsal** (alice, grace) was placed at Tue 2026-04-14 11:00–12:00. This was the earliest available P0 slot — both attendees are free after the morning standup (orig-007 ends 10:00) and before the lunch block.
- `req_006` **Offline Eval Deep Dive** (bob, carol, frank) was placed at Wed 2026-04-15 10:00–11:30. Wednesday is alice's unavailability day, but she is only an *optional* attendee here. All required attendees (bob, carol, frank) are free in this window.

**P0 meetings that could NOT be scheduled (highest concern):**
- `req_001` **CapRL Launch Readiness** (alice, bob, carol, P0) — In the first preferred window (Mon 10:30–12:00), alice is already in `Product Planning Sync` (orig-002, 10:30–11:30), leaving only a 30-minute gap before the lunch block — insufficient for a 60-minute meeting. In the second window (Mon 15:30–17:30), slot 15:30 conflicts with alice's `Customer Escalation Review` (orig-005, 15:00–16:00), and slot 16:00–17:00 conflicts with bob's `Infrastructure Operations Check` (orig-006, 16:30–17:30). No 60-minute window free for all three required attendees.
- `req_011` **Friday Research Sync** (alice, bob, carol, P0) — The only preferred window is Fri 10:30–12:00. Alice is occupied by `Hiring Debrief` (orig-026, 10:30–11:30), and the next slot at 11:30 would extend to 12:30, crossing the lunch break (12:00–13:00). No schedulable slot exists within the single requested window.

**Key trade-off note:** Because req_001 and req_011 are both P0, no lower-priority meeting displaced them — they failed strictly due to existing calendar conflicts and the lunch constraint. No deliberate priority preemption occurred for these two.

### Unscheduled Requests
- `req_001` — **time_conflict**: Alice (required) is booked in `Product Planning Sync` (10:30–11:30) blocking Window 1; Bob conflicts with `Infrastructure Operations Check` (16:30–17:30) blocking the only remaining slot in Window 2. No 60-minute slot exists across both windows where alice, bob, and carol are simultaneously free.
- `req_008` — **attendee_unavailable**: Alice (required) is fully unavailable on Wednesday (per hard constraint). Both preferred windows fall on Wednesday 2026-04-15, making scheduling impossible.
- `req_011` — **time_conflict**: Alice (required) is in `Hiring Debrief` (orig-026, 10:30–11:30 local) and the only candidate slot at 11:30–12:30 crosses the hard lunch break (12:00–13:00). No valid slot exists in the single preferred window (Fri 10:30–12:00).
- `req_013` — **attendee_unavailable**: Frank (required) is unavailable Friday from 17:00–23:59 (per hard constraint). The sole preferred window (Fri 2026-04-17 17:00–18:00) falls entirely within Frank's blocked period.
- `req_014` — **attendee_unavailable**: Alice (required) is fully unavailable on Wednesday (per hard constraint). Both preferred windows (2026-04-15 09:00–10:30 and 14:30–16:00) fall on Wednesday, making scheduling impossible regardless of other attendee availability.
