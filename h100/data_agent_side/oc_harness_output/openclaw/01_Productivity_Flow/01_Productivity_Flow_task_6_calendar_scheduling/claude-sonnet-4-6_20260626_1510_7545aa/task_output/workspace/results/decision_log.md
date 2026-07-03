### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Original Calendar Conflict Analysis
No attendee conflicts detected among original events.

### High Priority Decisions
**P0 meetings scheduled (2):**
- `req_004` — Vision Demo Rehearsal: scheduled 2026-04-14 Tue 11:00–12:00 (Asia/Shanghai). Weight: 5.
- `req_006` — Offline Eval Deep Dive: scheduled 2026-04-15 Wed 10:00–11:30 (Asia/Shanghai). Weight: 5.

**P1 meetings scheduled (4):**
- `req_002` — Dataset Labeling Escalation: scheduled 2026-04-13 Mon 13:00–13:30 (Asia/Shanghai). Weight: 3.
- `req_005` — Red Team Security Review: scheduled 2026-04-14 Tue 13:30–14:30 (Asia/Shanghai). Weight: 3.
- `req_009` — Infra Cost Reduction Workshop: scheduled 2026-04-16 Thu 13:00–14:30 (Asia/Shanghai). Weight: 3.
- `req_012` — Compliance Follow-up: scheduled 2026-04-17 Fri 13:30–14:00 (Asia/Shanghai). Weight: 3.

**P2 meetings scheduled (4):**
- `req_003` — Q2 Budget Alignment: scheduled 2026-04-13 Mon 09:00–09:45 (Asia/Shanghai). Weight: 1.
- `req_007` — Hiring Panel Calibration: scheduled 2026-04-15 Wed 14:30–15:15 (Asia/Shanghai). Weight: 1.
- `req_010` — Design Partner Feedback Review: scheduled 2026-04-16 Thu 15:00–16:00 (Asia/Shanghai). Weight: 1.
- `req_015` — Ops Overflow Triage: scheduled 2026-04-14 Tue 17:30–18:00 (Asia/Shanghai). Weight: 1.

**Trade-off notes:**
- `req_001` (CapRL Launch Readiness) could not be placed. Higher-priority meetings were scheduled first; this request was blocked by: alice@example.com has an existing meeting overlapping Mon 2026-04-13 11:00–12:00 (Asia/Shanghai).
- `req_008` (Customer Renewal Strategy) could not be placed. Higher-priority meetings were scheduled first; this request was blocked by: alice@example.com is marked unavailable on Wednesday (constraint covers the entire day or the period overlapping 17:30–18:30 local).
- `req_011` (Friday Research Sync) could not be placed. Higher-priority meetings were scheduled first; this request was blocked by: alice@example.com has an existing meeting overlapping Fri 2026-04-17 11:00–12:00 (Asia/Shanghai).
- `req_013` (Late Friday Launch Retro) could not be placed. Higher-priority meetings were scheduled first; this request was blocked by: alice@example.com has an existing meeting overlapping Fri 2026-04-17 17:00–18:00 (Asia/Shanghai).
- `req_014` (Wednesday Cross-team Planning) could not be placed. Higher-priority meetings were scheduled first; this request was blocked by: alice@example.com is marked unavailable on Wednesday (constraint covers the entire day or the period overlapping 09:30–10:30 local).

### Unscheduled Requests
- `req_001` (CapRL Launch Readiness) — [time_conflict] Window 1 (Mon 10:30–12:00): alice@example.com has orig-002 (10:30–11:30), leaving only 30 min — insufficient for 60-min meeting. Window 2 (Mon 15:30–17:30): alice blocked by orig-005 (15:00–16:00) until 16:00; starting at 16:00 or 16:15 runs into bob@example.com's orig-006 (16:30–17:30); no 60-min gap remains before 17:30 window close.
- `req_008` (Customer Renewal Strategy) — [attendee_unavailable] Window 1 (Wed Apr 15 17:00–18:30): alice@example.com is fully unavailable all day Wednesday per hard constraint. Window 2 (Thu Apr 16 09:00–10:30): alice@example.com has orig-019 (09:30–10:00), leaving only a 30-min gap (10:00–10:30) — insufficient for 60-min meeting.
- `req_011` (Friday Research Sync) — [time_conflict] Only window (Fri 10:30–12:00): alice@example.com has orig-026 (Hiring Debrief, 10:30–11:30), leaving only 30 min (11:30–12:00) before the lunch break — insufficient for 60-min meeting.
- `req_013` (Late Friday Launch Retro) — [time_conflict] Only window (Fri 17:00–18:00): alice@example.com has orig-030 (Weekly Wrap-up, 16:30–17:30) which overlaps the entire window, blocking all candidate slots.
- `req_014` (Wednesday Cross-team Planning) — [attendee_unavailable] Both windows (Wed Apr 15 09:00–10:30 and Wed Apr 15 14:30–16:00) fall on Wednesday, when alice@example.com is marked unavailable all day (00:00–23:59) per hard constraint. No alternative windows are available for this request.
