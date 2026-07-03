### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Scheduled Meetings
- `req_003` **Q2 Budget Alignment** (P2) → Monday 2026-04-13 09:00–09:45 CST
- `req_002` **Dataset Labeling Escalation** (P1) → Monday 2026-04-13 13:00–13:30 CST
- `req_004` **Vision Demo Rehearsal** (P0) → Tuesday 2026-04-14 11:00–12:00 CST
- `req_005` **Red Team Security Review** (P1) → Tuesday 2026-04-14 13:30–14:30 CST
- `req_015` **Ops Overflow Triage** (P2) → Tuesday 2026-04-14 17:30–18:00 CST
- `req_006` **Offline Eval Deep Dive** (P0) → Wednesday 2026-04-15 10:00–11:30 CST
- `req_007` **Hiring Panel Calibration** (P2) → Wednesday 2026-04-15 14:30–15:15 CST
- `req_009` **Infra Cost Reduction Workshop** (P1) → Thursday 2026-04-16 13:00–14:30 CST
- `req_010` **Design Partner Feedback Review** (P2) → Thursday 2026-04-16 15:00–16:00 CST
- `req_012` **Compliance Follow-up** (P1) → Friday 2026-04-17 13:30–14:00 CST

### High Priority Decisions
**P0 meetings (critical):**
- `req_004` (Vision Demo Rehearsal) was placed at 11:00–12:00 CST on 2026-04-14. All required attendees were available.
- `req_006` (Offline Eval Deep Dive) was placed at 10:00–11:30 CST on 2026-04-15. All required attendees were available.
**P0 meetings that could NOT be scheduled (critical failures):**
- `req_001` (CapRL Launch Readiness): alice@example.com is busy 10:30-11:30 CST on 2026-04-13 — conflicts with proposed slot 10:30-11:30. alice@example.com is busy 10:30-11:30 CST on 2026-04-13 — conflicts with proposed slot 11:00-12:00. alice@example.com is busy 15:00-16:00 CST on 2026-04-13 — conflicts with proposed slot 15:30-16:30. bob@example.com is busy 16:30-17:30 CST on 2026-04-13 — conflicts with proposed slot 16:00-17:00. bob@example.com is busy 16:30-17:30 CST on 2026-04-13 — conflicts with proposed slot 16:30-17:30.
- `req_011` (Friday Research Sync): alice@example.com is busy 10:30-11:30 CST on 2026-04-17 — conflicts with proposed slot 10:30-11:30. alice@example.com is busy 10:30-11:30 CST on 2026-04-17 — conflicts with proposed slot 11:00-12:00.

**P1/P2 meetings scheduled (no priority trade-offs were necessary — scheduling proceeded in strict priority order):**
- `req_002` (Dataset Labeling Escalation, P1) placed at 13:00 on 2026-04-13.
- `req_005` (Red Team Security Review, P1) placed at 13:30 on 2026-04-14.
- `req_009` (Infra Cost Reduction Workshop, P1) placed at 13:00 on 2026-04-16.
- `req_012` (Compliance Follow-up, P1) placed at 13:30 on 2026-04-17.
- `req_003` (Q2 Budget Alignment, P2) placed at 09:00 on 2026-04-13.
- `req_007` (Hiring Panel Calibration, P2) placed at 14:30 on 2026-04-15.
- `req_010` (Design Partner Feedback Review, P2) placed at 15:00 on 2026-04-16.
- `req_015` (Ops Overflow Triage, P2) placed at 17:30 on 2026-04-14.

### Unscheduled Requests
- `req_001` — alice@example.com is busy 10:30-11:30 CST on 2026-04-13 — conflicts with proposed slot 10:30-11:30. alice@example.com is busy 10:30-11:30 CST on 2026-04-13 — conflicts with proposed slot 11:00-12:00. alice@example.com is busy 15:00-16:00 CST on 2026-04-13 — conflicts with proposed slot 15:30-16:30. bob@example.com is busy 16:30-17:30 CST on 2026-04-13 — conflicts with proposed slot 16:00-17:00. bob@example.com is busy 16:30-17:30 CST on 2026-04-13 — conflicts with proposed slot 16:30-17:30.
- `req_011` — alice@example.com is busy 10:30-11:30 CST on 2026-04-17 — conflicts with proposed slot 10:30-11:30. alice@example.com is busy 10:30-11:30 CST on 2026-04-17 — conflicts with proposed slot 11:00-12:00.
- `req_008` — alice@example.com is unavailable on Wednesday (Wed 2026-04-15 17:00-18:30 CST) per scheduling constraints. alice@example.com is busy 09:30-10:00 CST on 2026-04-16 — conflicts with proposed slot 09:00-10:00. alice@example.com is busy 09:30-10:00 CST on 2026-04-16 — conflicts with proposed slot 09:30-10:30.
- `req_014` — alice@example.com is unavailable on Wednesday (Wed 2026-04-15 09:00-10:30 CST) per scheduling constraints. alice@example.com is unavailable on Wednesday (Wed 2026-04-15 14:30-16:00 CST) per scheduling constraints.
- `req_013` — frank@example.com is unavailable on Friday (Fri 2026-04-17 17:00-18:00 CST) per scheduling constraints.
