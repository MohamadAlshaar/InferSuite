### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### High Priority Decisions
- `req_004` **Vision Demo Rehearsal** (P0) scheduled on 2026-04-14 at 11:00-12:00 CST. Earliest valid slot in preferred windows selected (soft constraint: prefer_earlier_slot_for_p0).
- `req_006` **Offline Eval Deep Dive** (P0) scheduled on 2026-04-15 at 10:00-11:30 CST. Earliest valid slot in preferred windows selected (soft constraint: prefer_earlier_slot_for_p0).
- `req_001` **CapRL Launch Readiness** (P0) could NOT be scheduled: alice@example.com, bob@example.com has existing meetings that block every candidate slot within all preferred windows.
- `req_011` **Friday Research Sync** (P0) could NOT be scheduled: alice@example.com has existing meetings that block every candidate slot within all preferred windows.
- All requests were processed in strict priority order (P0 > P1 > P2), then by earliest window start. When attendees' daily meeting caps were consumed by higher-priority meetings, lower-priority requests competing for the same attendees on the same day were displaced.
- Alice is fully unavailable on Wednesday (2026-04-15). Requests with Wednesday windows requiring Alice (`req_008`, `req_014`) were forced to use alternate windows or were unscheduled.
- Frank is unavailable Friday after 17:00 CST. Any Friday evening windows for requests requiring Frank (`req_013`) were blocked by this constraint.

### Unscheduled Requests
- `req_001` — alice@example.com, bob@example.com has existing meetings that block every candidate slot within all preferred windows.
- `req_011` — alice@example.com has existing meetings that block every candidate slot within all preferred windows.
- `req_014` — alice@example.com is unavailable in all preferred windows per the scheduling constraints (weekday/time unavailability rule).
- `req_008` — alice@example.com has existing meetings that block every candidate slot within all preferred windows.
- `req_013` — frank@example.com is unavailable in all preferred windows per the scheduling constraints (weekday/time unavailability rule).

### Scheduled Meeting Details
| ID | Title | Priority | Date | Time (CST) | Required Attendees |
|---|---|---|---|---|---|
| `req_003` | Q2 Budget Alignment | P2 | 2026-04-13 (Mon) | 09:00-09:45 | dave@example.com, heidi@example.com |
| `req_002` | Dataset Labeling Escalation | P1 | 2026-04-13 (Mon) | 13:00-13:30 | carol@example.com, frank@example.com |
| `req_004` | Vision Demo Rehearsal | P0 | 2026-04-14 (Tue) | 11:00-12:00 | alice@example.com, grace@example.com |
| `req_005` | Red Team Security Review | P1 | 2026-04-14 (Tue) | 13:30-14:30 | frank@example.com, heidi@example.com |
| `req_015` | Ops Overflow Triage | P2 | 2026-04-14 (Tue) | 17:30-18:00 | frank@example.com, heidi@example.com |
| `req_006` | Offline Eval Deep Dive | P0 | 2026-04-15 (Wed) | 10:00-11:30 | bob@example.com, carol@example.com, frank@example.com |
| `req_007` | Hiring Panel Calibration | P2 | 2026-04-15 (Wed) | 14:30-15:15 | dave@example.com, heidi@example.com |
| `req_009` | Infra Cost Reduction Workshop | P1 | 2026-04-16 (Thu) | 13:00-14:30 | bob@example.com, frank@example.com |
| `req_010` | Design Partner Feedback Review | P2 | 2026-04-16 (Thu) | 15:00-16:00 | grace@example.com, heidi@example.com |
| `req_012` | Compliance Follow-up | P1 | 2026-04-17 (Fri) | 13:30-14:00 | dave@example.com, grace@example.com |
