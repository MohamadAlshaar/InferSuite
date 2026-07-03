### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Scheduled Meetings
- `req_003` **Q2 Budget Alignment** (P2) → 2026-04-13 09:00 CST – 2026-04-13 09:45 CST [attendees: dave@example.com, heidi@example.com]
- `req_002` **Dataset Labeling Escalation** (P1) → 2026-04-13 13:00 CST – 2026-04-13 13:30 CST [attendees: carol@example.com, frank@example.com]
- `req_004` **Vision Demo Rehearsal** (P0) → 2026-04-14 11:00 CST – 2026-04-14 12:00 CST [attendees: alice@example.com, grace@example.com]
- `req_005` **Red Team Security Review** (P1) → 2026-04-14 13:30 CST – 2026-04-14 14:30 CST [attendees: frank@example.com, heidi@example.com]
- `req_015` **Ops Overflow Triage** (P2) → 2026-04-14 17:30 CST – 2026-04-14 18:00 CST [attendees: frank@example.com, heidi@example.com]
- `req_006` **Offline Eval Deep Dive** (P0) → 2026-04-15 10:00 CST – 2026-04-15 11:30 CST [attendees: bob@example.com, carol@example.com, frank@example.com]
- `req_007` **Hiring Panel Calibration** (P2) → 2026-04-15 14:30 CST – 2026-04-15 15:15 CST [attendees: dave@example.com, heidi@example.com]
- `req_009` **Infra Cost Reduction Workshop** (P1) → 2026-04-16 13:00 CST – 2026-04-16 14:30 CST [attendees: bob@example.com, frank@example.com]
- `req_010` **Design Partner Feedback Review** (P2) → 2026-04-16 15:00 CST – 2026-04-16 16:00 CST [attendees: grace@example.com, heidi@example.com]
- `req_012` **Compliance Follow-up** (P1) → 2026-04-17 13:30 CST – 2026-04-17 14:00 CST [attendees: dave@example.com, grace@example.com]

### High Priority Decisions
**P0 meetings scheduled (highest priority, scheduled first):**
- `req_004` Vision Demo Rehearsal: placed at 2026-04-14 11:00 CST in earliest valid slot.
- `req_006` Offline Eval Deep Dive: placed at 2026-04-15 10:00 CST in earliest valid slot.
**P0 meetings that could NOT be scheduled:**
- `req_001` CapRL Launch Readiness: All preferred slots conflict with existing meetings for bob@example.com, alice@example.com.
- `req_011` Friday Research Sync: All preferred slots conflict with existing meetings for alice@example.com.

**Scheduling order:** Requests were processed in descending priority order (P0 → P1 → P2). Higher-priority requests claimed slots first; any lower-priority request that shared attendees or windows with an already-scheduled higher-priority meeting was forced into its remaining windows or marked unscheduled if none were valid.
- `req_014` (P1) lost window(s) to higher-priority meeting(s) `req_004`.
- `req_008` (P1) lost window(s) to higher-priority meeting(s) `req_004`.
- `req_013` (P2) lost window(s) to higher-priority meeting(s) `req_004`, `req_006`.

### Unscheduled Requests
- `req_001` — All preferred slots conflict with existing meetings for bob@example.com, alice@example.com.
- `req_011` — All preferred slots conflict with existing meetings for alice@example.com.
- `req_014` — alice@example.com is unavailable in all preferred windows.
- `req_008` — alice@example.com is unavailable in all preferred windows.
- `req_013` — frank@example.com is unavailable in all preferred windows.
