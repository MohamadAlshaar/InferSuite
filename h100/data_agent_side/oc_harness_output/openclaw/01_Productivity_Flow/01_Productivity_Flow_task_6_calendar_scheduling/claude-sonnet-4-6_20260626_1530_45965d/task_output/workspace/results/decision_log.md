### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Scheduling Details
- `req_004` **[P0]** Vision Demo Rehearsal → Tue 2026-04-14 11:00-12:00 CST  (attendees: alice@example.com, grace@example.com)
- `req_006` **[P0]** Offline Eval Deep Dive → Wed 2026-04-15 10:00-11:30 CST  (attendees: bob@example.com, carol@example.com, frank@example.com)
- `req_002` **[P1]** Dataset Labeling Escalation → Mon 2026-04-13 13:00-13:30 CST  (attendees: carol@example.com, frank@example.com)
- `req_005` **[P1]** Red Team Security Review → Tue 2026-04-14 13:30-14:30 CST  (attendees: frank@example.com, heidi@example.com)
- `req_009` **[P1]** Infra Cost Reduction Workshop → Thu 2026-04-16 13:00-14:30 CST  (attendees: bob@example.com, frank@example.com)
- `req_012` **[P1]** Compliance Follow-up → Fri 2026-04-17 13:30-14:00 CST  (attendees: dave@example.com, grace@example.com)
- `req_003` **[P2]** Q2 Budget Alignment → Mon 2026-04-13 09:00-09:45 CST  (attendees: dave@example.com, heidi@example.com)
- `req_007` **[P2]** Hiring Panel Calibration → Wed 2026-04-15 14:30-15:15 CST  (attendees: dave@example.com, heidi@example.com)
- `req_010` **[P2]** Design Partner Feedback Review → Thu 2026-04-16 15:00-16:00 CST  (attendees: grace@example.com, heidi@example.com)
- `req_015` **[P2]** Ops Overflow Triage → Tue 2026-04-14 17:30-18:00 CST  (attendees: frank@example.com, heidi@example.com)

### Original Calendar Conflicts Detected
- None detected.

### High Priority Decisions
- `req_004` **Vision Demo Rehearsal** (P0, weight=5): Scheduled at Tue 2026-04-14 11:00-12:00 CST. Required attendees alice@example.com, grace@example.com were all available.
- `req_006` **Offline Eval Deep Dive** (P0, weight=5): Scheduled at Wed 2026-04-15 10:00-11:30 CST. Required attendees bob@example.com, carol@example.com, frank@example.com were all available.
- `req_001` **CapRL Launch Readiness** (P0): Could not be scheduled — bob@example.com has a conflicting meeting from 16:30 to 17:30 CST on 2026-04-13. All preferred windows were exhausted after higher-priority meetings claimed the available slots or attendees.
- `req_011` **Friday Research Sync** (P0): Could not be scheduled — alice@example.com has a conflicting meeting from 10:30 to 11:30 CST on 2026-04-17. All preferred windows were exhausted after higher-priority meetings claimed the available slots or attendees.
- `req_008` **Customer Renewal Strategy** (P1): Could not be scheduled — alice@example.com has a conflicting meeting from 09:30 to 10:00 CST on 2026-04-16. All preferred windows were exhausted after higher-priority meetings claimed the available slots or attendees.
- `req_014` **Wednesday Cross-team Planning** (P1): Could not be scheduled — alice@example.com is marked unavailable during this slot. All preferred windows were exhausted after higher-priority meetings claimed the available slots or attendees.

### Unscheduled Requests
- `req_001` — [P0] **CapRL Launch Readiness**: bob@example.com has a conflicting meeting from 16:30 to 17:30 CST on 2026-04-13.
- `req_011` — [P0] **Friday Research Sync**: alice@example.com has a conflicting meeting from 10:30 to 11:30 CST on 2026-04-17.
- `req_008` — [P1] **Customer Renewal Strategy**: alice@example.com has a conflicting meeting from 09:30 to 10:00 CST on 2026-04-16.
- `req_014` — [P1] **Wednesday Cross-team Planning**: alice@example.com is marked unavailable during this slot.
- `req_013` — [P2] **Late Friday Launch Retro**: alice@example.com has a conflicting meeting from 16:30 to 17:30 CST on 2026-04-17.
