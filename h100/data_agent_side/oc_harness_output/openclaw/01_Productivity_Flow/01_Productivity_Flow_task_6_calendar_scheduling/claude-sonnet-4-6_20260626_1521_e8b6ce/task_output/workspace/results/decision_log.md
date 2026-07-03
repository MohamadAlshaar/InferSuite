### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Scheduled Meetings
- `req_004` **Vision Demo Rehearsal** (P0): Tue 2026-04-14 11:00 – 12:00 CST
- `req_006` **Offline Eval Deep Dive** (P0): Wed 2026-04-15 10:00 – 11:30 CST
- `req_002` **Dataset Labeling Escalation** (P1): Mon 2026-04-13 13:00 – 13:30 CST
- `req_005` **Red Team Security Review** (P1): Tue 2026-04-14 13:30 – 14:30 CST
- `req_009` **Infra Cost Reduction Workshop** (P1): Thu 2026-04-16 13:00 – 14:30 CST
- `req_012` **Compliance Follow-up** (P1): Fri 2026-04-17 13:30 – 14:00 CST
- `req_003` **Q2 Budget Alignment** (P2): Mon 2026-04-13 09:00 – 09:45 CST
- `req_007` **Hiring Panel Calibration** (P2): Wed 2026-04-15 14:30 – 15:15 CST
- `req_010` **Design Partner Feedback Review** (P2): Thu 2026-04-16 15:00 – 16:00 CST
- `req_015` **Ops Overflow Triage** (P2): Tue 2026-04-14 17:30 – 18:00 CST

### High Priority Decisions
- **req_001 (P0) 'CapRL Launch Readiness'** could not be scheduled despite highest priority. Window 1 (Mon Apr 13, 10:30-12:00): alice@example.com is in Product Planning Sync (10:30-11:30), leaving only a 30-min gap before the lunch block — insufficient for the 60-min requirement. Window 2 (Mon Apr 13, 15:30-17:30): alice is in Customer Escalation Review (15:00-16:00) and bob is in Infrastructure Operations Check (16:30-17:30); no contiguous 60-min gap exists for all three required attendees. No displacement of lower-priority meetings could open a valid window since the blockers are original (preserved) events.
- **req_011 (P0) 'Friday Research Sync'** has a single window (Fri Apr 17, 10:30-12:00). Alice is committed to Hiring Debrief (10:30-11:30), leaving only 30 min before noon — half the required 60 min. No alternative window was provided. This P0 loss was unavoidable given the calendar constraints.
- **req_004 (P0) 'Vision Demo Rehearsal'** was scheduled at 11:00-12:00 (Apr 14) in its first preferred window, taking priority over any P1/P2 that might have used that slot. No conflict arose.
- **req_006 (P0) 'Offline Eval Deep Dive'** was scheduled at 10:00-11:30 (Apr 15, Wed) in its first window. Note: alice@example.com is a required attendee here — but the Wednesday unavailability only applies to alice for req_008/req_014. alice is NOT a required attendee for req_006 (she is optional), so Wednesday scheduling was valid for bob, carol, and frank.
- **req_008 (P1) 'Customer Renewal Strategy'** is blocked on both windows. Window 1 (Wed Apr 15, 17:00-18:30): alice@example.com is fully unavailable all day Wednesday (hard constraint). Window 2 (Thu Apr 16, 09:00-10:30): the Engineering Standup occupies 09:30-10:00 for alice; only two 30-min fragments remain — insufficient for the 60-min duration. Unschedulable.
- **req_014 (P1) 'Wednesday Cross-team Planning'** both windows fall on Wednesday Apr 15. alice@example.com (required) is fully unavailable on Wednesdays. Both windows are blocked. This P1 meeting was sacrificed; no higher-priority meeting was affected by this decision.
- **req_013 (P2) 'Late Friday Launch Retro'** targets Fri Apr 17, 17:00-18:00. frank@example.com is unavailable on Fridays after 17:00 (hard constraint). No alternative window provided. Correctly unscheduled.

### Unscheduled Requests
- `req_001` — [time_conflict] Required attendees (e.g. alice@example.com) have conflicting existing meetings in all preferred windows; no contiguous 60-min gap is available.
- `req_011` — [time_conflict] Required attendees (e.g. alice@example.com) have conflicting existing meetings in all preferred windows; no contiguous 60-min gap is available.
- `req_008` — [attendee_unavailable] alice@example.com is unavailable in all preferred windows; no valid slot satisfies all hard constraints.
- `req_014` — [attendee_unavailable] alice@example.com is unavailable in all preferred windows; no valid slot satisfies all hard constraints.
- `req_013` — [attendee_unavailable] frank@example.com is unavailable in all preferred windows; no valid slot satisfies all hard constraints.
