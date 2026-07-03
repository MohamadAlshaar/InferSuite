### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### High Priority Decisions

- **req_004 [P0] Vision Demo Rehearsal** — Scheduled at Tue 2026-04-14 11:00-12:00 (UTC+8).
  Placed in the first preferred window. alice@example.com and grace@example.com are both free; the slot does not cross lunch and fits within the 60-min window before noon.

- **req_006 [P0] Offline Eval Deep Dive** — Scheduled at Wed 2026-04-15 10:00-11:30 (UTC+8).
  Placed in the first preferred window (10:00-12:00). All required attendees (bob, carol, frank) are free during this slot on Wednesday. Alice is unavailable Wednesday but is only optional here.

- **req_002 [P1] Dataset Labeling Escalation** — Scheduled at Mon 2026-04-13 13:00-13:30 (UTC+8).
  Placed in the first preferred window. carol@example.com and frank@example.com are both free immediately after the lunch break ends at 13:00.

- **req_005 [P1] Red Team Security Review** — Scheduled at Tue 2026-04-14 13:30-14:30 (UTC+8).
  Placed in the first preferred window. frank@example.com and heidi@example.com have no conflicts in this afternoon slot.

- **req_009 [P1] Infra Cost Reduction Workshop** — Scheduled at Thu 2026-04-16 13:00-14:30 (UTC+8).
  Placed in the sole preferred window. bob@example.com and frank@example.com are both free for the full 90-minute slot after lunch.

- **req_012 [P1] Compliance Follow-up** — Scheduled at Fri 2026-04-17 13:30-14:00 (UTC+8).
  Placed in the first preferred window. dave@example.com and grace@example.com are both free for the 30-min slot.

**Trade-offs and conflicts:**

- `req_001` [P0] could not be scheduled. Despite P0 priority, both Monday windows are blocked by existing events for the required attendees. No gap could be found to fit this meeting ahead of other already-scheduled events.
- `req_011` [P0] could not be scheduled. The sole Friday window is blocked by alice@example.com's prior commitment, making it impossible to fit all three required attendees.
- `req_014` [P1] could not be scheduled because alice@example.com — a required attendee — is unavailable all day Wednesday, and all preferred windows fall on Wednesday April 15.
- `req_008` [P1] could not be scheduled: the Wednesday window is blocked by alice's all-day unavailability, and the Thursday window contains no 60-min gap free for all three required attendees.
- `req_013` [P2] could not be scheduled due to double conflicts in its sole window (alice and erin blocked by Weekly Wrap-up, plus frank's Friday 17:00 unavailability constraint).

### Unscheduled Requests

- `req_001` — **CapRL Launch Readiness** (P0): Window 1 (Mon 10:30-12:00): alice@example.com has Product Planning Sync at 10:30-11:30, leaving insufficient time for the 60-min meeting. Window 2 (Mon 15:30-17:30): alice@example.com is busy 15:00-16:00 (Customer Escalation Review) and bob@example.com is busy 16:30-17:30 (Infrastructure Operations Check); no 60-min gap exists in which all of alice, bob, and carol are simultaneously free. [code: `time_conflict`]

- `req_011` — **Friday Research Sync** (P0): Window (Fri 10:30-12:00): alice@example.com has Hiring Debrief at 10:30-11:30, blocking the only available 60-min slot. No valid 60-min gap exists in the preferred window for all required attendees (alice, bob, carol). [code: `time_conflict`]

- `req_014` — **Wednesday Cross-team Planning** (P1): alice@example.com is fully unavailable on Wednesdays (all-day constraint). Both preferred windows fall on Wednesday April 15, making scheduling impossible since alice is a required attendee. [code: `attendee_unavailable`]

- `req_008` — **Customer Renewal Strategy** (P1): Window 1 (Wed Apr 15 17:00-18:30): alice@example.com is unavailable all day Wednesday. Window 2 (Thu Apr 16 09:00-10:30): alice@example.com has Engineering Standup at 09:30-10:00, leaving only a 30-min free gap (09:00-09:30) which is insufficient for the 60-min meeting. No valid slot found across all preferred windows. [code: `time_conflict`]

- `req_013` — **Late Friday Launch Retro** (P2): Window (Fri Apr 17 17:00-18:00): alice@example.com and erin@example.com have Weekly Wrap-up at 16:30-17:30 which overlaps the only candidate slot (17:00-18:00). Additionally, frank@example.com is unavailable on Fridays from 17:00 onward. The meeting cannot be placed in its sole preferred window. [code: `time_conflict`]
