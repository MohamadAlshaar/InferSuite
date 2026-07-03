### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### High Priority Decisions

- **req_001 (P0) — CapRL Launch Readiness** was **not scheduled** despite being P0.  
  Both preferred windows on 2026-04-13 are fully blocked for required attendees.  
  Window 1 (10:30–12:00) conflicts with existing `Product Planning Sync` (10:30–11:30, alice/bob) and hits the 12:00–13:00 lunch boundary with insufficient remaining time for a 60-min meeting.  
  Window 2 (15:30–17:30) conflicts with `Customer Escalation Review` (15:00–16:00, alice) and `Infrastructure Operations Check` (16:30–17:30, bob), leaving no clean 60-min gap.  
  *No lower-priority meeting was dropped* because the windows are blocked by original (protected) events, not by newly scheduled P1/P2 requests.

- **req_011 (P0) — Friday Research Sync** was **not scheduled**.  
  Window 10:30–12:00 on 2026-04-17 has `Hiring Debrief` (10:30–11:30, alice/dave/heidi) blocking alice.  
  Only 30 minutes remain (11:30–12:00), which is insufficient for the 60-min meeting before the lunch break.  
  All three required attendees (alice, bob, carol) are blocked by original protected events across this window.

- **req_004 (P0) — Vision Demo Rehearsal** was **scheduled** at 11:00–12:00 on Tue 2026-04-14 (window 1).  
  Window 1 (11:00–12:00) fit perfectly: alice's standup ends at 10:00, leaving a clean 60-min block before lunch. Grace is not subject to her Tuesday 09:00–10:00 unavailability at this time.

- **req_006 (P0) — Offline Eval Deep Dive** was **scheduled** at 10:00–11:30 on Wed 2026-04-15 (window 1).  
  Window 1 (10:00–12:00) accommodated the 90-min block (10:00–11:30) before the Engineering Standup boundary conflict. Wednesday unavailability only affects alice; she is not a required attendee for this meeting.

- **req_008 (P1) — Customer Renewal Strategy** was **not scheduled**.  
  Window 1 (Wed 04-15 17:00–18:30) and Window 2 (Thu 04-16 09:00–10:30) both require alice@example.com, who is entirely unavailable on Wednesdays per the hard constraint. Window 2 on Thursday 09:00–10:30 conflicts with the `Engineering Standup` (09:30–10:00, alice/bob/carol) — only a 30-min gap from 09:00–09:30 or 10:00–10:30 remains, each insufficient for the 60-min duration.

- **req_014 (P1) — Wednesday Cross-team Planning** was **not scheduled**.  
  Both preferred windows fall on Wednesday 2026-04-15, and alice@example.com (a required attendee) is unavailable the entire Wednesday per the hard attendee-unavailability constraint.

- **req_013 (P2) — Late Friday Launch Retro** was **not scheduled**.  
  Window 17:00–18:00 on Friday 2026-04-17 requires frank@example.com, who is subject to a hard unavailability from Friday 17:00 onward. This directly blocks the only preferred window.

### Unscheduled Requests
- `req_001` (**CapRL Launch Readiness**) — *time_conflict*: All preferred windows have time conflicts with existing events for required attendees.
- `req_011` (**Friday Research Sync**) — *time_conflict*: All preferred windows have time conflicts with existing events for required attendees.
- `req_008` (**Customer Renewal Strategy**) — *attendee_unavailable*: alice@example.com is unavailable during all preferred windows.
- `req_014` (**Wednesday Cross-team Planning**) — *attendee_unavailable*: alice@example.com is unavailable during all preferred windows.
- `req_013` (**Late Friday Launch Retro**) — *attendee_unavailable*: frank@example.com is unavailable during all preferred windows.
