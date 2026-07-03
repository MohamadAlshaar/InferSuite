### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### Existing Calendar Conflicts
- No conflicts detected among the 30 existing events.

### Scheduled Meetings
- `req_003` [P2] **Q2 Budget Alignment**  → 2026-04-13 09:00 CST–09:45 CST  _(attendees: dave@example.com, heidi@example.com)_
- `req_002` [P1] **Dataset Labeling Escalation**  → 2026-04-13 13:00 CST–13:30 CST  _(attendees: carol@example.com, frank@example.com)_
- `req_004` [P0] **Vision Demo Rehearsal**  → 2026-04-14 11:00 CST–12:00 CST  _(attendees: alice@example.com, grace@example.com)_
- `req_005` [P1] **Red Team Security Review**  → 2026-04-14 13:30 CST–14:30 CST  _(attendees: frank@example.com, heidi@example.com)_
- `req_015` [P2] **Ops Overflow Triage**  → 2026-04-14 17:30 CST–18:00 CST  _(attendees: frank@example.com, heidi@example.com)_
- `req_006` [P0] **Offline Eval Deep Dive**  → 2026-04-15 10:00 CST–11:30 CST  _(attendees: bob@example.com, carol@example.com, frank@example.com)_
- `req_007` [P2] **Hiring Panel Calibration**  → 2026-04-15 14:30 CST–15:15 CST  _(attendees: dave@example.com, heidi@example.com)_
- `req_009` [P1] **Infra Cost Reduction Workshop**  → 2026-04-16 13:00 CST–14:30 CST  _(attendees: bob@example.com, frank@example.com)_
- `req_010` [P2] **Design Partner Feedback Review**  → 2026-04-16 15:00 CST–16:00 CST  _(attendees: grace@example.com, heidi@example.com)_
- `req_012` [P1] **Compliance Follow-up**  → 2026-04-17 13:30 CST–14:00 CST  _(attendees: dave@example.com, grace@example.com)_

### High Priority Decisions
All requests were sorted by priority weight (P0=5, P1=3, P2=1) before scheduling. Higher-priority requests were allocated first, consuming slots that lower-priority requests might otherwise have used.

**Scheduled P0 meetings (highest priority):**
- `req_004` **Vision Demo Rehearsal** secured at 2026-04-14 11:00 CST–12:00 CST, consuming availability for: alice@example.com, grace@example.com.
- `req_006` **Offline Eval Deep Dive** secured at 2026-04-15 10:00 CST–11:30 CST, consuming availability for: bob@example.com, carol@example.com, frank@example.com.

**P0 meetings that could NOT be scheduled (critical):**
- `req_001` **CapRL Launch Readiness** — All preferred windows (Mon 04/13 10:30–12:00; Mon 04/13 15:30–17:30) are blocked by existing calendar events involving the required attendees (alice@example.com, bob@example.com, carol@example.com). No contiguous free 60-minute block exists within any window.
- `req_011` **Friday Research Sync** — All preferred windows (Fri 04/17 10:30–12:00) are blocked by existing calendar events involving the required attendees (alice@example.com, bob@example.com, carol@example.com). No contiguous free 60-minute block exists within any window.

**Notable trade-offs:**
- `req_001` (P0, CapRL Launch Readiness) required alice, bob, and carol simultaneously on Mon Apr 13. Both preferred windows were fully blocked: W1 (10:30–12:00) is occupied by orig-002 (alice/dave/erin standup) and the remaining gap (11:30–12:00) collides with the lunch block; W2 (15:30–17:30) is consumed by orig-005 (alice/grace 15:00–16:00) and orig-006 (bob/frank 16:30–17:30), leaving no contiguous 60-minute window for all three attendees. This P0 request could not be accommodated without modifying existing events.
- `req_011` (P0, Friday Research Sync) required alice, bob, and carol on Fri Apr 17 10:30–12:00. orig-026 (alice/dave/heidi Hiring Debrief 10:30–11:30) blocks alice; the next available start (11:30) would end at 12:30, crossing the lunch boundary. No valid 60-minute slot exists.
- `req_014` (P1, Wednesday Cross-team Planning) and `req_008` (P1, Customer Renewal Strategy — first window) both fall on Wednesday Apr 15. alice@example.com is declared unavailable for the entire day Wednesday, making scheduling impossible in those windows. req_008's Thu Apr 16 backup window is blocked by the morning standup (orig-019), leaving insufficient room for a 60-minute slot.
- `req_013` (P2, Late Friday Launch Retro) requires frank, but frank is declared unavailable from 17:00 onward on Fridays — exactly when the only preferred window falls.
- P1 requests `req_002`, `req_005`, `req_009`, `req_012` were all successfully scheduled by taking the earliest available slot in their respective windows before P2 requests could claim those times.

### Unscheduled Requests
- `req_001` **CapRL Launch Readiness** [P0] — _time_conflict_: All preferred windows (Mon 04/13 10:30–12:00; Mon 04/13 15:30–17:30) are blocked by existing calendar events involving the required attendees (alice@example.com, bob@example.com, carol@example.com). No contiguous free 60-minute block exists within any window.
- `req_008` **Customer Renewal Strategy** [P1] — _attendee_unavailable_: alice@example.com is unavailable during all preferred windows (Wed 04/15 17:00–18:30; Thu 04/16 09:00–10:30) due to declared unavailability constraints, making it impossible to schedule the required 60-minute slot.
- `req_011` **Friday Research Sync** [P0] — _time_conflict_: All preferred windows (Fri 04/17 10:30–12:00) are blocked by existing calendar events involving the required attendees (alice@example.com, bob@example.com, carol@example.com). No contiguous free 60-minute block exists within any window.
- `req_013` **Late Friday Launch Retro** [P2] — _attendee_unavailable_: frank@example.com is unavailable during all preferred windows (Fri 04/17 17:00–18:00) due to declared unavailability constraints, making it impossible to schedule the required 60-minute slot.
- `req_014` **Wednesday Cross-team Planning** [P1] — _attendee_unavailable_: alice@example.com is unavailable during all preferred windows (Wed 04/15 09:00–10:30; Wed 04/15 14:30–16:00) due to declared unavailability constraints, making it impossible to schedule the required 60-minute slot.
