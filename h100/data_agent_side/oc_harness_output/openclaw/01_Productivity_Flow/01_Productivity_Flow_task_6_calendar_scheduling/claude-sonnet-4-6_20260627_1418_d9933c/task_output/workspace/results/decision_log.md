### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### High Priority Decisions
- **req_001 – CapRL Launch Readiness** (P0, weight 5): Could NOT be scheduled. Reason `time_conflict`: alice@example.com has a conflicting meeting in this time slot. | bob@example.com has a conflicting meeting in this time slot.
- **req_004 – Vision Demo Rehearsal** (P0, weight 5): Scheduled at Tuesday 2026-04-14 11:00 - 12:00 CST. As the highest-priority tier, P0 meetings were processed first and given first pick of available slots.
- **req_006 – Offline Eval Deep Dive** (P0, weight 5): Scheduled at Wednesday 2026-04-15 10:00 - 11:30 CST. As the highest-priority tier, P0 meetings were processed first and given first pick of available slots.
- **req_011 – Friday Research Sync** (P0, weight 5): Could NOT be scheduled. Reason `time_conflict`: alice@example.com has a conflicting meeting in this time slot.

**Trade-off notes:**
- Meetings were processed strictly in priority order (P0 first, then P1, then P2). Each confirmed meeting immediately blocks its required attendees for that slot, potentially consuming capacity needed by lower-priority requests.
- **req_001** (P0 CapRL Launch Readiness) could not be scheduled due to existing calendar conflicts: on Monday 10:30-12:00 alice is busy with Product Planning Sync; on Monday 15:30-17:30 alice's Customer Escalation Review (15:00-16:00) and bob's Infrastructure Operations Check (16:30-17:30) leave no continuous 60-min gap for all three required attendees.
- **req_011** (P0 Friday Research Sync) is blocked by alice's Hiring Debrief (10:30-11:30 Fri), which occupies the only viable start times in the 10:30-12:00 window for a 60-minute slot.
- **req_014** (P1 Wednesday Cross-team Planning) is blocked entirely by alice's all-day Wednesday unavailability hard constraint — not by a competing meeting request.
- **req_008** (P1 Customer Renewal Strategy): Wednesday window blocked by alice's all-day Wednesday unavailability; Thursday 09:00-10:30 window blocked because alice's Engineering Standup (09:30-10:00) prevents any 60-minute slot from fitting.
- **req_013** (P2 Late Friday Launch Retro) is blocked by frank@example.com's hard unavailability rule (Fridays from 17:00 onward), which covers the entire 17:00-18:00 window.

### Unscheduled Requests
- `req_001` — **CapRL Launch Readiness**: `time_conflict` — alice@example.com has a conflicting meeting in this time slot. | bob@example.com has a conflicting meeting in this time slot.
- `req_011` — **Friday Research Sync**: `time_conflict` — alice@example.com has a conflicting meeting in this time slot.
- `req_008` — **Customer Renewal Strategy**: `attendee_unavailable` — alice@example.com is marked unavailable during this slot per weekly recurrence rule.
- `req_014` — **Wednesday Cross-team Planning**: `attendee_unavailable` — alice@example.com is marked unavailable during this slot per weekly recurrence rule.
- `req_013` — **Late Friday Launch Retro**: `time_conflict` — alice@example.com has a conflicting meeting in this time slot.
