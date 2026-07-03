### Scheduling Decision Log

### Summary
- Total requests: 15
- Scheduled: 10
- Unscheduled: 5
- Total priority weight achieved: 26

### High Priority Decisions

**P0 — CapRL Launch Readiness (req_001)** — *Not scheduled*
Although P0 (weight=5), all windows on Mon Apr 13 are blocked for alice@example.com:
window 1 (CST 10:30-12:00) collides with her 'Product Planning Sync' (10:30-11:30);
window 2 (CST 15:30-17:30) collides with her 'Customer Escalation Review' (15:00-16:00),
and after req_002 is scheduled she would hit the 4-meeting daily cap.
No lower-priority meeting was dropped in its favor because the conflict is with the
*original* immutable calendar, not a newly scheduled request.

**P0 — Vision Demo Rehearsal (req_004)** — *Scheduled Mon 2026-04-14 11:00-12:00 CST*
First-fit within window 1 (Tue Apr 14 11:00-12:00 CST). No attendee conflicts or cap
issues. Scheduled before any P1/P2 requests that share alice@example.com or grace@example.com.

**P0 — Offline Eval Deep Dive (req_006)** — *Scheduled Wed 2026-04-15 10:00-11:30 CST*
Fits in window 1 (Wed Apr 15 10:00-12:00 CST) with no required-attendee conflicts.
Note: alice is on the optional list; she is unavailable on Wednesday but is not required,
so the constraint does not apply. Scheduled ahead of P1/P2 requests on the same day.

**P0 — Friday Research Sync (req_011)** — *Not scheduled*
Only window: Fri Apr 17 10:30-12:00 CST. alice@example.com has 'Hiring Debrief'
10:30-11:30 CST (exact overlap). The window is too narrow for even a 15-min shift.
No alternative window given. No lower-priority meeting was competing for this slot.

**P1 trade-offs:**

- **req_014 (Wednesday Cross-team Planning)** was skipped because both windows fall on
  Wednesday Apr 15, when alice@example.com is entirely unavailable — a hard constraint
  that cannot be resolved by displacing any other meeting.

- **req_008 (Customer Renewal Strategy)** was blocked: window 1 (Wed Apr 15) triggers
  alice's Wednesday unavailability; window 2 (Thu Apr 16 09:00-10:30 CST) is fully
  covered by alice's 'Engineering Standup' (09:30-10:00) — every 60-min sub-slot
  overlaps it. No competing lower-priority request occupied this slot; the original
  calendar itself blocks it.

### Unscheduled Requests

- `req_001` — [P0] **CapRL Launch Readiness**: Window 1 conflicts with alice's 'Product Planning Sync' (10:30-11:30 CST); window 2 conflicts with alice's 'Customer Escalation Review' (15:00-16:00 CST) and would exceed her 4-meeting daily cap. (code: `time_conflict`)

- `req_011` — [P0] **Friday Research Sync**: Only window (Fri Apr 17 10:30-12:00 CST) fully overlaps alice's 'Hiring Debrief' (10:30-11:30 CST); no valid 60-min slot exists. (code: `time_conflict`)

- `req_008` — [P1] **Customer Renewal Strategy**: Window 1 (Wed Apr 15) blocked by alice's Wednesday-wide unavailability. Window 2 (Thu Apr 16 09:00-10:30 CST) blocked by alice's 'Engineering Standup' (09:30-10:00) overlapping every candidate slot. (code: `attendee_unavailable`)

- `req_014` — [P1] **Wednesday Cross-team Planning**: Both windows fall on Wednesday Apr 15; alice@example.com is unavailable the entire day (hard constraint). (code: `attendee_unavailable`)

- `req_013` — [P2] **Late Friday Launch Retro**: Only window (Fri Apr 17 17:00-18:00 CST) conflicts with alice's 'Weekly Wrap-up' (16:30-17:30 CST); the 60-min meeting cannot start at 17:00 without overlapping. (code: `time_conflict`)

### All Scheduling Decisions

- ❌ `req_001` [P0] **CapRL Launch Readiness** → `time_conflict`: Conflicts in both windows for alice@example.com; daily cap also applies in window 2.
- ✅ `req_004` [P0] **Vision Demo Rehearsal** → Tue 2026-04-14 11:00 - 12:00 (CST +08:00)
- ✅ `req_006` [P0] **Offline Eval Deep Dive** → Wed 2026-04-15 10:00 - 11:30 (CST +08:00)
- ❌ `req_011` [P0] **Friday Research Sync** → `time_conflict`: Single window fully blocked by alice's 'Hiring Debrief'.
- ✅ `req_002` [P1] **Dataset Labeling Escalation** → Mon 2026-04-13 13:00 - 13:30 (CST +08:00)
- ✅ `req_005` [P1] **Red Team Security Review** → Tue 2026-04-14 13:30 - 14:30 (CST +08:00)
- ❌ `req_008` [P1] **Customer Renewal Strategy** → `attendee_unavailable`: alice unavailable Wed; Thu window blocked by her existing standup.
- ✅ `req_009` [P1] **Infra Cost Reduction Workshop** → Thu 2026-04-16 13:00 - 14:30 (CST +08:00)
- ✅ `req_012` [P1] **Compliance Follow-up** → Fri 2026-04-17 13:30 - 14:00 (CST +08:00)
- ❌ `req_014` [P1] **Wednesday Cross-team Planning** → `attendee_unavailable`: alice unavailable all day Wednesday.
- ✅ `req_003` [P2] **Q2 Budget Alignment** → Mon 2026-04-13 09:00 - 09:45 (CST +08:00)
- ✅ `req_007` [P2] **Hiring Panel Calibration** → Wed 2026-04-15 14:30 - 15:15 (CST +08:00)
- ✅ `req_010` [P2] **Design Partner Feedback Review** → Thu 2026-04-16 15:00 - 16:00 (CST +08:00)
- ❌ `req_013` [P2] **Late Friday Launch Retro** → `time_conflict`: alice's 'Weekly Wrap-up' 16:30-17:30 overlaps the only 17:00-18:00 window.
- ✅ `req_015` [P2] **Ops Overflow Triage** → Tue 2026-04-14 17:30 - 18:00 (CST +08:00)
