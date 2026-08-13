# ADR-0016: Server-side `GET /api/metrics` analytics endpoint

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F04_dashboard_ui
- Supersedes: F04 design's "no separate aggregation endpoint or client-side
  JS needed" decision, and its "server-side aggregation cost grows with
  `since` window" Risk framing (both amended in `F04-dashboard-ui/design.md`
  alongside this ADR).

## Context

F04's original design computed the summary metrics band inline, in the same
request that renders the event-list HTML page — a pure function
(`compute_metrics()`) over the already-fetched event list, explicitly
reasoned as "no separate aggregation endpoint... needed" at hobby-project
volume.

The owner now wants a genuinely premium dashboard UI (charts, trend lines,
a breakdown of how outcomes relate to which vision path produced them) built
by a separate frontend-focused effort. That UI needs structured JSON to
drive a charting library — the existing HTML-only `compute_metrics()` output
isn't consumable outside a Jinja template, and the raw `GET /api/events`
list is capped at 500 rows (`storage._MAX_LIMIT`), which is the wrong shape
and the wrong cap for a rollup that should reflect the whole selected
window, not a page of individual rows.

Separately: the owner asked about a "confusion matrix" for the dashboard. A
true confusion matrix (TP/FP/TN/FN) requires a ground-truth label — was this
event *actually* an intrusion? — which this system does not capture; F04's
own Risks section already flags this ("dashboard reports what the pipeline
decided, not ground truth"). The owner confirmed (2026-07-14) that adding a
manual ground-truth-labeling workflow is out of scope for now; the chart
instead cross-tabulates decision outcome (real / false / non-escalating
wake) against `vision_source` (`gemini` / `failsafe` / none) — a **model
agreement / operational breakdown**, not an accuracy measurement. This
matters operationally regardless of ground truth: a high `failsafe` share
means Gemini itself is failing open often, which is worth seeing on its own.

## Decision

Add `GET /api/metrics?since=` to F01's route set (same Basic Auth, same
Container App, no new deployable unit — ADR-0007, ADR-0008, ADR-0009
unchanged). Aggregation logic moves into a new shared module,
`cloud/app/metrics.py`, with pure functions over an event-dict list:

- `summary_metrics()` — the existing real/false/non-escalating counts +
  email delivery rate (unchanged values, moved from `routes_dashboard.py`,
  which now delegates to this module so there is one implementation, not
  two).
- `daily_rollup()` — per-UTC-date counts (real/false/non-escalating/total),
  oldest first, for a trend chart.
- `vision_source_breakdown()` — the outcome × `vision_source` cross-tab
  described above (the "confusion-matrix-style" chart).
- `latency_stats()` — avg/p50/p95/max of `latency_s` across the window.

A new `storage.list_events_for_metrics(since)` backs the route: same
partition-key-range query as `list_events()`, but without the 500-row
display cap — aggregation must reflect the whole window, not a page.

## Alternatives Considered

### Compute everything client-side from `GET /api/events`
- **Pros:** no new backend route; the event list already returned to the
  page is "enough" at today's volume.
- **Cons:** duplicates `compute_metrics()`'s logic in JavaScript; breaks
  past 500 events (the existing display cap) since the client never sees
  events beyond that page; recomputes the same aggregation on every page
  load instead of once, server-side, where the data already lives.
- **Why not:** the owner explicitly asked for backend-driven metrics
  (2026-07-14) specifically so the frontend effort has one clean JSON
  contract to build charts against, not a second aggregation
  implementation to maintain.

### Real confusion matrix via manual ground-truth labeling
- **Pros:** would be an actual accuracy measurement, not a proxy.
- **Cons:** new schema field, new `PATCH /api/events/{id}` write path, and
  a labeling UI/workflow — meaningfully larger scope than "add charts to
  the existing dashboard."
- **Why not:** explicitly deferred by the owner (2026-07-14); the
  `vision_source_breakdown` chart ships the useful part (operational
  visibility into failsafe reliance) without the new write path. Revisit as
  a follow-up ADR if real accuracy tracking is wanted later.

## Consequences

### Positive
- One aggregation implementation (`cloud/app/metrics.py`), consumed by both
  the HTML summary band and the new JSON route — no drift between the two.
- The frontend effort gets a single, documented JSON contract instead of
  needing to re-derive aggregates from a paginated event list.
- `vision_source_breakdown` surfaces failsafe-reliance rate, which was
  previously only visible by reading raw event rows one at a time.

### Negative
- `list_events_for_metrics()` is uncapped, so the aggregation cost genuinely
  does grow with the `since` window and total event count — F04's original
  Risk note ("acceptable at hobby-project volume... that's the signal to
  add caching") now applies to this route specifically. Not mitigated here;
  accepted at today's volume (hundreds to low thousands of events/month).
- A second read path (`list_events_for_metrics` vs `list_events`) against
  the same Table — acceptable duplication, since the display route's 500-row
  cap is a deliberate, different contract (F01 design), not an oversight to
  reconcile.

## Risks (with mitigation)

- **Risk:** as event volume grows, `GET /api/metrics` (uncapped scan) gets
  slower than `GET /api/events` (capped at 500). **Mitigation:** none
  built preemptively (matches this project's established "simplest thing
  that works at this volume" pattern — ADR-0007, ADR-0009, ADR-0011); the
  signal to add caching or a precomputed daily aggregate is this route
  measurably slowing down, not anticipated now.
- **Risk:** `vision_source_breakdown` could be misread as a real accuracy
  matrix by someone not familiar with F04's ground-truth caveat.
  **Mitigation:** the frontend context handed to the UI implementation
  explicitly labels this chart as an operational/agreement breakdown, not
  an accuracy score (see `docs/handoff/dashboard-ui-premium-refresh.md`).

## Decisions

- ADR-0007, ADR-0008 (single Container App, same deployable unit).
- ADR-0009 (same shared Basic Auth credential protects this route too).
- F04 design's original "no aggregation endpoint needed" note — superseded
  by this ADR, not by a Tenet-level PR/FAQ change (this is a feature-scope
  extension, not a reversal of a Tenet).

## Addendum (2026-07-15): `trigger_breakdown`

After reviewing the first rendered dashboard, the owner asked whether
"non-escalating wakes" makes sense given the SNN's whole purpose is to
detect a candidate glass-break event — doesn't every wake mean something
was already detected?

Answer: `escalate` and `woken_by_trigger` are two independent signals.
`woken_by_trigger` (`agent/trigger.py`) records whether this specific boot
was actually caused by the Arduino latching a real SNN trigger, as opposed
to any other reason the Pi came up (manual/dev boot) — the deployed system
is designed so every wake *is* a trigger (README: "wakes on an SNN hardware
trigger"). `escalate` is a separate, later judgment: after capturing camera
frames (regardless of why the Pi woke), does the local motion/person
prefilter think the frames show something worth an expensive Gemini call?
A genuine SNN trigger can still land in "non-escalating" — the SNN is a
cheap, sensitive first-stage sensor (comparable to a smoke detector),
expected to latch on non-intrusion events (wind, ambient noise with a
similar spike signature); that's exactly why the prefilter and vision
stages exist downstream. A funnel shape (many SNN triggers, few
prefilter-escalations, fewer still confirmed alarms) is the system working
as intended, not a sign anything is broken.

`GET /api/metrics` gained a fourth field, `trigger_breakdown` (outcome x
`woken_by_trigger`, same shape as `vision_source_breakdown`), so the
dashboard can show this directly rather than leave it implicit — and so a
nonzero `not_triggered` count (which shouldn't normally happen) is visible
as the anomaly signal it actually is. `cloud/app/metrics.py::
trigger_breakdown()`, `cloud/app/schemas.py::TriggerBreakdown`/
`TriggerOutcomeCounts`, `MetricsResponse.trigger_breakdown` — same file set
this ADR already covers, no new module. `docs/handoff/
dashboard-ui-premium-refresh.md` was updated with this field so a follow-up
UI pass can surface it.

## Addendum (2026-07-15, same review pass): `gemini_call_success_rate`

Same session, a second request: the owner wants a single "how often does
the Gemini call itself actually succeed" number — e.g. "if I send 10
requests to Gemini and 8 succeed, show 80%." This is a call-reliability
metric, distinct from `vision_source_breakdown`'s per-outcome cross-tab:
`vision_source == "failsafe"` already means the `vision.verdict()` call
raised/timed out (`agent/machine.py`'s except block), so the data needed
was already present — this is a new aggregation over an existing field, not
new data collection on the Pi.

Added as `summary.gemini_call_success_rate` (same "rate over a subset,
0.0 when the subset is empty" shape as `email_delivery_rate`, same
`summary_metrics()` function so it renders as a KPI card automatically,
consistent with the other three): `successes / attempts`, where `attempts`
counts every event with `vision_source` in `("gemini", "failsafe")` and
`successes` counts `vision_source == "gemini"`. Non-escalating events
(`vision_source` is `None`, no vision call was ever attempted) are
excluded from both — including them in the denominator would understate
the rate for a reason that has nothing to do with Gemini's reliability.

## Addendum (2026-07-15, same review pass): `last_sync`

Third request from the same pass: the owner wants to see, at a glance, when
the dashboard last actually received data from the Pi — "so the user will
see the data and when it was sent from the Raspberry Pi wake up."

Added `last_sync` as a top-level field on `GET /api/metrics`'s response
(sibling to `summary`/`daily`/etc.): the most recent event in the queried
window, by `received_at` (cloud-side receipt time, not `ts_wall`'s Pi-side
wake time — the two can differ by however long the push/queue took, and
"last sync" should answer "when did the cloud last hear from the Pi," not
"when did the Pi last wake"). `null` when the window has no events.

Considered and rejected: a second, unbounded query across the whole table
(to answer "when did the Pi last sync, ever," regardless of the `since`
filter). Rejected because it would need a new Table access pattern
(`storage.py` has no "give me the single most recent row across all
partitions" query today, and Table Storage has no native support for
this without either a full scan or a second index), and because scoping
`last_sync` to the same window as everything else in the response is
actually more informative, not less: if the Pi hasn't synced within the
selected window at all, `last_sync: null` **is** the signal — the owner
widening `since` to look further back is the same action they'd already
take to investigate a gap in the `daily` trend chart. Consistent with how
every other field on this endpoint is already window-scoped; no special
case introduced.
