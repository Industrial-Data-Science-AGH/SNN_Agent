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
