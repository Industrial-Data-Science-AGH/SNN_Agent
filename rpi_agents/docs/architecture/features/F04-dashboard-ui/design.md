# F04 dashboard_ui

## Context

The pages the owner actually looks at — server-rendered (Jinja2) routes
inside the same FastAPI app as F01, not a separately hosted frontend
(ADR-0008). Reads only from F01's data-access functions in-process; never
talks to Storage from browser-side code (no storage keys ever reach the
browser). Reuses the overview diagram in `01-system-overview.md` for
component placement — this feature doesn't need its own flow diagram, it's a
straightforward fetch-and-render composition, not a multi-actor sequence.

## Contracts

- Consumes the same in-process data-access functions F01's `GET` routes use
  — no network hop, no CORS concern, since it's the same app/process
  (ADR-0008).
- Also exposes the same data as JSON via F01's `GET /api/events*` routes,
  for anyone who wants to script against it later — the HTML views and the
  JSON API share one implementation, not two.
- *(ADR-0016, revised 2026-07-14)* The summary/aggregate metrics are now
  also exposed as JSON via `GET /api/metrics` (F01 design) — a premium,
  chart-driven UI needs structured aggregate data, not just the rendered
  HTML band. `cloud/app/metrics.py` is the one aggregation implementation
  both the HTML band and the JSON route call — see "no separate aggregation
  endpoint... needed" below, which this supersedes.
- No other contracts of its own exposed to other features.

## Data model

Server-side view model derived from F01's data — no client-side persistent
state (no localStorage, per the project's general conventions; there's no
reason to cache home-security data in the browser anyway).

### Views

1. **Event list** — most recent first, default last 30 days (matches F01's
   `since` default). Each row: timestamp, thumbnail (from the SAS
   `image_url`), `alarm` badge (red/green), one-line `reason`.
2. **Summary metrics band** above the list, computed server-side. *(Revised
   2026-07-14, ADR-0016: originally "no separate aggregation endpoint or
   client-side JS needed, since it's already rendering HTML" — a premium
   chart-driven UI changed that; the same computation is now also reachable
   as JSON via `GET /api/metrics`, consumed by client-side charting.)*
   - **Real wakes**: count where `alarm == true`.
   - **False wakes**: count where `escalate == true and alarm == false`
     (prefilter or Gemini correctly suppressed a false trigger).
   - **Non-escalating wakes**: count where `escalate == false` (static
     scene, prefilter alone decided).
   - **Email delivery rate**: `count(email_sent == true) / count(alarm ==
     true)` — surfaces notifier failures distinctly from vision/prefilter
     behavior.
3. **Event detail** (click-through from the list): full snapshot image,
   Gemini's `reason` text verbatim, `vision_source` (`gemini` vs
   `failsafe` — visually distinguished, since a failsafe alarm means Gemini
   itself failed, which is operationally different from a confirmed
   intrusion), `email_sent` status, raw `score`/`motion`/`person` fields
   for anyone debugging prefilter behavior.

## Risks

- **"Real vs false" is only as good as Gemini's `is_intrusion` call** — the
  dashboard reports what the pipeline decided, not ground truth. Not a
  defect to fix here; noted so the metrics aren't over-trusted as an
  external validation of Gemini's accuracy (that's what
  `tests/test_vision_eval.py` / `tests/eval_harness.py` are for, and they're
  out of scope for this feature). *(ADR-0016, 2026-07-14: this is also why
  the new `vision_source_breakdown` chart in `GET /api/metrics` is
  documented as a model-agreement/operational breakdown, not a confusion
  matrix — a real confusion matrix needs a ground-truth label this system
  doesn't capture, and adding one was explicitly deferred by the owner.)*
- **Server-side aggregation cost grows with `since` window** — acceptable at
  hobby-project volume (hundreds to low thousands of events); if this ever
  needs a full-year view over tens of thousands of events, that's the signal
  to add caching or a precomputed aggregate (not built preemptively).
  *(ADR-0016: this now applies specifically to `GET /api/metrics`, which is
  intentionally uncapped unlike the 500-row `GET /api/events` list.)*

## Security

- Every page and every API route sits behind the same Basic Auth dependency
  (F05, ADR-0009) — no route in this app is reachable unauthenticated,
  dashboard pages included.
- Images loaded via short-lived SAS URLs the app mints per request — the
  browser never receives or stores a long-lived credential to Blob Storage;
  only the one Basic Auth credential (cached by the browser per the HTTP
  Basic Auth spec, not a custom session mechanism).

## Decisions

- ADR-0008 (one app serves API + dashboard UI, no separate frontend host).
- ADR-0009 (shared Basic Auth protects these pages too).
- ADR-0016 (`GET /api/metrics` server-side analytics endpoint for the
  premium chart-driven UI refresh).

## Branch

`feat/dashboard` (task T02)
