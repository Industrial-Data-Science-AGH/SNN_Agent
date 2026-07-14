# T05 — Dashboard analytics API (`GET /api/metrics`)

*(New task, added 2026-07-14 — ADR-0016. Backend half of a premium,
chart-driven dashboard UI refresh; the frontend half is handed to a
separate Claude Code effort, briefed by
`docs/handoff/dashboard-ui-premium-refresh.md`.)*

- **Branch:** `feat/dashboard`
- **Feature ID:** F01 (ingest_api), touches F04 (dashboard_ui)'s contracts
- **Depends on:** T02 (retrofits its already-shipped `cloud/app/`), T04
  (event schema/upsert semantics this reads from are already in place)
- **Blocks:** nothing new — `GET /api/events`/`GET /api/events/{id}` are
  unchanged; the frontend refresh consumes this task's output but that work
  happens outside this delivery plan
- **Source:** `docs/architecture/delivery-plan.json` (T05),
  `docs/architecture/features/F01-ingest-api/design.md`,
  `docs/architecture/features/F04-dashboard-ui/design.md`, ADR-0016

## Goal

Give the (separately-built) premium dashboard UI one clean JSON endpoint to
drive charts from, instead of making the frontend either recompute
aggregates client-side or reverse-engineer them from a paginated event list.
Also ship the "confusion-matrix-style" chart the owner asked for — as an
outcome × `vision_source` operational breakdown, since a true ground-truth
confusion matrix isn't computable today (see ADR-0016).

## Current state (brownfield — exact touch points)

- **`cloud/app/routes_dashboard.py::compute_metrics()`** — today the only
  aggregation logic in the app; computes real/false/non-escalating counts +
  email delivery rate, inline, over an already-fetched event list. **Change:**
  logic moves to `cloud/app/metrics.py::summary_metrics()`;
  `compute_metrics()` becomes a one-line delegate so
  `tests/test_dashboard.py`'s existing import (`from
  cloud.app.routes_dashboard import compute_metrics`) keeps working
  unchanged.
- **`cloud/app/storage.py::list_events()`** — capped at `_MAX_LIMIT = 500`
  (a deliberate display-page cap, F01 design). **New, not changed:**
  `list_events_for_metrics(since)` — same partition-key-range query, no cap,
  since aggregation must reflect the whole window.
- **`cloud/app/routes_api.py`** — currently one `APIRouter` at
  `prefix="/api/events"`. **New:** a second `APIRouter` at `prefix="/api"`
  in the same file, with one route, `GET /metrics`. Both routers are
  exported and included in `main.py`.

## Files to change

```
cloud/app/metrics.py       # new: summary_metrics, daily_rollup,
                            # vision_source_breakdown, latency_stats
cloud/app/storage.py         # new: list_events_for_metrics(since)
cloud/app/routes_api.py       # new: metrics_router, GET /api/metrics
cloud/app/routes_dashboard.py  # compute_metrics() delegates to metrics.py
cloud/app/main.py               # include metrics_router
```

## Contract (verbatim from F01 design)

`GET /api/metrics?since=` (Basic Auth, same as every route). No `limit` —
aggregation always covers the full matching window.

```json
{
  "since": "2026-06-14",
  "until": "2026-07-14",
  "summary": {
    "real_wakes": 12,
    "false_wakes": 34,
    "non_escalating_wakes": 210,
    "email_delivery_rate": 0.92
  },
  "daily": [
    {"date": "2026-07-01", "real_wakes": 1, "false_wakes": 3, "non_escalating_wakes": 20, "total": 24}
  ],
  "vision_source_breakdown": {
    "real_wakes": {"gemini": 10, "failsafe": 2, "none": 0},
    "false_wakes": {"gemini": 30, "failsafe": 0, "none": 4},
    "non_escalating_wakes": {"gemini": 0, "failsafe": 0, "none": 210}
  },
  "latency_s": {"avg": 8.2, "p50": 7.9, "p95": 14.1, "max": 22.0}
}
```

`daily` is sorted oldest-first (charts read left-to-right chronologically —
deliberately the opposite order from `GET /api/events`, which stays
newest-first for the scannable list view).

## Step-by-step

1. `cloud/app/metrics.py`: write the four pure functions over a
   `list[dict]` of already-fetched events (same dict shape
   `storage._to_summary_dict()` produces). No FastAPI/Storage imports here —
   keep it a plain, independently-testable module, same pattern as
   `routes_dashboard.py::compute_metrics()` today.
   - `summary_metrics(events) -> dict` — copy `compute_metrics()`'s current
     body verbatim.
   - `daily_rollup(events) -> list[dict]` — bucket by
     `datetime.fromtimestamp(ts_wall, tz=UTC).strftime("%Y-%m-%d")`, one
     dict per date with `real_wakes`/`false_wakes`/`non_escalating_wakes`/
     `total`, sorted ascending by date.
   - `vision_source_breakdown(events) -> dict` — for each of the three
     outcome buckets, count `vision_source` values, normalizing `None` to
     the string `"none"` (JSON has no `null`-as-key).
   - `latency_stats(events) -> dict` — avg/p50/p95/max over `latency_s`;
     implement percentile with a plain sorted-list + linear interpolation
     (no new numpy dependency in `cloud/app`); return all-zero dict on an
     empty list rather than dividing by zero.
2. `cloud/app/storage.py`: add `list_events_for_metrics(since: str | None =
   None) -> list[dict]` — same query/date-default logic as `list_events()`
   minus the `[:limit]` slice and the `limit` parameter entirely.
3. `cloud/app/routes_api.py`: add `metrics_router = APIRouter(prefix="/api",
   tags=["metrics"])` and `GET /metrics` on it, calling
   `storage.list_events_for_metrics()` then the four `metrics.py` functions;
   assemble the response dict per the contract above (`since`/`until` echo
   the effective window).
4. `cloud/app/routes_dashboard.py`: replace `compute_metrics()`'s body with
   `return metrics.summary_metrics(events)`.
5. `cloud/app/main.py`: import and `include_router(metrics_router)`
   alongside the existing `api_router`.
6. `tests/test_metrics.py` (new): unit tests for all four `metrics.py`
   functions (empty list, single event, multi-day rollup ordering, mixed
   vision sources, percentile edge cases) plus route-level tests
   (auth-rejected without credentials, correct JSON shape, empty-window
   response doesn't error).
7. `ruff check`/`format`; re-run the full suite — confirm
   `tests/test_dashboard.py`'s existing `compute_metrics` tests still pass
   unchanged (delegation must be behavior-preserving).

## Acceptance gate (from `delivery-plan.json`)

- Unit tests green, including new `test_metrics.py` cases and the unchanged
  `test_dashboard.py` cases (delegation didn't change behavior).
- `GET /api/metrics` requires auth like every other route (401 without
  credentials).
- Response shape matches the contract above, including on an empty/no-events
  window (no division-by-zero, no exception).

## Notes / risks to carry into implementation

- **This task does not build the premium UI itself.** That's a separate,
  frontend-only effort briefed by
  `docs/handoff/dashboard-ui-premium-refresh.md` — this task only ships the
  data contract that effort consumes. Do not touch
  `cloud/app/templates/*.html` styling as part of this task; `routes_
  dashboard.py`'s minimal delegation change is the only template-adjacent
  edit.
- **`vision_source_breakdown` is not a ground-truth confusion matrix** —
  see ADR-0016 and F04 design's Risks. Don't relabel it as one anywhere in
  code, comments, or the frontend context file.
- **Uncapped aggregation is an accepted, not ignored, scaling risk** — see
  ADR-0016 Consequences. Don't add caching/precomputation preemptively;
  that's explicitly deferred until it's actually slow.
