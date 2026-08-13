# T06 — Manual ground-truth review (`PATCH /api/events/{event_id}`)

*(New task, added 2026-07-15 — ADR-0018. Closes the gap ADR-0016's
Alternatives Considered explicitly deferred: "revisit as a follow-up ADR if
real accuracy tracking is wanted later." Backend-only, same division of
labor as T05 — the review UI itself is a separate, frontend-focused effort
briefed by `docs/handoff/dashboard-ui-premium-refresh.md`.)*

- **Branch:** `feat/dashboard`
- **Feature ID:** F01 (ingest_api), touches F04 (dashboard_ui)'s contracts
- **Depends on:** T02 (retrofits its already-shipped `cloud/app/`), T05
  (extends the same `GET /api/metrics` response this task adds a field to)
- **Blocks:** nothing new — every other route is unchanged
- **Source:** `docs/architecture/delivery-plan.json` (T06),
  `docs/architecture/features/F01-ingest-api/design.md`,
  `docs/architecture/features/F04-dashboard-ui/design.md`, ADR-0018

## Goal

Let the owner confirm, per event, whether the window was really broken and
whether an intruder/person was actually present — the ground truth this
system has never captured. Once that ground truth exists, `review_accuracy`
on `GET /api/metrics` becomes a genuine confusion matrix (TP/FP/TN/FN +
accuracy) scoring Gemini's `window_broken`/`is_intrusion` judgments against
it, rather than the operational/agreement breakdowns (`vision_source_
breakdown`, `trigger_breakdown`) this system has shipped so far.

## Current state (brownfield — exact touch points)

- **`cloud/app/storage.py::get_event()`** — already does a filtered Table
  scan by RowKey (`event_id`) since no PartitionKey is available from the
  URL alone. **Change:** the scan logic is extracted into a shared
  `_find_entity_by_row_key()` helper, so the new `review_event()` can reuse
  it instead of duplicating the query.
- **`cloud/app/storage.py::set_blob_name()`** — the existing precedent for
  patching a field onto an already-written entity via
  `update_entity(mode=MERGE)`. **New, same pattern:** `review_event()` does
  the same MERGE, with three fields instead of one.
- **`cloud/app/schemas.py::EventSummary`/`EventDetail`** — currently has no
  review-related fields. **Change:** gains `window_broken_confirmed: bool |
  None`, `intrusion_confirmed: bool | None`, `reviewed_at: float | None`
  (all `None` until reviewed).
- **`cloud/app/schemas.py::MetricsResponse`** — currently ends with
  `latency_s`. **Change:** gains `review_accuracy: ReviewAccuracy` (new
  `ConfusionCounts`/`ReviewAccuracy` models).
- **`cloud/app/metrics.py`** — currently has no accuracy-measuring function,
  only operational breakdowns. **New:** `_confusion()` (private TP/FP/TN/FN
  + accuracy helper) and `review_accuracy()` (public, calls `_confusion()`
  twice — once per judgment).
- **`cloud/app/routes_api.py::router`** (`prefix="/api/events"`) — has
  `POST`, `GET` (list), `GET /{event_id}` today. **New:** `PATCH
  /{event_id}`.

## Files to change

```
cloud/app/schemas.py       # EventReview, ConfusionCounts, ReviewAccuracy;
                            # EventSummary/EventDetail + MetricsResponse fields
cloud/app/storage.py       # _find_entity_by_row_key(), review_event();
                            # get_event() refactored to share the helper
cloud/app/metrics.py       # _confusion(), review_accuracy()
cloud/app/routes_api.py    # PATCH /api/events/{event_id}; get_metrics()
                            # wires review_accuracy(events) into the response
```

## Contract (verbatim from F01 design)

`PATCH /api/events/{event_id}` (Basic Auth, same as every route).

Request:
```json
{"window_broken_confirmed": true, "intrusion_confirmed": false}
```

Response `200` — the updated `EventDetail`, now including
`window_broken_confirmed`, `intrusion_confirmed`, `reviewed_at`. `404` if
`event_id` doesn't match any event. `422` if either field is missing or the
wrong type, or an unknown field is present (`extra="forbid"`).

`GET /api/metrics` response gains:
```json
"review_accuracy": {
  "reviewed_count": 18,
  "window_broken": {"tp": 10, "fp": 1, "tn": 6, "fn": 1, "accuracy": 0.889},
  "intrusion": {"tp": 8, "fp": 2, "tn": 7, "fn": 1, "accuracy": 0.833}
}
```

## Step-by-step

1. `cloud/app/schemas.py`:
   - Add `EventReview` (both fields required, `extra="forbid"`).
   - Add `window_broken_confirmed`, `intrusion_confirmed`, `reviewed_at`
     (all `| None`) to `EventSummary` (inherited by `EventDetail`).
   - Add `ConfusionCounts` (`tp`/`fp`/`tn`/`fn`/`accuracy`) and
     `ReviewAccuracy` (`reviewed_count` + the two `ConfusionCounts` blocks).
   - Add `review_accuracy: ReviewAccuracy` to `MetricsResponse`.
2. `cloud/app/storage.py`:
   - Extract `_find_entity_by_row_key(event_id) -> dict | None` from
     `get_event()`'s existing query logic; `get_event()` calls it then
     converts via `_to_summary_dict()`.
   - Add `review_event(event_id, *, window_broken_confirmed, intrusion_
     confirmed, reviewed_at) -> dict | None`: look up the entity via the
     shared helper (return `None` if not found), then `update_entity(
     mode=UpdateMode.MERGE)` with the three new fields plus the existing
     `PartitionKey`/`RowKey`, then return `get_event(event_id)`.
   - Add the three new keys to `_to_summary_dict()`
     (`entity.get("window_broken_confirmed")`, etc. — `None` if never
     reviewed).
3. `cloud/app/metrics.py`:
   - `_confusion(events, *, predicted_key, confirmed_key) -> dict` —
     iterate, classify each event as TP/FP/TN/FN by comparing
     `bool(event.get(predicted_key))` against
     `bool(event.get(confirmed_key))`; `accuracy = (tp+tn)/total` (`0.0` if
     `total == 0`).
   - `review_accuracy(events) -> dict` — filter to events with
     `vision_source == "gemini"` and `reviewed_at is not None`; call
     `_confusion()` twice (`window_broken`/`window_broken_confirmed`,
     `is_intrusion`/`intrusion_confirmed`); return `{"reviewed_count":
     len(reviewed), "window_broken": ..., "intrusion": ...}`.
4. `cloud/app/routes_api.py`:
   - Add `PATCH /{event_id}` on the existing `router`, calling
     `storage.review_event(event_id, window_broken_confirmed=payload.
     window_broken_confirmed, intrusion_confirmed=payload.intrusion_
     confirmed, reviewed_at=time.time())`; raise `404` (reusing
     `_MSG_EVENT_NOT_FOUND`) if it returns `None`; otherwise return
     `schemas.EventDetail(**event)`.
   - In `get_metrics()`, add `review_accuracy=metrics.review_accuracy(
     events)` to the `schemas.MetricsResponse(...)` construction (Pydantic
     coerces the plain dict into `ReviewAccuracy` automatically).
5. Tests:
   - `tests/test_review.py` (new) or additions to `tests/test_ingest.py`:
     `storage.review_event()` found/not-found; PATCH route 200 (fields
     reflected in the response), 404 (unknown `event_id`), 422 (missing
     field / wrong type / extra field), and a second PATCH overwriting the
     first (re-review).
   - `tests/test_auth.py`: add `("PATCH", "/api/events/some-id")` to both
     `test_every_route_class_rejects_missing_auth` and
     `test_every_route_class_rejects_bad_auth` parametrized lists.
   - `tests/test_metrics.py`: `_confusion()` (empty list, a mix covering all
     four TP/FP/TN/FN cells), `review_accuracy()` (excludes `failsafe`
     verdicts, excludes events with `reviewed_at is None`, correct
     `reviewed_count`); update the route-shape test to expect
     `review_accuracy` in `GET /api/metrics`'s JSON.
6. `ruff check`/`format`; re-run the full suite.

## Acceptance gate (from `delivery-plan.json`)

- Unit tests green, including new review/PATCH tests and the updated
  `test_auth.py`/`test_metrics.py` cases.
- `PATCH /api/events/{event_id}` requires auth like every other route (401
  without credentials).
- `PATCH` on an unknown `event_id` returns 404, not a 500 or a silent no-op.
- `GET /api/metrics`'s `review_accuracy` is all-zero/`reviewed_count: 0`
  on a window with no reviews (no division-by-zero, no exception) and
  correctly reflects TP/FP/TN/FN once reviews exist.

## Notes / risks to carry into implementation

- **This task does not build the review UI itself.** Same division of labor
  as T05 — a separate Claude Code prompt (round 3,
  `docs/handoff/claude-code-review-ui-prompt.txt`) briefs the frontend
  effort once this backend contract ships.
- **`review_accuracy` only scores real Gemini verdicts.** A `failsafe`
  verdict (`vision_source == "failsafe"`) has no real prediction — Gemini
  never actually judged the frame, so it's excluded from both the
  denominator and the confusion counts, same reasoning already applied to
  `window_break_confirmation_rate` (ADR-0017).
- **A review requires both fields, always.** No partial-review state to
  handle — `EventReview`'s `extra="forbid"` + both-required schema makes a
  half-submitted review a `422`, not a new data state `review_accuracy`
  would need to reason about.
- **Re-review is overwrite, not append.** No review history/audit trail —
  deliberately out of scope (see ADR-0018, Alternatives Considered).
