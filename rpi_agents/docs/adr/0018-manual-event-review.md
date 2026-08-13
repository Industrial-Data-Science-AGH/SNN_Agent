# ADR-0018: Manual ground-truth review (`PATCH /api/events/{event_id}`)

- Status: accepted
- Date: 2026-07-15
- Deciders: Wiktor
- Relates to: F01_ingest_api, F04_dashboard_ui

## Context

ADR-0016 explicitly deferred real accuracy tracking: `vision_source_breakdown`
and `trigger_breakdown` are operational/model-agreement cross-tabs, not
confusion matrices, because this system never captured ground truth — only
what the pipeline itself decided. ADR-0016's "Real confusion matrix via
manual ground-truth labeling" alternative was rejected at the time as
larger-scope than "add charts to the existing dashboard," with a note to
"revisit as a follow-up ADR if real accuracy tracking is wanted later."

The owner now wants exactly that follow-up: after seeing an event on the
dashboard (photo + Gemini's `window_broken`/`is_intrusion` verdict), they
want to confirm what actually happened — was the window really broken, and
was an intruder/person actually present. That confirmation is the ground
truth this system has never had, and it enables a real accuracy measurement
of Gemini's two judgments (ADR-0017 made these independent) against reality.

## Decision

Add `PATCH /api/events/{event_id}` to F01's route set (same Basic Auth, same
Container App, no new deployable unit — ADR-0007, ADR-0008, ADR-0009
unchanged). Request body (`schemas.EventReview`):

```json
{"window_broken_confirmed": true, "intrusion_confirmed": false}
```

Both fields are required — a review confirms both judgments together in one
pass (the owner looks at one photo and decides both at once), not a partial
update. `reviewed_at` is server-stamped (`time.time()`), never trusted from
the request, same pattern as `received_at` on ingest. A second PATCH of the
same `event_id` overwrites the prior review outright — the owner correcting
an earlier mistake is a normal, supported action, not an error.

Storage: `storage.review_event()` looks up the entity by RowKey (event_id) —
sharing the new `_find_entity_by_row_key()` helper with a refactored
`get_event()`, since a PATCH URL carries only `event_id`, not the
PartitionKey Table Storage needs to address an update — then
`update_entity(mode=UpdateMode.MERGE)` writes `window_broken_confirmed`,
`intrusion_confirmed`, `reviewed_at` onto the existing entity. Returns 404
(via the same `_MSG_EVENT_NOT_FOUND` used by `GET /api/events/{event_id}`)
if the event doesn't exist.

`cloud/app/metrics.py` gains `review_accuracy()`: TP/FP/TN/FN + accuracy for
both `window_broken` (vs `window_broken_confirmed`) and `is_intrusion` (vs
`intrusion_confirmed`), scoped to events with a real Gemini verdict
(`vision_source == "gemini"` — a failsafe verdict has no real prediction to
score) **and** a completed review (`reviewed_at is not None`). Surfaced as a
new top-level `review_accuracy` field on `GET /api/metrics`
(`{"reviewed_count": int, "window_broken": {...}, "intrusion": {...}}`),
alongside (not replacing) `vision_source_breakdown`/`trigger_breakdown` —
those remain useful for their own operational purpose even though this ADR
finally adds the real thing ADR-0016 said was out of scope.

## Alternatives Considered

### A separate review table/entity, not a PATCH on the existing event
- **Pros:** keeps the immutable ingest record untouched; a review becomes an
  audit trail (who reviewed what, when, and whether they changed their mind)
  rather than an overwrite.
- **Cons:** a second Table, a join at read time (`review_accuracy()` would
  need to fetch both tables and match by `event_id`), meaningfully more
  storage-layer complexity for a hobby-scale, single-owner dashboard where
  "the owner corrected their own earlier review" doesn't need a history.
- **Why not:** no requirement for a review audit trail; MERGE onto the
  existing entity is the same pattern `set_blob_name()` already uses to
  patch a field onto an event after ingest — no new access pattern needed.

### Partial review (allow submitting just one of the two fields)
- **Pros:** more flexible if the owner only wants to confirm one judgment.
- **Cons:** `review_accuracy()` would need to handle "reviewed for
  window_broken but not intrusion" as a distinct state per event, doubling
  the bookkeeping for a case the owner didn't ask for — they described
  reviewing "whether the window was really broken... and whether the
  intruder or person is in place" as one action.
- **Why not:** the owner's request describes a single combined review;
  `EventReview` (`extra="forbid"`, both fields required) matches that
  exactly and is simpler.

### Let the review also correct/overwrite the Gemini verdict itself
- **Pros:** would make `is_intrusion`/`window_broken` "self-healing" from
  owner feedback.
- **Cons:** conflates "what Gemini said" with "what actually happened" —
  the whole point of `review_accuracy()` is to compare the two, which
  requires keeping them as separate fields. Overwriting the original verdict
  would destroy the very data the accuracy metric needs.
- **Why not:** out of scope, and would break the feature's own premise.

## Consequences

### Positive
- Closes the gap ADR-0016 explicitly left open: `review_accuracy` is a real
  confusion-matrix accuracy measurement, not a proxy.
- No new Table/Storage account — one more field set (MERGE) on the existing
  `events` entity, same pattern already used for `blob_name`.
- `review_accuracy.reviewed_count` makes it obvious when the metric has no
  data yet (0, all-zero confusion blocks) rather than a misleading 0%/100%.

### Negative
- The `events` entity now has a "half-owned" lifecycle: fields written by
  the Pi at ingest (`agent/machine.py`'s record), fields written by the
  cloud (`blob_name`, `received_at`), and now fields written by the owner's
  browser (`*_confirmed`, `reviewed_at`) — three writers on one entity.
  Acceptable because each writer owns disjoint fields (no write conflicts
  possible) and MERGE mode already made this pattern safe for `blob_name`.
- `_find_entity_by_row_key()` is a filtered scan (no PartitionKey known from
  `event_id` alone), same accepted limitation `get_event()` already
  documented — unchanged by this ADR, just reused by a second caller.

## Risks (with mitigation)

- **Risk:** the owner's review is itself just an opinion, not infallible
  ground truth (they weren't physically present either, in the fully
  general case). **Mitigation:** none needed — this is the best available
  ground-truth signal for a single-owner hobby system, and the metric is
  explicitly framed as accuracy against the *owner's confirmed* judgment,
  not an independently verified objective truth.
- **Risk:** `review_accuracy` could look artificially perfect or terrible
  with a very small `reviewed_count` (e.g. 1-2 reviews) before enough data
  accumulates. **Mitigation:** `reviewed_count` is returned alongside the
  accuracy numbers specifically so the UI (and the owner reading it) can
  see the sample size, not just a bare percentage.

## Decisions

- ADR-0016 (`GET /api/metrics`) — `review_accuracy` lands in the same
  endpoint and directly fulfills the "revisit as a follow-up ADR" note left
  in that ADR's Alternatives Considered.
- ADR-0017 (`window_broken`) — `review_accuracy` scores exactly the two
  independent judgments that ADR introduced.
- ADR-0009 (same shared Basic Auth credential protects this route too).
