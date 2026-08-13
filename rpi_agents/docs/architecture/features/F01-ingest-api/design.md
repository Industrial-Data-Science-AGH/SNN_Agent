# F01 ingest_api

## Context

The FastAPI routes, inside the single Container App (ADR-0007, ADR-0008),
that both receive pushed events from the Pi and serve them to the dashboard.
Two concerns, one deployable unit and one process — cheaper and simpler than
splitting them, and every route (ingest and read alike) shares one Basic
Auth check (F05, ADR-0009).

![F01 ingest flow](../../diagrams/F01-ingest-flow.svg)

*Source: `docs/diagrams/F01-ingest-flow.dot` — edit and re-run `render_diagrams.sh`.*

## Current state *(brownfield — what this touches)*

T02 shipped this feature with server-generated `event_id`. **Change**
(ADR-0014, driven by F03's new local retry queue, ADR-0015): `event_id`
becomes a required, client-supplied field, and the ingest write becomes an
idempotent upsert instead of a strict create. Everything else in this
design — auth, the read routes, the Blob/Table split, the 2 MB image cap —
is **unchanged**.

**Change (2026-07-15, ADR-0018):** this feature gains its first
dashboard-initiated write. Every prior write (`POST /api/events`,
`set_blob_name()`) originated from the Pi; `PATCH /api/events/{event_id}`
originates from the owner's browser, submitting a manual ground-truth
review. Same auth, same entity, same MERGE-update mechanism `set_blob_name`
already established — not a new access pattern, just a new caller.

## Contracts

### `POST /api/events` (ingest — called by the Pi, F03)

- **Auth:** HTTP Basic Auth, same shared credential as every other route
  (ADR-0009) — checked by shared middleware, not per-route logic.
- **Body (JSON):**
  ```json
  {
    "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "ts_wall": 1784048796.83,
    "woken_by_trigger": false,
    "escalate": true,
    "motion": true,
    "person": false,
    "score": 0.1547,
    "vision_source": "gemini",
    "is_intrusion": false,
    "window_broken": false,
    "alarm": false,
    "reason": "vision: ...",
    "email_sent": false,
    "latency_s": 10.67,
    "image_jpeg_b64": "<base64, optional>"
  }
  ```
  `event_id` (ADR-0014, **new**, required) — a ULID the Pi generates once,
  at decision time, and reuses on every retry of this same event (immediate
  retry or a later retry from `sync_queue.jsonl`). `window_broken`
  (2026-07-15, ADR-0017, **new**) — a second, independent judgment from the
  same Gemini call as `is_intrusion`: does the frame show visible evidence
  the window/glass itself is broken, distinct from whether an intruder is
  present. `null` for non-escalating events, same nullability pattern as
  `is_intrusion`/`vision_source` (no vision call was ever made). Every
  other field is unchanged from T02.
- **Validation:** required fields present and correctly typed (a Pydantic
  model, matching FastAPI's native validation); `event_id` must be a
  well-formed ULID (26-char Crockford base32) — reject anything else with
  `422` before it can become a Table `RowKey`; `image_jpeg_b64` optional
  but if present, decoded size capped (reject > 2 MB — a 10-frame 320x180
  JPEG snapshot is a few hundred KB at most, 2 MB gives headroom without
  allowing abuse); reject unknown extra fields (schema is closed, Pydantic
  `model_config = {"extra": "forbid"}`).
- **Response:** `202 Accepted`, `{"event_id": "<ulid>"}` (echoes the
  request's `event_id`) on success; `422` on validation failure
  (FastAPI/Pydantic default); `401` on missing/bad Basic Auth credential.
- **Behavior:** `received_at` is still assigned server-side (ADR-0001,
  Risks: ordering tie-breaker). Writes the Table entity as an **upsert**
  keyed on the client-supplied `event_id` (ADR-0014) — a retry of the same
  `event_id` overwrites with identical data rather than erroring or
  duplicating — then writes the blob (if present), in that order; if the
  blob write fails after the table write succeeds, the entity's `blob_name`
  is left empty rather than the whole request failing (a metadata-only
  event is still useful; a fully-failed ingest is not). A blob write is
  itself idempotent for the same `event_id` (same blob name, overwrite),
  so a retried event with an image is also safe to re-send in full.

### `GET /api/events?since=&limit=` (dashboard list — F04)

- **Auth:** same Basic Auth middleware as every route — no separate
  delegation layer needed now that there's no Static Web Apps proxy in
  front of it (unlike the earlier SWA-delegated design).
- **Query params:** `since` (ISO date, optional, defaults to last 30 days),
  `limit` (default 100, max 500).
- **Response:** `200`, array of event summaries (all Table fields except the
  blob is not inlined — `image_url` is a signed SAS link, generated per item,
  15-minute expiry).

### `GET /api/events/{event_id}` (dashboard detail — F04)

- Same auth as the list route. Returns the full entity + a freshly generated
  15-minute SAS `image_url` (not the same URL as any previous request — SAS
  tokens are minted per-call, never cached/persisted).

### `GET /api/metrics?since=` (dashboard analytics — F04, ADR-0016, new)

- **Auth:** same Basic Auth middleware as every other route.
- **Query params:** `since` (ISO date, optional, defaults to last 30 days —
  same default as `GET /api/events`). No `limit`: aggregation always covers
  the full matching window (`storage.list_events_for_metrics()`, uncapped,
  unlike the 500-row display cap on `GET /api/events`).
- **Response `200`:**
  ```json
  {
    "since": "2026-06-14",
    "until": "2026-07-14",
    "summary": {
      "real_wakes": 12,
      "false_wakes": 34,
      "non_escalating_wakes": 210,
      "email_delivery_rate": 0.92,
      "gemini_call_success_rate": 0.80,
      "window_break_confirmation_rate": 0.75
    },
    "daily": [
      {"date": "2026-07-01", "real_wakes": 1, "false_wakes": 3, "non_escalating_wakes": 20, "total": 24}
    ],
    "vision_source_breakdown": {
      "real_wakes": {"gemini": 10, "failsafe": 2, "none": 0},
      "false_wakes": {"gemini": 30, "failsafe": 0, "none": 4},
      "non_escalating_wakes": {"gemini": 0, "failsafe": 0, "none": 210}
    },
    "trigger_breakdown": {
      "triggered": {"real_wakes": 12, "false_wakes": 34, "non_escalating_wakes": 210},
      "not_triggered": {"real_wakes": 0, "false_wakes": 0, "non_escalating_wakes": 0}
    },
    "last_sync": {
      "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
      "ts_wall": 1784048796.83,
      "received_at": 1784048801.2
    },
    "review_accuracy": {
      "reviewed_count": 18,
      "window_broken": {"tp": 10, "fp": 1, "tn": 6, "fn": 1, "accuracy": 0.889},
      "intrusion": {"tp": 8, "fp": 2, "tn": 7, "fn": 1, "accuracy": 0.833}
    },
    "latency_s": {"avg": 8.2, "p50": 7.9, "p95": 14.1, "max": 22.0}
  }
  ```
- **Behavior:** pure aggregation over `cloud/app/metrics.py`'s functions,
  fed by one uncapped Table query. `daily` is sorted oldest-first (trend
  charts read left-to-right chronologically; `GET /api/events` stays
  newest-first, a deliberate difference — that route is a list to scan, this
  one is a series to plot). `vision_source_breakdown` is **not** a
  ground-truth confusion matrix — see F04 design's Risks and ADR-0016; it
  cross-tabulates decision outcome against which vision path produced it
  (`gemini`, `failsafe`, or `none` for non-escalating events that never
  called vision). `trigger_breakdown` *(2026-07-15, ADR-0016 addendum)*
  cross-tabulates outcome against `woken_by_trigger` — the deployed system
  wakes only via the SNN hardware trigger (`agent/trigger.py`), so
  `not_triggered` should normally be all zeros; a nonzero value there is an
  anomaly signal (an unexpected non-trigger boot reaching the cloud), not a
  second normal wake path. This field also answers a natural question about
  a high `non_escalating_wakes` count: it does **not** mean the SNN trigger
  was wrong — the SNN is a deliberately cheap, sensitive first-stage sensor
  (comparable to a smoke detector), expected to latch on plenty of
  non-intrusion events; the prefilter and vision stages exist specifically
  to filter those out downstream. A funnel shape (many triggers, few
  confirmed alarms) is the system working as intended.
  `summary.gemini_call_success_rate` *(2026-07-15, ADR-0016 addendum)* —
  of every event that actually attempted a vision call (`vision_source` is
  `"gemini"` or `"failsafe"`; non-escalating events never call vision and
  are excluded from both numerator and denominator), what fraction got a
  real Gemini verdict rather than falling back to failsafe. Same "rate over
  a subset, 0.0 when empty" shape as `email_delivery_rate`. `last_sync`
  *(2026-07-15, ADR-0016 addendum)* is the most recent event in the queried
  window by `received_at` (when the cloud actually got the push, not
  `ts_wall`'s Pi-side wake time) — `null` when the window has no events,
  which is itself a meaningful "nothing has synced in this window" signal
  rather than an error. Deliberately scoped to the same `since` window as
  the rest of the response, not a second unbounded query.
  `summary.window_break_confirmation_rate` *(2026-07-15, ADR-0017, new)* —
  of every event with a *real* Gemini verdict (`vision_source == "gemini"`
  specifically; failsafe verdicts are excluded, since their
  `window_broken=true` is a conservative fail-open default, not an actual
  visual confirmation), what fraction did Gemini classify as showing a
  broken window. This is the metric that most directly validates the SNN's
  own detection target (acoustic glass-break), as opposed to
  `vision_source_breakdown`'s more general outcome/vision-path view.
  `review_accuracy` *(2026-07-15, ADR-0018, new)* — unlike every other field
  on this endpoint, this is a real confusion-matrix accuracy measurement
  against owner-confirmed ground truth (`PATCH /api/events/{event_id}`,
  below), not an operational/agreement breakdown. Scoped to events with both
  a real Gemini verdict (`vision_source == "gemini"`) and a completed review
  (`reviewed_at` set); `reviewed_count` is `0` (both confusion blocks
  all-zero, `accuracy: 0.0`) until the owner has reviewed at least one such
  event — read `reviewed_count` alongside the accuracy numbers rather than
  the percentage alone, since a small sample can look misleadingly perfect
  or bad.

### `PATCH /api/events/{event_id}` (manual ground-truth review — F04, ADR-0018, new)

- **Auth:** same Basic Auth middleware as every other route.
- **Body (JSON):**
  ```json
  {"window_broken_confirmed": true, "intrusion_confirmed": false}
  ```
  Both fields required (schema is closed, `extra="forbid"`) — a review
  confirms both judgments together in one submission, not a partial update.
- **Validation:** required fields present and boolean-typed; unknown extra
  fields rejected with `422`.
- **Response:** `200`, the updated `EventDetail` (same shape as `GET
  /api/events/{event_id}`, now including `window_broken_confirmed`,
  `intrusion_confirmed`, `reviewed_at`); `404` if `event_id` doesn't match
  any event; `401` on missing/bad Basic Auth credential.
- **Behavior:** `reviewed_at` is assigned server-side (`time.time()`), never
  trusted from the request — same pattern as `received_at` on ingest. Looks
  up the entity by `event_id` (RowKey) to discover its PartitionKey, then
  `update_entity(mode=MERGE)` writes the three review fields onto the
  existing entity without touching any Pi- or cloud-written field. A second
  PATCH of the same `event_id` overwrites the prior review outright — the
  owner correcting an earlier mistake is a normal, supported action.

### `GET /` and dashboard pages (F04)

- Same Basic Auth middleware, applied globally to the whole app (not just
  `/api/*`) — the simplification this single-app design enables over the
  earlier split (where auth was two different mechanisms on two different
  resources).

## Data model

Table Storage entity as defined in F02's design — this feature is the only
writer and the only reader.

## Risks

- **Container Apps cold starts after scale-to-zero** (low single-digit
  seconds, occasionally more) — not a threat to correctness (fire-and-forget
  from the Pi's side; dashboard tolerates a spinner). Accepted per the
  Evaluation Framework in `01-system-overview.md`.
- **Cost-of-abuse on a public endpoint** — mitigated by the Basic Auth
  requirement + the 2 MB body cap; residual risk accepted at hobby scale
  given the Consumption plan's large free monthly grant (see
  `01-system-overview.md` Security).

## Security

- Every route (ingest, read, and the dashboard pages) sits behind one
  shared Basic Auth dependency (ADR-0009) — there is no unauthenticated
  route in this app at all, including `/` itself.
- Ingest input is validated against a closed Pydantic schema, image size
  capped, no secrets or PII echoed in error responses (generic `401`/`422`
  bodies only).
- Unlike the earlier design (which delegated read-route protection to a
  Static Web Apps proxy in front of the API), this app enforces its own
  auth directly — there is no trust boundary to document here, because
  there is no longer a second resource in front of it. If this API is ever
  exposed to a second consumer (e.g. a future mobile app) that shouldn't
  share the dashboard's exact credential, that's the trigger to add a
  second auth mechanism — not needed today.
- **(ADR-0014, new)** Ingest now trusts a client-supplied `event_id` as an
  upsert key. This is not a new privilege beyond what the shared credential
  already grants — an authenticated caller already has full read/write
  access to the whole event history (ADR-0009); the credential, not
  per-event authorization, is this system's real trust boundary. The ULID
  format check rejects malformed values, not a malicious-but-well-formed
  one — accepted, consistent with the rest of this system's single-shared-
  credential trust model.

## Decisions

- ADR-0007 (Azure Container Apps, Consumption plan, Python/FastAPI).
- ADR-0008 (one app serves API + dashboard UI).
- ADR-0009 (shared fixed Basic Auth credential for every route).
- ADR-0006 (single combined JSON+base64 POST vs. two-step SAS upload).
- ADR-0014 (client-generated `event_id`, idempotent upsert ingest).
- ADR-0016 (`GET /api/metrics` server-side analytics endpoint).
- ADR-0018 (`PATCH /api/events/{event_id}` manual ground-truth review;
  `review_accuracy` on `GET /api/metrics`).

## Branch

`feat/dashboard` (task T02)
