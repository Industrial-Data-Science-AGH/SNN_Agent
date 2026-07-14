# T04 — Idempotent ingest retrofit (client-generated event_id)

*(New task, added 2026-07-14 alongside T03's revision — ADR-0014. Retrofits
code T02 already shipped; not part of the original three-task plan.)*

- **Branch:** `feat/dashboard` (off `feat/rpi`) — same branch as T02/T03
- **Feature ID:** F01 (ingest_api)
- **Depends on:** T02 (retrofits its already-shipped `cloud/app/`)
- **Blocks:** T03's live verification (the Pi's retries only succeed once
  the server accepts a client-supplied `event_id`); T03's *code* can be
  written/unit-tested independently
- **Source:** `docs/architecture/delivery-plan.json` (T04),
  `docs/architecture/features/F01-ingest-api/design.md`, ADR-0014, ADR-0015

## Goal

Make `POST /api/events` safe to retry. Today (T02) the server mints
`event_id` itself on every request, so a Pi-side retry of a request whose
response was lost — but which the server actually committed — creates a
duplicate row. T03's new local sync queue (ADR-0015) depends on retries
being safe; this task is the server-side half of that guarantee.

## Current state (brownfield — exact touch points)

This retrofits `cloud/app/schemas.py`, `cloud/app/routes_api.py`, and
`cloud/app/storage.py` as shipped by T02. Read all three in full before
starting.

- **`cloud/app/schemas.py::EventIn`** — currently has no `event_id` field
  (the server generates it). **Change:** add `event_id: str`, required, with
  a `field_validator` that checks it's a well-formed ULID (26 chars, every
  character in the Crockford base32 alphabet `0123456789ABCDEFGHJKMNPQRSTVWXYZ`)
  — reject anything else so a malformed value never reaches Table Storage as
  a `RowKey`.
- **`cloud/app/routes_api.py::ingest_event()`** — currently calls
  `storage.generate_ulid()` to produce `event_id`. **Change:** use
  `payload.event_id` instead; drop the `generate_ulid()` call from this
  function (the function itself can stay in `storage.py` — T03's Pi side
  needs the same ULID-generation logic; see Notes).
- **`cloud/app/storage.py::write_event()`** — currently calls
  `get_table_client().create_entity(entity=entity)`, which raises
  `ResourceExistsError` on a `PartitionKey`+`RowKey` collision. **Change:**
  use `get_table_client().upsert_entity(entity=entity, mode=UpdateMode.REPLACE)`
  instead — a retried `event_id` overwrites with identical data rather than
  raising.
- **Unchanged:** `write_blob()` is already `overwrite=True` (T02), so a
  retried event's image write is already idempotent — no change needed
  there. `set_blob_name()`, `list_events()`, `get_event()`, `mint_sas_url()`
  — untouched.

## Files to change

```
cloud/app/schemas.py      # EventIn.event_id: str, ULID-format validated
cloud/app/routes_api.py    # ingest_event(): use payload.event_id
cloud/app/storage.py        # write_event(): create_entity -> upsert_entity
```

## Contract change (verbatim delta from F01 design)

`POST /api/events` request body gains one required field:

```json
{
  "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
  "ts_wall": 1784048796.83,
  ...
}
```

Response is unchanged: `202 {"event_id": "<the same value>"}`. `422` now
also covers a malformed `event_id` (not just missing/mistyped other
fields). Behavior: the Table write is an upsert keyed on `event_id`
(`PartitionKey` is still derived from `ts_wall` as before) — posting the
same `event_id` twice with the same data is a no-op the second time from
the dashboard's point of view (one row, unchanged).

## Step-by-step

1. `cloud/app/schemas.py`: add `event_id: str` to `EventIn` with a
   `field_validator` checking length 26 and Crockford-base32-only
   characters (uppercase; reject lowercase too, since `generate_ulid()`
   only ever emits uppercase — no reason to accept a case the system never
   produces).
2. `cloud/app/storage.py::write_event()`: change `create_entity` to
   `upsert_entity(..., mode=UpdateMode.REPLACE)`. `write_event()`'s
   signature and return shape are unchanged — callers don't need to know
   this happened.
3. `cloud/app/routes_api.py::ingest_event()`: replace
   `event_id = storage.generate_ulid()` with `event_id = payload.event_id`.
4. Update `tests/test_ingest.py`'s `_VALID_PAYLOAD` fixture to include an
   `event_id` — every existing test in that file currently omits it and
   will start failing `422` (missing required field) the moment step 1
   lands. Add new tests:
   - Posting the same `event_id` twice (with `storage.write_event` mocked
     to a real in-memory dict, or asserting `upsert_entity` — not
     `create_entity` — was called) returns `202` both times, with no error
     on the second call.
   - A malformed `event_id` (wrong length, lowercase, non-Crockford
     character) is rejected with `422`.
5. `ruff check`/`format`; re-run the full `tests/test_auth.py`,
   `tests/test_ingest.py`, `tests/test_dashboard.py` suite — confirm
   nothing else regressed.

## Acceptance gate (from `delivery-plan.json`)

- Unit tests green, including the new upsert-idempotency and
  malformed-`event_id` cases.
- `tests/test_ingest.py`'s existing tests updated for the new required
  field and still passing.

## Notes / risks to carry into implementation

- **`storage.generate_ulid()` stays in `storage.py`.** It's no longer
  called from `routes_api.py`, but T03's Pi-side `agent/cloud_sync.py`
  needs the identical ULID-generation logic — rather than duplicating it
  across `cloud/app/` and `agent/`, keep `storage.generate_ulid()` as a
  reference implementation and have `agent/cloud_sync.py` implement its own
  copy (the two run in different processes/deployments, on different
  machines — there's no shared-import path between `cloud/app/` and
  `agent/`, so duplication here is unavoidable, not an oversight; keep both
  copies byte-for-byte identical in logic and note the pairing in each
  file's docstring).
- **Trust note (ADR-0014, Risks):** the ingest endpoint now trusts a
  client-supplied value as half of its primary key. This is not a new
  privilege beyond what the shared Basic Auth credential already grants —
  see F01 design's Security section for the full reasoning; no additional
  authorization check is needed here.
- **`UpdateMode` import:** `from azure.data.tables import UpdateMode` —
  already used by `storage.py::set_blob_name()` (T02), so no new
  dependency.
