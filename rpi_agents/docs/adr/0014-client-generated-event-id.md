# ADR-0014: Pi generates event_id (ULID); ingest is an idempotent upsert

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F03_pi_push_client
- Supersedes: the server-side `event_id` generation described in F01's
  original design and implemented by T02 (`storage.generate_ulid()` called
  inside `routes_api.ingest_event()`). No standalone ADR previously covered
  that narrow sub-decision, so there is nothing to mark superseded in the
  ADR index beyond this file.

## Context

T02's shipped `POST /api/events` generates `event_id` (a ULID) server-side
on every request (F01 design, "Behavior"). ADR-0015 adds a Pi-side local
queue that retries a push it couldn't confirm succeeded — including the
case where the server actually committed the write but the Pi never saw the
`202` (connection dropped mid-response, timeout on a slow-but-successful
request, etc.). With server-generated `event_id`, that retry is
indistinguishable from a brand-new event and creates a second row for the
same real-world wake cycle.

## Decision

The Pi generates `event_id` (a ULID) itself, once, at decision time in
`agent/machine.py::run_cycle()` — not inside `cloud_sync.push()` — so every
retry attempt for the same event (immediate retry within a cycle, or a
later retry from `sync_queue.jsonl`) reuses the exact same id. It travels in
the `POST /api/events` body as a required field. The server's ingest route
uses this value as the Table entity's `RowKey` and performs an **upsert**
(create-or-replace) instead of a strict create: a retry with the same
`event_id` overwrites the existing row with identical data rather than
erroring or duplicating. `storage.py` validates the value is a
well-formed ULID (26-char Crockford base32) before writing it, rejecting
anything else with `422`.

## Alternatives Considered

### Keep server-side generation; dedupe via a client-supplied nonce + a pre-write existence check
- **Pros:** `event_id` stays a purely server-owned identifier.
- **Cons:** needs an extra Table Storage read before every write (latency,
  cost of a second transaction) and still requires the client to send some
  stable identifier — which is exactly what a client-generated `event_id`
  already is, just with an extra layer of indirection on top.
- **Why not:** strictly more moving parts for the same outcome.

### Accept rare duplicates; no idempotency work at all
- **Pros:** zero contract change — T02's shipped code stays exactly as is.
- **Cons:** a small but real chance of a duplicate row surviving in the
  dashboard indefinitely, which also quietly inflates F04's real/false-wake
  counts (each duplicate double-counts one event in the metrics band).
- **Why not:** raised explicitly as the alternative to this ADR's approach
  and rejected by the owner — offline durability (ADR-0015) was requested
  specifically to make cloud delivery *more* reliable; leaving a
  duplicate-risk in place undercuts that goal.

## Consequences

### Positive
- Retries are always safe: POSTing the same `event_id` twice converges to
  the same end state, with no duplicate-row risk, at any retry distance
  (same cycle or many wake cycles later via the queue).
- An upsert is a single Table Storage call — the same cost as the create it
  replaces, no added round trip.
- ULID's original justification (F02 design: "sortable, unique, avoids
  clock-skew collisions between Pi and Azure clocks") is unaffected by
  moving generation from server to client — the Pi's clock is exactly as
  good a source for this as the server's was.

### Negative
- The ingest endpoint now trusts a client-supplied value as (half of) its
  primary key, a small expansion of what the client controls versus the
  server-generates-everything model.
- Needs new validation (`storage.py`, `schemas.py`) to reject a malformed
  `event_id` before it reaches Table Storage as a `RowKey`.

## Risks (with mitigation)

- **Risk:** a malicious or buggy client submits a fabricated `event_id`
  that collides with an existing, unrelated event, silently overwriting
  it. **Mitigation:** every client of this endpoint already authenticates
  with the one shared Basic Auth credential, which already has full
  read/write access to the entire event history (ADR-0009) — an
  authenticated caller overwriting a row is not a new privilege beyond what
  that credential already grants; the credential itself, not per-event
  authorization, is this system's actual trust boundary (F05 design).
  Format-validating the ULID rejects obviously garbage values, not a
  malicious-but-well-formed one — accepted, matching this project's general
  single-shared-credential trust model.
- **Risk:** clock skew on the Pi makes the ULID's embedded timestamp
  unreliable for chronological ordering. **Mitigation:** unchanged from the
  original design — `received_at` is still assigned server-side and remains
  the field trusted for ingestion-order tie-breaking; `ts_wall` (and now the
  ULID's timestamp component) are for display only, never for ordering
  guarantees (ADR-0001, Risks).

## Decisions

- ADR-0001 (push, not pull) — `received_at` still resolves ordering, unaffected.
- ADR-0009 (shared Basic Auth) — the trust boundary this ADR's Risks section relies on.
- ADR-0015 (bounded local sync queue) — the reason idempotent retries are needed at all.
