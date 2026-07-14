# ADR-0006: Single combined POST (JSON + base64 image) instead of two-step SAS upload

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F03_pi_push_client

## Context

The Pi needs to get both event metadata and a snapshot image to the cloud
per wake cycle. The two natural shapes are: (a) one request carrying both,
or (b) the "proper" large-file pattern — request a short-lived SAS upload
URL from the app, PUT the image directly to Blob Storage, then POST the
metadata referencing it.

## Decision

The Pi sends **one** `POST /api/events` request containing the JSON record
with the image inline as base64 (capped at 2 MB decoded). The app writes
both the Table entity and the Blob from that single request.

## Alternatives Considered

### Two-step: SAS-issue → direct-to-Blob PUT → metadata POST
- **Pros:** The standard pattern for larger files; avoids base64's ~33%
  size inflation and avoids routing image bytes through the app's own
  request handling (which has its own time/memory cost).
- **Cons:** Two round trips instead of one, each with its own timeout and
  failure mode to handle on the Pi (partial-failure states: image uploaded
  but metadata POST fails, or vice versa, need reconciliation logic); more
  code on the constrained, fail-safe-critical Pi side (ADR-0001, PR/FAQ
  Tenet 1) for a marginal efficiency gain on images that are, at most, a few
  hundred KB.
- **Why not:** At this image size, base64 inflation and the extra request
  handling are immaterial cost/latency, while a single request has exactly
  one failure mode (the whole push succeeded or it didn't) — which is
  simpler to reason about and test against the ≤5s-added SLO in
  `01-system-overview.md`, and keeps `agent/cloud_sync.py` small.

## Consequences

### Positive
- One request, one timeout, one failure mode on the Pi — matches the
  best-effort/non-blocking design goal directly.
- Simpler server implementation (one route does both writes).

### Negative
- ~33% larger request body than raw binary upload (irrelevant at these
  image sizes and this event frequency).
- Image bytes pass through the app's request handling, counted against the
  Container App's compute time (still negligible at this volume — see
  ADR-0007's cost reasoning).

### Risks (with mitigation)
- **Risk:** if snapshot images grow much larger in the future (e.g. full
  10-frame bursts instead of one snapshot), this approach stops scaling.
  **Mitigation:** the 2 MB cap in F01's validation is the explicit trigger —
  if a real need to exceed it appears, that's when to revisit this ADR and
  move to the two-step pattern, not before.
