# F01 ingest_api

## Context

The FastAPI routes, inside the single Container App (ADR-0007, ADR-0008),
that both receive pushed events from the Pi and serve them to the dashboard.
Two concerns, one deployable unit and one process — cheaper and simpler than
splitting them, and every route (ingest and read alike) shares one Basic
Auth check (F05, ADR-0009).

![F01 ingest flow](../../diagrams/F01-ingest-flow.svg)

*Source: `docs/diagrams/F01-ingest-flow.dot` — edit and re-run `render_diagrams.sh`.*

## Contracts

### `POST /api/events` (ingest — called by the Pi, F03)

- **Auth:** HTTP Basic Auth, same shared credential as every other route
  (ADR-0009) — checked by shared middleware, not per-route logic.
- **Body (JSON):**
  ```json
  {
    "ts_wall": 1784048796.83,
    "woken_by_trigger": false,
    "escalate": true,
    "motion": true,
    "person": false,
    "score": 0.1547,
    "vision_source": "gemini",
    "is_intrusion": false,
    "alarm": false,
    "reason": "vision: ...",
    "email_sent": false,
    "latency_s": 10.67,
    "image_jpeg_b64": "<base64, optional>"
  }
  ```
- **Validation:** required fields present and correctly typed (a Pydantic
  model, matching FastAPI's native validation); `image_jpeg_b64` optional
  but if present, decoded size capped (reject > 2 MB — a 10-frame 320x180
  JPEG snapshot is a few hundred KB at most, 2 MB gives headroom without
  allowing abuse); reject unknown extra fields (schema is closed, Pydantic
  `model_config = {"extra": "forbid"}`).
- **Response:** `202 Accepted`, `{"event_id": "<ulid>"}` on success;
  `422` on validation failure (FastAPI/Pydantic default); `401` on
  missing/bad Basic Auth credential.
- **Behavior:** generates the `event_id` (ULID) and `received_at` server-side,
  writes the Table entity and (if present) the blob, in that order — if the
  blob write fails after the table write succeeds, the entity's `blob_name`
  is left empty rather than the whole request failing (a metadata-only event
  is still useful; a fully-failed ingest is not).

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

## Decisions

- ADR-0007 (Azure Container Apps, Consumption plan, Python/FastAPI).
- ADR-0008 (one app serves API + dashboard UI).
- ADR-0009 (shared fixed Basic Auth credential for every route).
- ADR-0006 (single combined JSON+base64 POST vs. two-step SAS upload).

## Branch

`feat/dashboard` (task T02)
