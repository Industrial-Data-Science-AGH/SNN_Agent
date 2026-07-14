# T02 — FastAPI dashboard app

- **Branch:** `feat/dashboard` (off `feat/rpi`)
- **Feature IDs:** F01 (ingest_api), F05 (auth_and_secrets), F04 (dashboard_ui)
- **Depends on:** T01 (needs `cloud/infra` + the CD pipeline to deploy against;
  can be *written* in parallel with T01, just can't be verified live until
  T01 exists)
- **Can run in parallel with:** T03 (same branch, no code overlap)
- **Source:** `docs/architecture/delivery-plan.json` (T02), `docs/architecture/02-delivery.md`,
  `docs/architecture/features/F01-ingest-api/design.md`,
  `docs/architecture/features/F04-dashboard-ui/design.md`,
  `docs/architecture/features/F05-auth-and-secrets/design.md`,
  ADR-0006, ADR-0007, ADR-0008, ADR-0009

## Goal

Build the one FastAPI app (ingest API + read API + server-rendered
dashboard, all one deployable unit per ADR-0008) that the Pi pushes events
to and the owner views. Every route — ingest, read, and the dashboard pages
themselves — sits behind one shared Basic Auth dependency; there is no
unauthenticated route in this app.

## Files to create

```
cloud/app/
  main.py               # FastAPI() app instance, mounts routers, applies
                         # require_basic_auth as a global dependency
  auth.py                # require_basic_auth() — HTTPBasic dependency,
                          # secrets.compare_digest against DASHBOARD_USER/
                          # DASHBOARD_PASSWORD env vars (default ids/ids)
  schemas.py              # Pydantic models: EventIn (extra="forbid"),
                           # EventSummary, EventDetail
  storage.py               # data-access layer: Table read/write, Blob
                            # write, SAS URL minting — the only module that
                            # holds the Storage connection string
  routes_api.py             # POST/GET /api/events, GET /api/events/{id}
  routes_dashboard.py        # GET /, event list + detail pages (Jinja2)
  templates/
    base.html
    event_list.html          # metrics band + list, per F04 views
    event_detail.html
  Dockerfile                  # replaces T01's placeholder with the real app

tests/                         # (or cloud/app/tests/ if the toolchain
                                 # convention is per-package — check
                                 # pyproject.toml testpaths before deciding)
  test_auth.py                 # rejects missing/bad Basic Auth on every
                                # route class (ingest, read, dashboard)
  test_ingest.py                # Pydantic validation, 2MB image cap,
                                 # extra-field rejection, 202 + event_id
  test_dashboard.py              # metrics band math (real/false/
                                  # non-escalating/email-rate counts)
```

## Contracts (verbatim from F01 design — do not drift from these)

### `POST /api/events`

- Auth: Basic, same shared credential as every route.
- Body (Pydantic, `extra="forbid"`):
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
- `image_jpeg_b64` optional; if present, decoded size capped at 2 MB (reject
  over that).
- Response: `202 {"event_id": "<ulid>"}` on success; `422` on validation
  failure; `401` on missing/bad auth.
- Behavior: generate `event_id` (ULID) and `received_at` server-side. Write
  the Table entity first, then the Blob (if image present) — if the blob
  write fails after the table write succeeds, leave `blob_name` empty
  rather than failing the whole request.

### `GET /api/events?since=&limit=`

- Same auth. `since` (ISO date, default last 30 days), `limit` (default
  100, max 500).
- Response: array of summaries, `image_url` is a per-item SAS link,
  15-minute expiry, minted fresh on every call (never cached/persisted).

### `GET /api/events/{event_id}`

- Same auth. Full entity + a freshly minted 15-minute SAS `image_url`.

### `GET /` and dashboard pages

- Same Basic Auth dependency, applied globally — including `/` itself.

## Auth contract (F05 — do not build anything more elaborate than this)

- One FastAPI dependency, `require_basic_auth`, applied globally to the
  whole app.
- Compares `Authorization: Basic` credential against `DASHBOARD_USER`/
  `DASHBOARD_PASSWORD` env vars (default `ids`/`ids`) using
  `secrets.compare_digest` — constant-time, avoids a timing side-channel.
- Failure → `401` with `WWW-Authenticate: Basic` header.
- No lockout/rate-limiting — explicitly out of scope for this task
  (residual risk, accepted).

## Table entity schema (F02 — `storage.py` writes/reads this shape)

| Field | Type | Notes |
|---|---|---|
| `PartitionKey` | string | `YYYY-MM-DD`, UTC date from `ts_wall` |
| `RowKey` | string | ULID, = `event_id` |
| `ts_wall` | double | |
| `woken_by_trigger` | bool | |
| `escalate` | bool | |
| `motion` | bool | |
| `person` | bool | |
| `score` | double | |
| `vision_source` | string | `"gemini"` \| `"failsafe"` \| null |
| `is_intrusion` | bool \| null | |
| `alarm` | bool | |
| `reason` | string | |
| `email_sent` | bool | |
| `latency_s` | double | |
| `blob_name` | string | `{event_id}.jpg`, empty if no snapshot |
| `received_at` | double | set by the app on ingest |

## Dashboard views (F04 — server-rendered, computed in the same request, no separate endpoint)

1. **Event list** — most recent first, default last 30 days. Row:
   timestamp, thumbnail (SAS `image_url`), alarm badge (red/green),
   one-line `reason`.
2. **Metrics band** (computed server-side, same request):
   - Real wakes: `count(alarm == true)`.
   - False wakes: `count(escalate == true and alarm == false)`.
   - Non-escalating wakes: `count(escalate == false)`.
   - Email delivery rate: `count(email_sent == true) / count(alarm == true)`.
3. **Event detail**: full image, `reason` verbatim, `vision_source`
   (visually distinguish `gemini` vs `failsafe` — a failsafe alarm means
   Gemini itself failed), `email_sent` status, raw `score`/`motion`/
   `person` for debugging.

## Step-by-step

1. `cloud/app/schemas.py` — Pydantic models first; write `test_ingest.py`'s
   validation cases against them before wiring routes (closed schema,
   2 MB cap, required-field checks).
2. `cloud/app/auth.py` — `require_basic_auth`; write `test_auth.py` against
   it standalone (no Storage dependency needed to test auth rejection).
3. `cloud/app/storage.py` — Table + Blob read/write + SAS minting. This is
   the only module holding the Storage connection string; nothing else
   should touch `azure-storage-*` clients directly.
4. `cloud/app/routes_api.py` — wire schemas + auth + storage into the three
   API routes.
5. `cloud/app/routes_dashboard.py` + `templates/` — reuse `routes_api`'s
   data-access calls in-process (no network hop, no CORS — same app,
   per ADR-0008), compute the metrics band, render Jinja2.
6. `cloud/app/main.py` — assemble the app, apply `require_basic_auth` as a
   global dependency (not per-route), mount both routers.
7. Replace T01's placeholder `cloud/app/Dockerfile` with the real app's
   Dockerfile (same shape, real `CMD`/deps).
8. Run the full test suite locally against a dev Storage Account (from
   T01's `dev` Terraform environment) before opening the PR.
9. Manual check: confirm unauthenticated `GET /`, `GET /api/events`, and
   `POST /api/events` all return `401`.

## Acceptance gate (from `delivery-plan.json`)

- Unit tests green (auth rejection + payload validation, at minimum).
- App runs locally against a dev Storage Account.
- Manual check that unauthenticated requests are rejected on every route
  class.

## Notes / risks to carry into implementation

- No secrets or PII in error responses — `401`/`422` bodies stay generic.
- Container Apps cold start after scale-to-zero is expected (low
  single-digit seconds); not a correctness bug — the Pi's push is
  fire-and-forget and the dashboard tolerates a spinner.
- If this API ever needs a second consumer that shouldn't share the
  dashboard's exact credential, that's the trigger to add a second auth
  mechanism — don't build that now.
