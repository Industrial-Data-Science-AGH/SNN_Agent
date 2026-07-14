# T04 — how to test it

Companion to `T04-idempotent-ingest-retrofit.md`. Two levels: automated
(no Azure needed, run this first) and manual (a real running app, either
against Azurite locally or the real dev Storage Account once `feat/azure-cd`
exists).

## 1. Automated tests (do this first)

From the repo root, in your real `.venv` (uv-managed, Python 3.12):

```bash
uv run ruff check cloud/app/ tests/
uv run ruff format --check cloud/app/
uv run pytest tests/test_ingest.py tests/test_auth.py tests/test_dashboard.py -v
```

What to look for:

- `ruff check` — no output, exit 0.
- `pytest` — every test passes, including these T04-specific ones:
  - `test_event_in_rejects_malformed_event_id[...]` (4 cases: too short, too
    long, lowercase, contains a non-Crockford character like `I`)
  - `test_ingest_rejects_malformed_event_id_with_422`
  - `test_ingest_upsert_is_idempotent_across_duplicate_event_id`
  - `test_write_event_calls_upsert_entity_not_create_entity`

If `test_ingest_returns_202_and_event_id` or the blob-write tests fail,
check that nothing else in `cloud/app/` still calls `storage.generate_ulid()`
— it should only be defined in `storage.py` now, never called from
`routes_api.py`.

## 2. Manual smoke test (real app, no real Azure needed)

This runs the actual FastAPI app against **Azurite** (Microsoft's official
local Storage emulator) so you exercise the real `upsert_entity()` call, not
a mock. Needs Docker.

### 2.1 Start Azurite

```bash
docker run -d --name azurite -p 10000:10000 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite --blobHost 0.0.0.0 --tableHost 0.0.0.0 --skipApiVersionCheck
```

`--skipApiVersionCheck` is required: the `azure-data-tables`/`azure-storage-blob`
SDK versions pinned in `cloud/app/requirements.txt` send a newer Storage API
version than the `latest` Azurite image currently understands, and Azurite
rejects the request outright without this flag (`HttpResponseError: The API
version ... is not supported by Azurite`).

If you already started Azurite without the flag, recreate it:

```bash
docker stop azurite && docker rm azurite
docker run -d --name azurite -p 10000:10000 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite --blobHost 0.0.0.0 --tableHost 0.0.0.0 --skipApiVersionCheck
```

### 2.2 Create the table + container Terraform would normally provision

Azurite doesn't run your Terraform, so create the two resources by hand,
once:

```bash
python3 - <<'EOF'
from azure.data.tables import TableServiceClient
from azure.storage.blob import BlobServiceClient

conn_str = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)
TableServiceClient.from_connection_string(conn_str).create_table_if_not_exists("events")
BlobServiceClient.from_connection_string(conn_str).create_container("snapshots")
print("events table + snapshots container ready")
EOF
```

(That account key is Azurite's fixed, publicly documented default — not a
secret, and only reachable on your own machine.)

### 2.3 Run the app

```bash
export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
export DASHBOARD_USER=ids
export DASHBOARD_PASSWORD=ids

uv run uvicorn cloud.app.main:app --reload --port 8000
```

Leave this running; use a second terminal for the checks below.

### 2.4 Check 1 — a fresh event_id is accepted

```bash
curl -s -u ids:ids -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "ts_wall": 1784048796.83,
    "woken_by_trigger": false,
    "escalate": true,
    "motion": true,
    "person": false,
    "score": 0.15,
    "vision_source": "gemini",
    "is_intrusion": false,
    "alarm": false,
    "reason": "test event",
    "email_sent": false,
    "latency_s": 10.6
  }' | python3 -m json.tool
```

Expect: `202`, body `{"event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}`.

### 2.5 Check 2 — posting the *same* event_id again is a no-op, not a duplicate

Run the exact same `curl` command again. Expect: another `202` with the same
`event_id`, no error. Then confirm there's still only one row:

```bash
curl -s -u ids:ids http://localhost:8000/api/events | python3 -m json.tool
```

Expect exactly one entry with `"event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"` —
this is the actual bug T04 fixes (pre-T04, the second POST would have either
duplicated the row or raised `ResourceExistsError`, since `event_id` was
server-generated and `write_event()` used `create_entity`).

### 2.6 Check 3 — a malformed event_id is rejected with 422

```bash
curl -s -u ids:ids -X POST http://localhost:8000/api/events \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "not-a-real-ulid",
    "ts_wall": 1784048796.83,
    "woken_by_trigger": false,
    "escalate": true,
    "motion": true,
    "person": false,
    "score": 0.15,
    "reason": "should be rejected",
    "alarm": false,
    "email_sent": false,
    "latency_s": 10.6
  }' -w "\nHTTP %{http_code}\n"
```

Expect: `HTTP 422`.

### 2.7 Check 4 — missing event_id is rejected too

Same as above but drop `"event_id"` entirely from the body. Expect: `HTTP
422` (now a required field, same as any other missing field).

### 2.8 Clean up

```bash
docker stop azurite && docker rm azurite
```

## 3. Once `feat/azure-cd` exists and this is actually deployed

Same four checks as section 2.4–2.7, just point `curl` at the real Container
App FQDN instead of `localhost:8000`, using the real `DASHBOARD_PASSWORD`
you set in the GitHub Actions secret (not the `ids`/`ids` default). This is
also the point where you can confirm T03's Pi-side retry logic actually
lands correctly: trigger two wake cycles that reuse a queued `event_id`
(e.g. disable networking for one wake, then reconnect) and confirm the
dashboard shows one row for that event, not two.

## Acceptance gate reminder (from `delivery-plan.json`)

- Automated: unit tests green, including upsert-idempotency and
  malformed-`event_id` cases.
- Manual: same-`event_id` posted twice → one row; malformed `event_id` →
  `422`.
