# F03 pi_push_client

## Context

The only code that runs on the Pi for this feature. A new module,
`agent/cloud_sync.py`, called from `agent/machine.py::run_cycle()` right
after the local `_log_event()` write, on the same record. Must never delay
`agent/power.py::resleep()` meaningfully — this is the feature most directly
constrained by Tenet 1 in the PR/FAQ.

![F03 push flow](../../diagrams/F03-push-flow.svg)

*Source: `docs/diagrams/F03-push-flow.dot` — edit and re-run `render_diagrams.sh`.*

## Current state *(brownfield — what this touches)*

- **Change:** `agent/machine.py::run_cycle()` — capture whether
  `notifier.notify()` succeeded or raised (it's already wrapped in a
  try/except that logs and continues; today the outcome isn't retained
  anywhere). Add `email_sent: bool` alongside the other fields already
  assembled for `_log_event()`, and pass that same record to
  `cloud_sync.push()` after the local log write.
- **Change:** `agent/machine.py::_log_event()` — add `email_sent` to the
  local JSON record (keeps the local log and the cloud copy on one schema,
  per `01-system-overview.md`).
- **New:** `agent/cloud_sync.py` — `build_payload()` and `push()`.
- **New:** `agent/config.py` gains `CLOUD_SYNC_ENABLED` (bool, env-driven,
  default `True`), `CLOUD_SYNC_URL`, `CLOUD_SYNC_TIMEOUT_S` (default
  `(3, 5)` connect/read), matching the existing `os.getenv(...)` constant
  style already used throughout `config.py`.
- **New:** `Settings` dataclass in `config.py` gains `cloud_sync_user` and
  `cloud_sync_password` (the same shared Basic Auth credential the
  dashboard uses, ADR-0009), loaded the same way as `gemini_api_key` etc.
  from `~/.config/snn-agent/.env` — **not** required fields (unlike the
  existing four), since a missing credential should disable the push (log
  once, skip), not crash the wake cycle.
- **Unchanged:** `agent/vision.py`, `agent/notifier.py` themselves — F03 only
  reads `notifier.notify()`'s success/failure from `machine.py`'s existing
  try/except, it doesn't change `notifier.py`.

## Contracts

`agent/cloud_sync.py`:

```python
def build_payload(record: dict, snapshot: np.ndarray | None) -> dict: ...
def push(payload: dict) -> bool:
    """Best-effort POST to CLOUD_SYNC_URL. Returns True on 2xx, False on
    any failure (timeout, connection error, non-2xx). Never raises."""
```

Called from `machine.py` as:

```python
if config.CLOUD_SYNC_ENABLED:
    cloud_sync.push(cloud_sync.build_payload(record, snapshot))
```

wrapped the same way `notifier.notify()` already is in the ALARM branch —
except unlike `notifier.notify()` (which is allowed to raise and is caught by
the caller), `push()` itself swallows all exceptions internally and returns a
bool, because unlike email delivery, a failed cloud push is not something any
caller needs to react to or retry.

## Data model

Same record shape as F02's Table entity, minus `PartitionKey`/`RowKey`/
`received_at`/`blob_name` (server-assigned), plus `image_jpeg_b64` (only
included when the decision escalated far enough that a snapshot exists —
`None`/omitted for a plain "static scene" cycle, keeping those pushes tiny).

## Risks

- **Timeout budget vs. SLO.** `(3, 5)` connect/read timeout bounds worst case
  at ~8s, inside the ≤5s-added target only if the connect phase succeeds
  quickly; on a fully dead network the OS-level connect timeout dominates.
  Mitigation: use `requests` with explicit `timeout=` (never the default of
  "wait forever") — this is the one hard requirement, not a nice-to-have.
- **Missing cloud_sync credential** shouldn't be a crash — validated by a
  unit test asserting `push()` returns `False` and logs, rather than
  raising, when either the user or password is absent.

## Security

- Credential read from `~/.config/snn-agent/.env`, same file/permission
  pattern (`chmod 600`) as the four existing secrets — no new secrets-storage
  mechanism introduced on the Pi side. This is the *same* credential the
  owner types into the dashboard (ADR-0009) — the Pi is just another Basic
  Auth client of the one app.
- Sent as a standard `Authorization: Basic` header (via `requests`'
  built-in `auth=(user, password)` parameter), never logged (mirrors the
  existing rule in `config.py`'s `Settings` docstring: "Never log or print
  these values").
- Image bytes sent over HTTPS only; `CLOUD_SYNC_URL` must be validated to be
  `https://` at config-load time (reject `http://` outright — home security
  snapshots, and the Basic Auth credential itself, must not go over
  plaintext HTTP even accidentally).

## Decisions

- ADR-0001 (push, not pull/poll).
- ADR-0006 (single combined POST with inline base64 image).
- ADR-0009 (shared Basic Auth credential, same one the dashboard uses).

## Branch

`feat/dashboard` (task T03)
