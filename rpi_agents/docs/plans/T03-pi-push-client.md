# T03 — Pi push client

- **Branch:** `feat/dashboard` (off `feat/rpi`) — same branch as T02
- **Feature ID:** F03 (pi_push_client)
- **Depends on:** T01 for the live end-to-end check only (needs a deployed
  ingest endpoint to push to); the code itself can be written and unit
  tested with no dependency on T01 or T02 being done first (mocked HTTP).
- **Can run in parallel with:** T02 (same branch, no code overlap — T02
  touches `cloud/`, T03 touches `agent/`)
- **Source:** `docs/architecture/delivery-plan.json` (T03), `docs/architecture/02-delivery.md`,
  `docs/architecture/features/F03-pi-push-client/design.md`,
  ADR-0001, ADR-0006, ADR-0009

## Goal

Add a best-effort, non-blocking push of every wake-cycle event to the cloud
dashboard, without ever meaningfully delaying `agent/power.py::resleep()` —
this is the feature most directly constrained by the PR/FAQ's Tenet 1
(power/latency budget). A failed push must never raise into `machine.py`.

## Current state (brownfield — exact touch points)

Read `agent/machine.py` and `agent/config.py` in full before starting;
the changes below are additive and must not alter existing behavior or
the `Decision`/`PrefilterResult`/`VisionVerdict` contracts in
`agent/types.py`.

- **`agent/machine.py::run_cycle()`** (lines ~68–75 today): the ALARM
  branch already wraps `notifier.notify(verdict.reason, snapshot)` in a
  `try/except Exception` that logs and continues — but the outcome
  (success vs. exception) isn't retained anywhere today. Capture it into a
  local `email_sent: bool` (default `True` outside the ALARM branch is
  wrong — `email_sent` should be `False` whenever `notify()` wasn't even
  attempted, i.e. every non-alarm decision; only set `True` when
  `notify()` returns without raising inside the ALARM branch).
- **`agent/machine.py::_log_event()`** (lines ~88–109 today): add
  `email_sent` to the `record` dict passed to `json.dumps` — same JSONL
  schema addition on the local log as the cloud Table entity (F02 already
  has `email_sent` in its schema).
- **After the existing `_log_event(...)` call** in `run_cycle()`: call
  `cloud_sync.push(cloud_sync.build_payload(record, snapshot))` — needs
  the same `record` dict `_log_event` builds, plus the raw snapshot
  (`np.ndarray | None`, `None` when `pf.escalate` was `False` and no
  snapshot was ever captured). Cleanest approach: refactor `_log_event` to
  return the `record` dict it built (or build it once in `run_cycle` and
  pass it to both `_log_event` and `cloud_sync`), rather than duplicating
  the dict-assembly logic in two places.
- **`agent/config.py`**: add module constants matching the existing
  `os.getenv(...)` style already used throughout this file —
  `CLOUD_SYNC_ENABLED` (bool, default `True`), `CLOUD_SYNC_URL`,
  `CLOUD_SYNC_TIMEOUT_S` (default `(3, 5)` connect/read tuple).
- **`agent/config.py::Settings`**: add `cloud_sync_user` and
  `cloud_sync_password` fields. Unlike `gemini_api_key`/`gmail_user`/
  `gmail_app_password`/`alert_to` (which are required — `load_settings()`
  raises `ValueError` if any is missing), these two must be **optional**:
  a missing cloud_sync credential should disable the push (log once, skip)
  rather than crash the wake cycle. This means `load_settings()`'s
  `if not all([...])` required-fields check must NOT include these two.
- **Unchanged:** `agent/vision.py`, `agent/notifier.py` — F03 only reads
  `notifier.notify()`'s outcome via `machine.py`'s existing try/except; it
  does not modify `notifier.py` itself.

## Files to create

```
agent/cloud_sync.py
tests/test_cloud_sync.py   # mirrors tests/test_notifier.py's structure —
                            # FakeSMTP-style fake there; here, mock
                            # requests.post directly with monkeypatch
```

## Files to change

```
agent/machine.py    # run_cycle(): capture email_sent, call cloud_sync.push()
                     # after _log_event(); _log_event(): add email_sent field
agent/config.py      # CLOUD_SYNC_* constants; Settings gains
                      # cloud_sync_user/cloud_sync_password (optional)
```

## Contract (`agent/cloud_sync.py`, verbatim from F03 design)

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

`push()` differs from `notifier.notify()` in one important way:
`notify()` is allowed to raise and is caught by its caller; `push()`
swallows *all* exceptions internally and returns a `bool`, because unlike
email delivery, a failed cloud push is not something any caller needs to
react to or retry.

## Payload shape (F03 — mirrors F02's Table entity minus server-assigned fields)

Same fields as the local JSONL record (`ts_wall`, `woken_by_trigger`,
`escalate`, `motion`, `person`, `score`, `vision_source`, `is_intrusion`,
`alarm`, `reason`, `email_sent`, `latency_s`), minus `PartitionKey`/
`RowKey`/`received_at`/`blob_name` (server-assigned by T02's ingest route),
plus `image_jpeg_b64` — included only when a snapshot exists (i.e. the
decision escalated far enough to have one); omitted/`None` for a plain
static-scene cycle, keeping those pushes tiny.

## Step-by-step

1. `agent/config.py`: add the three `CLOUD_SYNC_*` constants and the two
   new optional `Settings` fields. Validate `CLOUD_SYNC_URL` is `https://`
   at config-load time — reject `http://` outright (F03 Security: image
   bytes and the Basic Auth credential must never go over plaintext HTTP).
2. `agent/cloud_sync.py`: implement `build_payload()` — pure function,
   easy to unit test without any network mocking.
3. `agent/cloud_sync.py`: implement `push()` using `requests` with an
   **explicit** `timeout=CLOUD_SYNC_TIMEOUT_S` (never the default
   "wait forever" — this is the one hard requirement in F03's Risks, not
   a nice-to-have) and `auth=(user, password)` (standard
   `Authorization: Basic` header, never logged — mirrors the "Never log
   or print these values" rule already in `Settings`' docstring). Catch
   every exception inside `push()`; return `False`, log once, never
   propagate.
4. `tests/test_cloud_sync.py`, mirroring `tests/test_notifier.py`'s
   fixture style:
   - `build_payload()` shape tests (with and without a snapshot).
   - `push()` returns `True` on a mocked 2xx.
   - `push()` returns `False` (not raises) on: timeout, connection error,
     non-2xx, and on a missing `cloud_sync_user`/`cloud_sync_password`
     (this last case is explicitly called out in F03's Risks as needing
     its own test).
   - Config validation: `CLOUD_SYNC_URL="http://..."` rejected at
     load time.
5. `agent/machine.py`: refactor so the event-record dict is built once and
   available to both `_log_event()` and the new `cloud_sync.push()` call;
   add `email_sent` to that record (see "Current state" above for exact
   semantics — `False` by default, `True` only after a successful
   `notify()` inside the ALARM branch).
6. Run `tests/test_machine.py` — confirm no existing assertions on
   `Decision`/`PrefilterResult`/`VisionVerdict` break; the `email_sent`
   field is additive to the logged record only, never a new field on
   those dataclasses in `agent/types.py`.
7. Once T01's pipeline has deployed T02's app (PR from `feat/dashboard`
   into `feat/azure-cd` open and merged/deployed): manually trigger a real
   wake cycle on the Pi and confirm the event appears in the dashboard
   within 30s.

## Acceptance gate (from `delivery-plan.json`)

- Unit tests green (mocked HTTP, mirrors `tests/test_notifier.py`).
- Manual E2E on the Pi: a real wake cycle, with `feat/dashboard`'s PR
  already deployed via T01's pipeline, produces a visible row in the
  dashboard within 30s.

## Notes / risks to carry into implementation

- Timeout budget: `(3, 5)` connect/read bounds worst case at ~8s inside
  the ≤5s-added target only if the connect phase succeeds quickly; on a
  fully dead network the OS-level connect timeout dominates. The
  `timeout=` kwarg is non-negotiable — do not use `requests`' default.
- Credential source: same `~/.config/snn-agent/.env` file, same
  `chmod 600` pattern as the four existing secrets — no new
  secrets-storage mechanism on the Pi side. This is the *same* credential
  the owner types into the dashboard (ADR-0009); the Pi is just another
  Basic Auth client of T02's app.
