# T03 — Pi push client (with bounded offline sync queue)

*(Revised 2026-07-14 — adds the local retry queue described below; T03 was
not yet implemented when this revision landed, so this plan replaces the
original version in place rather than layering a patch on top of shipped
code. See ADR-0014, ADR-0015.)*

- **Branch:** `feat/dashboard` (off `feat/rpi`) — same branch as T02/T04
- **Feature ID:** F03 (pi_push_client)
- **Depends on:** T01 for the live end-to-end check only (needs a deployed
  ingest endpoint to push to); **T04** for the live push/flush to actually
  succeed (the server must accept a client-supplied `event_id` first —
  ADR-0014). The code itself can be written and unit tested with no
  dependency on T01/T02/T04 being done first (mocked HTTP).
- **Can run in parallel with:** T02/T04 for *writing* the code (same
  branch, no file overlap — T02/T04 touch `cloud/`, T03 touches `agent/`);
  not for the *live* manual E2E check, which needs T04 deployed.
- **Source:** `docs/architecture/delivery-plan.json` (T03), `docs/architecture/02-delivery.md`,
  `docs/architecture/features/F03-pi-push-client/design.md`,
  ADR-0001, ADR-0006, ADR-0009, ADR-0014, ADR-0015

## Goal

Add a best-effort push of every wake-cycle event to the cloud dashboard,
without ever meaningfully delaying `agent/power.py::resleep()` (Tenet 1). A
failed push must never raise into `machine.py`. Unlike the original design,
a failed push is no longer simply dropped: it's queued locally (bounded)
and retried on later wake cycles, so an offline stretch delays delivery
rather than losing it outright — up to the queue's cap (Tenet 2, revised;
ADR-0015).

## Current state (brownfield — exact touch points)

Read `agent/machine.py` and `agent/config.py` in full before starting;
the changes below are additive and must not alter existing behavior or
the `Decision`/`PrefilterResult`/`VisionVerdict` contracts in
`agent/types.py`.

- **`agent/machine.py::run_cycle()`** (lines ~68–85 today): the ALARM
  branch already wraps `notifier.notify(verdict.reason, snapshot)` in a
  `try/except Exception` that logs and continues — but the outcome
  (success vs. exception) isn't retained anywhere today. Capture it into a
  local `email_sent: bool` (`False` whenever `notify()` wasn't even
  attempted, i.e. every non-alarm decision; `True` only when `notify()`
  returns without raising inside the ALARM branch). **Also generate
  `event_id` here** — `storage_ulid = cloud_sync.generate_event_id()` (or
  equivalent; see Contracts) — once, before `_log_event()` is called, so
  the same id is reused for every retry of this event, immediate or queued.
- **`agent/machine.py::_log_event()`** (lines ~88–109 today): add
  `email_sent` and `event_id` to the `record` dict passed to `json.dumps`
  — same JSONL schema addition on the local log as the cloud Table entity.
- **After the existing `_log_event(...)` call** in `run_cycle()`: build the
  payload, push it, enqueue on failure, then flush the backlog:
  ```python
  if config.CLOUD_SYNC_ENABLED:
      payload = cloud_sync.build_payload(record, snapshot)
      if not cloud_sync.push(payload):
          sync_queue.enqueue(payload)
      cloud_sync.flush_queue()
  ```
  Cleanest approach: refactor `_log_event` to return the `record` dict it
  built (or build it once in `run_cycle` and pass it to both `_log_event`
  and `cloud_sync`), rather than duplicating the dict-assembly logic in two
  places.
- **`agent/config.py`**: add module constants matching the existing
  `os.getenv(...)` style already used throughout this file —
  `CLOUD_SYNC_ENABLED` (bool, default `True`), `CLOUD_SYNC_URL`,
  `CLOUD_SYNC_TIMEOUT_S` (default `(3, 5)` connect/read tuple),
  `SYNC_QUEUE_PATH` (default `VAR_DIR / "sync_queue.jsonl"`),
  `SYNC_QUEUE_MAX_SIZE` (default `20`), `SYNC_QUEUE_MAX_FLUSH_PER_CYCLE`
  (default `5`), `SYNC_QUEUE_MAX_ATTEMPTS` (default `5`).
- **`agent/config.py::Settings`**: add `cloud_sync_user` and
  `cloud_sync_password` fields. Unlike `gemini_api_key`/`gmail_user`/
  `gmail_app_password`/`alert_to` (which are required — `load_settings()`
  raises `ValueError` if any is missing), these two must be **optional**:
  a missing cloud_sync credential should disable the push (log once, skip)
  rather than crash the wake cycle.
- **Unchanged:** `agent/vision.py`, `agent/notifier.py` — F03 only reads
  `notifier.notify()`'s outcome via `machine.py`'s existing try/except; it
  does not modify `notifier.py` itself.

## Files to create

```
agent/cloud_sync.py
agent/sync_queue.py
tests/test_cloud_sync.py   # mirrors tests/test_notifier.py's structure —
                            # FakeSMTP-style fake there; here, mock
                            # requests.post directly with monkeypatch
tests/test_sync_queue.py   # cap eviction, attempts tracking, corrupted-
                            # last-line tolerance — no network involved
```

## Files to change

```
agent/machine.py    # run_cycle(): generate event_id, capture email_sent,
                     # push + enqueue-on-failure + flush_queue() after
                     # _log_event(); _log_event(): add email_sent/event_id
agent/config.py      # CLOUD_SYNC_*/SYNC_QUEUE_* constants; Settings gains
                      # cloud_sync_user/cloud_sync_password (optional)
```

## Contracts (verbatim from F03 design)

```python
# agent/cloud_sync.py
def build_payload(record: dict, snapshot: np.ndarray | None) -> dict: ...

def push(payload: dict) -> bool:
    """Best-effort POST to CLOUD_SYNC_URL. Returns True on 2xx, False on
    any failure (timeout, connection error, non-2xx). Never raises."""

def flush_queue(max_items: int = config.SYNC_QUEUE_MAX_FLUSH_PER_CYCLE) -> None:
    """Attempt to push up to max_items queued payloads, oldest first.
    Stops at the first failure in this call. Never raises."""
```

```python
# agent/sync_queue.py
def enqueue(payload: dict) -> None:
    """Append payload. Drops the oldest entry first if already at
    SYNC_QUEUE_MAX_SIZE, to make room."""

def dequeue_batch(max_items: int) -> list[dict]:
    """Return up to max_items oldest queued payloads without removing them."""

def remove(event_id: str) -> None:
    """Remove one entry after a confirmed successful push."""

def record_failure(event_id: str) -> None:
    """Increment one entry's attempts; drop it (logged) at
    SYNC_QUEUE_MAX_ATTEMPTS."""
```

`push()`/`flush_queue()` swallow *all* exceptions internally and never
raise — unlike `notifier.notify()`, which is allowed to raise and is caught
by its caller. **Ordering is fixed:** the current cycle's own event is
always pushed (or queued) first; `flush_queue()` runs strictly after, using
whatever of the cycle's time budget remains — backlog work must never delay
the current event's own push.

## Payload shape

Same fields as the local JSONL record (`ts_wall`, `woken_by_trigger`,
`escalate`, `motion`, `person`, `score`, `vision_source`, `is_intrusion`,
`alarm`, `reason`, `email_sent`, `latency_s`), **plus `event_id`
(ADR-0014, a ULID generated in `machine.py` at decision time)**, minus
`PartitionKey`/`RowKey`/`received_at`/`blob_name` (server-assigned by T04's
retrofitted ingest route), plus `image_jpeg_b64` (included only when a
snapshot exists).

## `sync_queue.jsonl` shape (ADR-0015)

One JSON object per line: `{"event_id": ..., "payload": {...}, "attempts":
0, "queued_at": 1784048800.1}`. Capped at `SYNC_QUEUE_MAX_SIZE`; oldest
dropped first once full. Append-only; a torn last line (e.g. from a power
loss mid-write) is skipped on read, not treated as a fatal error.

## Step-by-step

1. `agent/config.py`: add the `CLOUD_SYNC_*`/`SYNC_QUEUE_*` constants and
   the two new optional `Settings` fields. Validate `CLOUD_SYNC_URL` is
   `https://` at config-load time — reject `http://` outright (F03
   Security).
2. `agent/cloud_sync.py`: implement `build_payload()` — pure function,
   takes `event_id` from the already-built `record` rather than minting
   its own.
3. `agent/sync_queue.py`: implement the JSONL-backed queue —
   `enqueue()`/`dequeue_batch()`/`remove()`/`record_failure()`. Tolerant
   read: skip a line that fails `json.loads` rather than raising.
4. `agent/cloud_sync.py`: implement `push()` using `requests` with an
   **explicit** `timeout=CLOUD_SYNC_TIMEOUT_S` and `auth=(user, password)`.
   Implement `flush_queue()` on top of step 3's functions: pull up to
   `SYNC_QUEUE_MAX_FLUSH_PER_CYCLE` oldest entries, push each in order,
   `remove()` on success, `record_failure()` and **stop the loop** on the
   first failure.
5. `tests/test_cloud_sync.py` + `tests/test_sync_queue.py`, mirroring
   `tests/test_notifier.py`'s fixture style:
   - `build_payload()` shape tests (with/without a snapshot; `event_id`
     present and unchanged from the input record).
   - `push()` returns `True` on a mocked 2xx; `False` (never raises) on
     timeout, connection error, non-2xx, and on a missing
     `cloud_sync_user`/`cloud_sync_password`.
   - `flush_queue()`: pushes up to the per-cycle cap; stops at the first
     failure (assert a second, still-pending mock isn't even called);
     removes confirmed successes; increments `attempts` on failure and
     drops the entry once `SYNC_QUEUE_MAX_ATTEMPTS` is hit.
   - `sync_queue`: cap eviction drops the oldest entry when appending past
     `SYNC_QUEUE_MAX_SIZE`; a corrupted/truncated last line is skipped, not
     fatal, on read.
   - Config validation: `CLOUD_SYNC_URL="http://..."` rejected at load
     time.
6. `agent/machine.py`: build the event record once (including `event_id`),
   pass it to `_log_event()` and `cloud_sync.build_payload()`; on push
   failure call `sync_queue.enqueue()`; always call
   `cloud_sync.flush_queue()` afterward regardless of the current event's
   own push outcome.
7. Run `tests/test_machine.py` — confirm no existing assertions on
   `Decision`/`PrefilterResult`/`VisionVerdict` break; `email_sent` and
   `event_id` are additive to the logged record only.
8. Once T01's pipeline has deployed T04 + this task (PR from
   `feat/dashboard` into `feat/azure-cd` open and merged/deployed):
   manually trigger a real wake cycle, confirm the event appears in the
   dashboard within 30s. Then run the offline-then-reconnect check: disable
   networking, trigger 2-3 wakes, confirm they land in `sync_queue.jsonl`
   and not the dashboard; re-enable networking, trigger (or wait for) the
   next wake, confirm the backlog flushes and all queued events appear.

## Acceptance gate (from `delivery-plan.json`)

- Unit tests green (mocked HTTP, mirrors `tests/test_notifier.py`, plus
  queue cap/attempts/corruption cases).
- Manual E2E on the Pi: a real wake cycle, with T04 already deployed via
  T01's pipeline, produces a visible row in the dashboard within 30s.
- Manual offline-then-reconnect check (above) passes.

## Notes / risks to carry into implementation

- Timeout budget: `(3, 5)` connect/read bounds worst case at ~8s per push
  attempt; on a fully dead network the OS-level connect timeout dominates.
  `flush_queue()` stopping at the first failure bounds a dead-network
  cycle's added cost to roughly one extra timeout, not `SYNC_QUEUE_MAX_FLUSH_PER_CYCLE`
  of them.
- Credential source: same `~/.config/snn-agent/.env` file, same
  `chmod 600` pattern as the four existing secrets.
- **T03 cannot be verified live until T04 is deployed** — the server
  rejects (or, pre-T04, silently mishandles) an ingest payload it doesn't
  yet expect an `event_id` field on. Write and unit-test T03 in parallel
  with T04 if convenient; sequence the *deploy* so T04 lands first.
