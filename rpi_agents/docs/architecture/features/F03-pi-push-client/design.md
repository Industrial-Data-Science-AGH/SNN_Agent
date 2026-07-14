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

Not yet implemented (T03 is still pending) — this design supersedes an
earlier version of itself that had no local queue (revised 2026-07-14,
ADR-0014, ADR-0015, at the owner's explicit request). Nothing here has
shipped yet, so this is the design T03 should build directly; there is no
prior F03 code to reconcile against.

- **Change:** `agent/machine.py::run_cycle()` — capture whether
  `notifier.notify()` succeeded or raised (it's already wrapped in a
  try/except that logs and continues; today the outcome isn't retained
  anywhere). Add `email_sent: bool` alongside the other fields already
  assembled for `_log_event()`, and pass that same record to
  `cloud_sync.push()` after the local log write. **Also generate
  `event_id` (a ULID) here** — once, at decision time — so the same id is
  reused on every retry attempt for this event, immediate or queued
  (ADR-0014).
- **Change:** `agent/machine.py::_log_event()` — add `email_sent` and
  `event_id` to the local JSON record (keeps the local log and the cloud
  copy on one schema, per `01-system-overview.md`).
- **New:** `agent/cloud_sync.py` — `build_payload()`, `push()`,
  `flush_queue()`.
- **New:** `agent/sync_queue.py` — the bounded local queue backing
  `sync_queue.jsonl` (ADR-0015): `enqueue()`, `dequeue_batch()`,
  `remove()`, `record_failure()`.
- **New:** `agent/config.py` gains `CLOUD_SYNC_ENABLED` (bool, env-driven,
  default `True`), `CLOUD_SYNC_URL`, `CLOUD_SYNC_TIMEOUT_S` (default
  `(3, 5)` connect/read), `SYNC_QUEUE_PATH` (default
  `VAR_DIR / "sync_queue.jsonl"`), `SYNC_QUEUE_MAX_SIZE` (default `20`),
  `SYNC_QUEUE_MAX_FLUSH_PER_CYCLE` (default `5`),
  `SYNC_QUEUE_MAX_ATTEMPTS` (default `5`) — matching the existing
  `os.getenv(...)` constant style already used throughout `config.py`.
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

def flush_queue(max_items: int = config.SYNC_QUEUE_MAX_FLUSH_PER_CYCLE) -> None:
    """Attempt to push up to max_items queued payloads, oldest first.
    Stops at the first failure in this call (a dead network fails every
    later attempt identically, ADR-0015). Never raises."""
```

`agent/sync_queue.py`:

```python
def enqueue(payload: dict) -> None:
    """Append payload to the queue. If already at SYNC_QUEUE_MAX_SIZE,
    drops the oldest entry first (logged) to make room."""

def dequeue_batch(max_items: int) -> list[dict]:
    """Return up to max_items oldest queued payloads, without removing
    them — caller removes explicitly via remove() only after a confirmed
    push, or via record_failure() once an entry's attempts are exhausted."""

def remove(event_id: str) -> None:
    """Remove one entry (by event_id) after a confirmed successful push."""

def record_failure(event_id: str) -> None:
    """Increment one entry's attempts counter; drop it (logged) if this
    was its SYNC_QUEUE_MAX_ATTEMPTS-th failure."""
```

Called from `machine.py` as:

```python
if config.CLOUD_SYNC_ENABLED:
    payload = cloud_sync.build_payload(record, snapshot)
    if not cloud_sync.push(payload):
        sync_queue.enqueue(payload)
    cloud_sync.flush_queue()
```

`push()` differs from `notifier.notify()` in one important way: `notify()`
is allowed to raise and is caught by its caller; `push()` swallows *all*
exceptions internally and returns a `bool`, because unlike email delivery,
a failed cloud push is not something any caller needs to react to
synchronously — it needs to be queued, which `run_cycle()` does explicitly
above, not something `push()` does itself. `flush_queue()` follows the same
never-raises contract.

**Ordering is fixed and load-bearing:** the current cycle's own event is
always attempted first; backlog flush runs strictly after, using whatever
of the cycle's time budget remains. Backlog work must never delay or
preempt the current event's own push attempt (ADR-0015, Risks).

## Data model

### `POST /api/events` payload (unchanged shape, one new field)

Same record shape as F02's Table entity, minus `PartitionKey`/`RowKey`/
`received_at`/`blob_name` (server-assigned), **plus `event_id`
(ADR-0014, new — a ULID generated on the Pi, not the server)**, plus
`image_jpeg_b64` (only included when the decision escalated far enough that
a snapshot exists — `None`/omitted for a plain "static scene" cycle,
keeping those pushes tiny).

### `sync_queue.jsonl` (ADR-0015, new)

One JSON object per line, appended by `sync_queue.enqueue()`:

| Field | Type | Notes |
|---|---|---|
| `event_id` | string | ULID; same value used for every retry of this entry |
| `payload` | object | the exact `POST /api/events` body `build_payload()` produced |
| `attempts` | int | failed-push count so far; entry is dropped once this reaches `SYNC_QUEUE_MAX_ATTEMPTS` |
| `queued_at` | float | epoch seconds, for operator visibility only (not used for ordering — file order is queue order) |

Capped at `SYNC_QUEUE_MAX_SIZE` (20) entries; oldest dropped first to make
room for a new one once full. This file is a cloud-delivery worklist only —
`event.log` remains the complete, uncapped local record regardless of what
this queue drops (Tenet 2, unchanged in that respect).

## Step-by-step

1. `agent/config.py`: add the `CLOUD_SYNC_*` and `SYNC_QUEUE_*` constants
   and the two new optional `Settings` fields. Validate `CLOUD_SYNC_URL` is
   `https://` at config-load time — reject `http://` outright (Security,
   below).
2. `agent/cloud_sync.py`: implement `build_payload()` — pure function,
   easy to unit test without any network mocking. It now takes the
   `event_id` already present on `record` (generated in `machine.py`,
   see Current state) rather than minting its own.
3. `agent/sync_queue.py`: implement the JSONL-backed queue functions
   (`enqueue`/`dequeue_batch`/`remove`/`record_failure`), append-only,
   tolerant of a torn last line (ADR-0015, Risks) — skip a line that fails
   to `json.loads`, don't abort the whole read.
4. `agent/cloud_sync.py`: implement `push()` using `requests` with an
   **explicit** `timeout=CLOUD_SYNC_TIMEOUT_S` (never the default
   "wait forever") and `auth=(user, password)`. Implement `flush_queue()`
   on top of `sync_queue`'s functions per the Contracts above.
5. `tests/test_cloud_sync.py` and `tests/test_sync_queue.py`, mirroring
   `tests/test_notifier.py`'s fixture style:
   - `build_payload()` shape tests (with and without a snapshot;
     `event_id` present).
   - `push()` returns `True` on a mocked 2xx; `False` (not raises) on
     timeout, connection error, non-2xx, and on a missing
     `cloud_sync_user`/`cloud_sync_password`.
   - `flush_queue()`: pushes up to the per-cycle cap, stops at the first
     failure, removes successes, increments/drops on repeated failure.
   - `sync_queue`: cap eviction (oldest dropped when full), attempts
     tracking + drop-after-`SYNC_QUEUE_MAX_ATTEMPTS`, tolerant read of a
     corrupted/truncated last line.
   - Config validation: `CLOUD_SYNC_URL="http://..."` rejected at load
     time.
6. `agent/machine.py`: build the event record once (including `event_id`),
   pass it to `_log_event()`, `cloud_sync.build_payload()`, and — on push
   failure — `sync_queue.enqueue()`; call `cloud_sync.flush_queue()` after.
7. Run `tests/test_machine.py` — confirm no existing assertions on
   `Decision`/`PrefilterResult`/`VisionVerdict` break; `email_sent` and
   `event_id` are additive to the logged record only, never new fields on
   those dataclasses in `agent/types.py`.
8. Once T01's pipeline has deployed T04's retrofit + this task
   (`feat/dashboard` → `feat/azure-cd` merged/deployed): manually trigger a
   real wake cycle, confirm the event appears in the dashboard within 30s;
   then run the offline-then-reconnect check from
   `01-system-overview.md`'s Evaluation Framework (2-3 wakes with network
   disabled, confirm queuing, re-enable, confirm the backlog flushes).

## Risks

- **Timeout budget vs. SLO.** `(3, 5)` connect/read timeout bounds worst case
  at ~8s per push attempt, inside the ≤5s-added target only if the connect
  phase succeeds quickly; on a fully dead network the OS-level connect
  timeout dominates. Mitigation: `timeout=` is non-negotiable, and backlog
  flush is capped at 5 attempts/cycle and stops at the first failure
  (ADR-0015) so a dead network costs roughly one timeout, not six.
- **Missing cloud_sync credential** shouldn't be a crash — validated by a
  unit test asserting `push()` returns `False` and logs, rather than
  raising, when either the user or password is absent. A missing
  credential also means every push fails immediately, so the queue fills
  to its cap quickly — accepted; this is a configuration problem the owner
  needs to notice and fix, not something the queue should paper over
  indefinitely.
- **(ADR-0015, new) Queue file corruption** from a write interrupted by
  power loss. Mitigation: append-only writes, one JSON object per line, a
  torn final line is skipped on read rather than aborting the whole queue.
- **(ADR-0015, new) A permanently-broken queued entry blocking everything
  behind it.** Mitigation: `SYNC_QUEUE_MAX_ATTEMPTS` (5) — dropped and
  logged once exhausted, rather than retried forever at the head of the
  queue.

## Security

- Credential read from `~/.config/snn-agent/.env`, same file/permission
  pattern (`chmod 600`) as the four existing secrets — no new
  secrets-storage mechanism introduced on the Pi side. This is the *same*
  credential the owner types into the dashboard (ADR-0009) — the Pi is just
  another Basic Auth client of the one app.
- Sent as a standard `Authorization: Basic` header (via `requests`'
  built-in `auth=(user, password)` parameter), never logged (mirrors the
  existing rule in `config.py`'s `Settings` docstring: "Never log or print
  these values").
- Image bytes sent over HTTPS only; `CLOUD_SYNC_URL` must be validated to be
  `https://` at config-load time (reject `http://` outright — home security
  snapshots, and the Basic Auth credential itself, must not go over
  plaintext HTTP even accidentally).
- **(ADR-0015, new)** `sync_queue.jsonl` holds the same event data (and,
  when present, the base64 snapshot image) as `event.log` — no new
  sensitivity class introduced, same local file already holds this data
  today. No credential is ever written into the queue file; `push()` reads
  `cloud_sync_user`/`cloud_sync_password` from `Settings` at call time, not
  from the queued payload.

## Decisions

- ADR-0001 (push, not pull/poll).
- ADR-0006 (single combined POST with inline base64 image).
- ADR-0009 (shared Basic Auth credential, same one the dashboard uses).
- ADR-0014 (client-generated `event_id`) — what makes retrying from this
  queue safe.
- ADR-0015 (bounded local sync queue) — the queue design itself.

## Branch

`feat/dashboard` (task T03)
