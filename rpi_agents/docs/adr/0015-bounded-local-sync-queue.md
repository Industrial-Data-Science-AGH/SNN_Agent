# ADR-0015: Bounded local sync queue, flushed on every wake cycle

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F03_pi_push_client, F01_ingest_api
- Supersedes: the PR/FAQ's original Tenet 2 ("best-effort telemetry... no
  retry queue, no store-and-forward, no delivery guarantees") and removes
  "Retry/outbox queue for failed pushes (Tenet 2)" from the PR/FAQ's Out of
  scope (MVP) list. Both are updated in `00-prfaq.md` alongside this ADR.

## Context

The original design (Tenet 2; ADR-0001 Consequences) accepted that a failed
push is "simply lost from the cloud view" — `event.log` remains the
complete local record, but the cloud copy silently goes stale for the
duration of any offline stretch, with no attempt to catch up afterward. The
owner has asked for this to change: the Pi should retain events it couldn't
push and deliver them once connectivity returns, checked at least once per
wake cycle.

Every wake cycle on this hardware is already a literal fresh boot, not a
resume from sleep — `deploy/snn-agent.service` is a `systemd` **oneshot**
unit that fires on boot, and `agent/power.py::resleep()` calls `sudo halt`
(full power-down) at the end of every cycle. There is no "boot" event
distinct from "wake" to hook separately — "synced on powerup," as requested,
and "attempted once per wake cycle" are the same thing on this system.

## Decision

`agent/cloud_sync.py` gains a small bounded local queue,
`sync_queue.jsonl` (JSON Lines, one pending event per line, written under
`config.VAR_DIR` alongside `event.log`). Each line is the same payload
`build_payload()` produces (now including the client-generated `event_id`,
ADR-0014) plus an `attempts: int` counter.

Every wake cycle, after the current cycle's own event has been pushed (or
queued, if that push failed):

1. Read `sync_queue.jsonl`, oldest entry first.
2. Attempt to push up to **5** queued entries this cycle. Stop at the first
   failure in the loop — a genuinely offline network fails every subsequent
   attempt identically, so continuing only spends timeout budget for
   nothing (Tenet 1).
3. On success, remove that entry from the queue. On failure, increment its
   `attempts`; if `attempts` has now reached **5**, drop the entry (log an
   error) rather than let one permanently-broken record block every future
   cycle's flush attempt — the queue always processes oldest-first, so a
   stuck head-of-line entry would otherwise wedge the entire backlog
   indefinitely.

The queue is capped at **20** pending entries total. When a new entry would
exceed the cap, the oldest queued entry is dropped (logged) to make room —
`event.log` remains the authoritative, uncapped local record regardless
(Tenet 2, unchanged in this respect); the queue is only a cloud-delivery
worklist, never a second copy of history.

## Alternatives Considered

### Unbounded backlog, flush the entire backlog every cycle
- **Pros:** conceptually simple ("keep everything until it's confirmed"),
  never silently drops a queued event.
- **Cons:** a long offline stretch (say, a multi-week Wi-Fi outage) could
  accumulate hundreds of entries; the wake cycle that finally reconnects
  would try to flush all of them in one pass, which could blow well past
  Tenet 1's bounded-latency requirement depending on how many succeed vs.
  time out before `power.resleep()` needs to run.
- **Why not:** raised explicitly and rejected by the owner in favor of a
  bounded backlog + bounded per-cycle flush count.

### A separate boot-time sync hook, distinct from the wake-cycle path
- **Pros:** conceptually separates "sync" from "decide."
- **Cons:** no such distinction exists on this hardware — every wake *is* a
  fresh boot (see Context). A separate hook would be the same code path
  behind extra indirection, not a different one.
- **Why not:** nothing to gain; folding backlog-flush into the existing
  wake-cycle push step is the only place this makes sense here.

### SQLite-backed queue instead of a JSONL file
- **Pros:** transactional removal, no risk of a torn write corrupting an entry.
- **Cons:** a new dependency and schema on a device that already has a
  working append-only JSONL convention (`event.log`) this can mirror
  exactly; at a few events/day, JSONL's simplicity outweighs SQLite's
  transactional guarantees.
- **Why not:** consistency with the existing `event.log` pattern and this
  project's established "simplest thing that works at this volume"
  reasoning (ADR-0007, ADR-0009, ADR-0011).

## Consequences

### Positive
- The dashboard eventually reflects events pushed during an offline
  stretch, once the Pi reconnects — closes the gap the original Tenet 2
  explicitly accepted.
- Bounded per-cycle flush count (5) keeps each cycle's added latency
  predictable even immediately after a long outage.
- Bounded queue size (20) and per-entry attempt cap (5) bound both memory
  and "stuck forever" failure modes.
- No new storage dependency on the Pi (same JSONL convention as `event.log`).

### Negative
- Tenet 2 is no longer strictly "no retry queue, no store-and-forward" —
  this ADR is the explicit record of that revision; `00-prfaq.md` is
  updated in the same change, not left describing stale behavior.
- The queue file is another artifact that can, in principle, suffer a
  torn write on power loss mid-write — the same class of risk `event.log`
  already carries and already accepts.
- A sufficiently long offline stretch (more than 20 pending events) still
  silently drops the oldest queued entries — this is a *bounded*, not
  unlimited, durability improvement; genuinely unbounded store-and-forward
  remains explicitly out of scope.

## Risks (with mitigation)

- **Risk:** queue file corruption from a write interrupted by power loss
  (a wake cycle calls `sudo halt` at its end; an earlier failure could in
  principle race a write). **Mitigation:** append-only writes, one JSON
  object per line, mirroring `event.log`'s existing convention — a torn
  final line is detected and skipped on read rather than aborting the
  whole queue.
- **Risk:** a single malformed or permanently-rejected (e.g. persistent
  `422`) queued entry blocks every entry behind it forever, since the queue
  is processed oldest-first and a cycle stops at the first failure.
  **Mitigation:** the per-entry `attempts` cap (5) — once exhausted, the
  entry is dropped and logged rather than retried indefinitely.
- **Risk:** the current cycle's own event competes with backlog flush for
  the same cycle's bounded time budget. **Mitigation:** ordering is fixed —
  the current event is always pushed (or queued) first; backlog flush only
  runs afterward, using whatever budget remains, never the reverse. The
  current event's own delivery is never delayed by backlog work.

## Decisions

- ADR-0001 (push, not pull) — this ADR narrows, not replaces, "no delivery
  guarantee": delivery is now *bounded-eventually-consistent*, not instant
  and not infinite.
- ADR-0014 (client-generated `event_id`) — what makes retrying from this
  queue safe.
- ADR-0006 (single combined POST) — unchanged; each queued entry is still
  one self-contained request.
