# ADR-0001: Pi pushes events to the cloud; dashboard does not poll the Pi

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F03_pi_push_client

## Context

The dashboard needs to show event history "any time," but the Pi is fully
halted (not sleeping — actually powered down between wakes) for the large
majority of its life by explicit design (the whole point of the SNN
wake-trigger architecture is near-zero idle power). Any design where the
dashboard or backend initiates contact with the Pi assumes the Pi is
reachable, which is false almost all the time.

## Decision

The Pi pushes a compact event record (plus snapshot image) to a cloud HTTP
endpoint immediately after each wake-cycle decision, best-effort with a
bounded timeout. The dashboard and its backend never attempt to reach the Pi.

## Alternatives Considered

### Dashboard/backend polls or SSHes into the Pi on demand
- **Pros:** No new code path on the Pi at wake time; Pi stays "dumb."
- **Cons:** Fails outright — the Pi is unreachable (halted) essentially
  whenever a human would want to check the dashboard. This isn't a latency
  trade-off, it's a correctness failure against the stated goal.
- **Why not:** Directly contradicts the PR/FAQ's core promise ("viewable any
  time, even while the Pi is powered off").

### Pi uploads its whole event.log periodically (batch sync) instead of per-event push
- **Pros:** Fewer, larger requests; simpler dedup logic.
- **Cons:** "Periodically" requires the Pi to be awake on a schedule, which
  conflicts with the wake-only-on-trigger design; batching adds latency
  between "decision happened" and "visible on dashboard" for no benefit,
  since each wake cycle is already a natural, infrequent batch boundary of
  size one.
- **Why not:** A single wake cycle already produces exactly one event; there
  is no batching opportunity to exploit, only complexity to add.

## Consequences

### Positive
- Dashboard reflects reality "any time" as promised, independent of Pi power
  state.
- The push is naturally rate-limited to "once per wake cycle" — no risk of a
  runaway polling loop or unbounded request volume.

### Negative
- No delivery guarantee — a push that fails (network down at wake time) is
  simply lost from the cloud view; `event.log` on the Pi remains the only
  complete record (see PR/FAQ Tenet 2, accepted explicitly).

### Risks (with mitigation)
- **Risk:** Pi's clock drift could produce out-of-order `ts_wall` values.
  **Mitigation:** the Function assigns `received_at` server-side for
  ingestion-order tie-breaking; `ts_wall` is trusted for display but not for
  ordering guarantees.
