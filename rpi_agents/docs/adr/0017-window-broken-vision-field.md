# ADR-0017: Gemini returns `window_broken` as a second, independent judgment

- Status: accepted
- Date: 2026-07-15
- Deciders: Wiktor
- Relates to: F01_ingest_api, F03_pi_push_client (agent/vision.py, P3)

## Context

The SNN hardware trigger (`agent/trigger.py`) exists specifically to detect
a candidate glass-break event — that's the whole point of this project. Once
the Pi wakes, the vision stage (`agent/vision.py`) sends the captured frame
to Gemini and asks a single, generic question: is this a real break-in or
intruder, vs. a false alarm? That question conflates two things that are
not the same: whether a window is visibly broken, and whether a person is
present. A window can show clear damage with no one currently in frame; a
person can be intruding through an already-open door with no broken glass
at all. The owner asked for a second, independent classification —
"is the window broken, yes or no" — specifically so the dashboard can
report on the thing the SNN itself is designed to detect, not just a
generic intrusion call.

## Decision

`agent/vision.py`'s Gemini prompt and structured-output schema gain a
second required boolean field, `window_broken`, asked as an explicit,
separate judgment from `is_intrusion`:

```
You are a home-security analyst reviewing a single frame captured after
a glass-break sensor triggered. Make two independent judgments:
1. window_broken: does this frame show visible evidence the window or
glass itself is broken (a shattered pane, a hole, cracks, glass shards
on the sill or floor)? Judge the window's physical state only -- not
whether anyone is present.
2. is_intrusion: does this frame show a real break-in or intruder, as
opposed to a false alarm (pet, headlights, curtain, empty scene, or a
window that looks broken but with no other sign of intrusion)?
Return JSON {"window_broken": bool, "is_intrusion": bool, "confidence": 0..1,
"reason": short string}. When uncertain on either judgment, prefer the
value that leads to an alert (window_broken=true or is_intrusion=true).
```

`VisionVerdict` (`agent/types.py`) gains a matching `window_broken: bool`
field. It defaults to `True` at the dataclass level purely so pre-existing
test fixtures that construct a `VisionVerdict` without caring about this
field keep compiling — every real code path sets it explicitly:
`_parse_verdict()` requires it in Gemini's response (a missing
`window_broken` raises `ValueError`, same as the other three fields, no
silent default in production), and both failsafe branches (`agent/
vision.py`'s own except-block and `agent/machine.py`'s outer one) set
`window_broken=True` explicitly — the same conservative fail-open value as
`is_intrusion`, since a failed vision call gives no real evidence either
way and this project's established philosophy (Tenet 1, `is_intrusion`'s
own fail-open behavior) is to assume the worst when uncertain.

**`window_broken` does not drive the alarm decision.** `is_intrusion`
remains the sole input to `Decision.alarm` in `agent/machine.py` — this is
a deliberate, conservative scope boundary: changing what triggers the
actual alarm/notify/actuator path is a safety-relevant decision the owner
did not ask for here, and folding a second signal into that path without
being asked would risk silently changing real-world alarm behavior (e.g.
suppressing an alarm because `window_broken=false` on a frame where an
intruder is nonetheless present). `window_broken` is additive and
informational: it flows into the local event log, the cloud push payload,
storage, and a new dashboard metric, but the alarm path is untouched.

The field flows end-to-end: `agent/vision.py` (Gemini call) →
`agent/machine.py::_build_record()` (local event log + cloud payload
source, `null` when no verdict exists — same nullability as
`is_intrusion`) → `agent/cloud_sync.py::_PAYLOAD_FIELDS` (pushed to the
cloud) → `cloud/app/schemas.py::EventIn`/`EventSummary` (ingest + read
contracts) → `cloud/app/storage.py` (Table entity field, same
None-is-dropped write semantics as every other nullable field) →
`cloud/app/metrics.py::summary_metrics()`'s new
`window_break_confirmation_rate` (`GET /api/metrics`, ADR-0016's
endpoint).

## Alternatives Considered

### Fold `window_broken` into the existing `is_intrusion` decision (e.g. `alarm = is_intrusion AND window_broken`)
- **Pros:** would make the alarm decision more specific to this project's
  actual glass-break use case.
- **Cons:** changes safety-relevant, already-tuned alarm behavior without
  being asked to; risks suppressing a genuine alarm on a frame where an
  intruder is present but glass damage isn't visible in that single frame
  (e.g. they're already inside, frame shows the room not the window).
- **Why not:** out of scope for this request, which was specifically "add
  this as a classification and a metric," not "change when the alarm
  fires." Revisit as its own explicit decision if the owner asks for it.

### A separate, second Gemini call dedicated to window-breakage only
- **Pros:** cleanly separates the two concerns at the API-call level.
- **Cons:** doubles Gemini API cost and latency per escalating event, for a
  judgment the same single call can already make reliably from the same
  frame.
- **Why not:** no evidence a single combined call produces worse judgments
  than two separate ones for this task; not worth the added cost/latency
  budget (Tenet 1's bounded-latency requirement) without a demonstrated
  need.

## Consequences

### Positive
- The dashboard can report on the SNN's actual detection target (window
  breakage) rather than only a generic intrusion call.
- No new Gemini API cost — same single call, one more field in the
  response schema.
- Alarm/notify/actuator behavior is provably unchanged (no code in that
  path reads `window_broken`).

### Negative
- `VisionVerdict`'s default (`window_broken: bool = True`) means a
  hand-constructed `VisionVerdict` in test code that forgets to set this
  field silently gets `True` rather than failing loudly — a deliberate
  trade-off to avoid rewriting every pre-existing `VisionVerdict(...)` call
  site in `tests/test_machine.py`, acceptable because no production code
  path relies on the default (see Decision).
- One more field for the Pi and the cloud schema to keep in sync
  (`_PAYLOAD_FIELDS`, `EventIn`, `EventSummary`) — same maintenance
  pattern as every other event field already established by F01/F03.

## Risks (with mitigation)

- **Risk:** Gemini's `window_broken` judgment could be wrong in either
  direction (misses real damage, or flags intact glass as broken) just
  like `is_intrusion` already can be. **Mitigation:** none needed beyond
  what already exists — this field doesn't gate the alarm, so a wrong
  `window_broken` value affects a dashboard metric, not physical-world
  alarm behavior.
- **Risk:** someone later wires `window_broken` into the alarm decision
  without going through the same deliberation this ADR documents.
  **Mitigation:** this ADR's Decision section states the boundary
  explicitly; a future change to `Decision.alarm`'s inputs should reference
  and update this ADR, not silently diverge from it.

## Decisions

- ADR-0001 (Tenet 1: the Pi's own contract, including the alarm path,
  comes first — this ADR's scope boundary is a direct application of that).
- ADR-0016 (`GET /api/metrics`) — `window_break_confirmation_rate` lands
  in the same endpoint and `summary_metrics()` function.
