"""Pure aggregation functions over already-fetched event dicts (F04 design,
"Summary metrics band"; ADR-0016). One implementation, consumed by both the
HTML dashboard's metrics band (`routes_dashboard.py::compute_metrics()`,
which now delegates here) and the JSON `GET /api/metrics` route
(`routes_api.py`) — no drift between the two.

Every function takes a `list[dict]` in the shape
`storage._to_summary_dict()` produces (or the Table entity dict itself, for
the fields this module reads: `alarm`, `escalate`, `email_sent`,
`vision_source`, `ts_wall`, `latency_s`) and returns plain JSON-serializable
data. No FastAPI/Storage imports here, deliberately — independently
testable without a Table Storage account or a running app, same pattern the
original `compute_metrics()` used.
"""

import math
from datetime import UTC, datetime
from typing import Literal

Outcome = Literal["real_wakes", "false_wakes", "non_escalating_wakes"]

_OUTCOMES: tuple[Outcome, ...] = ("real_wakes", "false_wakes", "non_escalating_wakes")
_VISION_SOURCES = ("gemini", "failsafe", "none")


def _outcome(event: dict) -> Outcome:
    """Classify one event into the same three buckets F04's metrics band
    has always used (real / false / non-escalating wake).
    """
    if event["alarm"]:
        return "real_wakes"
    if event["escalate"]:
        return "false_wakes"
    return "non_escalating_wakes"


def summary_metrics(events: list[dict]) -> dict:
    """Summary metrics band (F04 design, "Summary metrics band").

    Unchanged from the original `routes_dashboard.py::compute_metrics()` —
    moved here (ADR-0016) so the JSON route and the HTML band share one
    implementation. `gemini_call_success_rate` *(2026-07-15 addition, ADR-
    0016 addendum)* is new: of every event that actually attempted a vision
    call (`vision_source` is `"gemini"` or `"failsafe"` — non-escalating
    events never call vision at all and are excluded), what fraction
    returned a real Gemini verdict rather than falling back to failsafe
    (`agent/machine.py`: a `vision.verdict()` exception/timeout sets
    `source="failsafe"`, not a `"gemini"` value). Same "rate over a subset,
    0.0 when the subset is empty" pattern as `email_delivery_rate`.
    `window_break_confirmation_rate` *(2026-07-15, ADR-0017)* is also new:
    of every event with a *real* Gemini verdict (`vision_source ==
    "gemini"` specifically — failsafe verdicts are excluded, since their
    `window_broken=True` is a conservative fail-open default, not an actual
    visual confirmation), what fraction did Gemini classify as showing a
    broken window. This is the metric that most directly validates the
    SNN's own detection target (acoustic glass-break), as opposed to
    `vision_source_breakdown`'s more general outcome/vision-path view.
    """
    real_wakes = sum(1 for event in events if event["alarm"])
    false_wakes = sum(1 for event in events if event["escalate"] and not event["alarm"])
    non_escalating = sum(1 for event in events if not event["escalate"])
    email_sent_count = sum(1 for event in events if event["email_sent"])
    email_delivery_rate = email_sent_count / real_wakes if real_wakes else 0.0
    vision_attempts = sum(
        1 for event in events if event.get("vision_source") in ("gemini", "failsafe")
    )
    gemini_successes = sum(1 for event in events if event.get("vision_source") == "gemini")
    gemini_call_success_rate = gemini_successes / vision_attempts if vision_attempts else 0.0
    gemini_verdicts = [event for event in events if event.get("vision_source") == "gemini"]
    windows_confirmed_broken = sum(1 for event in gemini_verdicts if event.get("window_broken"))
    window_break_confirmation_rate = (
        windows_confirmed_broken / len(gemini_verdicts) if gemini_verdicts else 0.0
    )
    return {
        "real_wakes": real_wakes,
        "false_wakes": false_wakes,
        "non_escalating_wakes": non_escalating,
        "email_delivery_rate": email_delivery_rate,
        "gemini_call_success_rate": gemini_call_success_rate,
        "window_break_confirmation_rate": window_break_confirmation_rate,
    }


def daily_rollup(events: list[dict]) -> list[dict]:
    """Per-UTC-date counts, oldest date first (a trend chart reads
    left-to-right chronologically — the opposite order from `GET
    /api/events`'s newest-first list, which is a deliberate difference, not
    an inconsistency; see F01 design, `GET /api/metrics`).
    """
    by_date: dict[str, dict] = {}
    for event in events:
        date = datetime.fromtimestamp(event["ts_wall"], tz=UTC).strftime("%Y-%m-%d")
        bucket = by_date.setdefault(
            date,
            {
                "date": date,
                "real_wakes": 0,
                "false_wakes": 0,
                "non_escalating_wakes": 0,
                "total": 0,
            },
        )
        bucket[_outcome(event)] += 1
        bucket["total"] += 1
    return [by_date[date] for date in sorted(by_date)]


def vision_source_breakdown(events: list[dict]) -> dict:
    """Outcome x vision_source cross-tab — an operational/model-agreement
    breakdown, **not** a ground-truth confusion matrix (ADR-0016, F04
    design Risks: this system never captures whether an event was actually
    an intrusion, only what the pipeline itself decided).

    `vision_source` is `None` for non-escalating events (no vision call was
    ever made) and is normalized to the string `"none"` here since JSON
    object keys can't be `null`.
    """
    breakdown = {outcome: dict.fromkeys(_VISION_SOURCES, 0) for outcome in _OUTCOMES}
    for event in events:
        source = event.get("vision_source") or "none"
        breakdown[_outcome(event)][source] += 1
    return breakdown


def trigger_breakdown(events: list[dict]) -> dict:
    """Outcome x `woken_by_trigger` cross-tab (2026-07-15 addition, ADR-0016
    addendum) — distinguishes genuine SNN-hardware-latched wakes
    (`agent/trigger.py`: `WAKE_CONFIRM_PIN` low means the Arduino actually
    latched a trigger) from any other reason the Pi came up (manual/dev
    boot). The deployed system is designed so every wake *is* an SNN
    trigger (README: "wakes on an SNN hardware trigger") — so a nonzero
    `not_triggered` count is itself an anomaly signal worth noticing, not
    just descriptive data.

    This is also the answer to "does a high non-escalating count mean
    something is wrong?": no. The SNN is a deliberately cheap, sensitive
    first-stage sensor — like a smoke detector, it is *expected* to latch
    on plenty of things that aren't real intrusions (wind-driven vibration,
    ambient noise with a spike signature similar to glass breaking, etc).
    That's exactly why the prefilter and vision stages exist downstream: a
    funnel shape where triggers vastly outnumber confirmed alarms is the
    system working as intended, not a defect.
    """
    breakdown = {
        "triggered": dict.fromkeys(_OUTCOMES, 0),
        "not_triggered": dict.fromkeys(_OUTCOMES, 0),
    }
    for event in events:
        key = "triggered" if event.get("woken_by_trigger") else "not_triggered"
        breakdown[key][_outcome(event)] += 1
    return breakdown


def last_sync(events: list[dict]) -> dict | None:
    """The most recent event in the queried window, by `received_at` (when
    the cloud actually got the push, not `ts_wall`'s Pi-side wake time) —
    lets the dashboard show "last synced X ago" (2026-07-15 addition).

    `None` when `events` is empty. Deliberately scoped to the *same* window
    the rest of `GET /api/metrics` already queried (`since`), not a second
    unbounded Table scan: if the Pi hasn't synced within the selected
    window, "no sync in this window" (a `None` here) is itself the useful
    signal — the owner can widen `since` if they want to look further back.
    """
    if not events:
        return None
    latest = max(events, key=lambda event: event["received_at"])
    return {
        "event_id": latest["event_id"],
        "ts_wall": latest["ts_wall"],
        "received_at": latest["received_at"],
    }


def _confusion(events: list[dict], *, predicted_key: str, confirmed_key: str) -> dict:
    """TP/FP/TN/FN + accuracy of `predicted_key` (Gemini's classification)
    against `confirmed_key` (the owner's manually reviewed ground truth).
    """
    tp = fp = tn = fn = 0
    for event in events:
        predicted = bool(event.get(predicted_key))
        confirmed = bool(event.get(confirmed_key))
        if predicted and confirmed:
            tp += 1
        elif predicted and not confirmed:
            fp += 1
        elif not predicted and confirmed:
            fn += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "accuracy": (tp + tn) / total if total else 0.0}


def review_accuracy(events: list[dict]) -> dict:
    """Real confusion-matrix accuracy of Gemini's predictions against the
    owner's manually confirmed ground truth (2026-07-15, ADR-0018) — the
    accuracy measurement F04 design's Risks section and ADR-0016's addendum
    both noted this system couldn't compute until reviews existed
    (`PATCH /api/events/{event_id}`, `storage.review_event()`).

    Only counts events with BOTH a real Gemini verdict (`vision_source ==
    "gemini"` — a failsafe verdict has no real prediction to score) AND a
    completed review (`reviewed_at` is not `None` — an unreviewed event has
    no ground truth to compare against). Unlike `vision_source_breakdown`
    and `trigger_breakdown`, this genuinely measures correctness, not just
    an operational/agreement breakdown — it only exists because the owner
    can now supply ground truth, closing the gap those two metrics were
    explicit about not closing.
    """
    reviewed = [
        event
        for event in events
        if event.get("vision_source") == "gemini" and event.get("reviewed_at") is not None
    ]
    return {
        "reviewed_count": len(reviewed),
        "window_broken": _confusion(
            reviewed, predicted_key="window_broken", confirmed_key="window_broken_confirmed"
        ),
        "intrusion": _confusion(
            reviewed, predicted_key="is_intrusion", confirmed_key="intrusion_confirmed"
        ),
    }


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation-between-closest-ranks percentile (matches
    numpy's default `interpolation="linear"`) — avoids a numpy dependency
    in `cloud/app` for a single-use calculation.
    """
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return sorted_values[int(rank)]
    return sorted_values[lower] * (upper - rank) + sorted_values[upper] * (rank - lower)


def latency_stats(events: list[dict]) -> dict:
    """avg/p50/p95/max of `latency_s` across the window. All-zero on an
    empty list rather than dividing by zero.
    """
    values = sorted(event["latency_s"] for event in events)
    if not values:
        return {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "avg": sum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "max": values[-1],
    }
