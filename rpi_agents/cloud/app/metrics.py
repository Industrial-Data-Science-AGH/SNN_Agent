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
    implementation.
    """
    real_wakes = sum(1 for event in events if event["alarm"])
    false_wakes = sum(1 for event in events if event["escalate"] and not event["alarm"])
    non_escalating = sum(1 for event in events if not event["escalate"])
    email_sent_count = sum(1 for event in events if event["email_sent"])
    email_delivery_rate = email_sent_count / real_wakes if real_wakes else 0.0
    return {
        "real_wakes": real_wakes,
        "false_wakes": false_wakes,
        "non_escalating_wakes": non_escalating,
        "email_delivery_rate": email_delivery_rate,
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
