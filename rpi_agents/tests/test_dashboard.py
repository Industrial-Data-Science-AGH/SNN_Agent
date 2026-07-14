"""Tests for the metrics-band computation in routes_dashboard.py (F04
design, "Summary metrics band"; T02 plan, Step 5 + acceptance gate). Pure
function over a plain list of dicts — no FastAPI app, no Storage.
"""

from cloud.app.routes_dashboard import compute_metrics


def _event(**overrides) -> dict:
    base = {"alarm": False, "escalate": False, "email_sent": False}
    base.update(overrides)
    return base


def test_compute_metrics_on_empty_list():
    assert compute_metrics([]) == {
        "real_wakes": 0,
        "false_wakes": 0,
        "non_escalating_wakes": 0,
        "email_delivery_rate": 0.0,
    }


def test_real_wakes_counts_alarm_true():
    events = [_event(alarm=True), _event(alarm=True), _event(alarm=False)]
    assert compute_metrics(events)["real_wakes"] == 2


def test_false_wakes_counts_escalate_true_and_alarm_false():
    events = [
        _event(escalate=True, alarm=False),
        _event(escalate=True, alarm=True),
        _event(escalate=False, alarm=False),
    ]
    assert compute_metrics(events)["false_wakes"] == 1


def test_non_escalating_wakes_counts_escalate_false():
    events = [_event(escalate=False), _event(escalate=False), _event(escalate=True)]
    assert compute_metrics(events)["non_escalating_wakes"] == 2


def test_email_delivery_rate_is_fraction_of_real_wakes():
    # email_sent is only ever True on a real wake in practice (the Pi only
    # calls notifier.notify() when alarm fires) — this fixture reflects
    # that invariant rather than an impossible alarm=False/email_sent=True
    # combination.
    events = [
        _event(alarm=True, email_sent=True),
        _event(alarm=True, email_sent=False),
    ]
    assert compute_metrics(events)["email_delivery_rate"] == 0.5


def test_email_delivery_rate_is_zero_when_no_real_wakes():
    events = [_event(alarm=False), _event(alarm=False)]
    assert compute_metrics(events)["email_delivery_rate"] == 0.0
