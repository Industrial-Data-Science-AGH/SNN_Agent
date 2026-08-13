"""Tests for cloud/app/metrics.py's aggregation functions and the GET
/api/metrics route (T05 plan, Step 6 + acceptance gate; ADR-0016). The
pure-function tests need no FastAPI app or Storage account, mirroring
tests/test_dashboard.py's existing pattern for compute_metrics(). Route
tests monkeypatch storage.list_events_for_metrics().
"""

import pytest
from fastapi.testclient import TestClient

from cloud.app import metrics, storage
from cloud.app.main import app

client = TestClient(app)


def _event(**overrides) -> dict:
    base = {
        "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "alarm": False,
        "escalate": False,
        "email_sent": False,
        "vision_source": None,
        "window_broken": None,
        "woken_by_trigger": True,  # every wake IS an SNN trigger in the deployed system
        "ts_wall": 1784048796.83,  # 2026-07-14T13:...Z
        "received_at": 1784048801.0,
        "latency_s": 10.0,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _fixed_credentials(monkeypatch):
    monkeypatch.setenv("DASHBOARD_USER", "testuser")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "testpass")


def _auth() -> tuple[str, str]:
    return ("testuser", "testpass")


# --- summary_metrics (moved from routes_dashboard.py, ADR-0016) ---


def test_summary_metrics_matches_compute_metrics_delegate():
    """The delegate in routes_dashboard.py must behave identically —
    this is the single-source-of-truth check for that refactor.
    """
    from cloud.app.routes_dashboard import compute_metrics

    events = [
        _event(alarm=True, email_sent=True),
        _event(escalate=True, alarm=False),
        _event(escalate=False),
    ]
    assert metrics.summary_metrics(events) == compute_metrics(events)


def test_summary_metrics_gemini_call_success_rate():
    """2026-07-15 addition: of every event that actually attempted a vision
    call, what fraction got a real Gemini verdict vs. fell back to
    failsafe (agent/machine.py: an exception/timeout during
    vision.verdict() sets source="failsafe", never "gemini").
    Non-escalating events (vision_source=None) never attempted a call and
    must not count in either the numerator or the denominator.
    """
    events = [
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="gemini"),
        _event(escalate=True, alarm=True, vision_source="failsafe"),
        _event(escalate=True, alarm=True, vision_source="failsafe"),
        _event(escalate=False, vision_source=None),  # never attempted -- excluded
    ]
    assert metrics.summary_metrics(events)["gemini_call_success_rate"] == pytest.approx(0.8)


def test_summary_metrics_window_break_confirmation_rate():
    """2026-07-15 addition (ADR-0017): of every event with a REAL Gemini
    verdict, what fraction did Gemini classify as showing a broken window.
    Failsafe verdicts are excluded even though window_broken defaults to
    True there -- that default is a conservative fail-open value, not an
    actual visual confirmation, and must not inflate this rate.
    """
    events = [
        _event(escalate=True, alarm=True, vision_source="gemini", window_broken=True),
        _event(escalate=True, alarm=True, vision_source="gemini", window_broken=True),
        _event(escalate=True, alarm=True, vision_source="gemini", window_broken=True),
        _event(escalate=True, alarm=False, vision_source="gemini", window_broken=False),
        # failsafe: window_broken=True (fail-open default) but must be excluded
        _event(escalate=True, alarm=True, vision_source="failsafe", window_broken=True),
        _event(escalate=False, vision_source=None, window_broken=None),  # excluded
    ]
    assert metrics.summary_metrics(events)["window_break_confirmation_rate"] == pytest.approx(0.75)


def test_summary_metrics_window_break_confirmation_rate_zero_when_no_gemini_verdicts():
    events = [
        _event(escalate=True, alarm=True, vision_source="failsafe", window_broken=True),
        _event(escalate=False, vision_source=None, window_broken=None),
    ]
    assert metrics.summary_metrics(events)["window_break_confirmation_rate"] == 0.0


def test_summary_metrics_gemini_call_success_rate_zero_when_no_attempts():
    events = [_event(escalate=False, vision_source=None)]
    assert metrics.summary_metrics(events)["gemini_call_success_rate"] == 0.0


# --- daily_rollup ---


def test_daily_rollup_on_empty_list():
    assert metrics.daily_rollup([]) == []


def test_daily_rollup_buckets_by_utc_date_oldest_first():
    day1 = 1783900800.0  # 2026-07-13T00:00:00Z
    day2 = 1783987200.0  # 2026-07-14T00:00:00Z
    events = [
        _event(ts_wall=day2, alarm=True),
        _event(ts_wall=day1, escalate=True, alarm=False),
        _event(ts_wall=day1, escalate=False),
    ]
    rollup = metrics.daily_rollup(events)
    assert [row["date"] for row in rollup] == ["2026-07-13", "2026-07-14"]
    assert rollup[0] == {
        "date": "2026-07-13",
        "real_wakes": 0,
        "false_wakes": 1,
        "non_escalating_wakes": 1,
        "total": 2,
    }
    assert rollup[1]["real_wakes"] == 1
    assert rollup[1]["total"] == 1


# --- vision_source_breakdown (ADR-0016: operational, not ground-truth) ---


def test_vision_source_breakdown_on_empty_list():
    breakdown = metrics.vision_source_breakdown([])
    assert breakdown == {
        "real_wakes": {"gemini": 0, "failsafe": 0, "none": 0},
        "false_wakes": {"gemini": 0, "failsafe": 0, "none": 0},
        "non_escalating_wakes": {"gemini": 0, "failsafe": 0, "none": 0},
    }


def test_vision_source_breakdown_cross_tabs_outcome_and_source():
    events = [
        _event(alarm=True, vision_source="gemini"),
        _event(alarm=True, vision_source="failsafe"),
        _event(escalate=True, alarm=False, vision_source="gemini"),
        _event(escalate=False, vision_source=None),  # non-escalating: no vision call
    ]
    breakdown = metrics.vision_source_breakdown(events)
    assert breakdown["real_wakes"] == {"gemini": 1, "failsafe": 1, "none": 0}
    assert breakdown["false_wakes"] == {"gemini": 1, "failsafe": 0, "none": 0}
    assert breakdown["non_escalating_wakes"] == {"gemini": 0, "failsafe": 0, "none": 1}


# --- trigger_breakdown (2026-07-15 addition, ADR-0016 addendum) ---


def test_trigger_breakdown_on_empty_list():
    breakdown = metrics.trigger_breakdown([])
    assert breakdown == {
        "triggered": {"real_wakes": 0, "false_wakes": 0, "non_escalating_wakes": 0},
        "not_triggered": {"real_wakes": 0, "false_wakes": 0, "non_escalating_wakes": 0},
    }


def test_trigger_breakdown_all_triggered_is_the_common_case():
    """The deployed system wakes only via the SNN hardware trigger --
    normal event streams should have everything under "triggered"."""
    events = [
        _event(escalate=True, alarm=True, woken_by_trigger=True),
        _event(escalate=False, woken_by_trigger=True),
    ]
    breakdown = metrics.trigger_breakdown(events)
    assert breakdown["triggered"] == {
        "real_wakes": 1,
        "false_wakes": 0,
        "non_escalating_wakes": 1,
    }
    assert breakdown["not_triggered"] == {
        "real_wakes": 0,
        "false_wakes": 0,
        "non_escalating_wakes": 0,
    }


def test_trigger_breakdown_flags_a_non_trigger_wake_as_an_anomaly():
    """A woken_by_trigger=False event (manual/dev boot reaching the cloud)
    lands under "not_triggered" -- distinguishable from a real SNN latch,
    not merged into the same bucket."""
    events = [
        _event(escalate=False, woken_by_trigger=True),
        _event(escalate=False, woken_by_trigger=False),
    ]
    breakdown = metrics.trigger_breakdown(events)
    assert breakdown["triggered"]["non_escalating_wakes"] == 1
    assert breakdown["not_triggered"]["non_escalating_wakes"] == 1


# --- last_sync (2026-07-15 addition) ---


def test_last_sync_on_empty_list_is_none():
    assert metrics.last_sync([]) is None


def test_last_sync_picks_the_most_recent_by_received_at():
    events = [
        _event(event_id="A", ts_wall=100.0, received_at=105.0),
        _event(event_id="B", ts_wall=300.0, received_at=310.0),  # most recent
        _event(event_id="C", ts_wall=200.0, received_at=205.0),
    ]
    assert metrics.last_sync(events) == {
        "event_id": "B",
        "ts_wall": 300.0,
        "received_at": 310.0,
    }


# --- _confusion / review_accuracy (2026-07-15 addition, ADR-0018) ---


def test_confusion_on_empty_list_is_all_zero():
    result = metrics._confusion(
        [], predicted_key="window_broken", confirmed_key="window_broken_confirmed"
    )
    assert result == {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "accuracy": 0.0}


def test_confusion_covers_all_four_cells():
    events = [
        _event(window_broken=True, window_broken_confirmed=True),  # TP
        _event(window_broken=True, window_broken_confirmed=False),  # FP
        _event(window_broken=False, window_broken_confirmed=True),  # FN
        _event(window_broken=False, window_broken_confirmed=False),  # TN
    ]
    result = metrics._confusion(
        events, predicted_key="window_broken", confirmed_key="window_broken_confirmed"
    )
    assert result == {"tp": 1, "fp": 1, "tn": 1, "fn": 1, "accuracy": 0.5}


def test_review_accuracy_on_empty_list():
    zero = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "accuracy": 0.0}
    assert metrics.review_accuracy([]) == {
        "reviewed_count": 0,
        "window_broken": zero,
        "intrusion": zero,
    }


def test_review_accuracy_excludes_failsafe_verdicts():
    """A failsafe verdict has no real prediction to score -- Gemini never
    actually judged the frame -- so even a reviewed failsafe event must not
    count toward reviewed_count.
    """
    events = [
        _event(
            vision_source="failsafe",
            window_broken=True,
            is_intrusion=True,
            window_broken_confirmed=True,
            intrusion_confirmed=True,
            reviewed_at=123.0,
        ),
    ]
    assert metrics.review_accuracy(events)["reviewed_count"] == 0


def test_review_accuracy_excludes_unreviewed_events():
    events = [
        _event(vision_source="gemini", window_broken=True, is_intrusion=True, reviewed_at=None),
    ]
    assert metrics.review_accuracy(events)["reviewed_count"] == 0


def test_review_accuracy_scores_only_reviewed_gemini_verdicts():
    events = [
        _event(
            vision_source="gemini",
            window_broken=True,
            is_intrusion=False,
            window_broken_confirmed=True,
            intrusion_confirmed=False,
            reviewed_at=100.0,
        ),
        _event(
            vision_source="gemini",
            window_broken=False,
            is_intrusion=True,
            window_broken_confirmed=True,  # Gemini missed this one (FN)
            intrusion_confirmed=False,  # Gemini false-positived this one (FP)
            reviewed_at=200.0,
        ),
        # not reviewed -- excluded
        _event(vision_source="gemini", window_broken=True, is_intrusion=True),
        # failsafe -- excluded even though reviewed
        _event(
            vision_source="failsafe",
            window_broken=True,
            is_intrusion=True,
            window_broken_confirmed=True,
            intrusion_confirmed=True,
            reviewed_at=300.0,
        ),
    ]
    result = metrics.review_accuracy(events)
    assert result["reviewed_count"] == 2
    assert result["window_broken"] == {"tp": 1, "fp": 0, "tn": 0, "fn": 1, "accuracy": 0.5}
    assert result["intrusion"] == {"tp": 0, "fp": 1, "tn": 1, "fn": 0, "accuracy": 0.5}


# --- latency_stats ---


def test_latency_stats_on_empty_list():
    assert metrics.latency_stats([]) == {"avg": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0}


def test_latency_stats_single_event():
    stats = metrics.latency_stats([_event(latency_s=5.0)])
    assert stats == {"avg": 5.0, "p50": 5.0, "p95": 5.0, "max": 5.0}


def test_latency_stats_avg_and_max_over_multiple_events():
    events = [_event(latency_s=v) for v in (2.0, 4.0, 6.0, 8.0, 10.0)]
    stats = metrics.latency_stats(events)
    assert stats["avg"] == pytest.approx(6.0)
    assert stats["max"] == 10.0
    assert stats["p50"] == pytest.approx(6.0)
    # p95 of [2,4,6,8,10] via linear interpolation between closest ranks:
    # rank = (5-1)*0.95 = 3.8 -> interpolate between index 3 (8.0) and 4 (10.0)
    assert stats["p95"] == pytest.approx(9.6)


# --- route-level: GET /api/metrics ---


def test_get_metrics_returns_200_with_expected_shape(monkeypatch):
    # escalate=True on the alarm=True event: mirrors machine.py's actual
    # invariant (alarm can only become True inside the escalate branch) --
    # same discipline test_dashboard.py's fixtures already follow.
    events = [
        _event(
            escalate=True, alarm=True, email_sent=True, vision_source="gemini", window_broken=True
        ),
        _event(escalate=True, alarm=False, vision_source="gemini", window_broken=False),
        _event(escalate=False, vision_source=None),
    ]
    monkeypatch.setattr(storage, "list_events_for_metrics", lambda since=None: events)

    response = client.get("/api/metrics", auth=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["summary"] == {
        "real_wakes": 1,
        "false_wakes": 1,
        "non_escalating_wakes": 1,
        "email_delivery_rate": 1.0,
        "gemini_call_success_rate": 1.0,
        "window_break_confirmation_rate": 0.5,
    }
    assert body["vision_source_breakdown"]["real_wakes"]["gemini"] == 1
    assert body["trigger_breakdown"]["triggered"]["real_wakes"] == 1
    assert body["trigger_breakdown"]["not_triggered"] == {
        "real_wakes": 0,
        "false_wakes": 0,
        "non_escalating_wakes": 0,
    }
    assert body["last_sync"]["event_id"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert body["review_accuracy"]["reviewed_count"] == 0  # none of the fixture events reviewed
    assert set(body["review_accuracy"]) == {"reviewed_count", "window_broken", "intrusion"}
    assert "daily" in body and isinstance(body["daily"], list)
    assert set(body["latency_s"]) == {"avg", "p50", "p95", "max"}
    assert "since" in body and "until" in body


def test_get_metrics_empty_window_does_not_error(monkeypatch):
    monkeypatch.setattr(storage, "list_events_for_metrics", lambda since=None: [])

    response = client.get("/api/metrics", auth=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["summary"]["real_wakes"] == 0
    assert body["daily"] == []
    assert body["latency_s"]["avg"] == 0.0
    assert body["last_sync"] is None
    assert body["review_accuracy"]["reviewed_count"] == 0


def test_get_metrics_respects_since_query_param(monkeypatch):
    captured = {}

    def _fake_list(since=None):
        captured["since"] = since
        return []

    monkeypatch.setattr(storage, "list_events_for_metrics", _fake_list)

    response = client.get("/api/metrics?since=2026-01-01", auth=_auth())

    assert response.status_code == 200
    assert captured["since"] == "2026-01-01"
