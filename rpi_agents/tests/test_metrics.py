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
        "alarm": False,
        "escalate": False,
        "email_sent": False,
        "vision_source": None,
        "ts_wall": 1784048796.83,  # 2026-07-14T13:...Z
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
        _event(escalate=True, alarm=True, email_sent=True, vision_source="gemini"),
        _event(escalate=True, alarm=False, vision_source="gemini"),
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
    }
    assert body["vision_source_breakdown"]["real_wakes"]["gemini"] == 1
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


def test_get_metrics_respects_since_query_param(monkeypatch):
    captured = {}

    def _fake_list(since=None):
        captured["since"] = since
        return []

    monkeypatch.setattr(storage, "list_events_for_metrics", _fake_list)

    response = client.get("/api/metrics?since=2026-01-01", auth=_auth())

    assert response.status_code == 200
    assert captured["since"] == "2026-01-01"
