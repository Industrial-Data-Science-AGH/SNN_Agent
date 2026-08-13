"""Tests for the manual ground-truth review feature (T06 plan; ADR-0018):
storage.review_event() and the PATCH /api/events/{event_id} route. Route
tests monkeypatch storage.* so no real Storage account is needed to run
them, mirroring tests/test_ingest.py's existing pattern.
"""

import pytest
from azure.data.tables import UpdateMode
from fastapi.testclient import TestClient

from cloud.app import storage
from cloud.app.main import app

client = TestClient(app)

_EVENT_ID = "01ARZ3NDEKTSV4RRFFQ69G5FAV"

_REVIEW_PAYLOAD = {"window_broken_confirmed": True, "intrusion_confirmed": False}


def _full_event(**overrides) -> dict:
    """A dict shaped like storage._to_summary_dict()'s output -- what
    storage.get_event()/review_event() return and schemas.EventDetail expects.
    """
    base = {
        "event_id": _EVENT_ID,
        "ts_wall": 1784048796.83,
        "woken_by_trigger": True,
        "escalate": True,
        "motion": True,
        "person": False,
        "score": 0.1547,
        "vision_source": "gemini",
        "is_intrusion": False,
        "window_broken": True,
        "alarm": False,
        "reason": "vision: ...",
        "email_sent": False,
        "latency_s": 10.67,
        "received_at": 1784048801.0,
        "image_url": None,
        "window_broken_confirmed": None,
        "intrusion_confirmed": None,
        "reviewed_at": None,
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _fixed_credentials(monkeypatch):
    monkeypatch.setenv("DASHBOARD_USER", "testuser")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "testpass")


def _auth() -> tuple[str, str]:
    return ("testuser", "testpass")


# --- storage.review_event() ---


def test_review_event_returns_none_when_entity_not_found(monkeypatch):
    monkeypatch.setattr(storage, "_find_entity_by_row_key", lambda event_id: None)

    result = storage.review_event(
        _EVENT_ID, window_broken_confirmed=True, intrusion_confirmed=False, reviewed_at=100.0
    )

    assert result is None


def test_review_event_merges_review_fields_onto_the_existing_entity(monkeypatch):
    entity = {"PartitionKey": "2026-07-14", "RowKey": _EVENT_ID, "alarm": False}
    monkeypatch.setattr(storage, "_find_entity_by_row_key", lambda event_id: entity)

    calls = {}

    class _FakeTableClient:
        def update_entity(self, mode, entity):
            calls["update"] = (mode, entity)

    monkeypatch.setattr(storage, "get_table_client", lambda: _FakeTableClient())
    monkeypatch.setattr(storage, "get_event", lambda event_id: _full_event())

    result = storage.review_event(
        _EVENT_ID, window_broken_confirmed=True, intrusion_confirmed=False, reviewed_at=555.0
    )

    mode, updated_entity = calls["update"]
    assert mode == UpdateMode.MERGE
    assert updated_entity == {
        "PartitionKey": "2026-07-14",
        "RowKey": _EVENT_ID,
        "window_broken_confirmed": True,
        "intrusion_confirmed": False,
        "reviewed_at": 555.0,
    }
    assert result == _full_event()


# --- route-level: PATCH /api/events/{event_id} ---


def test_submit_event_review_returns_200_with_updated_fields(monkeypatch):
    reviewed = _full_event(
        window_broken_confirmed=True, intrusion_confirmed=False, reviewed_at=999.0
    )
    monkeypatch.setattr(
        storage,
        "review_event",
        lambda event_id, *, window_broken_confirmed, intrusion_confirmed, reviewed_at: reviewed,
    )

    response = client.patch(f"/api/events/{_EVENT_ID}", json=_REVIEW_PAYLOAD, auth=_auth())

    assert response.status_code == 200
    body = response.json()
    assert body["window_broken_confirmed"] is True
    assert body["intrusion_confirmed"] is False
    assert body["reviewed_at"] == 999.0


def test_submit_event_review_returns_404_for_unknown_event(monkeypatch):
    monkeypatch.setattr(
        storage,
        "review_event",
        lambda event_id, *, window_broken_confirmed, intrusion_confirmed, reviewed_at: None,
    )

    response = client.patch("/api/events/does-not-exist", json=_REVIEW_PAYLOAD, auth=_auth())

    assert response.status_code == 404


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"window_broken_confirmed": True},  # missing intrusion_confirmed
        {"intrusion_confirmed": False},  # missing window_broken_confirmed
        # not a valid Pydantic-bool-coercible value (unlike "yes"/"true"/1)
        {"window_broken_confirmed": ["not", "a", "bool"], "intrusion_confirmed": False},
        {"window_broken_confirmed": True, "intrusion_confirmed": False, "extra": "nope"},
    ],
)
def test_submit_event_review_rejects_invalid_body_with_422(bad_payload, monkeypatch):
    """Validation happens before storage is ever reached -- track whether
    storage.review_event was called so a regression that skips validation
    shows up as a broken assertion, not a silent pass.
    """
    calls = []
    monkeypatch.setattr(
        storage, "review_event", lambda *a, **k: calls.append((a, k)) or _full_event()
    )

    response = client.patch(f"/api/events/{_EVENT_ID}", json=bad_payload, auth=_auth())

    assert response.status_code == 422
    assert calls == []


def test_submit_event_review_overwrites_a_prior_review(monkeypatch):
    """A second PATCH of the same event_id is a normal, supported
    correction -- not an error -- and the response reflects the latest
    submission, not the first.
    """
    calls = []

    def _fake_review_event(event_id, *, window_broken_confirmed, intrusion_confirmed, reviewed_at):
        calls.append((window_broken_confirmed, intrusion_confirmed))
        return _full_event(
            window_broken_confirmed=window_broken_confirmed,
            intrusion_confirmed=intrusion_confirmed,
            reviewed_at=reviewed_at,
        )

    monkeypatch.setattr(storage, "review_event", _fake_review_event)

    first = client.patch(
        f"/api/events/{_EVENT_ID}",
        json={"window_broken_confirmed": False, "intrusion_confirmed": False},
        auth=_auth(),
    )
    second = client.patch(
        f"/api/events/{_EVENT_ID}",
        json={"window_broken_confirmed": True, "intrusion_confirmed": True},
        auth=_auth(),
    )

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["window_broken_confirmed"] is True
    assert second.json()["intrusion_confirmed"] is True
    assert calls == [(False, False), (True, True)]
