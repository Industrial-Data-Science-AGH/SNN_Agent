"""Tests for cloud/app/schemas.py validation and the POST /api/events route
(T02 plan, Step 1 + acceptance gate: "payload validation... 202 +
event_id"). Route-level tests monkeypatch storage.* so no real Storage
account is needed to run them.
"""

import base64

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cloud.app import schemas, storage
from cloud.app.main import app

client = TestClient(app)

_VALID_PAYLOAD = {
    "ts_wall": 1784048796.83,
    "woken_by_trigger": False,
    "escalate": True,
    "motion": True,
    "person": False,
    "score": 0.1547,
    "vision_source": "gemini",
    "is_intrusion": False,
    "alarm": False,
    "reason": "vision: ...",
    "email_sent": False,
    "latency_s": 10.67,
}


@pytest.fixture(autouse=True)
def _fixed_credentials(monkeypatch):
    monkeypatch.setenv("DASHBOARD_USER", "testuser")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "testpass")


def _auth() -> tuple[str, str]:
    return ("testuser", "testpass")


# --- schema-level validation (F01 design, "POST /api/events" Validation) ---


def test_event_in_accepts_a_valid_payload():
    event = schemas.EventIn(**_VALID_PAYLOAD)
    assert event.ts_wall == _VALID_PAYLOAD["ts_wall"]
    assert event.image_jpeg_b64 is None


def test_event_in_rejects_unknown_extra_field():
    with pytest.raises(ValidationError):
        schemas.EventIn(**_VALID_PAYLOAD, unexpected_field="nope")


def test_event_in_rejects_missing_required_field():
    payload = dict(_VALID_PAYLOAD)
    del payload["reason"]
    with pytest.raises(ValidationError):
        schemas.EventIn(**payload)


def test_event_in_accepts_image_under_cap():
    small_image = base64.b64encode(b"x" * 1024).decode()
    event = schemas.EventIn(**_VALID_PAYLOAD, image_jpeg_b64=small_image)
    assert event.image_jpeg_b64 == small_image


def test_event_in_rejects_image_over_2mb_cap():
    too_big = base64.b64encode(b"x" * (2 * 1024 * 1024 + 1)).decode()
    with pytest.raises(ValidationError):
        schemas.EventIn(**_VALID_PAYLOAD, image_jpeg_b64=too_big)


def test_event_in_rejects_non_base64_image():
    with pytest.raises(ValidationError):
        schemas.EventIn(**_VALID_PAYLOAD, image_jpeg_b64="not-base64-!!!")


# --- route-level: POST /api/events ---


def test_ingest_returns_202_and_event_id(monkeypatch):
    monkeypatch.setattr(storage, "generate_ulid", lambda: "01ARZ3NDEKTSV4RRFFQ69G5FAV")
    monkeypatch.setattr(
        storage,
        "write_event",
        lambda fields, event_id, received_at: {"PartitionKey": "2026-07-14", "RowKey": event_id},
    )

    response = client.post("/api/events", json=_VALID_PAYLOAD, auth=_auth())

    assert response.status_code == 202
    assert response.json() == {"event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV"}


def test_ingest_writes_blob_when_image_present(monkeypatch):
    calls = {}
    monkeypatch.setattr(storage, "generate_ulid", lambda: "01ARZ3NDEKTSV4RRFFQ69G5FAV")
    monkeypatch.setattr(
        storage,
        "write_event",
        lambda fields, event_id, received_at: {"PartitionKey": "2026-07-14", "RowKey": event_id},
    )

    def _fake_write_blob(event_id: str, image_b64: str) -> str:
        calls["blob"] = event_id
        return f"{event_id}.jpg"

    monkeypatch.setattr(storage, "write_blob", _fake_write_blob)

    def _fake_set_blob_name(event_id: str, pk: str, blob_name: str) -> None:
        calls["set_blob_name"] = (event_id, pk, blob_name)

    monkeypatch.setattr(storage, "set_blob_name", _fake_set_blob_name)

    payload = dict(_VALID_PAYLOAD, image_jpeg_b64=base64.b64encode(b"fake-jpeg").decode())
    response = client.post("/api/events", json=payload, auth=_auth())

    assert response.status_code == 202
    assert calls["blob"] == "01ARZ3NDEKTSV4RRFFQ69G5FAV"
    assert calls["set_blob_name"] == (
        "01ARZ3NDEKTSV4RRFFQ69G5FAV",
        "2026-07-14",
        "01ARZ3NDEKTSV4RRFFQ69G5FAV.jpg",
    )


def test_ingest_does_not_set_blob_name_when_blob_write_fails(monkeypatch):
    """F01 design, Behavior: a failed blob write leaves blob_name empty
    rather than failing the whole request.
    """
    calls = {}
    monkeypatch.setattr(storage, "generate_ulid", lambda: "01ARZ3NDEKTSV4RRFFQ69G5FAV")
    monkeypatch.setattr(
        storage,
        "write_event",
        lambda fields, event_id, received_at: {"PartitionKey": "2026-07-14", "RowKey": event_id},
    )
    monkeypatch.setattr(storage, "write_blob", lambda event_id, image_b64: None)
    monkeypatch.setattr(
        storage,
        "set_blob_name",
        lambda *a: calls.setdefault("called", True),
    )

    payload = dict(_VALID_PAYLOAD, image_jpeg_b64=base64.b64encode(b"fake-jpeg").decode())
    response = client.post("/api/events", json=payload, auth=_auth())

    assert response.status_code == 202
    assert "called" not in calls


def test_ingest_rejects_extra_field_with_422():
    response = client.post("/api/events", json=dict(_VALID_PAYLOAD, extra="nope"), auth=_auth())
    assert response.status_code == 422


def test_ingest_rejects_oversized_image_with_422():
    too_big = base64.b64encode(b"x" * (2 * 1024 * 1024 + 1)).decode()
    response = client.post(
        "/api/events", json=dict(_VALID_PAYLOAD, image_jpeg_b64=too_big), auth=_auth()
    )
    assert response.status_code == 422
