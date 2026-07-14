"""Tests for cloud/app/schemas.py validation and the POST /api/events route
(T02 plan, Step 1 + acceptance gate: "payload validation... 202 +
event_id"; T04 plan, Step 4: client-supplied event_id + upsert idempotency).
Route-level tests monkeypatch storage.* so no real Storage account is
needed to run them.
"""

import base64

import pytest
from azure.data.tables import UpdateMode
from fastapi.testclient import TestClient
from pydantic import ValidationError

from cloud.app import schemas, storage
from cloud.app.main import app

client = TestClient(app)

_VALID_PAYLOAD = {
    "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
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


@pytest.mark.parametrize(
    "bad_event_id",
    [
        "01ARZ3NDEKTSV4RRFFQ69G5FA",  # 25 chars -- too short
        "01ARZ3NDEKTSV4RRFFQ69G5FAVX",  # 27 chars -- too long
        "01arz3ndektsv4rrffq69g5fav",  # lowercase -- rejected (server never emits this case)
        "01ARZ3NDEKTSV4RRFFQ69G5FAI",  # 'I' is not in the Crockford base32 alphabet
    ],
)
def test_event_in_rejects_malformed_event_id(bad_event_id):
    """ADR-0014: a malformed event_id must never reach Table Storage as a
    RowKey.
    """
    payload = dict(_VALID_PAYLOAD, event_id=bad_event_id)
    with pytest.raises(ValidationError):
        schemas.EventIn(**payload)


# --- route-level: POST /api/events ---


def test_ingest_returns_202_and_event_id(monkeypatch):
    monkeypatch.setattr(
        storage,
        "write_event",
        lambda fields, event_id, received_at: {"PartitionKey": "2026-07-14", "RowKey": event_id},
    )

    response = client.post("/api/events", json=_VALID_PAYLOAD, auth=_auth())

    assert response.status_code == 202
    assert response.json() == {"event_id": _VALID_PAYLOAD["event_id"]}


def test_ingest_writes_blob_when_image_present(monkeypatch):
    calls = {}
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


def test_ingest_rejects_malformed_event_id_with_422():
    payload = dict(_VALID_PAYLOAD, event_id="not-a-valid-ulid-at-all!!")
    response = client.post("/api/events", json=payload, auth=_auth())
    assert response.status_code == 422


# --- upsert idempotency (ADR-0014, T04 plan Step 4) ---


def test_ingest_upsert_is_idempotent_across_duplicate_event_id(monkeypatch):
    """Posting the same event_id twice (a queued retry, ADR-0015) must be a
    no-op the second time from the dashboard's point of view: one row,
    not two, and no error either time. Fakes storage.write_event() with a
    real in-memory dict so the "overwrite, don't duplicate" behavior is
    actually exercised, not just assumed.
    """
    written: dict[str, dict] = {}

    def _fake_write_event(fields: dict, event_id: str, received_at: float) -> dict:
        entity = {"PartitionKey": "2026-07-14", "RowKey": event_id, **fields}
        written[event_id] = entity  # overwrite semantics, mirrors upsert_entity()
        return entity

    monkeypatch.setattr(storage, "write_event", _fake_write_event)

    first = client.post("/api/events", json=_VALID_PAYLOAD, auth=_auth())
    second = client.post("/api/events", json=_VALID_PAYLOAD, auth=_auth())

    assert first.status_code == 202
    assert second.status_code == 202
    assert first.json() == second.json() == {"event_id": _VALID_PAYLOAD["event_id"]}
    assert len(written) == 1


def test_write_event_calls_upsert_entity_not_create_entity(monkeypatch):
    """storage.write_event() itself (ADR-0014): the Table client call must
    be upsert_entity(mode=REPLACE), never create_entity() — the latter
    raises ResourceExistsError on a retried event_id.
    """
    calls = {}

    class _FakeTableClient:
        def upsert_entity(self, entity: dict, mode: UpdateMode) -> None:
            calls["upsert"] = (entity["RowKey"], mode)

        def create_entity(self, entity: dict) -> None:
            calls["create"] = entity["RowKey"]

    monkeypatch.setattr(storage, "get_table_client", lambda: _FakeTableClient())

    fields = {k: v for k, v in _VALID_PAYLOAD.items() if k not in ("event_id", "image_jpeg_b64")}
    storage.write_event(fields, "01ARZ3NDEKTSV4RRFFQ69G5FAV", 1784048800.0)

    assert calls == {"upsert": ("01ARZ3NDEKTSV4RRFFQ69G5FAV", UpdateMode.REPLACE)}
    assert "create" not in calls
