"""Unit tests for agent.cloud_sync (F03 design; ADR-0001, ADR-0006,
ADR-0009, ADR-0014, ADR-0015). Mirrors tests/test_notifier.py's fixture
style; requests.post is mocked directly rather than a fake transport.
"""

import numpy as np
import pytest
import requests

from agent import cloud_sync, config, sync_queue

_RECORD = {
    "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "ts_wall": 123.0,
    "woken_by_trigger": True,
    "escalate": True,
    "motion": True,
    "person": False,
    "score": 0.5,
    "vision_source": "gemini",
    "is_intrusion": True,
    "alarm": True,
    "reason": "test reason",
    "email_sent": True,
    "clip": "/tmp/clip.mp4",
    "latency_s": 1.23,
}


@pytest.fixture(autouse=True)
def _queue_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(config, "SYNC_QUEUE_PATH", tmp_path / "sync_queue.jsonl")


@pytest.fixture()
def configured(monkeypatch: pytest.MonkeyPatch) -> config.Settings:
    """Cloud sync enabled, URL + credentials set."""
    monkeypatch.setattr(config, "CLOUD_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "CLOUD_SYNC_URL", "https://example.invalid/api/events")
    settings = config.Settings(
        gemini_api_key="x",
        gmail_user="y",
        gmail_app_password="z",
        alert_to="w",
        cloud_sync_user="ids",
        cloud_sync_password="ids",
    )
    monkeypatch.setattr(config, "load_settings", lambda: settings)
    return settings


class _FakeResponse:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


# --- generate_event_id() ---


def test_generate_event_id_is_26_char_crockford_base32() -> None:
    event_id = cloud_sync.generate_event_id()
    assert len(event_id) == 26
    assert set(event_id) <= set(cloud_sync._CROCKFORD_ALPHABET)


def test_generate_event_id_is_unique_across_calls() -> None:
    ids = {cloud_sync.generate_event_id() for _ in range(50)}
    assert len(ids) == 50


# --- build_payload() ---


def test_build_payload_without_snapshot_omits_image() -> None:
    payload = cloud_sync.build_payload(_RECORD, None)
    assert "image_jpeg_b64" not in payload
    assert payload["event_id"] == _RECORD["event_id"]
    assert "clip" not in payload  # local-only field, never pushed


def test_build_payload_with_snapshot_includes_base64_image() -> None:
    snapshot = np.zeros((4, 4, 3), dtype=np.uint8)
    payload = cloud_sync.build_payload(_RECORD, snapshot)
    assert isinstance(payload["image_jpeg_b64"], str)
    assert len(payload["image_jpeg_b64"]) > 0


# --- is_configured() ---


def test_is_configured_false_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLOUD_SYNC_ENABLED", False)
    monkeypatch.setattr(config, "CLOUD_SYNC_URL", "https://example.invalid")
    assert cloud_sync.is_configured() is False


def test_is_configured_false_when_no_url(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLOUD_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "CLOUD_SYNC_URL", "")
    assert cloud_sync.is_configured() is False


def test_is_configured_true_when_enabled_and_url_set(configured: config.Settings) -> None:
    assert cloud_sync.is_configured() is True


# --- push() ---


def test_push_returns_false_when_not_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLOUD_SYNC_ENABLED", False)
    assert cloud_sync.push({"event_id": "x"}) is False


def test_push_returns_false_when_credentials_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLOUD_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "CLOUD_SYNC_URL", "https://example.invalid")
    settings = config.Settings(
        gemini_api_key="x", gmail_user="y", gmail_app_password="z", alert_to="w"
    )  # cloud_sync_user/password left at their None default
    monkeypatch.setattr(config, "load_settings", lambda: settings)

    assert cloud_sync.push({"event_id": "x"}) is False


def test_push_returns_false_when_load_settings_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "CLOUD_SYNC_ENABLED", True)
    monkeypatch.setattr(config, "CLOUD_SYNC_URL", "https://example.invalid")

    def _raise() -> config.Settings:
        raise ValueError("missing required secrets")

    monkeypatch.setattr(config, "load_settings", _raise)

    assert cloud_sync.push({"event_id": "x"}) is False


def test_push_returns_true_on_2xx(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fake_post(url, json, auth, timeout):
        calls.append((url, json, auth, timeout))
        return _FakeResponse(202)

    monkeypatch.setattr(requests, "post", fake_post)

    payload = cloud_sync.build_payload(_RECORD, None)
    assert cloud_sync.push(payload) is True

    url, sent_json, auth, timeout = calls[0]
    assert url == config.CLOUD_SYNC_URL
    assert sent_json["event_id"] == _RECORD["event_id"]
    assert auth == ("ids", "ids")
    assert timeout == config.CLOUD_SYNC_TIMEOUT_S


def test_push_returns_false_on_non_2xx(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(500))
    assert cloud_sync.push({"event_id": "x"}) is False


def test_push_returns_false_on_timeout(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_post(*args, **kwargs):
        raise requests.Timeout("timed out")

    monkeypatch.setattr(requests, "post", fake_post)
    assert cloud_sync.push({"event_id": "x"}) is False


def test_push_returns_false_on_connection_error(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_post(*args, **kwargs):
        raise requests.ConnectionError("refused")

    monkeypatch.setattr(requests, "post", fake_post)
    assert cloud_sync.push({"event_id": "x"}) is False


def test_push_never_raises_on_unexpected_request_exception(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """push()'s contract is 'never raises' -- even a RequestException
    subclass we didn't explicitly enumerate must still be swallowed."""

    def fake_post(*args, **kwargs):
        raise requests.exceptions.SSLError("bad cert")

    monkeypatch.setattr(requests, "post", fake_post)
    assert cloud_sync.push({"event_id": "x"}) is False


# --- flush_queue() ---


def test_flush_queue_pushes_and_removes_successes(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_queue.enqueue("e1", {"event_id": "e1"})
    sync_queue.enqueue("e2", {"event_id": "e2"})
    monkeypatch.setattr(requests, "post", lambda *a, **k: _FakeResponse(202))

    cloud_sync.flush_queue()

    assert sync_queue.dequeue_batch(10) == []


def test_flush_queue_stops_at_first_failure(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    sync_queue.enqueue("e1", {"event_id": "e1"})
    sync_queue.enqueue("e2", {"event_id": "e2"})
    sync_queue.enqueue("e3", {"event_id": "e3"})

    responses = iter([_FakeResponse(202), _FakeResponse(500)])
    monkeypatch.setattr(requests, "post", lambda *a, **k: next(responses))

    cloud_sync.flush_queue()

    remaining = sync_queue.dequeue_batch(10)
    # e1 succeeded and was removed; e2 failed and stays (attempts=1);
    # e3 was never attempted this cycle.
    assert [entry["event_id"] for entry in remaining] == ["e2", "e3"]
    assert remaining[0]["attempts"] == 1
    assert remaining[1]["attempts"] == 0


def test_flush_queue_respects_max_items(
    configured: config.Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    for i in range(5):
        sync_queue.enqueue(f"e{i}", {"event_id": f"e{i}"})

    call_count = 0

    def fake_post(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return _FakeResponse(202)

    monkeypatch.setattr(requests, "post", fake_post)

    cloud_sync.flush_queue(max_items=2)

    assert call_count == 2
    assert len(sync_queue.dequeue_batch(10)) == 3


def test_flush_queue_on_empty_queue_does_nothing(configured: config.Settings) -> None:
    cloud_sync.flush_queue()
    assert sync_queue.dequeue_batch(10) == []
