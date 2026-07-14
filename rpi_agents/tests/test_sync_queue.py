"""Unit tests for agent.sync_queue (F03 design; ADR-0015).

No network involved -- pure filesystem behavior against a tmp_path-backed
queue file.
"""

import json

import pytest

from agent import config, sync_queue


@pytest.fixture(autouse=True)
def _queue_path(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """Redirect config.SYNC_QUEUE_PATH to a tmp file for every test."""
    monkeypatch.setattr(config, "SYNC_QUEUE_PATH", tmp_path / "sync_queue.jsonl")


def test_dequeue_batch_on_missing_file_is_empty() -> None:
    assert sync_queue.dequeue_batch(5) == []


def test_enqueue_then_dequeue_preserves_order() -> None:
    sync_queue.enqueue("e1", {"x": 1})
    sync_queue.enqueue("e2", {"x": 2})

    batch = sync_queue.dequeue_batch(10)

    assert [entry["event_id"] for entry in batch] == ["e1", "e2"]
    assert batch[0]["payload"] == {"x": 1}
    assert batch[0]["attempts"] == 0
    assert "queued_at" in batch[0]


def test_dequeue_batch_does_not_remove() -> None:
    sync_queue.enqueue("e1", {"x": 1})
    sync_queue.dequeue_batch(10)
    assert len(sync_queue.dequeue_batch(10)) == 1


def test_dequeue_batch_respects_max_items() -> None:
    for i in range(5):
        sync_queue.enqueue(f"e{i}", {"i": i})
    assert len(sync_queue.dequeue_batch(2)) == 2


def test_remove_deletes_only_matching_entry() -> None:
    sync_queue.enqueue("e1", {})
    sync_queue.enqueue("e2", {})

    sync_queue.remove("e1")

    remaining = sync_queue.dequeue_batch(10)
    assert [entry["event_id"] for entry in remaining] == ["e2"]


def test_enqueue_evicts_oldest_when_full(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SYNC_QUEUE_MAX_SIZE", 3)

    for i in range(4):
        sync_queue.enqueue(f"e{i}", {"i": i})

    remaining = sync_queue.dequeue_batch(10)
    assert [entry["event_id"] for entry in remaining] == ["e1", "e2", "e3"]


def test_record_failure_increments_attempts() -> None:
    sync_queue.enqueue("e1", {})
    sync_queue.record_failure("e1")

    remaining = sync_queue.dequeue_batch(10)
    assert remaining[0]["attempts"] == 1


def test_record_failure_drops_entry_after_max_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "SYNC_QUEUE_MAX_ATTEMPTS", 3)
    sync_queue.enqueue("e1", {})

    sync_queue.record_failure("e1")
    sync_queue.record_failure("e1")
    assert len(sync_queue.dequeue_batch(10)) == 1  # not dropped yet

    sync_queue.record_failure("e1")  # 3rd failure -> dropped
    assert sync_queue.dequeue_batch(10) == []


def test_record_failure_does_not_affect_other_entries() -> None:
    sync_queue.enqueue("e1", {})
    sync_queue.enqueue("e2", {})

    sync_queue.record_failure("e1")

    remaining = {entry["event_id"]: entry for entry in sync_queue.dequeue_batch(10)}
    assert remaining["e1"]["attempts"] == 1
    assert remaining["e2"]["attempts"] == 0


def test_corrupted_last_line_is_skipped_not_fatal() -> None:
    sync_queue.enqueue("e1", {})
    with config.SYNC_QUEUE_PATH.open("a") as fh:
        fh.write("{this is not valid json\n")

    # A subsequent enqueue reads the file first (to check the cap) -- must
    # not raise on the corrupted line.
    sync_queue.enqueue("e2", {})

    remaining = sync_queue.dequeue_batch(10)
    assert [entry["event_id"] for entry in remaining] == ["e1", "e2"]


def test_blank_lines_are_ignored() -> None:
    config.SYNC_QUEUE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with config.SYNC_QUEUE_PATH.open("w") as fh:
        fh.write(json.dumps({"event_id": "e1", "payload": {}, "attempts": 0, "queued_at": 0.0}))
        fh.write("\n\n")

    assert len(sync_queue.dequeue_batch(10)) == 1
