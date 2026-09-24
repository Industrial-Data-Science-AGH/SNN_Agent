import subprocess
import sys
import threading

import pytest

from rpi_agents.agent.outbox import MAX_PAYLOAD_BYTES, Outbox, OutboxConflict


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "outbox.db")


def ids(box, limit=50):
    return [e.request_id for e in box.pending(limit)]


def test_delivery_order_is_insertion_order_and_payload_round_trips(db):
    box = Outbox(db)
    for i in (3, 1, 2):
        assert box.put(f"r{i}", {"n": i, "spikes": [{"dt_us": 5, "channel": "zcr"}]}).stored
    assert ids(box) == ["r3", "r1", "r2"]
    assert box.pending()[0].payload == {"n": 3, "spikes": [{"dt_us": 5, "channel": "zcr"}]}


def test_exact_retry_is_a_noop_and_different_content_conflicts(db):
    box = Outbox(db)
    box.put("r1", {"a": 1, "b": 2})
    assert box.put("r1", {"b": 2, "a": 1}).stored is False  # key order does not matter
    with pytest.raises(OutboxConflict):
        box.put("r1", {"a": 1, "b": 3})
    assert ids(box) == ["r1"]


def test_sent_entries_leave_the_queue_and_are_still_recognised_as_duplicates(db):
    box = Outbox(db)
    box.put("r1", {"n": 1})
    box.put("r2", {"n": 2})
    assert box.mark_sent("r1") is True and box.mark_sent("r1") is False and box.mark_sent("nope") is False
    assert ids(box) == ["r2"]
    assert box.put("r1", {"n": 1}).stored is False  # a late duplicate must not be queued again
    assert (box.stats().pending, box.stats().sent) == (1, 1)


def test_state_survives_reopen_in_the_original_order(db):
    box = Outbox(db)
    for i in range(4):
        box.put(f"r{i}", {"n": i})
    box.mark_sent("r1")
    box.record_failure("r2", "HTTP 503")
    box.close()
    again = Outbox(db)
    assert ids(again) == ["r0", "r2", "r3"]
    assert [e.attempts for e in again.pending()] == [0, 1, 0]
    assert again.pending()[1].last_error == "HTTP 503"


def test_an_entry_written_by_a_process_that_dies_abruptly_is_not_lost(db):
    code = (
        "import os, sys; from rpi_agents.agent.outbox import Outbox;"
        "b = Outbox(sys.argv[1]); b.put('crash-1', {'n': 1}); os._exit(1)"  # no close, no cleanup
    )
    subprocess.run([sys.executable, "-c", code, db], check=False)
    assert ids(Outbox(db)) == ["crash-1"]


def test_bound_drops_the_oldest_visibly_and_counts_persistently(db):
    box = Outbox(db, max_pending=3)
    results = [box.put(f"r{i}", {"n": i}) for i in range(5)]
    assert [r.dropped for r in results] == [(), (), (), ("r0",), ("r1",)]
    assert ids(box) == ["r2", "r3", "r4"]
    assert box.stats().dropped_total == 2
    box.close()
    assert Outbox(db, max_pending=3).stats().dropped_total == 2


def test_delivered_entries_do_not_count_against_the_bound(db):
    box = Outbox(db, max_pending=2)
    for i in range(6):
        assert box.put(f"r{i}", {"n": i}).dropped == ()
        box.mark_sent(f"r{i}")
    assert box.stats().dropped_total == 0


def test_purge_keeps_the_newest_delivered_entries(db):
    box = Outbox(db)
    for i in range(6):
        box.put(f"r{i}", {"n": i})
        box.mark_sent(f"r{i}")
    box.put("r6", {"n": 6})
    assert box.purge_sent(keep=2) == 4
    assert box.stats().sent == 2 and ids(box) == ["r6"]
    assert box.put("r5", {"n": 5}).stored is False  # still remembered


def test_bad_payloads_and_settings_are_refused(db):
    box = Outbox(db)
    with pytest.raises(ValueError, match="limit"):
        box.put("big", {"blob": "x" * MAX_PAYLOAD_BYTES})
    with pytest.raises(ValueError):
        box.put("nan", {"v": float("nan")})
    with pytest.raises(ValueError):
        Outbox(db, max_pending=0)
    assert ids(box) == []


def test_concurrent_writers_lose_nothing(db):
    box = Outbox(db)
    errors = []

    def worker(k):
        try:
            for i in range(50):
                box.put(f"t{k}-{i}", {"k": k, "i": i})
        except Exception as exc:  # pragma: no cover - would fail the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors and box.stats().pending == 200


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.outbox"], check=True)
