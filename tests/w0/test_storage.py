"""Contract tests for the storage interfaces. Any implementation must pass all of them.

Two implementations run this suite: the in-memory one (always, with a fake clock) and Azure Storage through the
Azurite emulator (only when SNN_TEST_AZURITE=1 and the Azure SDKs are installed; real time, so a few tests wait
a few seconds). `storage_env` is the only thing that differs between them.
"""

import os
import subprocess
import sys
import time
import uuid

import pytest

from rpi_agents.cloud.app.storage import (
    Conflict,
    NotFound,
    Op,
    PreconditionFailed,
    Storage,
    memory_storage,
)

SHORT = 2  # seconds of queue visibility used by the timing tests: real on Azurite, fake on memory
AZURITE = "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;QueueEndpoint=http://127.0.0.1:10001/devstoreaccount1;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;"


class MemoryEnv:
    def __init__(self):
        self.now = 1000.0
        self.storage = memory_storage(lambda: self.now)

    def advance(self, seconds):
        self.now += seconds


class Scoped:
    """Renames tables, containers and queues per test so runs on a shared emulator cannot collide."""

    def __init__(self, inner, suffix, joiner):
        self._inner, self._suffix, self._joiner = inner, suffix, joiner

    def _n(self, name):
        return f"{name}{self._joiner}{self._suffix}"

    def __getattr__(self, method):
        target = getattr(self._inner, method)

        def call(name, *args, **kwargs):
            return target(self._n(name), *args, **kwargs)

        return call


class AzuriteEnv:
    def __init__(self):
        from rpi_agents.cloud.app.storage_azure import azure_storage

        real, suffix = azure_storage(connection_string=AZURITE), uuid.uuid4().hex[:8]
        self.storage = Storage(Scoped(real.tables, suffix, "x"), Scoped(real.blobs, suffix, "-"), Scoped(real.queues, suffix, "-"))

    def advance(self, seconds):
        time.sleep(seconds)


@pytest.fixture(params=["memory", "azurite"])
def storage_env(request):
    if request.param == "memory":
        return MemoryEnv()
    if not os.environ.get("SNN_TEST_AZURITE"):
        pytest.skip("set SNN_TEST_AZURITE=1 with Azurite running to test the Azure adapters")
    pytest.importorskip("azure.data.tables")
    return AzuriteEnv()


@pytest.fixture
def tables(storage_env):
    return storage_env.storage.tables


@pytest.fixture
def blobs(storage_env):
    return storage_env.storage.blobs


@pytest.fixture
def queues(storage_env):
    return storage_env.storage.queues


# ------------------------------------------------------------------------------------------------ tables


def test_insert_get_roundtrip_returns_copies(tables):
    row = tables.insert("tab", "p", "r1", {"a": 1, "nested": {"x": [1, 2]}})
    got = tables.get("tab", "p", "r1")
    assert got.data == {"a": 1, "nested": {"x": [1, 2]}} and got.etag == row.etag
    got.data["nested"]["x"].append(3)  # mutating what was returned must not change the store
    assert tables.get("tab", "p", "r1").data["nested"]["x"] == [1, 2]
    assert tables.get("tab", "p", "missing") is None and tables.get("tab", "other", "r1") is None


def test_insert_of_an_existing_row_conflicts_and_changes_nothing(tables):
    tables.insert("tab", "p", "r", {"v": 1})
    with pytest.raises(Conflict):
        tables.insert("tab", "p", "r", {"v": 2})
    assert tables.get("tab", "p", "r").data == {"v": 1}


def test_replace_needs_the_current_etag(tables):
    first = tables.insert("tab", "p", "r", {"v": 1})
    second = tables.replace("tab", "p", "r", {"v": 2}, first.etag)
    assert second.etag != first.etag and tables.get("tab", "p", "r").data == {"v": 2}
    with pytest.raises(PreconditionFailed):
        tables.replace("tab", "p", "r", {"v": 3}, first.etag)  # stale
    with pytest.raises(NotFound):
        tables.replace("tab", "p", "nope", {"v": 1}, first.etag)
    assert tables.get("tab", "p", "r").data == {"v": 2}


def test_two_writers_racing_on_one_etag_only_one_wins(tables):
    row = tables.insert("tab", "p", "r", {"n": 0})
    tables.replace("tab", "p", "r", {"n": 1}, row.etag)
    with pytest.raises(PreconditionFailed):
        tables.replace("tab", "p", "r", {"n": 2}, row.etag)


def test_delete_with_a_stale_etag_fails_and_a_missing_row_is_not_found(tables):
    row = tables.insert("tab", "p", "r", {"v": 1})
    tables.replace("tab", "p", "r", {"v": 2}, row.etag)
    with pytest.raises(PreconditionFailed):
        tables.delete("tab", "p", "r", row.etag)
    tables.delete("tab", "p", "r")
    assert tables.get("tab", "p", "r") is None
    with pytest.raises(NotFound):
        tables.delete("tab", "p", "r")


def test_query_orders_filters_and_limits(tables):
    for rk in ("b1", "a2", "a1", "a3", "c1"):
        tables.insert("tab", "p", rk, {"rk": rk})
    tables.insert("tab", "other", "a9", {})
    assert [r.rk for r in tables.query("tab", "p")] == ["a1", "a2", "a3", "b1", "c1"]
    assert [r.rk for r in tables.query("tab", "p", rk_prefix="a")] == ["a1", "a2", "a3"]
    assert [r.rk for r in tables.query("tab", "p", rk_from="a3")] == ["a3", "b1", "c1"]
    assert [r.rk for r in tables.query("tab", "p", limit=2)] == ["a1", "a2"]
    assert [r.rk for r in tables.query("tab", "p", reverse=True, limit=2)] == ["c1", "b1"]
    assert [r.rk for r in tables.query("tab", "p", reverse=True, rk_from="b1")] == ["b1", "a3", "a2", "a1"]
    assert tables.query("tab", "empty") == []


def test_tables_and_partitions_are_isolated(tables):
    tables.insert("tabone", "p", "r", {"t": 1})
    tables.insert("tabtwo", "p", "r", {"t": 2})
    tables.insert("tabone", "q", "r", {"t": 3})
    assert [tables.get(t, p, "r").data["t"] for t, p in (("tabone", "p"), ("tabtwo", "p"), ("tabone", "q"))] == [1, 2, 3]


def test_a_transaction_applies_everything_or_nothing(tables):
    existing = tables.insert("tab", "p", "old", {"v": 1})
    rows = tables.transaction("tab", "p", [Op("insert", "a", {"v": "a"}), Op("replace", "old", {"v": 2}, existing.etag),
                                         Op("insert", "b", {"v": "b"})])  # fmt: skip
    assert [r.rk for r in rows] == ["a", "old", "b"] and tables.get("tab", "p", "old").data == {"v": 2}
    with pytest.raises(Conflict):  # the second operation fails, so the first must not be applied
        tables.transaction("tab", "p", [Op("insert", "c", {"v": "c"}), Op("insert", "a", {"v": "again"})])
    assert tables.get("tab", "p", "c") is None and tables.get("tab", "p", "a").data == {"v": "a"}
    stale = existing.etag
    with pytest.raises(PreconditionFailed):
        tables.transaction("tab", "p", [Op("insert", "d", {}), Op("replace", "old", {"v": 3}, stale)])
    assert tables.get("tab", "p", "d") is None and tables.get("tab", "p", "old").data == {"v": 2}
    with pytest.raises(NotFound):
        tables.transaction("tab", "p", [Op("insert", "e", {}), Op("delete", "ghost")])
    assert tables.get("tab", "p", "e") is None
    assert tables.transaction("tab", "p", []) == []


def test_a_transaction_may_not_touch_the_same_row_twice_or_be_too_large(tables):
    tables.insert("tab", "p", "x", {"v": 1})
    with pytest.raises(ValueError, match="only once"):
        tables.transaction("tab", "p", [Op("insert", "y", {"v": 0}), Op("delete", "x"), Op("insert", "x", {"v": 2})])
    assert tables.get("tab", "p", "x").data == {"v": 1} and tables.get("tab", "p", "y") is None  # nothing was applied
    with pytest.raises(ValueError, match="100"):
        tables.transaction("tab", "p", [Op("insert", f"r{i:03d}", {}) for i in range(101)])
    assert len(tables.transaction("tab", "p", [Op("insert", f"r{i:03d}", {}) for i in range(100)])) == 100


# ------------------------------------------------------------------------------------------------- blobs


def test_blobs_are_create_only_and_return_what_was_written(blobs):
    blobs.put("cont", "n", b"\xff\xd8data", "image/jpeg")
    assert blobs.get("cont", "n") == b"\xff\xd8data" and blobs.exists("cont", "n")
    with pytest.raises(Conflict):
        blobs.put("cont", "n", b"other", "image/jpeg")
    assert blobs.get("cont", "n") == b"\xff\xd8data"
    assert not blobs.exists("cont", "missing") and not blobs.exists("othercont", "n")
    with pytest.raises(NotFound):
        blobs.get("cont", "missing")
    blobs.delete("cont", "n")
    blobs.delete("cont", "n")  # deleting twice is fine
    assert not blobs.exists("cont", "n")


# ------------------------------------------------------------------------------------------------ queues


def test_a_message_is_invisible_while_in_flight_and_returns_if_not_deleted(queues, storage_env):
    queues.send("que", {"job": 1})
    (m,) = queues.receive("que", visibility_s=SHORT)
    assert m.body == {"job": 1} and m.dequeue_count == 1
    assert queues.receive("que", visibility_s=SHORT) == [] and queues.depth("que") == 1  # in flight, not gone
    storage_env.advance(SHORT + 1)
    (again,) = queues.receive("que", visibility_s=SHORT)
    assert again.id == m.id and again.dequeue_count == 2 and again.receipt != m.receipt


def test_only_the_latest_receipt_may_delete_or_renew(queues, storage_env):
    queues.send("que", {"job": 1})
    (first,) = queues.receive("que", visibility_s=SHORT)
    storage_env.advance(SHORT + 1)
    (second,) = queues.receive("que", visibility_s=SHORT)
    with pytest.raises(PreconditionFailed):
        queues.delete("que", first)  # the first worker lost its claim
    with pytest.raises(PreconditionFailed):
        queues.renew("que", first, 10)
    queues.delete("que", second)
    assert queues.depth("que") == 0
    with pytest.raises(NotFound):
        queues.delete("que", second)


def test_renewing_extends_the_claim_and_hands_out_a_new_receipt(queues, storage_env):
    queues.send("que", {"job": 1})
    (m,) = queues.receive("que", visibility_s=3)
    storage_env.advance(2)
    old, m = m, queues.renew("que", m, 3)  # now held until 5 s, not 3 s
    assert m.receipt != old.receipt
    with pytest.raises(PreconditionFailed):
        queues.delete("que", old)  # renewing invalidates the previous receipt
    storage_env.advance(2)  # 4 s in: without the renewal it would be visible again since 3 s
    assert queues.receive("que", visibility_s=SHORT) == []
    queues.delete("que", m)
    assert queues.depth("que") == 0


def test_send_delay_max_messages_fifo_and_isolation(queues, storage_env):
    for i in range(3):
        queues.send("que", {"i": i})
    queues.send("que", {"i": "late"}, delay_s=SHORT)
    queues.send("otherq", {"i": "x"})
    got = queues.receive("que", visibility_s=SHORT, max_messages=2)
    assert [m.body["i"] for m in got] == [0, 1]
    got += queues.receive("que", visibility_s=SHORT, max_messages=5)
    assert [m.body["i"] for m in got] == [0, 1, 2]
    for message in got:
        queues.delete("que", message)  # handled: only the delayed message is left
    storage_env.advance(SHORT + 1)
    assert [m.body["i"] for m in queues.receive("que", visibility_s=SHORT)] == ["late"]
    assert queues.depth("que") == 1 and queues.depth("otherq") == 1 and queues.depth("nothingq") == 0


def test_message_bodies_are_copied(queues):
    body = {"nested": {"a": 1}}
    queues.send("que", body)
    body["nested"]["a"] = 2
    (m,) = queues.receive("que", visibility_s=SHORT)
    assert m.body == {"nested": {"a": 1}}


def test_storage_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.cloud.app.storage"], check=True)
