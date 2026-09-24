import pytest

from rpi_agents.cloud.app.publisher import outbox_op
from rpi_agents.cloud.app.records import CORE, Q_NOTIFY, Q_VISION, T_DEVICE_COMMANDS, T_EVENTS
from rpi_agents.cloud.app.storage import NotFound
from tests.w0.backend_env import TRIGGER, Env

SPIKES = [(1200, "zcr")]


def record_only(env, *ops):
    """Record outbox rows the way a request does, then 'crash' before anything is published."""
    env.storage.tables.transaction(T_EVENTS, CORE, list(ops))


def test_a_crash_before_enqueue_is_recovered_by_the_reconciler_after_the_grace_period():
    env = Env(publish_grace_s=10)
    record_only(env, outbox_op(env.ctx, "e1", 0, "vision_job", {"event_id": "e1", "image_id": "e1-0"}))
    assert not env.publisher.drained() and env.storage.queues.depth(Q_VISION) == 0  # recorded, never enqueued
    assert env.publisher.publish_pending(min_age_s=10) == 0  # too young: the request that wrote it may still be on it
    env.clock.advance(11)
    assert env.publisher.publish_pending(min_age_s=10) == 1
    (message,) = env.storage.queues.receive(Q_VISION, visibility_s=30)
    assert message.body == {"event_id": "e1", "image_id": "e1-0"}
    assert env.publisher.drained()  # the row is gone only after it was published


def test_a_crash_after_the_send_but_before_the_delete_causes_a_duplicate_message_never_a_lost_one():
    env = Env()
    record_only(env, outbox_op(env.ctx, "e1", 0, "vision_job", {"event_id": "e1", "image_id": "e1-0"}))
    tables, real = env.storage.tables, env.storage.tables.delete

    def dying(*a, **k):
        raise KeyboardInterrupt

    tables.delete = dying
    with pytest.raises(KeyboardInterrupt):
        env.publisher.publish_pending()
    tables.delete = real
    assert env.storage.queues.depth(Q_VISION) == 1 and not env.publisher.drained()
    assert env.publisher.publish_pending() == 1  # the reconciler sends it again
    assert env.storage.queues.depth(Q_VISION) == 2 and env.publisher.drained()  # at-least-once, by design


def test_a_command_becomes_visible_to_the_device_only_when_its_index_row_is_published():
    env = Env()
    state = env.open_session()
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    env.publisher._publish = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("storage down"))
    (command,) = env.send(state, 0, spikes=SPIKES)["commands"]
    assert env.storage.tables.query(T_DEVICE_COMMANDS, "demo-pi") == []
    del env.publisher._publish  # the storage is back
    assert env.publisher.publish_pending() == 1
    assert env.storage.tables.get(T_DEVICE_COMMANDS, "demo-pi", command["command_id"]).data["status"] == "issued"


def test_publishing_the_same_row_twice_is_harmless_for_indexes():
    env = Env()
    state = env.open_session()
    env.ctx.runtimes[state["session_id"]].decisions = [TRIGGER]
    (command,) = env.send(state, 0, spikes=SPIKES)["commands"]
    env.publisher._publish("index_command", {"device_id": "demo-pi", "command_id": command["command_id"]})
    env.publisher._publish("index_command", {"device_id": "demo-pi", "command_id": command["command_id"]})
    assert len(env.storage.tables.query(T_DEVICE_COMMANDS, "demo-pi")) == 1


def test_one_poisoned_row_never_blocks_the_rest():
    env = Env()
    record_only(
        env,
        outbox_op(env.ctx, "e1", 0, "no_such_kind", {"x": 1}),
        outbox_op(env.ctx, "e1", 1, "index_command", {"device_id": "demo-pi", "command_id": "missing-command"}),
        outbox_op(env.ctx, "e1", 2, "notify_job", {"event_id": "e1", "reason": "GLASS_AND_PERSON"}),
    )
    assert env.publisher.publish_pending() == 1  # the good one went out
    assert env.storage.queues.depth(Q_NOTIFY) == 1
    assert len(env.publisher.pending()) == 2  # the two bad ones stay for inspection and retry, and are not dropped


def test_rows_are_published_oldest_first_and_the_limit_is_respected():
    env = Env()
    for n in range(5):
        record_only(env, outbox_op(env.ctx, "e1", n, "notify_job", {"event_id": "e1", "reason": f"r{n}"}))
        env.clock.advance(1)
    assert env.publisher.publish_pending(limit=3) == 3
    order = [m.body["reason"] for m in env.storage.queues.receive(Q_NOTIFY, visibility_s=30, max_messages=10)]
    assert order == ["r0", "r1", "r2"] and len(env.publisher.pending()) == 2


def test_a_row_another_publisher_already_finished_is_not_an_error():
    env = Env()
    record_only(env, outbox_op(env.ctx, "e1", 0, "notify_job", {"event_id": "e1", "reason": "x"}))
    tables, real = env.storage.tables, env.storage.tables.delete

    def raced(table, pk, rk, etag=None):
        real(table, pk, rk)  # the other publisher deleted it first...
        raise NotFound(rk)  # ...so ours finds nothing

    tables.delete = raced
    assert env.publisher.publish_pending() == 1
