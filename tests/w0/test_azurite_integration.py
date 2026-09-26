"""A whole alarm chain on Azure Storage (through the Azurite emulator) instead of memory.

Runs only with SNN_TEST_AZURITE=1 and the Azure SDKs installed. It exercises what memory cannot: real Table
transactions and ETags, create-only blobs, and real queues with visibility timeouts, all through the adapters.
"""

import os
import uuid

import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("SNN_TEST_AZURITE"), reason="set SNN_TEST_AZURITE=1 with Azurite running")

from rpi_agents.cloud.app.policy import ARMED  # noqa: E402
from rpi_agents.cloud.app.records import CORE, Q_VISION, T_DEVICE_COMMANDS, T_EVENTS  # noqa: E402
from rpi_agents.cloud.app.storage import Storage  # noqa: E402
from rpi_agents.cloud.app.vision import Observation  # noqa: E402
from tests.w0.backend_env import Env  # noqa: E402
from tests.w0.fakes import FakeNotifier, FakeVision  # noqa: E402
from tests.w0.test_storage import AZURITE, Scoped  # noqa: E402


@pytest.fixture
def env():
    pytest.importorskip("azure.data.tables")
    from rpi_agents.cloud.app.storage_azure import azure_storage

    real, suffix = azure_storage(connection_string=AZURITE), uuid.uuid4().hex[:8]
    storage = Storage(Scoped(real.tables, suffix, "x"), Scoped(real.blobs, suffix, "-"), Scoped(real.queues, suffix, "-"))
    return Env(storage=storage, policy_version=ARMED, recipients=("owner@example.com",), event_cooldown_s=0)


def test_the_alarm_chain_runs_on_azure_storage(env):
    state = env.open_session()
    event_id, capture = env.trigger(state)  # the SNN trigger: one transaction writes event, command and outbox rows
    assert env.publisher.drained()  # published inline: the device_commands index row and nothing left over
    assert env.storage.tables.get(T_DEVICE_COMMANDS, "demo-pi", capture["command_id"]).data["status"] == "issued"

    env.ack(capture, "accepted")
    image = env.upload(event_id)  # blob written create-only, event and vision job recorded in one transaction
    assert image["status"] == "queued" and env.storage.queues.depth(Q_VISION) == 1
    assert env.upload(event_id) == image and env.storage.queues.depth(Q_VISION) == 1  # a retry: no second job
    env.ack(capture, "completed", image_id=image["image_id"])

    notifier = FakeNotifier()
    vision = FakeVision([Observation(True, True, "good", "Broken glass and a person.")])
    counts = env.worker(vision, notifier).run_once()
    assert counts["vision"] == 1 and len(vision.calls) == 1 and len(notifier.sent) == 1

    event = env.events.get(event_id)
    assert event["status"] == "alarm_confirmed" and [c["type"] for c in event["commands"]] == ["capture", "alarm"]
    alarm = event["commands"][1]
    assert alarm["command_id"] in {c["command_id"] for c in env.commands.poll("demo-pi")}
    env.ack(alarm, "accepted")
    env.ack(alarm, "completed")
    assert env.commands.poll("demo-pi") == [] and env.storage.queues.depth(Q_VISION) == 0
    assert env.storage.tables.query(T_EVENTS, CORE, rk_prefix="outbox:") == []  # nothing left unpublished


def test_a_duplicate_vision_message_is_absorbed_on_azure_storage(env):
    state = env.open_session()
    event_id, capture = env.trigger(state)
    env.upload(event_id)
    env.storage.queues.send(Q_VISION, {"event_id": event_id, "image_id": f"{event_id}-0"})  # at-least-once: a second copy
    vision = FakeVision([Observation(False, False, "good", "Empty room."), Observation(False, False, "good", "Empty room.")])
    env.worker(vision, FakeNotifier()).run_once()
    assert len(vision.calls) == 1 and env.events.get(event_id)["status"] == "no_alarm"
