"""The runtime behind the real backend, not behind a test double (task P2, point 3).

``tests/w0`` exercises the backend against ``FakeRuntime``; ``tests/runtime``
exercises the runtime on its own. Neither proves they fit. This file builds the
backend's own services around ``LuiRuntime`` and the network we actually
trained, and drives ``SpikeBatch`` in through ``IngestService.ingest`` the way
the edge bridge does, so the seam is covered from both sides.

It also pins the deployment string. The backend imports a runtime by
``module:attribute`` (``backend_config._runtime_factory``), and the attribute
this package offers is ``snn_runtime.runtime:from_environment``, not the class:
the factory is what reads ``SNN_MODEL_ARTIFACT_ROOT``, and a deployment that
named the class directly would silently run without verifying the weights.
"""

from __future__ import annotations

import pytest

from contracts.validation import content_hash, fixture, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.ingest import IngestService
from rpi_agents.cloud.app.publisher import Publisher
from rpi_agents.cloud.app.sessions import SessionService, provision_device
from rpi_agents.cloud.app.settings import Settings
from rpi_agents.cloud.app.storage import memory_storage
from snn_runtime import LuiRuntime

DEVICE = "demo-pi"


@pytest.fixture
def backend(lui8):
    manifest = validate("ModelManifest", lui8)
    ctx = Context(
        memory_storage(),
        Settings(),
        manifest,
        lambda: LuiRuntime(allow_unverified_artifacts=True),
    )
    provision_device(ctx, DEVICE)
    sessions = SessionService(ctx)
    return ctx, sessions, IngestService(ctx, sessions, Publisher(ctx))


def test_the_backend_can_run_the_real_runtime(backend, lui8, stream, make_batch):
    ctx, sessions, ingest = backend
    state = sessions.create(
        DEVICE,
        fixture("session-create")
        | {
            "device_id": DEVICE,
            "model_hash": content_hash(ctx.manifest),
            "encoder_hash": ctx.manifest["encoder_hash"],
            "source_start_us": 0,
        },
    )

    acks = []
    for seq, first in enumerate(range(0, 200, 25)):
        batch = make_batch(stream, first, 25, seq=seq) | {
            "device_id": DEVICE,
            "session_id": state["session_id"],
            "epoch": state["epoch"],
            "boot_id": state["boot_id"],
        }
        acks.append(ingest.ingest(DEVICE, state["session_id"], batch))

    for ack in acks:
        validate("BatchAck", ack)
        decision = ack["decision"]
        assert decision["status"] == "valid"
        assert decision["score_kind"] == "uncalibrated"
        assert decision["provenance"] == "simulated"
        assert decision["model_hash"] == content_hash(ctx.manifest)
    assert sum(ack["decision"]["score"] for ack in acks) > 0, "the runtime never spiked behind the backend"
    assert any(ack["decision"]["trigger"] for ack in acks), "no decision reached the event path"


def test_the_deployment_string_resolves_to_the_factory():
    from rpi_agents.cloud.app.backend_config import _runtime_factory

    factory = _runtime_factory("snn_runtime.runtime:from_environment")
    assert isinstance(factory(), LuiRuntime)


def test_the_backend_refuses_a_session_on_a_package_the_runtime_rejects(backend, lui8, mutate):
    """A model the runtime will not integrate must fail at session start, not at the first batch."""
    ctx, sessions, _ = backend
    broken = mutate(lui8, lambda m: m["topology"]["neurons"].clear())
    ctx.manifest = broken
    with pytest.raises(Exception) as excinfo:
        sessions.create(
            DEVICE,
            fixture("session-create")
            | {
                "device_id": DEVICE,
                "model_hash": content_hash(broken),
                "encoder_hash": broken["encoder_hash"],
                "source_start_us": 0,
            },
        )
    assert getattr(excinfo.value, "code", "") == "RUNTIME_REFUSED"
