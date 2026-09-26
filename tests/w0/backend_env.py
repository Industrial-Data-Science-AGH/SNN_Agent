"""Test environment for the backend services: in-memory storage, a controllable clock, a fake runtime."""

import itertools
from datetime import datetime, timedelta, timezone

from contracts.validation import content_hash, fixture, validate
from rpi_agents.cloud.app.context import Context
from rpi_agents.cloud.app.device_commands import DeviceCommandService
from rpi_agents.cloud.app.events import EventReader
from rpi_agents.cloud.app.images import ImageService
from rpi_agents.cloud.app.ingest import IngestService
from rpi_agents.cloud.app.publisher import Publisher
from rpi_agents.cloud.app.sessions import SessionService, provision_device
from rpi_agents.cloud.app.settings import Settings
from rpi_agents.cloud.app.storage import memory_storage
from rpi_agents.cloud.app.worker import Worker

START = datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


class TestClock:
    __test__ = False  # not a pytest class

    def __init__(self, start=START):
        self.dt = start

    def __call__(self):
        return self.dt

    def advance(self, seconds):
        self.dt += timedelta(seconds=seconds)

    def seconds(self):
        return self.dt.timestamp()


class FakeRuntime:
    """A test double for Patryk's SNNRuntime. `decisions` are returned by step() in order."""

    def __init__(self, *, load_error=None, reset_error=None, step_error=None, decisions=None):
        self.load_error, self.reset_error, self.step_error = load_error, reset_error, step_error
        self.loaded, self.resets, self.steps = None, [], []
        self.decisions = list(decisions or [])

    def load(self, manifest):
        if self.load_error:
            raise self.load_error
        self.loaded = manifest

    def reset(self, *, epoch, source_time_us):
        if self.reset_error:
            raise self.reset_error
        self.resets.append((epoch, source_time_us))

    def step(self, batch):
        if self.step_error:
            raise self.step_error
        self.steps.append(batch)
        if self.decisions:
            return self.decisions.pop(0)
        return {"trigger": False, "status": "valid", "score": None, "score_kind": "unavailable", "provenance": "demo"}

    def snapshot(self):
        return {}

    def checkpoint(self):
        return b""

    def restore(self, checkpoint):
        pass


QUIET = {"trigger": False, "status": "valid", "score": None, "score_kind": "unavailable", "provenance": "demo"}
class SpikeRuntime(FakeRuntime):
    """A test double that fires whenever a batch carries at least `threshold` spikes (a stand-in, not an SNN)."""

    threshold = 3

    def step(self, batch):
        self.steps.append(batch)
        fire = len(batch["spikes"]) >= self.threshold
        return {"trigger": fire, "status": "valid", "score": float(len(batch["spikes"])), "score_kind": "spike_count", "provenance": "demo"}


TRIGGER = {"trigger": True, "status": "valid", "score": 3.0, "score_kind": "spike_count", "provenance": "demo"}


class Env:
    def __init__(self, *, storage=None, **settings):
        self.clock = TestClock()
        self.storage = storage if storage is not None else memory_storage(self.clock.seconds)
        self.manifest = validate("ModelManifest", fixture("model-manifest"))
        self.runtime_kwargs, self.created, self.runtime_class = {}, [], FakeRuntime
        counter = itertools.count(1)

        def factory():
            runtime = self.runtime_class(**self.runtime_kwargs)
            self.created.append(runtime)
            return runtime

        self.ctx = Context(self.storage, Settings(**settings), self.manifest, factory, clock=self.clock,
                           new_id=lambda: f"id-{next(counter)}")  # fmt: skip
        provision_device(self.ctx, "demo-pi")
        self.sessions = SessionService(self.ctx)
        self.publisher = Publisher(self.ctx)
        self.ingest = IngestService(self.ctx, self.sessions, self.publisher)
        self.commands = DeviceCommandService(self.ctx, self.sessions)
        self.images = ImageService(self.ctx, self.publisher)
        self.events = EventReader(self.ctx)

    def create_body(self, **changes):
        return fixture("session-create") | {"device_id": "demo-pi", "model_hash": content_hash(self.manifest)} | changes

    def open_session(self, **changes):
        return self.sessions.create("demo-pi", self.create_body(**changes))

    def stop_body(self, state, **changes):
        return {"schema_version": "1.0", "request_id": "stop-1", "device_id": state["device_id"],
                "session_id": state["session_id"], "epoch": state["epoch"]} | changes  # fmt: skip

    def batch(self, state, seq, *, spikes=(), start=None, end=None, dropped=0, clipped=False, request_id=None, **changes):
        start = seq * 250_000 if start is None else start
        end = start + 250_000 if end is None else end
        return {
            "schema_version": "1.0", "request_id": request_id or f"b{seq}", "device_id": state["device_id"],
            "session_id": state["session_id"], "epoch": state["epoch"], "boot_id": state["boot_id"],
            "batch_seq": seq, "encoder_hash": state["encoder_hash"], "source_start_us": start, "source_end_us": end,
            "spikes": [{"dt_us": dt, "channel": channel} for dt, channel in spikes],
            "quality": {"dropped_events": dropped, "adc_clipped": clipped},
        } | changes  # fmt: skip

    def send(self, state, seq, **kw):
        return self.ingest.ingest(state["device_id"], state["session_id"], self.batch(state, seq, **kw))

    def trigger(self, state, seq=0):
        """Drive one SNN trigger through ingest and return (event_id, capture command)."""
        self.ctx.runtimes[state["session_id"]].decisions.append(TRIGGER)
        (command,) = self.send(state, seq, spikes=[(1200, "zcr")])["commands"]
        return command["event_id"], command

    def ack_body(self, command, status, **changes):
        base = {
            "schema_version": "1.0", "request_id": f"ack-{command['command_id']}-{status}", "device_id": command["device_id"],
            "session_id": command["session_id"], "epoch": command["epoch"], "command_id": command["command_id"],
            "status": status, "completed_at": "2026-09-24T12:00:05Z" if status == "completed" else None,
            "error_code": "X_FAILED" if status in ("failed", "expired") else None, "image_id": None,
        }  # fmt: skip
        return base | changes

    def ack(self, command, status, **changes):
        return self.commands.acknowledge(command["device_id"], command["command_id"], self.ack_body(command, status, **changes))

    def upload(self, event_id, index=0, data=None, **changes):
        import hashlib

        from tests.w0.fakes import make_jpeg

        data = make_jpeg() if data is None else data
        args = {"index": index, "sha256": hashlib.sha256(data).hexdigest(), "captured_at": "2026-09-24T12:00:04Z", "data": data}
        return self.images.upload("demo-pi", event_id, **(args | changes))

    def worker(self, vision, notifier=None, **kw):
        return Worker(self.ctx, self.publisher, vision, notifier, **kw)

    def pending_event(self, **session):
        """A session, a trigger and an uploaded photo: an event that waits for its analysis."""
        state = self.open_session(**session)
        event_id, command = self.trigger(state)
        self.upload(event_id)
        return state, event_id, command
