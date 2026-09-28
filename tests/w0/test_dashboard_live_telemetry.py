"""GET /v1/sessions/{session_id}/telemetry -- the SSE feed the dashboard's LiveRuntime
depends on. Was only ever a one-shot stub in mock_api.py ("production will stream"); this
tests the real, streaming version wired into api.py against the live `ctx.runtimes[session_id]`
the ingest path already steps."""
import json

import pytest
from fastapi.testclient import TestClient

from rpi_agents.cloud.app.api import ApiSettings, Services, create_app
from rpi_agents.cloud.app.auth import OperatorAuth, hash_password
from rpi_agents.cloud.app.status import StatusService
from snn_runtime.errors import RuntimeStateError
from tests.w0.backend_env import Env

PASSWORD = "pw-for-tests"


class SnapshotRuntime:
    """A fake SNNRuntime whose snapshot() hands back real NeuronFrame-shaped dicts in
    order, then removes itself from ctx.runtimes once exhausted -- the same way a real
    session ending (stop, or the device retiring it) makes the endpoint see it vanish."""

    def __init__(self, frames, *, ctx, session_id):
        self._frames, self._ctx, self._session_id = list(frames), ctx, session_id

    def load(self, manifest):
        pass

    def reset(self, *, epoch, source_time_us):
        pass

    def step(self, batch):
        return {"trigger": False, "status": "valid", "score": None, "score_kind": "unavailable", "provenance": "demo"}

    def snapshot(self):
        frame = self._frames.pop(0)
        if not self._frames:
            self._ctx.runtimes.pop(self._session_id, None)
        return frame

    def checkpoint(self):
        return b""

    def restore(self, checkpoint):
        pass


def _frame(*, session_id, frame_seq, source_time_us, status="running"):
    return {
        "schema_version": "1.0", "device_id": "demo-pi", "session_id": session_id, "epoch": 1,
        "source_time_us": source_time_us, "model_hash": "sha256:" + "0" * 64, "frame_seq": frame_seq,
        "topology_version": "t1", "status": status, "provenance": "simulated", "potential_unit": "a.u.",
        "neurons": [{"neuron_id": "n0", "v_mem": 0.5, "v_threshold": 1.0, "v_reset": 0.0, "spiked": False}],
    }  # fmt: skip


@pytest.fixture
def env_and_client():
    env = Env()
    services = Services(env.sessions, env.ingest, env.commands, env.images, env.events, StatusService(env.ctx), env.ctx)
    operator = OperatorAuth(env.ctx, username="operator", password_hash=hash_password(PASSWORD, log2_n=14))
    client = TestClient(create_app(services, operator, ApiSettings(trusted_proxies=0)), base_url="https://testserver")
    return env, client


def _data_lines(lines):
    return [line[len("data:") :].strip() for line in lines if line.startswith("data:")]


def test_telemetry_requires_an_operator_session(env_and_client):
    env, client = env_and_client
    state = env.open_session()
    r = client.get(f"/v1/sessions/{state['session_id']}/telemetry")
    assert r.status_code == 401


def test_telemetry_404s_for_a_session_that_is_not_live(env_and_client):
    _env, client = env_and_client
    client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    r = client.get("/v1/sessions/no-such-session/telemetry")
    assert r.status_code == 404


def test_telemetry_streams_neuron_frames_then_ends_when_the_session_does(env_and_client):
    env, client = env_and_client
    state = env.open_session()
    session_id = state["session_id"]
    env.ctx.runtimes[session_id] = SnapshotRuntime(
        [_frame(session_id=session_id, frame_seq=0, source_time_us=0),
         _frame(session_id=session_id, frame_seq=1, source_time_us=200_000)],
        ctx=env.ctx, session_id=session_id,
    )  # fmt: skip
    client.post("/auth/login", json={"username": "operator", "password": PASSWORD})

    lines = []
    with client.stream("GET", f"/v1/sessions/{session_id}/telemetry") as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        for line in r.iter_lines():
            lines.append(line)
            if len(_data_lines(lines)) >= 2:
                break

    data = [json.loads(d) for d in _data_lines(lines)]
    assert [f["frame_seq"] for f in data] == [0, 1]  # the feed's own sequence, not the runtime's internal one
    assert data[0]["session_id"] == session_id
    assert data[0]["neurons"][0]["neuron_id"] == "n0"
    # the runtime removed itself from ctx.runtimes after the 2nd frame (session ended, same as a real stop)
    assert session_id not in env.ctx.runtimes


def test_telemetry_waits_out_no_stream_identity_instead_of_dying(env_and_client):
    """snapshot() raises NO_STREAM_IDENTITY before the first batch (runtime.py's own docstring:
    "take the snapshot after the first batch"). The feed must poll through that, not 500 or end."""
    env, client = env_and_client
    state = env.open_session()
    session_id = state["session_id"]

    class NotYetRuntime:
        def __init__(self):
            self.calls = 0

        def snapshot(self):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeStateError("NO_STREAM_IDENTITY", "no batch yet")
            frame = _frame(session_id=session_id, frame_seq=0, source_time_us=0)
            env.ctx.runtimes.pop(session_id, None)  # end the stream right after the one real frame
            return frame

    env.ctx.runtimes[session_id] = NotYetRuntime()
    client.post("/auth/login", json={"username": "operator", "password": PASSWORD})

    lines = []
    with client.stream("GET", f"/v1/sessions/{session_id}/telemetry") as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            lines.append(line)
            if _data_lines(lines):
                break

    data = [json.loads(d) for d in _data_lines(lines)]
    assert len(data) == 1
    assert data[0]["frame_seq"] == 0
