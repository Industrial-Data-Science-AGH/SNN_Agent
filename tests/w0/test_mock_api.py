import json

import pytest
from fastapi.testclient import TestClient

from contracts.validation import fixture, validate
from rpi_agents.cloud.app.mock_api import create_app


@pytest.fixture
def client():
    with TestClient(create_app(), base_url="http://127.0.0.1") as c:
        yield c


def post(client, path, body):
    return client.post(path, json=body, headers={"Idempotency-Key": body["request_id"]})


def session(client, scenario="trigger", mode="demo"):
    body = fixture("session-create") | {"mode": mode}
    r = post(client, f"/v1/sessions?scenario={scenario}", body)
    assert r.status_code == 201, r.text
    validate("SessionState", r.json())
    return r.json()


def batch(s, name):
    return fixture(name) | {k: s[k] for k in ("device_id", "session_id", "epoch", "boot_id")}


def test_end_to_end_demo_and_idempotence(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}/batches"
    silence = post(client, path, batch(s, "silence"))
    assert silence.status_code == 200
    assert silence.json()["decision"]["trigger"] is False
    b = batch(s, "spike")
    r = post(client, path, b)
    assert r.status_code == 200, r.text
    validate("BatchAck", r.json())
    assert r.json()["decision"]["trigger"] is True
    assert r.json()["durable_seq"] is None
    assert post(client, path, b).json() == r.json()
    events = client.get("/v1/events").json()
    assert len(events["items"]) == 1
    e = events["items"][0]
    validate("Event", e)
    assert client.get(f"/v1/events/{e['event_id']}").json() == e
    cmds = client.get(f"/v1/devices/{s['device_id']}/commands").json()["items"]
    assert len(cmds) == 1 and cmds[0]["type"] == "capture" and cmds[0]["mode"] == "demo"
    assert post(client, path, b | {"source_end_us": 510000}).status_code == 409


def test_gap_quality_epoch_and_out_of_order(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}/batches"
    assert post(client, path, batch(s, "silence")).status_code == 200
    r = post(client, path, batch(s, "gap"))
    assert r.status_code == 200
    assert r.json()["status"] == "gap"
    assert r.json()["gaps"][0]["source_start_us"] == 250000
    assert r.json()["decision"]["trigger"] is False
    assert post(client, path, batch(s, "spike")).status_code == 409
    for key, value in [("epoch", 2), ("boot_id", "reboot"), ("device_id", "another")]:
        assert post(client, path, batch(s, "spike") | {key: value}).status_code == 409


def test_validation_and_payload_limits(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}/batches"
    b = batch(s, "silence")
    assert client.post(path, json=b).status_code == 422
    for change, status in [
        ({"schema_version": "2.0"}, 409),
        ({"spikes": [{"dt_us": 1, "channel": "bad"}]}, 422),
    ]:
        r = post(client, path, b | change)
        assert r.status_code == status
        validate("Error", r.json())
    assert (
        client.post(path, content=b" " * 65537, headers={"Content-Type": "application/json"}).status_code
        == 413
    )
    assert (
        client.post(path, content='{"bad":NaN}', headers={"Content-Type": "application/json"}).status_code
        == 422
    )
    assert (
        client.post(path, content='{"x":1,"x":2}', headers={"Content-Type": "application/json"}).status_code
        == 422
    )


def test_unavailable_vision_and_replay_cannot_alarm(client):
    s = session(client, "vision_unavailable")
    path = f"/v1/sessions/{s['session_id']}/batches"
    post(client, path, batch(s, "silence"))
    r = post(client, path, batch(s, "spike"))
    event = client.get("/v1/events").json()["items"][0]
    assert event["status"] == "review_required"
    assert event["vision"]["glass_visible"] == "unknown"
    assert all(c["type"] != "alarm" for c in r.json()["commands"])


def test_replay_has_no_commands(client):
    s = session(client, mode="replay")
    path = f"/v1/sessions/{s['session_id']}/batches"
    post(client, path, batch(s, "silence"))
    assert post(client, path, batch(s, "spike")).json()["commands"] == []
    assert client.get(f"/v1/devices/{s['device_id']}/commands").json()["items"] == []


def test_telemetry_stop_ack(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}"
    post(client, path + "/batches", batch(s, "silence"))
    r = post(client, path + "/batches", batch(s, "spike"))
    c = r.json()["commands"][0]
    a = fixture("command-ack") | {k: s[k] for k in ("device_id", "session_id", "epoch")}
    a.update(command_id=c["command_id"])
    assert post(client, f"/v1/commands/{c['command_id']}/ack", a).status_code == 200
    assert client.get(f"/v1/devices/{s['device_id']}/commands").json()["items"] == []
    stream = client.get(path + "/telemetry")
    assert "event: snapshot" in stream.text
    data = json.loads(next(line[6:] for line in stream.text.splitlines() if line.startswith("data: ")))
    validate("NeuronFrame", data)
    control = {k: s[k] for k in ("schema_version", "device_id", "session_id", "epoch")} | {
        "request_id": "stop-1"
    }
    stop = post(client, path + "/stop", control)
    assert stop.json()["state"] == "stopped"
    assert post(client, path + "/stop", control).json() == stop.json()
    assert post(client, path + "/batches", batch(s, "gap")).status_code == 409


def test_local_only_demo_and_stable_openapi(client):
    assert client.get("/healthz").json()["demo"] is True
    assert client.get("/healthz", headers={"Host": "evil.example"}).status_code == 403
    assert client.get("/healthz", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/v1/events/absent").status_code == 404
    spec = client.get("/openapi.json").json()
    assert "SpikeBatch" in spec["components"]["schemas"]
    assert spec["paths"]["/v1/sessions/{session_id}/batches"]["post"]["requestBody"]["content"][
        "application/json"
    ]["schema"]["$ref"].endswith("/SpikeBatch")
    body = fixture("session-create") | {"mode": "live"}
    assert post(client, "/v1/sessions", body).status_code == 409


def test_create_retry_and_scenario_conflict(client):
    body = fixture("session-create")
    first = post(client, "/v1/sessions?scenario=trigger", body)
    assert post(client, "/v1/sessions?scenario=trigger", body).json() == first.json()
    assert post(client, "/v1/sessions?scenario=silence", body).status_code == 409
    assert post(client, "/v1/sessions", body | {"request_id": "another-create"}).status_code == 409


def test_rejected_batch_does_not_advance_session(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}"
    bad = batch(s, "silence") | {"spikes": [{"dt_us": 250000, "channel": "zcr"}]}
    assert post(client, path + "/batches", bad).status_code == 422
    assert client.get(path).json()["received_seq"] is None
    assert post(client, path + "/batches", batch(s, "silence")).status_code == 200


@pytest.mark.parametrize(
    "quality", [{"adc_clipped": True, "dropped_events": 0}, {"adc_clipped": False, "dropped_events": 1}]
)
def test_corrupt_observation_never_triggers(client, quality):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}/batches"
    post(client, path, batch(s, "silence"))
    r = post(client, path, batch(s, "spike") | {"quality": quality})
    assert r.status_code == 200
    assert r.json()["status"] == "gap"
    assert not r.json()["decision"]["trigger"]
    assert r.json()["commands"] == []


def test_ack_identity_and_terminal_state(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}/batches"
    post(client, path, batch(s, "silence"))
    c = post(client, path, batch(s, "spike")).json()["commands"][0]
    body = fixture("command-ack") | {k: s[k] for k in ("device_id", "session_id", "epoch")}
    body["command_id"] = c["command_id"]
    route = f"/v1/commands/{c['command_id']}/ack"
    assert post(client, route, body | {"device_id": "foreign"}).status_code == 409
    assert post(client, route, body).status_code == 200
    assert post(client, route, body).status_code == 200
    assert (
        post(
            client, route, body | {"request_id": "ack2", "status": "accepted", "error_code": None}
        ).status_code
        == 409
    )


def test_expired_command_not_returned(client):
    s = session(client)
    path = f"/v1/sessions/{s['session_id']}/batches"
    post(client, path, batch(s, "silence"))
    cmd = post(client, path, batch(s, "spike")).json()["commands"][0]
    client.app.state.demo_store.commands[cmd["command_id"]]["expires_at"] = "2000-01-01T00:00:00Z"
    assert client.get(f"/v1/devices/{s['device_id']}/commands").json()["items"] == []
    ack = fixture("command-ack") | {k: s[k] for k in ("device_id", "session_id", "epoch")}
    ack.update(command_id=cmd["command_id"], status="accepted", error_code=None)
    assert post(client, f"/v1/commands/{cmd['command_id']}/ack", ack).status_code == 409


def test_body_attacks_are_bounded_and_do_not_leak_values(client):
    route = "/v1/sessions"
    for body in [b"\xff", b"[" * 1200 + b"0" + b"]" * 1200, b"null", b"[]"]:
        r = client.post(route, content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 422
    body = fixture("session-create") | {"secret": "DO_NOT_ECHO_THIS"}
    r = post(client, route, body)
    assert r.status_code == 422 and "DO_NOT_ECHO_THIS" not in r.text
    assert client.post(route, content="{}").status_code == 415


def test_demo_capacity_does_not_evict_retry_records(client):
    store = client.app.state.demo_store
    store.retries.update({("reserved", str(i), "r"): ("digest", {}) for i in range(1024)})
    response = post(client, "/v1/sessions", fixture("session-create"))
    assert response.status_code == 429
    assert store.sessions == {}


def test_device_status_latest_wins_and_never_uses_the_idempotency_budget(client):
    body = fixture("device-status")
    r = post(client, "/v1/devices/demo-pi/status", body)
    assert r.status_code == 200
    validate("DeviceStatus", r.json())
    assert client.get("/v1/devices/demo-pi/status").json() == body
    for i in range(1100):  # more heartbeats than the 1024 idempotent-mutation budget
        sent = post(client, "/v1/devices/demo-pi/status", body | {"request_id": f"status-{i + 2}", "uptime_s": i})
        assert sent.status_code == 200
    assert client.get("/v1/devices/demo-pi/status").json()["uptime_s"] == 1099
    assert session(client)["state"] == "running"  # sessions are still creatable: the budget is untouched


def test_device_status_rejects_mismatch_unknown_device_and_bad_payloads(client):
    body = fixture("device-status")
    r = post(client, "/v1/devices/other-pi/status", body)
    assert (r.status_code, r.json()["error"]["code"]) == (409, "DEVICE_MISMATCH")
    assert client.get("/v1/devices/nobody/status").status_code == 404
    assert post(client, "/v1/devices/demo-pi/status", body | {"epoch": None}).status_code == 422
    assert post(client, "/v1/devices/demo-pi/status", body | {"surprise": 1}).status_code == 422
    assert post(client, "/v1/devices/demo-pi/status", body | {"reported_at": "2026-09-24T00:00:05"}).status_code == 422
    assert post(client, "/v1/devices/demo-pi/status", body | {"state": "asleep"}).status_code == 422
    assert client.get("/v1/devices/demo-pi/status").status_code == 404  # nothing invalid was stored


def test_device_status_is_capped_at_sixteen_devices(client):
    body = fixture("device-status")
    for i in range(16):
        assert post(client, f"/v1/devices/pi-{i}/status", body | {"device_id": f"pi-{i}"}).status_code == 200
    r = post(client, "/v1/devices/pi-16/status", body | {"device_id": "pi-16"})
    assert (r.status_code, r.json()["error"]["code"]) == (429, "DEMO_CAPACITY")
    assert post(client, "/v1/devices/pi-3/status", body | {"device_id": "pi-3", "request_id": "again"}).status_code == 200
