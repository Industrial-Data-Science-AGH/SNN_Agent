"""C1 dashboard: shell, demo fixtures and the sign-in contract.

Runs against the local dev harness (`create_dashboard_app`), which serves the
same dashboard router the real backend mounts plus a demo `/auth` stub. No
hardware, storage or SNN inference is involved.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rpi_agents.cloud.app.routes_dashboard import create_dashboard_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_dashboard_app())


# ------------------------------------------------------------------ shell

def test_shell_serves_english_sections(client: TestClient) -> None:
    body = client.get("/").text
    assert client.get("/").status_code == 200
    for label in ("Network", "Events", "Experiments", "Energy", "Device"):
        assert f">{label}" in body or label in body
    assert "Sign in" in body
    assert "Log out" in body


def test_shell_has_no_polish_product_strings(client: TestClient) -> None:
    body = client.get("/").text
    # A few obvious Polish labels that must never ship in the product UI.
    for polish in ("Zaloguj", "Sieć", "Zdarzenia", "Urządzenie", "Wyloguj"):
        assert polish not in body


# ------------------------------------------------------------------ demo fixtures

def test_fixture_index_is_demo_tagged(client: TestClient) -> None:
    body = client.get("/dashboard/fixtures").json()
    assert body["demo"] is True
    assert "device-status" in body["items"]
    assert "neuron-frame" in body["items"]


def test_device_status_fixture_shape(client: TestClient) -> None:
    body = client.get("/dashboard/fixtures/device-status").json()
    assert body["demo"] is True
    assert body["fixture"] == "device-status"
    assert body["data"]["device_id"] == "demo-pi"
    assert "serial" in body["data"]


def test_unknown_fixture_is_404(client: TestClient) -> None:
    res = client.get("/dashboard/fixtures/not-a-real-fixture")
    assert res.status_code == 404
    assert res.json()["error"]["code"] == "UNKNOWN_FIXTURE"


# ------------------------------------------------------------------ sign-in contract

def test_session_requires_login(client: TestClient) -> None:
    res = client.get("/auth/session")
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "UNAUTHORIZED"


def test_login_rejects_wrong_credentials(client: TestClient) -> None:
    res = client.post("/auth/login", json={"username": "operator", "password": "wrong"})
    assert res.status_code == 401
    assert res.json()["error"]["code"] == "INVALID_CREDENTIALS"


#  C2 network editor

def test_neuron_svg_has_distinct_layers(client: TestClient) -> None:
    body = client.get("/static/img/neuron.svg").text
    assert client.get("/static/img/neuron.svg").status_code == 200
    # Not a plain circle: named layers the editor/runtime address separately.
    for layer in ("lui-board", "board-sel", "board-ports", "led-potential", "led-spike"):
        assert layer in body


def test_network_module_is_served(client: TestClient) -> None:
    res = client.get("/static/js/network.js")
    assert res.status_code == 200
    assert "mountNetworkEditor" in res.text


# C3 runtime signals

def test_golden_frames_served_and_consistent(client: TestClient) -> None:
    doc = client.get("/static/demo/neuron-frames.json").json()
    assert doc["demo"] is True
    ids = doc["neuron_ids"]
    assert ids == [f"n{i}" for i in range(1, 9)]
    assert doc["frames"], "golden replay has frames"
    assert "v_threshold" in doc and "potential_unit" in doc
    # Every frame carries exactly the declared neurons with runtime fields —
    # this is what makes the LEDs and the raster agree on neuron/time.
    for frame in doc["frames"]:
        assert [n["neuron_id"] for n in frame["neurons"]] == ids
        for n in frame["neurons"]:
            assert isinstance(n["spiked"], bool)
            assert isinstance(n["v_mem"], (int, float))
    # There is real spiking to render (not a dead/random signal).
    spikes = sum(1 for f in doc["frames"] for n in f["neurons"] if n["spiked"])
    assert spikes > 0


def test_runtime_modules_served(client: TestClient) -> None:
    assert "createRuntime" in client.get("/static/js/runtime.js").text
    assert "mountInspector" in client.get("/static/js/inspector.js").text
    assert "mountRaster" in client.get("/static/js/raster.js").text


# C4 events / experiments / energy

def test_events_fixture_shape(client: TestClient) -> None:
    doc = client.get("/static/demo/events.json").json()
    assert doc["demo"] is True
    ids = {e["event_id"] for e in doc["items"]}
    assert {"evt-001", "evt-002", "evt-004"} <= ids
    for ev in doc["items"]:
        assert "decision" in ev and "vision" in ev and "commands" in ev
    # a failed event with no vision result exercises the "Not available" path
    failed = next(e for e in doc["items"] if e["event_id"] == "evt-004")
    assert failed["vision"] is None


def test_experiments_metrics_are_separated_with_ci(client: TestClient) -> None:
    doc = client.get("/static/demo/experiments.json").json()
    assert doc["demo"] is True
    runs = doc["runs"]
    assert len(runs) >= 2
    # different models/datasets so the filter must not mix them
    assert len({r["model_hash"] for r in runs}) >= 2
    for r in runs:
        for scope in ("snn", "system"):  # SNN vs whole-system kept separate
            fa = r[scope]["fa_per_h"]
            assert {"value", "ci_low", "ci_high"} <= fa.keys()
            assert "recall" in r[scope]


def test_energy_keeps_sources_separate_and_allows_missing(client: TestClient) -> None:
    doc = client.get("/static/demo/energy.json").json()
    assert doc["demo"] is True
    kinds = {s["source"] for s in doc["sources"]}
    assert "measured" in kinds and "estimated" in kinds  # not mixed into one number
    for s in doc["sources"]:
        assert "boundary" in s
    # a source with no measurement is null, never 0
    missing = [s for s in doc["sources"] if s["power_w"] is None]
    assert missing and all(s["energy_j"] is None for s in missing)


def test_c4_module_served(client: TestClient) -> None:
    body = client.get("/static/js/c4.js").text
    for fn in ("mountEvents", "mountExperiments", "mountEnergy"):
        assert fn in body


def test_login_then_session_and_logout(client: TestClient) -> None:
    res = client.post("/auth/login", json={"username": "operator", "password": "demo"})
    assert res.status_code == 200
    assert res.json()["csrf_token"]
    assert res.json()["actor"] == "shared_operator"

    who = client.get("/auth/session")
    assert who.status_code == 200
    assert who.json()["authenticated"] is True

    out = client.post("/auth/logout", headers={"X-CSRF-Token": "demo-csrf-token"})
    assert out.status_code == 204
    # Cookie cleared → session no longer valid.
    assert client.get("/auth/session").status_code == 401
