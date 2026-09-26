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
