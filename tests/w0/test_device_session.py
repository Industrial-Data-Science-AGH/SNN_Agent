"""GET /v1/devices/{device_id}/session -- what the dashboard's LiveSource needs before it
can point LiveRuntime at a real session_id. Without this, createRuntime's own default
("demo-session") was the only value ever used in Live mode, and that session never exists
(see static/js/runtime.js createRuntime / static/js/data.js LiveSource)."""
import pytest
from fastapi.testclient import TestClient

from rpi_agents.cloud.app.api import ApiSettings, Services, create_app
from rpi_agents.cloud.app.auth import OperatorAuth, hash_password
from rpi_agents.cloud.app.status import StatusService
from tests.w0.backend_env import Env

PASSWORD = "pw-for-tests"


@pytest.fixture
def env_and_client():
    env = Env()
    services = Services(env.sessions, env.ingest, env.commands, env.images, env.events, StatusService(env.ctx), env.ctx)
    operator = OperatorAuth(env.ctx, username="operator", password_hash=hash_password(PASSWORD, log2_n=14))
    client = TestClient(create_app(services, operator, ApiSettings(trusted_proxies=0)), base_url="https://testserver")
    return env, client


def test_current_session_requires_an_operator_session(env_and_client):
    _env, client = env_and_client
    r = client.get("/v1/devices/demo-pi/session")
    assert r.status_code == 401


def test_current_session_404s_for_an_unknown_device(env_and_client):
    _env, client = env_and_client
    client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    r = client.get("/v1/devices/no-such-device/session")
    assert r.status_code == 404


def test_current_session_is_null_before_any_session_is_open(env_and_client):
    _env, client = env_and_client
    client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    r = client.get("/v1/devices/demo-pi/session")
    assert r.status_code == 200
    body = r.json()
    assert body["device_id"] == "demo-pi"
    assert body["session_id"] is None


def test_current_session_returns_the_live_session_id_once_one_is_open(env_and_client):
    env, client = env_and_client
    state = env.open_session()
    client.post("/auth/login", json={"username": "operator", "password": PASSWORD})
    r = client.get("/v1/devices/demo-pi/session")
    assert r.status_code == 200
    assert r.json()["session_id"] == state["session_id"]
