"""Tests for cloud/app/auth.py and F05's global-auth contract (T02 plan,
Step 2): every route class — dashboard, read API, ingest API — must reject
missing/bad Basic Auth with 401. No Storage dependency needed for any of
this (`require_basic_auth` never touches storage.py).
"""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from cloud.app.auth import require_basic_auth
from cloud.app.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _fixed_credentials(monkeypatch):
    """Pin DASHBOARD_USER/PASSWORD so tests don't depend on ambient env vars."""
    monkeypatch.setenv("DASHBOARD_USER", "testuser")
    monkeypatch.setenv("DASHBOARD_PASSWORD", "testpass")


def _credentials(username: str, password: str) -> SimpleNamespace:
    """Stand-in for `HTTPBasicCredentials` — same two attributes, no FastAPI
    request machinery needed to call `require_basic_auth` directly.
    """
    return SimpleNamespace(username=username, password=password)


def test_require_basic_auth_accepts_correct_credentials():
    assert require_basic_auth(_credentials("testuser", "testpass")) == "testuser"


def test_require_basic_auth_rejects_wrong_password():
    with pytest.raises(HTTPException) as exc_info:
        require_basic_auth(_credentials("testuser", "wrong"))
    assert exc_info.value.status_code == 401
    assert exc_info.value.headers == {"WWW-Authenticate": "Basic"}


def test_require_basic_auth_rejects_wrong_username():
    with pytest.raises(HTTPException) as exc_info:
        require_basic_auth(_credentials("wrong", "testpass"))
    assert exc_info.value.status_code == 401


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/"),
        ("GET", "/events/some-id"),
        ("GET", "/api/events"),
        ("POST", "/api/events"),
        ("GET", "/api/events/some-id"),
        ("GET", "/api/metrics"),
        ("PATCH", "/api/events/some-id"),
    ],
)
def test_every_route_class_rejects_missing_auth(method, path):
    """T02 plan, acceptance gate: unauthenticated dashboard, read, and
    ingest requests all return 401. T05 (ADR-0016) adds /api/metrics to
    this enumeration — it sits behind the same global dependency as every
    other route, no per-route auth logic to forget. T06 (ADR-0018) adds
    the PATCH review route the same way.
    """
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Basic"


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("GET", "/"),
        ("GET", "/api/events"),
        ("POST", "/api/events"),
        ("GET", "/api/metrics"),
        ("PATCH", "/api/events/some-id"),
    ],
)
def test_every_route_class_rejects_bad_auth(method, path):
    response = client.request(method, path, auth=("wrong", "wrong"))
    assert response.status_code == 401
