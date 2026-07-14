"""Shared HTTP Basic Auth dependency (F05 design, ADR-0009).

One dependency, applied globally to the whole app in `main.py` — every
route, including the dashboard pages and `/` itself, sits behind this. There
is no unauthenticated route anywhere in `cloud/app`.
"""

import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

_MSG_INVALID_CREDENTIALS = "Invalid credentials"

_security = HTTPBasic()


def require_basic_auth(credentials: HTTPBasicCredentials = Depends(_security)) -> str:
    """Validate the request's Basic Auth credential; raise 401 on mismatch.

    Compares against `DASHBOARD_USER`/`DASHBOARD_PASSWORD` env vars (default
    `ids`/`ids`, ADR-0009) using `secrets.compare_digest` for both the
    username and the password, so neither comparison leaks timing
    information about how much of the guess was correct. Env vars are read
    per-call (not cached at import time) so tests can set/monkeypatch them
    per test without reloading this module.

    Returns the authenticated username (unused by callers today, but a
    natural place for a future audit-log field without changing the
    dependency's shape).
    """
    expected_user = os.getenv("DASHBOARD_USER", "ids")
    expected_password = os.getenv("DASHBOARD_PASSWORD", "ids")

    user_ok = secrets.compare_digest(credentials.username, expected_user)
    password_ok = secrets.compare_digest(credentials.password, expected_password)

    if not (user_ok and password_ok):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_MSG_INVALID_CREDENTIALS,
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
