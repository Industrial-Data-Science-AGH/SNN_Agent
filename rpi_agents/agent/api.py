"""HTTP transport and API client for the edge. Standard library only (urllib, ssl).

Security rules enforced here, not left to callers:
- https is required for any non-loopback host; a credential never travels in clear text.
- Redirects are never followed, so an Authorization header cannot be forwarded to another host.
- Response bodies are bounded; the token is never part of an error message or a log line.

Every call ends in a Result whose outcome tells the caller what to do: OK, RETRY (network failure, 5xx,
429, 408, 425 and 401/403, so a rotated credential never costs data) or PERMANENT (a 4xx that will not
change on retry, for example a 409 conflict). Callers keep order and never drop on RETRY.
"""

from __future__ import annotations

import http.client
import json
import random
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Mapping, Protocol
from urllib.parse import quote, urlsplit

LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}
MAX_RESPONSE_BYTES = 1 << 20
MAX_RETRY_AFTER_S = 300.0
_RETRYABLE = {401, 403, 408, 425, 429}


class TransportError(ConnectionError):
    """The request never produced an HTTP response (refused, reset, timeout, TLS failure)."""


@dataclass(frozen=True)
class Response:
    status: int
    body: dict | None
    retry_after_s: float | None = None


class Transport(Protocol):
    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> Response: ...


def validate_base_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("backend url must be http(s)://host[:port]")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ValueError("backend url must not contain credentials, a query or a fragment")
    if parts.scheme == "http" and parts.hostname not in LOOPBACK_HOSTS:
        raise ValueError("plain http is only allowed for loopback; use https")
    return url.rstrip("/")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # noqa: D401 - urllib hook
        return None


def _retry_after(headers) -> float | None:
    try:
        value = float(headers.get("Retry-After", ""))
    except (TypeError, ValueError):
        return None
    return min(max(value, 0.0), MAX_RETRY_AFTER_S)


class UrllibTransport:
    def __init__(self, base_url: str, *, token: str | None = None, ca_file: str | None = None):
        self._base = validate_base_url(base_url)
        self._token = token
        handlers: list = [_NoRedirect()]
        if self._base.startswith("https"):
            handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context(cafile=ca_file)))
        self._opener = urllib.request.build_opener(*handlers)

    def request(
        self,
        method: str,
        path: str,
        body: dict | None = None,
        *,
        headers: Mapping[str, str] | None = None,
        timeout_s: float = 5.0,
    ) -> Response:
        if not path.startswith("/"):
            raise ValueError("path must start with '/'")
        data = None if body is None else json.dumps(body, separators=(",", ":"), allow_nan=False).encode()
        req = urllib.request.Request(self._base + path, data=data, method=method)
        req.add_header("Accept", "application/json")
        if data is not None:
            req.add_header("Content-Type", "application/json")
        if self._token:
            req.add_header("Authorization", f"Bearer {self._token}")
        for name, value in (headers or {}).items():
            req.add_header(name, value)
        try:
            try:
                reply = self._opener.open(req, timeout=timeout_s)
            except urllib.error.HTTPError as exc:  # an HTTP status is a response, not a transport failure
                reply = exc
            with reply:
                raw = reply.read(MAX_RESPONSE_BYTES + 1)
                status, reply_headers = reply.status, reply.headers
        except (urllib.error.URLError, OSError, ssl.SSLError, http.client.HTTPException) as exc:
            raise TransportError(type(exc).__name__) from None
        parsed = None
        if len(raw) <= MAX_RESPONSE_BYTES:
            try:
                value = json.loads(raw.decode("utf-8")) if raw else None
                parsed = value if isinstance(value, dict) else None
            except (ValueError, UnicodeDecodeError):
                parsed = None
        return Response(status, parsed, _retry_after(reply_headers))


class Outcome(str, Enum):
    OK = "ok"
    RETRY = "retry"
    PERMANENT = "permanent"


@dataclass(frozen=True)
class Result:
    outcome: Outcome
    status: int | None
    code: str | None
    body: dict | None
    retry_after_s: float | None = None
    detail: str = ""


def classify(response: Response) -> Result:
    error = (response.body or {}).get("error")
    code = error.get("code") if isinstance(error, dict) and isinstance(error.get("code"), str) else None
    if 200 <= response.status < 300:
        outcome = Outcome.OK
    elif response.status in _RETRYABLE or response.status >= 500:
        outcome = Outcome.RETRY
    else:
        outcome = Outcome.PERMANENT
    return Result(outcome, response.status, code, response.body, response.retry_after_s, f"HTTP {response.status}")


class ApiClient:
    """The endpoints the bridge uses. Ids are quoted into paths; POSTs carry Idempotency-Key = request_id."""

    def __init__(self, transport: Transport, *, timeout_s: float = 5.0):
        self._transport, self._timeout = transport, timeout_s

    def _call(self, method: str, path: str, body: dict | None = None) -> Result:
        headers = {"Idempotency-Key": body["request_id"]} if method == "POST" and body else None
        try:
            response = self._transport.request(method, path, body, headers=headers, timeout_s=self._timeout)
        except TransportError as exc:
            return Result(Outcome.RETRY, None, "TRANSPORT_ERROR", None, None, str(exc) or "transport error")
        return classify(response)

    def create_session(self, body: dict, scenario: str | None = None) -> Result:
        suffix = f"?scenario={quote(scenario, safe='')}" if scenario else ""
        return self._call("POST", f"/v1/sessions{suffix}", body)

    def post_batch(self, session_id: str, body: dict) -> Result:
        return self._call("POST", f"/v1/sessions/{quote(session_id, safe='')}/batches", body)

    def stop_session(self, session_id: str, body: dict) -> Result:
        return self._call("POST", f"/v1/sessions/{quote(session_id, safe='')}/stop", body)

    def poll_commands(self, device_id: str) -> Result:
        return self._call("GET", f"/v1/devices/{quote(device_id, safe='')}/commands")

    def ack_command(self, command_id: str, body: dict) -> Result:
        return self._call("POST", f"/v1/commands/{quote(command_id, safe='')}/ack", body)

    def post_status(self, device_id: str, body: dict) -> Result:
        return self._call("POST", f"/v1/devices/{quote(device_id, safe='')}/status", body)


class Backoff:
    """Exponential delay with 50-100 % jitter; a server Retry-After hint is a floor, capped."""

    def __init__(self, base: float = 0.5, cap: float = 30.0, rand: Callable[[], float] = random.random):
        self._base, self._cap, self._rand, self._attempts = base, cap, rand, 0

    def next_delay(self, hint: float | None = None) -> float:
        delay = min(self._cap, self._base * 2**self._attempts) * (0.5 + self._rand() / 2)
        self._attempts = min(self._attempts + 1, 30)
        return min(max(delay, hint or 0.0), MAX_RETRY_AFTER_S)

    def reset(self) -> None:
        self._attempts = 0
