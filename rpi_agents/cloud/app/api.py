"""The production HTTP API. Requires the backend dependencies (fastapi, jsonschema); the demo mock stays separate.

Two kinds of caller, two kinds of credential, never mixed:
- a DEVICE presents `Authorization: Bearer <device_id>~<secret>` and can only reach its own device's data;
- an OPERATOR (the team's shared login) presents the session cookie, and every request that changes state must
  also carry the session's CSRF token and a same-origin Origin header.
Nothing is public except /healthz (which reveals nothing) and the login itself.

Bodies are size-limited while they are read, JSON is parsed strictly (no duplicate keys, no NaN), every mutation
needs Idempotency-Key equal to its request_id, and error responses never carry internals. Interactive API docs
are off unless explicitly enabled for development.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from contracts.validation import ContractError
from rpi_agents.cloud.app.auth import (
    OperatorAuth,
    WebSession,
    authenticate_device,
    check_csrf,
    client_ip,
)
from rpi_agents.cloud.app.device_commands import DeviceCommandService
from rpi_agents.cloud.app.events import EventReader
from rpi_agents.cloud.app.images import ImageService
from rpi_agents.cloud.app.imaging import MAX_IMAGE_BYTES
from rpi_agents.cloud.app.ingest import IngestService
from rpi_agents.cloud.app.records import DEVICES_PK, T_DEVICES
from rpi_agents.cloud.app.sessions import SessionService
from rpi_agents.cloud.app.status import StatusService

log = logging.getLogger("snn_backend.api")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}")
MAX_JSON_BYTES = 65536
COOKIE_SECURE, COOKIE_DEV = "__Host-snn_session", "snn_session"


@dataclass
class Services:
    sessions: SessionService
    ingest: IngestService
    commands: DeviceCommandService
    images: ImageService
    events: EventReader
    status: StatusService
    ctx: object


@dataclass(frozen=True)
class ApiSettings:
    trusted_proxies: int = 1
    allowed_hosts: tuple[str, ...] = ()  # empty = any host (tests); set it in production
    insecure_dev: bool = False  # plain-http development: drops Secure and the __Host- cookie prefix
    enable_docs: bool = False


def _hostname(raw: str) -> str:
    """The host part of a Host header, without the port; IPv6 literals keep their brackets."""
    raw = raw.strip().lower()
    if raw.startswith("["):
        return raw[: raw.index("]") + 1] if "]" in raw else raw
    return raw.rsplit(":", 1)[0] if ":" in raw else raw


def _error(exc: ContractError, headers: dict | None = None) -> JSONResponse:
    body = {"schema_version": "1.0", "error": {"code": exc.code, "message": str(exc)}}
    return JSONResponse(body, status_code=exc.status, headers=headers)


def _unique(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("Duplicate JSON key")
        out[key] = value
    return out


def _no_constant(value):
    raise ValueError("Non-finite JSON number")


async def read_body(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length")
    if declared is not None and (not declared.isdigit() or int(declared) > limit):
        raise ContractError("PAYLOAD_TOO_LARGE", f"Maximum body size is {limit} bytes", 413)
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > limit:
            raise ContractError("PAYLOAD_TOO_LARGE", f"Maximum body size is {limit} bytes", 413)
    return bytes(data)


async def read_json(request: Request, *, keyed: bool = True) -> dict:
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise ContractError("CONTENT_TYPE", "Expected application/json", 415)
    raw = await read_body(request, MAX_JSON_BYTES)
    try:
        body = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique, parse_constant=_no_constant)
    except (ValueError, UnicodeDecodeError, RecursionError):
        raise ContractError("INVALID_JSON", "Expected strict UTF-8 JSON", 422) from None
    if not isinstance(body, dict):
        raise ContractError("INVALID_JSON", "Expected a JSON object", 422)
    if keyed and request.headers.get("idempotency-key") != body.get("request_id"):
        raise ContractError("IDEMPOTENCY_REQUIRED", "Idempotency-Key must equal request_id", 422)
    return body


def create_app(services: Services, operator: OperatorAuth, settings: ApiSettings = ApiSettings()) -> FastAPI:
    docs = {} if settings.enable_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="SNN Agent API", version="1.0", **docs)
    cookie_name = COOKIE_DEV if settings.insecure_dev else COOKIE_SECURE
    allowed = {h.lower() for h in settings.allowed_hosts}

    # ---------------------------------------------------------------------- cross-cutting

    @app.middleware("http")
    async def boundary(request: Request, call_next):
        host = _hostname(request.headers.get("host", ""))
        # The health probe of a container platform calls the pod by its internal address, whatever the public host is; it
        # answers with nothing but "ok", so it is exempt from the host allow-list. Everything else is not.
        if allowed and host not in allowed and request.url.path != "/healthz":
            return _error(ContractError("BAD_HOST", "Unknown host", 400))
        try:
            response = await call_next(request)
        except ContractError as exc:
            response = _error(exc)
        except Exception:
            log.exception("unhandled error on %s %s", request.method, request.url.path)
            response = JSONResponse({"schema_version": "1.0", "error": {"code": "INTERNAL", "message": "Internal error"}}, status_code=500)
        headers = response.headers
        headers.setdefault("Cache-Control", "no-store")
        headers["X-Content-Type-Options"] = "nosniff"
        headers["Referrer-Policy"] = "no-referrer"
        headers["X-Frame-Options"] = "DENY"
        headers.setdefault("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'")
        if not settings.insecure_dev:
            headers["Strict-Transport-Security"] = "max-age=31536000"
        return response

    @app.exception_handler(ContractError)
    async def contract_error(request: Request, exc: ContractError):
        return _error(exc)

    def device(request: Request) -> str:
        header = request.headers.get("authorization", "")
        token = header[7:].strip() if header[:7].lower() == "bearer " else ""
        device_id = authenticate_device(services.ctx, token)
        if device_id is None:
            raise ContractError("UNAUTHORIZED", "Missing or invalid device credential", 401)
        return device_id

    def own(request: Request, device_id: str) -> str:
        who = device(request)
        if who != device_id:
            raise ContractError("FORBIDDEN", "This credential does not belong to that device", 403)
        return who

    def operator_session(request: Request, *, unsafe: bool = False) -> WebSession:
        web = operator.session(request.cookies.get(cookie_name))
        if web is None:
            raise ContractError("UNAUTHORIZED", "Sign in required", 401)
        if unsafe:
            origin = request.headers.get("origin")
            if origin is not None and origin != f"{request.url.scheme}://{request.headers.get('host', '')}":
                raise ContractError("CSRF_FAILED", "Cross-origin request refused", 403)
            if not check_csrf(web, request.headers.get("x-csrf-token")):
                raise ContractError("CSRF_FAILED", "Missing or invalid CSRF token", 403)
        return web

    def either(request: Request, device_id: str | None = None):
        """A device (restricted to its own data) or an operator: whichever credential the request carries."""
        if request.headers.get("authorization"):
            who = device(request)
            if device_id is not None and who != device_id:
                raise ContractError("FORBIDDEN", "This credential does not belong to that device", 403)
            return who
        operator_session(request)
        return None

    def id_ok(value: str, what: str) -> str:
        if not _ID.fullmatch(value):
            raise ContractError("INVALID_ID", f"Invalid {what}", 422)
        return value

    # ---------------------------------------------------------------------------- public

    @app.get("/healthz")
    def health():
        return {"status": "ok", "schema_version": "1.0"}

    @app.post("/auth/login")
    async def login(request: Request):
        body = await read_json(request, keyed=False)
        username, password = body.get("username"), body.get("password")
        if not isinstance(username, str) or not isinstance(password, str) or len(username) > 200 or len(password) > 1000:
            raise ContractError("INVALID_CREDENTIALS", "Invalid credentials", 401)
        client = client_ip(request.headers.get("x-forwarded-for"), request.client.host if request.client else "unknown", settings.trusted_proxies)
        try:
            token, web = operator.login(username, password, client)
        except ContractError as exc:
            headers = {"Retry-After": str(operator.retry_after_s(client))} if exc.status == 429 else None
            return _error(exc, headers)
        response = JSONResponse({"schema_version": "1.0", "csrf_token": web.csrf_token, "expires_at": web.expires_at, "actor": web.actor})
        response.set_cookie(cookie_name, token, max_age=8 * 3600, path="/", secure=not settings.insecure_dev, httponly=True, samesite="strict")
        return response

    @app.post("/auth/logout")
    async def logout(request: Request):
        operator_session(request, unsafe=True)
        operator.logout(request.cookies.get(cookie_name))
        response = Response(status_code=204)
        response.delete_cookie(cookie_name, path="/", secure=not settings.insecure_dev, httponly=True, samesite="strict")
        return response

    @app.get("/auth/session")
    def whoami(request: Request):
        web = operator_session(request)
        return {"schema_version": "1.0", "authenticated": True, "csrf_token": web.csrf_token, "expires_at": web.expires_at, "actor": web.actor}

    # ---------------------------------------------------------------------------- devices

    @app.post("/v1/sessions", status_code=201)
    async def create_session(request: Request):
        who = device(request)
        return services.sessions.create(who, await read_json(request))

    @app.post("/v1/sessions/{session_id}/batches")
    async def batches(session_id: str, request: Request):
        who = device(request)
        return services.ingest.ingest(who, id_ok(session_id, "session id"), await read_json(request))

    @app.post("/v1/sessions/{session_id}/stop")
    async def stop(session_id: str, request: Request):
        who = device(request)
        return services.sessions.stop(who, id_ok(session_id, "session id"), await read_json(request))

    @app.get("/v1/devices/{device_id}/commands")
    def commands(device_id: str, request: Request):
        who = own(request, id_ok(device_id, "device id"))
        return {"schema_version": "1.0", "items": services.commands.poll(who)}

    @app.post("/v1/commands/{command_id}/ack")
    async def acknowledge(command_id: str, request: Request):
        who = device(request)
        return services.commands.acknowledge(who, id_ok(command_id, "command id"), await read_json(request))

    @app.post("/v1/devices/{device_id}/status")
    async def report_status(device_id: str, request: Request):
        who = own(request, id_ok(device_id, "device id"))
        return services.status.report(who, await read_json(request))

    @app.post("/v1/events/{event_id}/image")
    async def upload_image(event_id: str, request: Request):
        who = device(request)
        id_ok(event_id, "event id")
        if request.headers.get("content-type", "").split(";")[0].strip() != "image/jpeg":
            raise ContractError("CONTENT_TYPE", "Expected image/jpeg", 415)
        try:
            index = int(request.headers.get("x-image-index", ""))
        except ValueError:
            raise ContractError("BAD_IMAGE_INDEX", "X-Image-Index must be an integer", 422) from None
        if request.headers.get("idempotency-key") != f"{event_id}-{index}":
            raise ContractError("IDEMPOTENCY_REQUIRED", "Idempotency-Key must be <event_id>-<index>", 422)
        data = await read_body(request, MAX_IMAGE_BYTES)
        return services.images.upload(who, event_id, index=index, sha256=request.headers.get("x-image-sha256", ""),
                                      captured_at=request.headers.get("x-captured-at", ""), data=data)  # fmt: skip

    # ---------------------------------------------------------------------------- reading

    @app.get("/v1/sessions/{session_id}")
    def session_state(session_id: str, request: Request):
        who = either(request)
        id_ok(session_id, "session id")
        tables = services.ctx.storage.tables
        for row in tables.query(T_DEVICES, DEVICES_PK, limit=1000):
            if who is not None and row.rk != who:
                continue
            found = tables.get("sessions", row.rk, session_id)
            if found is not None:
                return services.sessions.get(row.rk, session_id)
        raise ContractError("NOT_FOUND", "Session not found", 404)

    @app.get("/v1/devices/{device_id}/status")
    def latest_status(device_id: str, request: Request):
        either(request, id_ok(device_id, "device id"))
        return services.status.latest(device_id)

    @app.get("/v1/events")
    def list_events(request: Request, limit: int = 20, offset: int = 0):
        operator_session(request)
        return services.events.list(limit, offset)

    @app.get("/v1/events/{event_id}")
    def get_event(event_id: str, request: Request):
        operator_session(request)
        return services.events.get(id_ok(event_id, "event id"))

    @app.get("/v1/events/{event_id}/images/{index}")
    def get_image(event_id: str, index: int, request: Request):
        operator_session(request)
        id_ok(event_id, "event id")
        services.events.get(event_id)
        data = services.images.read(event_id, index)
        return Response(data, media_type="image/jpeg", headers={
            "Content-Disposition": f'inline; filename="{event_id}-{index}.jpg"', "Cache-Control": "private, no-store",
            "Content-Security-Policy": "default-src 'none'; sandbox",
        })  # fmt: skip

    return app
