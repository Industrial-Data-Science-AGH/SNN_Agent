"""Local W0 mock. Start only with: python -m rpi_agents.cloud.app.mock_api."""

from __future__ import annotations

import copy
import json
import re

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response

from contracts.validation import MAX_BATCH_BYTES, ROOT, ContractError, fixture, schema, validate
from rpi_agents.cloud.app.mock_store import DemoStore

ORIGINS = [f"http://{host}:{port}" for host in ("localhost", "127.0.0.1") for port in (3000, 5173, 8000)]


def error_response(exc):
    body = {"schema_version": "1.0", "error": {"code": exc.code, "message": str(exc)}, "demo": True}
    return JSONResponse(body, status_code=exc.status, headers={"X-SNN-Demo": "true"})


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("Non-finite JSON number")


async def payload(request, contract):
    data = bytearray()
    async for chunk in request.stream():
        data.extend(chunk)
        if len(data) > MAX_BATCH_BYTES:
            raise ContractError("PAYLOAD_TOO_LARGE", "Maximum body size is 65536 bytes", 413)
    try:
        body = json.loads(
            data.decode("utf-8"), object_pairs_hook=unique_object, parse_constant=reject_constant
        )
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ContractError("INVALID_JSON", "Expected strict UTF-8 JSON") from exc
    validate(contract, body)
    if request.headers.get("idempotency-key") != body["request_id"]:
        raise ContractError("IDEMPOTENCY_REQUIRED", "Idempotency-Key must equal request_id")
    return body


def body_spec(name):
    return {
        "requestBody": {
            "required": True,
            "content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}},
        },
        "parameters": [
            {
                "name": "Idempotency-Key",
                "in": "header",
                "required": True,
                "schema": {"type": "string"},
                "description": "Same stable value as request_id; preserve it on retries.",
            }
        ],
    }


def responses(name, code=200):
    return {
        code: {"content": {"application/json": {"schema": {"$ref": f"#/components/schemas/{name}"}}}},
        **{
            c: {
                "description": label,
                "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Error"}}},
            }
            for c, label in [
                (409, "Version or state conflict"),
                (413, "Body too large"),
                (422, "Invalid contract"),
                (429, "Demo capacity reached"),
            ]
        },
    }


def create_app():
    app = FastAPI(
        title="SNN Agent W0 — Demo API",
        version="1.0",
        description="Local scripted fixtures. No SNN inference, authentication, hardware, storage, email or cloud calls. Never deploy this mock.",
    )
    store = DemoStore()
    app.state.demo_store = store
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Idempotency-Key"],
        allow_credentials=False,
    )

    @app.middleware("http")
    async def local_boundary(request: Request, call_next):
        if not re.fullmatch(r"(localhost|127\.0\.0\.1)(:[0-9]{1,5})?", request.headers.get("host", "")):
            return error_response(ContractError("LOCAL_ONLY", "Mock accepts localhost only", 403))
        if (
            request.headers.get("origin")
            and request.headers["origin"] not in ORIGINS
            and request.headers["origin"] != str(request.base_url).rstrip("/")
        ):
            return error_response(ContractError("LOCAL_ONLY", "Origin is not allowed", 403))
        if request.method == "POST":
            if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
                return error_response(ContractError("CONTENT_TYPE", "Expected application/json", 415))
        response = await call_next(request)
        response.headers["X-SNN-Demo"] = "true"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ContractError)
    async def contract_error(request, exc):
        return error_response(exc)

    @app.get("/healthz")
    def health():
        return {"status": "ok", "demo": True, "schema_version": "1.0"}

    @app.get("/demo/fixtures/{name}")
    def get_fixture(name: str):
        return fixture(name)

    @app.post(
        "/v1/sessions",
        status_code=201,
        openapi_extra=body_spec("SessionCreate"),
        responses=responses("SessionState", 201),
    )
    async def create_session(request: Request, scenario: str = "silence"):
        return store.create(await payload(request, "SessionCreate"), scenario)

    @app.get("/v1/sessions/{session_id}", responses=responses("SessionState"))
    def session_state(session_id: str):
        with store.lock:
            return copy.deepcopy(store.state(store.get_session(session_id)))

    @app.post(
        "/v1/sessions/{session_id}/batches",
        openapi_extra=body_spec("SpikeBatch"),
        responses=responses("BatchAck"),
    )
    async def batches(session_id: str, request: Request):
        return store.ingest(session_id, await payload(request, "SpikeBatch"))

    @app.post(
        "/v1/sessions/{session_id}/stop",
        openapi_extra=body_spec("SessionControl"),
        responses=responses("SessionState"),
    )
    async def stop(session_id: str, request: Request):
        return store.stop(session_id, await payload(request, "SessionControl"))

    @app.get("/v1/devices/{device_id}/commands")
    def commands(device_id: str):
        return {"schema_version": "1.0", "items": store.pending(device_id), "demo": True}

    @app.post(
        "/v1/commands/{command_id}/ack",
        openapi_extra=body_spec("CommandAck"),
        responses=responses("CommandAck"),
    )
    async def acknowledge(command_id: str, request: Request):
        return store.acknowledge(command_id, await payload(request, "CommandAck"))

    @app.get("/v1/events")
    def events(limit: int = 20, offset: int = 0):
        if not 1 <= limit <= 100 or offset < 0:
            raise ContractError("INVALID_PAGINATION", "Use limit 1..100 and offset >= 0")
        with store.lock:
            items = list(reversed(list(store.events.values())))
            next_offset = offset + limit if offset + limit < len(items) else None
            return {
                "schema_version": "1.0",
                "items": copy.deepcopy(items[offset : offset + limit]),
                "next_offset": next_offset,
                "demo": True,
            }

    @app.get("/v1/events/{event_id}", responses=responses("Event"))
    def event(event_id: str):
        with store.lock:
            if event_id not in store.events:
                raise ContractError("NOT_FOUND", "Event not found", 404)
            return copy.deepcopy(store.events[event_id])

    @app.get(
        "/v1/sessions/{session_id}/telemetry",
        response_class=Response,
        responses={
            200: {
                "content": {"text/event-stream": {"schema": {"type": "string"}}},
                "description": "One snapshot then EOF; demo client reconnects after retry:1000. Production will stream.",
            }
        },
    )
    def telemetry(session_id: str):
        with store.lock:
            frame = copy.deepcopy(store.get_session(session_id)["frame"])
        return Response(
            f"retry: 1000\nevent: snapshot\nid: {frame['epoch']}:{frame['frame_seq']}\ndata: {json.dumps(frame)}\n\n",
            media_type="text/event-stream",
        )

    def openapi():
        if app.openapi_schema is None:
            spec = get_openapi(
                title=app.title, version=app.version, description=app.description, routes=app.routes
            )
            components = spec.setdefault("components", {}).setdefault("schemas", {})
            for p in (ROOT / "v1").glob("*.schema.json"):
                name = p.stem.removesuffix(".schema")
                components[name] = schema(name)
            app.openapi_schema = spec
        return app.openapi_schema

    app.openapi = openapi
    return app


if __name__ == "__main__":
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Local W0 demo API; never deploy publicly")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if not 1024 <= args.port <= 65535:
        parser.error("port must be between 1024 and 65535")

    # Fixed loopback and one worker: this mock deliberately has no deployment mode.
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, workers=1)
