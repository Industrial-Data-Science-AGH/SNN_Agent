"""Server side of the operator dashboard (task C1: shell + sign-in).

The dashboard is a *static shell* that is rendered entirely in the browser
(``templates/index.html`` + ``static/js/*``). This module does two things:

1. ``router`` — the routes that belong to the dashboard when it is mounted
   into the real backend app (Wiktor's ``rpi_agents.cloud.app.api``):
     * ``GET /``                       → the shell HTML
     * ``GET /dashboard/fixtures``     → the list of demo fixture names
     * ``GET /dashboard/fixtures/{n}`` → one contract fixture, tagged ``demo``

   The fixtures are read straight from ``contracts/fixtures`` via
   ``contracts.validation.fixture`` — one source of truth, never a copy. This
   is the ``DemoSource`` the front-end talks to in demo mode; the real backend
   exposes the same *shape* of data under ``/v1/...`` so the UI does not need
   to be rewritten to go live.

2. ``create_dashboard_app`` / ``__main__`` — a **development-only** harness so
   the UI can be run and clicked without the full cloud backend or any
   hardware::

       python -m rpi_agents.cloud.app.routes_dashboard   # http://127.0.0.1:8080

   The harness mounts ``/static`` and includes ``router``, and adds a *demo*
   ``/auth/*`` stub that speaks the exact same contract as Wiktor's real
   endpoints (so the sign-in JS is identical in both worlds) but holds no real
   data: it accepts only the published demo credential and rejects everything
   else with ``INVALID_CREDENTIALS``. This harness is never deployed.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from contracts.validation import ContractError, fixture

_APP_DIR = Path(__file__).parent
_TEMPLATES_DIR = _APP_DIR / "templates"
_STATIC_DIR = _APP_DIR / "static"

# The published demo credential for the local harness. It is intentionally not
# a secret and unlocks nothing but scripted fixtures. Real access control lives
# in rpi_agents.cloud.app.auth (Wiktor) and is never reproduced here.
_DEMO_USERNAME = "operator"
_DEMO_PASSWORD = "demo"
_DEMO_COOKIE = "snn_session"

router = APIRouter(tags=["dashboard"])


def _index_html() -> str:
    return (_TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")


@router.get("/", response_class=HTMLResponse)
def shell() -> HTMLResponse:
    """The dashboard shell. All data is fetched client-side after it loads."""
    return HTMLResponse(_index_html())


@router.get("/dashboard/fixtures")
def fixture_index() -> JSONResponse:
    """The demo fixture names the UI may request (from the contract index)."""
    import json

    from contracts.validation import ROOT

    names = sorted(json.loads((ROOT / "fixtures/index.json").read_text()))
    return JSONResponse({"schema_version": "1.0", "items": names, "demo": True})


@router.get("/dashboard/fixtures/{name}")
def demo_fixture(name: str) -> JSONResponse:
    """One contract fixture, explicitly tagged as demo data.

    ``contracts.validation.fixture`` raises ``UNKNOWN_FIXTURE`` (404) for an
    unknown name; the app's ContractError handler turns that into JSON.
    """
    data = fixture(name)  # raises ContractError(UNKNOWN_FIXTURE, 404) if absent
    return JSONResponse({"schema_version": "1.0", "demo": True, "fixture": name, "data": data})


# --------------------------------------------------------------------------- dev harness


def create_dashboard_app() -> FastAPI:
    """A local, development-only app that serves the dashboard on its own.

    Not for deployment: it carries a demo ``/auth`` stub and no real storage,
    hardware or SNN inference. In production the dashboard ``router`` is mounted
    into Wiktor's backend, which supplies the real ``/auth`` and ``/v1`` routes.
    """
    app = FastAPI(title="SNN Lab dashboard — DEV harness", version="0.9.0")

    @app.exception_handler(ContractError)
    async def _contract_error(request: Request, exc: ContractError) -> JSONResponse:
        return JSONResponse(
            {"schema_version": "1.0", "error": {"code": exc.code, "message": str(exc)}},
            status_code=exc.status,
        )

    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")
    app.include_router(router)

    # ---- demo sign-in stub (same contract as api.py, no real data) ----------

    def _authenticated(request: Request) -> bool:
        return request.cookies.get(_DEMO_COOKIE) == "demo-session"

    @app.post("/auth/login")
    async def demo_login(request: Request) -> Response:
        body = await request.json()
        username, password = body.get("username"), body.get("password")
        # Constant-ish delay so the front-end "loading" state is visible locally.
        time.sleep(0.4)
        if username == _DEMO_USERNAME and password == _DEMO_PASSWORD:
            resp = JSONResponse({
                "schema_version": "1.0",
                "csrf_token": "demo-csrf-token",
                "expires_at": "2099-01-01T00:00:00Z",
                "actor": "shared_operator",
            })
            resp.set_cookie(_DEMO_COOKIE, "demo-session", httponly=True, samesite="strict", path="/")
            return resp
        return JSONResponse(
            {"schema_version": "1.0", "error": {"code": "INVALID_CREDENTIALS", "message": "Invalid credentials"}},
            status_code=401,
        )

    @app.get("/auth/session")
    def demo_session(request: Request) -> JSONResponse:
        if not _authenticated(request):
            return JSONResponse(
                {"schema_version": "1.0", "error": {"code": "UNAUTHORIZED", "message": "Sign in required"}},
                status_code=401,
            )
        return JSONResponse({
            "schema_version": "1.0", "authenticated": True,
            "csrf_token": "demo-csrf-token", "expires_at": "2099-01-01T00:00:00Z", "actor": "shared_operator",
        })

    @app.post("/auth/logout")
    def demo_logout() -> Response:
        resp = Response(status_code=204)
        resp.delete_cookie(_DEMO_COOKIE, path="/")
        return resp

    return app


if __name__ == "__main__":
    import uvicorn

    print("SNN Lab dashboard (DEV) — http://127.0.0.1:8080")
    print(f"Demo sign-in: username '{_DEMO_USERNAME}', password '{_DEMO_PASSWORD}' (or use 'Explore demo').")
    uvicorn.run(create_dashboard_app(), host="127.0.0.1", port=8080)
