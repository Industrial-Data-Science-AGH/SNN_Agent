"""cloud/app entrypoint (T02) — assembles the one FastAPI app that serves
the ingest/read API and the dashboard UI in a single deployable unit
(ADR-0008). `require_basic_auth` is applied here, once, as a global
dependency — every route in the app, dashboard pages included, inherits it;
no route opts out (F05 design, F01 design Security: "there is no
unauthenticated route in this app at all, including `/` itself").

Replaces T01's placeholder `/healthz`-only app inside the same Dockerfile
shape (T01 plan, "T02 replaces the app code, not the Dockerfile shape").
"""

from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.staticfiles import StaticFiles

from .auth import require_basic_auth
from .routes_api import metrics_router
from .routes_api import router as api_router
from .routes_dashboard import router as dashboard_router

_STATIC_DIR = Path(__file__).parent / "static"
_STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(
    title="wakeup-ai-cloud-dashboard",
    dependencies=[Depends(require_basic_auth)],
)

app.include_router(api_router)
app.include_router(metrics_router)
app.include_router(dashboard_router)
# CSS/JS/fonts/brand assets for the dashboard UI (T05, premium refresh).
# Deliberately NOT behind require_basic_auth: FastAPI's app-level
# `dependencies=` only wraps routes added via include_router/@app.get, not
# an app.mount()'d ASGI sub-app like StaticFiles -- protecting it would need
# its own auth-checking wrapper. Left unauthenticated on purpose rather than
# built around: this directory must only ever hold generic UI assets (CSS,
# JS, fonts, the company logo) -- never snapshot images or anything else
# Tenet 4 (00-prfaq.md) calls privacy-sensitive. Actual event photos are
# never placed here; they only ever reach the browser via Blob Storage's
# short-lived, per-request SAS URLs (F01/F04 design), which stay behind
# auth the whole way. If this directory is ever asked to hold anything
# user-specific, that is the trigger to protect this mount properly, not to
# keep treating "just static files" as automatically safe.
app.mount(
    "/static",
    StaticFiles(directory=str(_STATIC_DIR)),
    name="static",
)
