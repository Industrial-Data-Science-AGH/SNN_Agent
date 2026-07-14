"""cloud/app entrypoint (T02) — assembles the one FastAPI app that serves
the ingest/read API and the dashboard UI in a single deployable unit
(ADR-0008). `require_basic_auth` is applied here, once, as a global
dependency — every route in the app, dashboard pages included, inherits it;
no route opts out (F05 design, F01 design Security: "there is no
unauthenticated route in this app at all, including `/` itself").

Replaces T01's placeholder `/healthz`-only app inside the same Dockerfile
shape (T01 plan, "T02 replaces the app code, not the Dockerfile shape").
"""

from fastapi import Depends, FastAPI

from .auth import require_basic_auth
from .routes_api import router as api_router
from .routes_dashboard import router as dashboard_router

app = FastAPI(
    title="wakeup-ai-cloud-dashboard",
    dependencies=[Depends(require_basic_auth)],
)

app.include_router(api_router)
app.include_router(dashboard_router)
