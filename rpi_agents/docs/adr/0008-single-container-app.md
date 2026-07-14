# ADR-0008: One FastAPI app serves both the API and the dashboard UI (no separate Static Web App)

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F04_dashboard_ui
- Supersedes: ADR-0003

## Context

ADR-0003 chose Azure Static Web Apps specifically for its built-in Entra ID
auth and same-origin Functions proxy. With Entra ID gone (ADR-0009) and
Functions gone (ADR-0007), neither reason to keep a separate frontend
hosting resource still applies — a second Azure resource for the frontend
would now only add cost and deployment surface for no benefit.

## Decision

The dashboard UI is a small set of server-rendered pages (Jinja2 templates)
or a minimal static bundle served directly by the same FastAPI app that
implements `POST/GET /api/events`, inside the one Container App from
ADR-0007. One image, one deploy, one place Basic Auth (ADR-0009) is enforced.

## Alternatives Considered

### Keep a separate Static Web App (Free tier) for just the frontend
- **Pros:** Clean separation of concerns; SWA Free tier is still $0.
- **Cons:** A second resource to provision in Terraform, a second deploy
  step in the GitHub Actions workflow, and — the real issue — SWA's
  linked-backend proxy pattern was built around Functions integration and
  Entra ID; using it with a plain Container App backend and Basic Auth loses
  most of its original benefit while keeping its complexity.
- **Why not:** Contradicts "deployed very easily" — one container is a
  strictly simpler deploy than two coordinated resources for a
  single-owner dashboard this small.

### A separate Vite/React SPA build, served as static files from the same container
- **Pros:** Richer frontend tooling if the UI grows complex.
- **Cons:** Adds a Node build step to the CD pipeline for a dashboard that's
  fundamentally a list + a detail view + a metrics band — no interactivity
  complex enough to need a SPA framework.
- **Why not:** Server-rendered templates (or a handful of hand-written
  static HTML/JS pages) cover the actual UI requirements from F04's design
  without a second toolchain/build step in the pipeline.

## Consequences

### Positive
- One Dockerfile, one image, one Container App, one GitHub Actions deploy
  job — matches "deployed very easily."
- No CORS configuration (same origin, same process).

### Negative
- If the dashboard UI ever needs genuinely rich client-side interactivity,
  this will need revisiting (see F04 design, Risks) — not a concern at
  today's scope (list, filter by date, click into detail).

### Risks (with mitigation)
- **Risk:** conflating API and UI code in one app can get messy as it grows.
  **Mitigation:** keep them in clearly separate modules/routers within the
  one FastAPI app (`api/` vs `web/`) from the start, so splitting them into
  separate deployments later is a refactor, not a rewrite.
