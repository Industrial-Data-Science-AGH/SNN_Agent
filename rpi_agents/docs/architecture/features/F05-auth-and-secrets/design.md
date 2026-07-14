# F05 auth_and_secrets

## Context

Cross-cutting feature covering the one auth mechanism (shared HTTP Basic
Auth, ADR-0009) and secrets flow (ADR-0010) for the whole app. Unlike the
earlier Entra-ID-based design, auth here is simple enough to build alongside
F01/F04 from the start rather than as a hardening pass bolted on afterward
— it's one FastAPI dependency, not a separate identity-provider integration.

![F05 auth flow](../../diagrams/F05-auth-flow.svg)

*Source: `docs/diagrams/F05-auth-flow.dot` — edit and re-run `render_diagrams.sh`.*

## Contracts

- **Auth mechanism:** a single FastAPI dependency (`require_basic_auth`)
  applied globally to the app (every route: dashboard pages, list/detail,
  and ingest). Compares the request's `Authorization: Basic` credential
  against `DASHBOARD_USER`/`DASHBOARD_PASSWORD` env vars (default
  `ids`/`ids`) using a constant-time comparison
  (`secrets.compare_digest`) to avoid a timing side-channel. Failure →
  `401` with a `WWW-Authenticate: Basic` header (triggers the browser's
  native login prompt for dashboard use; the Pi's `cloud_sync.py` sends the
  header directly, no prompt involved).
- **One credential for everything:** the dashboard viewer and the Pi's
  ingest push use the *same* credential pair (ADR-0009) — simpler than the
  earlier two-credential (Entra ID + function key) split.
- **Secrets inventory and where each lives:**

  | Secret | Lives in | Notes |
  |---|---|---|
  | `DASHBOARD_PASSWORD` | GitHub Actions secret → Terraform sensitive var → Container App secret | Same value the Pi sends; rotate by updating the GitHub secret and re-running the deploy (ADR-0010) |
  | Storage Account connection string | GitHub Actions secret → Terraform sensitive var → Container App secret | Never in Terraform source, never in the repo (ADR-0010) |
  | `DASHBOARD_USER` | Terraform variable (non-sensitive, defaults to `ids`) | Not secret — the password is what protects the credential, per HTTP Basic Auth's design |
  | Pi-side copy of `DASHBOARD_USER`/`DASHBOARD_PASSWORD` | `~/.config/snn-agent/.env` | Same file/`chmod 600` pattern as `GEMINI_API_KEY` etc. |
  | GitHub PAT (`read:packages`) for the Container App to pull from `ghcr.io` | GitHub Actions secret → Terraform sensitive var → Container App registry credential | See F06, ADR-0012 |

## Data model

N/A — this feature is configuration, not data.

## Risks

- **Single shared credential, weak default value.** If left at `ids`/`ids`
  and exposed publicly, anyone can read the full event history/images and
  write fake events. Accepted trade-off for a single-owner hobby project at
  the owner's explicit request (ADR-0009); the deployment runbook (F06)
  calls out changing it from the default as the first post-deploy step.
- **No lockout/rate-limiting on auth attempts.** Not implemented in this
  revision — flagged as a residual risk in `01-system-overview.md`, not
  solved preemptively.
- **Basic Auth sends the credential (base64, not encrypted by the scheme
  itself) on every request.** Mitigated entirely by mandatory HTTPS —
  Container Apps' default ingress terminates TLS, so the credential is
  never on the wire in plaintext; it would only be exposed if HTTPS were
  somehow disabled, which Terraform's ingress config does not permit.

## Security

*(Direct manual STRIDE-lite pass; the dedicated `cybersecurity` skill's
routing table was not available in this environment — flagged as a residual
risk in `01-system-overview.md`.)*

- **Trust boundaries:** Internet → Container App (single auth boundary — no
  second resource in front of it, unlike the earlier SWA+Function split) →
  Storage (no direct internet exposure; only the app's connection string
  can reach it).
- **Authn/authz model:** one shared credential for both the human viewer and
  the one trusted machine (the Pi). No signup flow, no per-user roles — a
  deliberate simplification for a single-owner system (ADR-0009).
- **Data classification:** snapshot images and event timestamps are
  privacy-sensitive (occupancy-revealing); treated as confidential
  throughout — private blob container, SAS-only image access, auth-gated
  on every route including the dashboard's own pages.
- **Rate-limiting posture:** none implemented; accepted residual risk at
  hobby scale (see `01-system-overview.md` Security and Risks).
- **Secrets handling:** see table above; validated by task T01's finalization
  step — `git grep` for accidentally-committed credentials/connection
  strings before the PR merges (see `delivery-plan.json`).

## Decisions

- ADR-0009 (fixed HTTP Basic Auth, one shared credential, instead of
  Entra ID + function key).
- ADR-0010 (secrets via GitHub Actions + Terraform into Container App
  native secrets, instead of Function App Application Settings).

## Branch

`feat/dashboard` (task T02)
