# ADR-0009: Fixed HTTP Basic Auth credential instead of Entra ID

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F03_pi_push_client, F05_auth_and_secrets
- Supersedes: ADR-0004

## Context

The owner explicitly asked for a fixed username/password (`ids` / `ids`)
instead of Entra ID, and for the system to be as simple/cheap to deploy as
possible. Entra ID was chosen in ADR-0004 specifically because it came free
with Static Web Apps (ADR-0003), which is no longer part of the design
(ADR-0008). There is no longer a "free, zero-effort" identity provider
paired with the hosting choice, so the trade-off changes.

## Decision

The single FastAPI app (ADR-0008) enforces **HTTP Basic Auth** on every
route — both the dashboard pages and the API (list/detail *and* ingest) —
using one credential pair, read from an environment variable at startup
(`DASHBOARD_USER` / `DASHBOARD_PASSWORD`), defaulting to `ids` / `ids` if
unset. The **same** credential is used by the Pi's push client (F03) to
authenticate its `POST /api/events` calls — one credential pair for the
whole system, not a separate machine credential, since the owner asked for
simplicity over the earlier design's two-credential split (ADR-0004's
Entra ID + function key).

## Alternatives Considered

### Keep two credentials — one for the dashboard, one for Pi ingest (as ADR-0004 did with Entra ID + function key)
- **Pros:** Slightly better isolation — a leaked dashboard password doesn't
  also grant write access, and vice versa.
- **Cons:** Two secrets to generate, store (on the Pi *and* in GitHub
  Actions/Terraform), and keep in sync — directly working against "as
  simple as possible." At this system's threat model (single owner, hobby
  project, read-only dashboard data plus a narrow "add one event" write
  capability) the extra isolation buys little.
- **Why not:** The owner's explicit ask was for *one* fixed user/password,
  not a credential scheme — take that at face value; the single-credential
  design is documented here so it's easy to split later if the threat model
  changes (see Risks).

### Entra ID (unchanged from ADR-0004)
- **Pros:** Strongest option, no credential to leak or brute-force, no
  password to remember/type.
- **Cons:** Explicitly what the owner asked to remove — it added the login
  friction and Azure AD app-registration setup they don't want for a
  personal hobby dashboard, and it was only "free" as a Static Web Apps
  feature, which no longer applies (ADR-0008).
- **Why not:** Overruled by explicit instruction. Documented here rather
  than silently dropped, so the trade-off is visible if priorities change.

## Consequences

### Positive
- Trivial to implement (one FastAPI dependency/middleware), trivial to
  operate (one credential, one env var, one GitHub Actions secret).
- Matches the "as simple/cheap as possible" instruction directly.

### Negative
- HTTP Basic Auth over a bare `ids`/`ids` default is a genuinely weak
  credential — no lockout, no MFA, guessable if left at the default. This is
  the single biggest security trade-off in this revision.
- The Pi and the dashboard now share one credential — a leak from either
  side (e.g. the Pi's `.env`, or a browser's saved-password store) affects
  both read and write access.

### Risks (with mitigation)
- **Risk:** credential brute-forcing or exposure at the default value.
  **Mitigation:** (1) HTTPS is mandatory (Container Apps' default ingress is
  TLS-terminated) so the credential is never sent in plaintext over the
  network; (2) the value is environment-driven, not hardcoded in source, so
  changing it from the `ids`/`ids` default is a one-line Terraform variable
  / GitHub Actions secret change, not a code change — the owner is
  encouraged to change it from the placeholder before exposing the app
  publicly, called out explicitly in F05's design and the deployment
  runbook; (3) documented as the concrete trigger to revisit this ADR if the
  system ever holds higher-value data or gets a second user.
- **Risk:** no rate limiting on login attempts. **Mitigation:** Container
  Apps' own scale-to-zero/cold-start behavior provides mild natural
  friction; formal rate limiting is out of scope for this revision, flagged
  as a residual risk in `01-system-overview.md`.
