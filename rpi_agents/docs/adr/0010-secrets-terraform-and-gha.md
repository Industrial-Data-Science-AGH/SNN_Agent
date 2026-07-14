# ADR-0010: Secrets flow through GitHub Actions secrets + Terraform variables into Container App env

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F05_auth_and_secrets
- Supersedes: ADR-0005

## Context

ADR-0005 stored secrets in Function App Application Settings, configured
manually via `az` CLI (no CI/CD existed yet). The system is now deployed via
Terraform + GitHub Actions CD (ADR-0013), so secrets need a path from
"where the owner sets them once" through the pipeline into the running
container without ever touching the repo.

## Decision

Secrets (`DASHBOARD_PASSWORD`, the Storage Account connection string) are
stored as **GitHub Actions repository secrets**, passed into `terraform
apply` as `-var` values (marked `sensitive = true` in the Terraform
variable definitions so they never print in plan/apply logs), and written by
Terraform into the Container App's **secret** block (Container Apps' native
secret storage, referenced by env vars in the container spec — distinct from
plain env vars, encrypted at rest by Azure). `DASHBOARD_USER` (non-sensitive,
defaults to `ids`) is a plain Terraform variable, not a secret.

## Alternatives Considered

### Azure Key Vault (as originally considered and rejected in ADR-0005)
- **Pros:** Centralized rotation/audit, the enterprise-correct pattern.
- **Cons:** Same reasoning as ADR-0005 still applies — one secret pair, one
  consumer, disproportionate setup cost for what it buys here. Adding it now
  would also mean the Terraform module and the GitHub Actions workflow both
  need Key Vault read permissions configured, more setup than "deployed very
  easily" wants.
- **Why not:** ADR-0005's reasoning carries forward unchanged; still the
  documented upgrade path if secret/consumer count grows.

### Hardcode the default credential in the Dockerfile/source
- **Pros:** Zero configuration.
- **Cons:** Bakes a credential into the container image and git history —
  even a weak default credential (ADR-0009) shouldn't be *unchangeable*
  without a rebuild, and definitely shouldn't be permanently visible in the
  image layer/history.
- **Why not:** Environment-variable injection costs almost nothing extra and
  keeps the credential changeable without a code change (see ADR-0009
  Consequences).

## Consequences

### Positive
- No secret ever appears in the repo, a Terraform `.tf` file, or a Docker
  image layer — only in GitHub's encrypted secrets store and Azure's
  encrypted Container App secret store.
- Rotating the password is: update the GitHub secret, re-run the workflow
  (or `terraform apply`) — no rebuild needed since it's injected at
  container start, not build time.

### Negative
- Terraform state itself will contain the secret value (Terraform stores
  resource attributes, including ones sourced from sensitive variables, in
  state) — state must be treated as sensitive.

### Risks (with mitigation)
- **Risk:** Terraform state exposure leaks secrets. **Mitigation:** remote
  state backend is the same Storage Account (private container, no public
  access — same posture as F02's Blob container for images), never local
  state committed to the repo; `.gitignore` covers `*.tfstate*` as a
  defense-in-depth backstop even though state isn't meant to be local.
