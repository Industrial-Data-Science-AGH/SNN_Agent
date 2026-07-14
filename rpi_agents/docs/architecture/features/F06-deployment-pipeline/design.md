# F06 deployment_pipeline

## Context

New feature (not in the original design) — the Terraform module and GitHub
Actions workflow that make "deployed very easily via Terraform and GitHub
CD" true. Lives entirely on `feat/azure-cd`; this is task T01, and it must
exist and work (even against a placeholder image) before `feat/dashboard`'s
PR has anything to deploy to.

Reuses the overview diagram in `01-system-overview.md` (nodes `gha`, `pr`,
`dashboard_branch`) for the high-level shape — no separate flow diagram
needed here; the trigger mechanics are fully specified in ADR-0013.

## Contracts

- **Terraform** (`cloud/infra/*.tf`, `azurerm` provider, ADR-0011):
  - `azurerm_storage_account` + `azurerm_storage_container` (`snapshots`) +
    `azurerm_storage_table` (`events`) — F02.
  - `azurerm_container_app_environment` + `azurerm_container_app` — F01/F04/F05's
    runtime, Consumption plan, scale-to-zero, `min_replicas = 0`.
  - `azurerm_container_app` `registry` block pointing at `ghcr.io` with the
    PAT-based pull credential (ADR-0012).
  - `azurerm_container_app` `secret` blocks for `dashboard-password` and the
    Storage connection string (ADR-0010); `env` blocks referencing those
    secrets plus the plain `DASHBOARD_USER` variable.
  - Remote state: `azurerm` backend, pointing at a private Blob container in
    the same Storage Account (bootstrapped once, manually, before the first
    `terraform init` — a one-time chicken-and-egg step documented in this
    feature's runbook, not automated).
- **GitHub Actions** (`.github/workflows/deploy.yml`):
  - Trigger: `pull_request` (`opened`, `synchronize`) targeting
    `feat/azure-cd`.
  - Concurrency group: one, keyed on the PR branch, so overlapping pushes to
    the same PR queue rather than race (ADR-0011, Risks).
  - Steps: checkout → build `cloud/app/Dockerfile` → push to
    `ghcr.io/<owner>/<repo>/dashboard:${{ github.sha }}` using
    `GITHUB_TOKEN` → `terraform init`/`plan`/`apply` in `cloud/infra/`,
    passing the image tag and the secrets (GitHub Actions repo secrets) as
    `-var` values.

## Data model

N/A — this feature is infrastructure-as-code and pipeline config, not
application data.

## Risks

- **One-time manual bootstrap** (the Terraform state backend container, and
  the initial GitHub PAT for `ghcr.io` pulls) isn't itself automated.
  **Mitigation:** documented as explicit numbered steps in this feature's
  runbook (below) — small, one-time, and acceptable to do by hand for a
  single-owner project (consistent with ADR-0012's/ADR-0010's stance on not
  automating single-credential setup).
- **No staging environment.** Every successful PR check is a live deploy to
  the one Container App. Accepted (ADR-0013) — revisit if this project ever
  needs to protect uptime for someone other than the owner.

## Security

- The workflow's only long-lived secret is the `ghcr.io` pull PAT and the
  Terraform-consumed secrets (Storage connection string, dashboard
  password) — all stored as GitHub Actions repository secrets, never in
  workflow YAML or Terraform source (ADR-0010).
- `terraform plan` output is visible in the workflow run before `apply`
  executes, giving the owner a way to notice an unexpected change even
  without a manual approval gate (ADR-0013).
- The workflow only triggers on PRs targeting `feat/azure-cd` specifically
  — a PR against any other branch does not deploy anything, limiting the
  blast radius of an accidental workflow trigger from unrelated work.

## Decisions

- ADR-0011 (Terraform, not Bicep).
- ADR-0012 (ghcr.io, not Azure Container Registry).
- ADR-0013 (GitHub Actions CD, PR-triggered).

## Branch

`feat/azure-cd` (task T01)

## Runbook (one-time manual bootstrap, before the first CD run)

1. `az login`; create the Storage Account manually or via a bootstrap
   `terraform apply` run locally (chicken-and-egg: the backend container
   must exist before Terraform can use it as a backend).
2. Create a GitHub PAT (`read:packages` scope only) for the Container App's
   `ghcr.io` pull credential; store as repo secret `GHCR_PULL_TOKEN`.
3. Store `DASHBOARD_PASSWORD` and the Storage connection string as repo
   secrets.
4. Open the PR from `feat/dashboard` into `feat/azure-cd` — the workflow
   takes it from there.
