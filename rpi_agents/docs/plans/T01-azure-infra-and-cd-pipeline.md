# T01 — Azure infra + CD pipeline

- **Branch:** `feat/azure-cd` (off `feat/rpi`)
- **Feature IDs:** F06 (deployment_pipeline), F02 (storage_infra)
- **Depends on:** nothing (first task; everything else deploys against this)
- **Blocks:** T02, T03 (their PR into this branch is what triggers a real deploy)
- **Source:** `docs/architecture/delivery-plan.json` (T01), `docs/architecture/02-delivery.md`, `docs/architecture/features/F02-storage-infra/design.md`, `docs/architecture/features/F06-deployment-pipeline/design.md`, ADR-0007, ADR-0010, ADR-0011, ADR-0012, ADR-0013

## Goal

Stand up the Azure infrastructure and the PR-triggered CD pipeline, and prove
the pipeline works end-to-end against a placeholder container image before
any real application code exists. Nothing in T02/T03 has anywhere to deploy
until this is done and merged.

## One-time manual bootstrap (do first, before any Terraform run)

These are explicitly *not* automated (F06 design, "Runbook"):

1. `az login`.
2. Create the Storage Account that will hold Terraform remote state, either
   by hand or via a local `terraform apply` run once — chicken-and-egg,
   since Terraform's `azurerm` backend needs a container that already
   exists before it can use it.
3. Create a GitHub PAT scoped to `read:packages` only (ADR-0012) for the
   Container App's `ghcr.io` pull credential. Store as repo secret
   `GHCR_PULL_TOKEN`.
4. Store `DASHBOARD_PASSWORD` (pick something better than the `ids`/`ids`
   default before this leaves dev — F05 flags this explicitly) and the
   Storage Account connection string as GitHub Actions repo secrets.

## Files to create

```
cloud/infra/
  main.tf            # provider block (azurerm), backend "azurerm" block
  storage.tf          # azurerm_storage_account, azurerm_storage_container "snapshots",
                       # azurerm_storage_table "events"
  container_app.tf    # azurerm_container_app_environment, azurerm_container_app
  variables.tf         # environment (dev/prod), image tag, dashboard_password (sensitive),
                        # storage connection string (sensitive), ghcr_pull_token (sensitive)
  outputs.tf           # container app FQDN, storage account name (non-sensitive outputs only)

cloud/app/
  Dockerfile           # placeholder for this task — a minimal FastAPI "hello" app is enough
                        # to prove the pipeline; T02 replaces the app code, not the Dockerfile shape

.github/workflows/
  deploy.yml
```

## Terraform contracts (from F02 + F06 design)

- `azurerm_storage_account`: one account, parameterized by `environment`
  var (`dev`/`prod`) so a throwaway dev account can exist alongside prod
  without duplicating modules.
- `azurerm_storage_container` `"snapshots"`: private, no public/anonymous
  read access at the container level.
- `azurerm_storage_table` `"events"`.
- `azurerm_container_app_environment` + `azurerm_container_app`:
  Consumption plan, `min_replicas = 0` (scale-to-zero).
- `azurerm_container_app` `registry` block: `ghcr.io`, PAT-based pull
  credential (`GHCR_PULL_TOKEN`).
- `azurerm_container_app` `secret` blocks: `dashboard-password`, storage
  connection string — referenced by `env` blocks, plus a plain
  (non-secret) `DASHBOARD_USER` env var, default `ids`.
- Remote state: `azurerm` backend pointing at a private Blob container in
  the same Storage Account (ADR-0011). Bootstrapped manually per the
  runbook above.

## GitHub Actions contract (`.github/workflows/deploy.yml`)

- Trigger: `pull_request`, types `[opened, synchronize]`, targeting
  `feat/azure-cd` specifically (not any other branch — limits blast
  radius per F06's Security section).
- Concurrency group: one, keyed on the PR branch/number, so overlapping
  pushes to the same PR queue instead of racing each other in the same
  Terraform state (ADR-0011 risk).
- Steps: checkout → build `cloud/app/Dockerfile` → push to
  `ghcr.io/<owner>/<repo>/dashboard:${{ github.sha }}` using
  `GITHUB_TOKEN` → `terraform init`/`plan`/`apply` in `cloud/infra/`,
  passing the image tag and secrets as `-var` values. `terraform plan`
  output must be visible in the run log before `apply` executes (no
  manual approval gate exists — the plan output is the owner's only
  chance to notice something unexpected, per ADR-0013).

## Step-by-step

1. Complete the one-time manual bootstrap above.
2. Write `cloud/infra/*.tf`. Run `terraform validate` and `terraform plan`
   locally against the `dev` environment var before touching CI.
3. Write a placeholder `cloud/app/Dockerfile` (e.g. a one-route FastAPI
   `/healthz` app) — just enough to prove build → push → deploy works.
   T02 will replace the app code inside this same Dockerfile shape.
4. Write `.github/workflows/deploy.yml`.
5. Open a throwaway PR into `feat/azure-cd` (e.g. from a scratch branch
   with only the placeholder Dockerfile) to prove the pipeline once,
   end-to-end, before `feat/dashboard` exists — this is the explicit
   point of T01's gate.
6. Confirm the Container App is reachable and returns something (even a
   401, once F05's auth exists, or a 200 on `/healthz` for the
   placeholder) at its FQDN.
7. Before merging: `git grep` the repo for accidentally-committed secrets
   or connection strings (F05 Security section calls this out as T01's
   finalization step).

## Acceptance gate (from `delivery-plan.json`)

- `terraform validate`/`plan` clean.
- Workflow YAML lints.
- A PR from `feat/dashboard` into `feat/azure-cd` deploys successfully.
- Unauthenticated dashboard/ingest access is rejected (once F05 auth
  lands in T02 — T01 itself just needs the placeholder to deploy).
- A real Pi wake cycle produces a visible dashboard row within 30s (only
  fully verifiable once T02/T03 exist — T01's own gate is the pipeline
  working end-to-end against the placeholder).
- `ruff`+`mypy` clean across `cloud/` (once `cloud/app/` has real code —
  T01 itself only needs the placeholder to pass).

## Notes / risks to carry into implementation

- No staging environment — every successful PR check is a live deploy
  (ADR-0013, accepted at single-owner hobby scale). Don't build an
  approval gate; it's explicitly out of scope.
- Terraform state contains the Storage connection string as a resource
  attribute — it lives in a private Blob container, never local, never
  committed (ADR-0010, ADR-0011).
