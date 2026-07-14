# Architecture Decision Records

Only the ADRs still needed to build and operate the current design are kept
here. Earlier ADR-0002 through ADR-0005 (Azure Functions, Static Web Apps,
Entra ID, Application Settings) were superseded when the owner asked for the
cheapest possible container-based stack with fixed-credential auth, and have
been removed rather than kept as dead files — their still-relevant reasoning
(e.g. why one Storage Account with Blob+Table, not Cosmos DB) is restated
inline in the ADRs below, mainly ADR-0007.

| ADR | Title | Status | Date | Features |
|-----|-------|--------|------|----------|
| [0001](0001-push-not-pull.md) | Pi pushes events; dashboard does not poll the Pi | accepted | 2026-07-14 | F01_ingest_api, F03_pi_push_client |
| [0006](0006-single-post-inline-image.md) | Single combined POST (JSON + base64 image), not two-step SAS upload | accepted | 2026-07-14 | F01_ingest_api, F03_pi_push_client |
| [0007](0007-container-apps-not-functions.md) | Azure Container Apps (Consumption, scale-to-zero); storage stays one Storage Account (Blob+Table) | accepted | 2026-07-14 | F01_ingest_api, F02_storage_infra |
| [0008](0008-single-container-app.md) | One FastAPI app serves API + dashboard UI (no separate Static Web App) | accepted | 2026-07-14 | F01_ingest_api, F04_dashboard_ui |
| [0009](0009-fixed-basic-auth.md) | Fixed HTTP Basic Auth credential instead of Entra ID | accepted | 2026-07-14 | F01_ingest_api, F03_pi_push_client, F05_auth_and_secrets |
| [0010](0010-secrets-terraform-and-gha.md) | Secrets via GitHub Actions secrets + Terraform variables into Container App env | accepted | 2026-07-14 | F01_ingest_api, F05_auth_and_secrets |
| [0011](0011-terraform-not-bicep.md) | Terraform (azurerm provider) instead of Bicep | accepted | 2026-07-14 | F02_storage_infra, F01_ingest_api, F06_deployment_pipeline |
| [0012](0012-ghcr-not-acr.md) | GitHub Container Registry (ghcr.io) instead of Azure Container Registry | accepted | 2026-07-14 | F06_deployment_pipeline |
| [0013](0013-github-actions-cd.md) | GitHub Actions CD: PR from `feat/dashboard` into `feat/azure-cd` builds + deploys the Azure infra | accepted | 2026-07-14 | F06_deployment_pipeline |
| [0014](0014-client-generated-event-id.md) | Pi generates `event_id` (ULID); ingest is an idempotent upsert | accepted | 2026-07-14 | F01_ingest_api, F03_pi_push_client |
| [0015](0015-bounded-local-sync-queue.md) | Bounded local sync queue, flushed on every wake cycle (supersedes Tenet 2's original no-retry stance) | accepted | 2026-07-14 | F03_pi_push_client, F01_ingest_api |
