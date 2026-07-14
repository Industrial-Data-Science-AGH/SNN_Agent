# ADR-0011: Terraform (azurerm provider) instead of Bicep

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F02_storage_infra, F01_ingest_api, F06_deployment_pipeline

## Context

The earlier design assumed Bicep, Azure's native IaC language, deployed
manually via `az`. The owner explicitly asked for Terraform, driven from
GitHub Actions.

## Decision

All Azure infrastructure (Storage Account, Container App Environment,
Container App, registry credential) is defined in Terraform (`azurerm`
provider), stored under `cloud/infra/`, with state in a remote backend
(an Azure Storage Account blob container — see ADR-0010, Risks).

## Alternatives Considered

### Bicep (the original plan)
- **Pros:** Native to Azure, no separate state-backend concern (ARM manages
  deployment history itself), slightly less to install.
- **Cons:** The owner explicitly asked for Terraform — this is a direct
  instruction, not a close call on technical merits. Terraform also has the
  practical advantage of being cloud-agnostic tooling the owner may already
  know from other contexts, and its GitHub Actions integration
  (`hashicorp/setup-terraform`) is equally mature.
- **Why not:** Overruled by explicit instruction; also a reasonable choice
  on its own merits, not just deferred to.

## Consequences

### Positive
- Matches the explicit request; well-trodden path for GitHub Actions CD
  (plan on PR, apply on merge is a standard, well-documented pattern).

### Negative
- Introduces a state-management concern Bicep wouldn't have had (remote
  backend, state locking) — handled via the Storage Account backend, see
  ADR-0010.

### Risks (with mitigation)
- **Risk:** state lock contention or corruption if `terraform apply` runs
  concurrently (e.g. two pushes racing). **Mitigation:** the Storage Account
  backend supports native blob-lease locking; the GitHub Actions workflow
  is configured to run deploys on a single concurrency group so only one
  apply runs at a time (see F06 design).
