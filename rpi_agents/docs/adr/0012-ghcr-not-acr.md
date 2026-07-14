# ADR-0012: GitHub Container Registry (ghcr.io) instead of Azure Container Registry

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F06_deployment_pipeline

## Context

The Container App (ADR-0007) needs to pull its image from somewhere. Azure's
native option, Azure Container Registry (ACR), is not free — even the
smallest Basic tier is a fixed ~$5/month, a real recurring cost that
conflicts directly with "cheapest possible." The build pipeline is already
GitHub Actions, which has a first-party, genuinely free registry.

## Decision

Container images are built and pushed to **GitHub Container Registry**
(`ghcr.io/<owner>/<repo>`) by the GitHub Actions workflow, using the
automatically-provided `GITHUB_TOKEN` (no separate registry credential to
create/rotate for the *push* side). The Container App is configured with a
registry credential (a GitHub PAT with `read:packages` scope, stored as a
GitHub Actions secret / Terraform sensitive variable per ADR-0010) to *pull*
the image.

## Alternatives Considered

### Azure Container Registry (Basic tier)
- **Pros:** Same-cloud resource, integrates with Container Apps via managed
  identity (no PAT to manage for pulls).
- **Cons:** ~$5/month fixed cost, every month, forever — for a system whose
  every other component targets $0. This is the single largest recurring
  cost in the whole design if chosen.
- **Why not:** Fails "cheapest possible" outright; ghcr.io is free at this
  project's scale (private images included) and the pipeline is already
  GitHub-native.

## Consequences

### Positive
- Zero registry cost, keeping the full system at ~$0/month.
- Push side needs no new credential (`GITHUB_TOKEN` is automatic).

### Negative
- Pull side needs a GitHub PAT stored in Azure (one more secret, vs. ACR's
  managed-identity-based pull which needs none) — a modest complexity trade
  for avoiding the $5/month.

### Risks (with mitigation)
- **Risk:** the pull-side PAT expires or is revoked, breaking deploys/scale
  events. **Mitigation:** use a fine-scoped (`read:packages` only),
  long-lived PAT, documented in F06's design with a renewal reminder; not
  automated in this revision (acceptable at hobby scale — the same
  not-automating-rotation stance applies to the Basic Auth credential,
  ADR-0009/ADR-0010).
