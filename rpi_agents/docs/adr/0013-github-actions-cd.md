# ADR-0013: GitHub Actions CD pipeline (build → push → terraform apply)

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F06_deployment_pipeline

## Context

The original PR/FAQ (`00-prfaq.md`) explicitly put CI/CD out of scope for
this phase, assuming manual `az`/`azd` deploys. The owner has now explicitly
asked for GitHub CD, and reduced the whole delivery to one branch, one day,
three tasks — a pipeline that deploys on every push is a better fit for that
shape of work than manual deploys would be, since it removes a manual step
from an already-tight scope.

## Decision

A single GitHub Actions workflow (`.github/workflows/deploy.yml`), defined
on `feat/azure-cd` and triggered on **pull request events (opened/
synchronize) targeting `feat/azure-cd`** — concretely, the PR from
`feat/dashboard` into `feat/azure-cd` — does: (1) build the FastAPI app's
Docker image, (2) push it to `ghcr.io` (ADR-0012), (3) run `terraform apply`
(ADR-0011) with the new image tag and the secrets from ADR-0010 as
variables. One job, run with a concurrency group of 1 so overlapping PR
updates queue rather than race (ADR-0011, Risks). This means the Container
App is already running `feat/dashboard`'s code before that PR is even
merged — merging is a bookkeeping step, not the deploy trigger, which suits
a single-owner, single-PR, one-day build.

## Alternatives Considered

### Manual deploy (`az`/`terraform apply` run by hand, as originally scoped)
- **Pros:** No pipeline to build or debug; matches the original PR/FAQ scope
  exactly.
- **Cons:** Explicitly what the owner asked to replace — one more manual
  step in a workflow they want to be "very easy," and one more thing to
  forget/get wrong by hand each time.
- **Why not:** Overruled by explicit instruction; a single simple workflow
  is not disproportionate effort for a one-day, three-task build.

## Consequences

### Positive
- Push-to-deploy — the whole point of asking for this; no manual
  `terraform apply` step for the owner to run themselves.
- Forces the Terraform + container build to actually work end-to-end as part
  of "done," rather than being validated only by hand once.

### Negative
- Adds one new file/concern (the workflow itself) to the single day's scope
  — folded into task 3 of the 3-task plan (see `delivery-plan.json`) rather
  than treated as a separate phase, to keep the "one day" framing honest.

### Risks (with mitigation)
- **Risk:** a bad `terraform apply` (e.g. a misconfigured variable) could
  break the live Container App with no manual gate in front of it.
  **Mitigation:** `terraform plan` output is surfaced in the workflow run
  before `apply`; for a single-owner hobby project with no other users
  depending on uptime, this residual risk is accepted rather than building
  a staging environment or approval gate (would contradict "as simple as
  possible").
