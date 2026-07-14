# Task plans — wakeup-ai-cloud-dashboard

Per-task implementation plans derived from `docs/architecture/` (PR/FAQ,
system overview, feature designs, ADRs, `delivery-plan.json`,
`module-map.json`). One plan per task in the current delivery plan
(`docs/architecture/02-delivery.md`: "one day, two branches, three tasks").

| Task | Plan | Branch | Feature(s) | Depends on |
|---|---|---|---|---|
| T01 | [T01-azure-infra-and-cd-pipeline.md](T01-azure-infra-and-cd-pipeline.md) | `feat/azure-cd` | F06, F02 | — |
| T02 | [T02-fastapi-dashboard-app.md](T02-fastapi-dashboard-app.md) | `feat/dashboard` | F01, F05, F04 | T01 (to verify live) |
| T03 | [T03-pi-push-client.md](T03-pi-push-client.md) | `feat/dashboard` | F03 | T01 (to verify live) |

T01 must merge/exist first since T02 and T03 have nothing to deploy against
otherwise. T02 and T03 can be worked in either order, or together, on the
same `feat/dashboard` branch — neither depends on the other's code. Opening
the PR from `feat/dashboard` into `feat/azure-cd` is what triggers T01's CD
pipeline and is the "done" signal for the whole build.

If a task's implementation needs to deviate from its `design.md` or from
`delivery-plan.json`/`module-map.json`, update those architecture files in
the same PR as the code change (per `02-delivery.md`'s branch strategy) and
update the corresponding plan here to match.
