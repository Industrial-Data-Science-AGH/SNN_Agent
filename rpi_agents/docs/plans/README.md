# Task plans — wakeup-ai-cloud-dashboard

Per-task implementation plans derived from `docs/architecture/` (PR/FAQ,
system overview, feature designs, ADRs, `delivery-plan.json`,
`module-map.json`). One plan per task in the current delivery plan
(`docs/architecture/02-delivery.md`: "one day, one working branch, four
tasks" — T04 added 2026-07-14, ADR-0014/ADR-0015, alongside T03's queue
design; branch strategy revised the same day to match actual execution —
see `02-delivery.md`, "Branch & Commit Strategy"). T05 added later the same
day (ADR-0016) — the backend half of a premium, chart-driven dashboard UI
refresh; see `T05-dashboard-analytics-api.md` and
`docs/handoff/dashboard-ui-premium-refresh.md`. T06 added 2026-07-15
(ADR-0018) — manual ground-truth review, the follow-up ADR-0016 explicitly
deferred; see `T06-manual-event-review.md`.

All four tasks are developed, committed, and unit-tested locally on
**`feat/dashboard`** (branched off `feat/rpi` before any of this work
started) — there is no separate `feat/azure-cd` branch built up in
parallel. `feat/azure-cd` is created later, once `feat/dashboard` is done,
specifically because `.github/workflows/deploy.yml` only triggers on a PR
targeting a branch named `feat/azure-cd` (ADR-0013).

| Task | Plan | Branch | Feature(s) | Depends on |
|---|---|---|---|---|
| T01 | [T01-azure-infra-and-cd-pipeline.md](T01-azure-infra-and-cd-pipeline.md) | `feat/dashboard` | F06, F02 | — |
| T02 | [T02-fastapi-dashboard-app.md](T02-fastapi-dashboard-app.md) | `feat/dashboard` | F01, F05, F04 | T01 (code lives on the same branch; local Terraform validate/plan stands in for "live" until deploy time) |
| T04 | [T04-idempotent-ingest-retrofit.md](T04-idempotent-ingest-retrofit.md) | `feat/dashboard` | F01 | T02 (retrofits its shipped ingest route) |
| T03 | [T03-pi-push-client.md](T03-pi-push-client.md) | `feat/dashboard` | F03 | T01, T04 (server must accept client `event_id` before retries can succeed) |
| T05 | [T05-dashboard-analytics-api.md](T05-dashboard-analytics-api.md) | `feat/dashboard` | F01, F04 | T02, T04 (reads the same event schema/storage layer) |
| T06 | [T06-manual-event-review.md](T06-manual-event-review.md) | `feat/dashboard` | F01, F04 | T02, T05 (extends the same `GET /api/metrics` response) |

T02 and T04 touch the same files (`cloud/app/schemas.py`/`routes_api.py`/
`storage.py`) so are naturally sequential. T03's code can be written and
unit-tested (mocked HTTP) any time, but its live push only succeeds once
T04's server-side change exists. Once all four are done on
`feat/dashboard`, `feat/azure-cd` gets created from that finished state,
and whatever PR the owner opens against it is what triggers T01's CD
pipeline — the "done" signal for the whole build.

If a task's implementation needs to deviate from its `design.md` or from
`delivery-plan.json`/`module-map.json`, update those architecture files in
the same PR as the code change (per `02-delivery.md`'s branch strategy) and
update the corresponding plan here to match.
