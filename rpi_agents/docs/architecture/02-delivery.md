# Delivery

*(Revised 2026-07-14 — collapsed from a 4-phase/5-branch plan to a one-day,
two-branch, three-task plan at the owner's explicit request. See ADR-0007
through ADR-0013. A fourth task, T04, was added the same day when the owner
asked for offline durability on the Pi side — see ADR-0014, ADR-0015.
Branch strategy revised again the same day — see "Branch & Commit Strategy"
below — to match how the owner is actually executing the build: everything
on one branch, `feat/azure-cd` created at the end, not two branches in
parallel from the start.)*

## Shape: one day, one working branch, four tasks

| Task | Branch | Content | Gate |
|------|--------|---------|------|
| T01 | `feat/dashboard` | Terraform (`cloud/infra/`): Storage Account (Blob `snapshots` + Table `events`), Container App Environment + Container App shell, remote state backend; GitHub Actions workflow (`.github/workflows/deploy.yml`), pre-written to trigger on a future PR into `feat/azure-cd` (a branch that doesn't exist yet — see Branch & Commit Strategy) | `terraform validate`/`plan` clean; workflow YAML lints; a trivial placeholder image deploys successfully end-to-end once, proving the pipeline itself works before any real app code exists |
| T02 | `feat/dashboard` | FastAPI app (`cloud/app/`): `POST/GET /api/events`, `GET /api/events/{id}` (F01), Basic Auth middleware shared by every route (F05), server-rendered dashboard pages — event list, metrics band, detail view (F04) | Unit tests green; app runs locally against a dev Storage Account; manual check that unauthenticated requests are rejected |
| T04 | `feat/dashboard` | Retrofit T02's ingest route (F01, ADR-0014): `event_id` becomes a required, client-supplied ULID; ingest becomes an idempotent upsert keyed on it instead of a strict server-generated create | Unit tests green: same `event_id` posted twice yields one row, not two; malformed `event_id` rejected with 422 |
| T03 | `feat/dashboard` | `agent/cloud_sync.py` + `agent/sync_queue.py` (F03, ADR-0015): `build_payload()`/`push()`/`flush_queue()` with bounded timeout, wired into `machine.run_cycle()`; bounded local queue (`sync_queue.jsonl`, cap 20, 5 flushed/cycle, 5 attempts/event) retries pushes that failed on an earlier wake; `email_sent` + `event_id` added to `agent/machine.py`'s local event record | Unit tests green (mocked HTTP, mirrors `tests/test_notifier.py`, plus queue cap/attempts/corruption cases); manual E2E on the Pi: a real wake cycle produces a visible row within 30s, and 2-3 wakes triggered with network disabled are visible in the dashboard once the next connected wake cycle flushes the backlog |

All four tasks are built and unit-tested locally on `feat/dashboard` — see
Branch & Commit Strategy for how the deploy actually gets triggered once
everything above is done. T02 and T04 touch the same files
(`cloud/app/routes_api.py`/`storage.py`/`schemas.py`) so are naturally
sequential; T03's live push only succeeds once T04's server-side change is
in place, though T03's code can be written and unit-tested (mocked HTTP) in
parallel with T04.

## Branch & Commit Strategy

*(Revised 2026-07-14 to match actual execution — see the note at the top of
this file. The original plan below this line described two branches
existing in parallel from day one; that's not what happened, and this
section now documents what did.)*

- **`feat/dashboard`** (branched off `feat/rpi` before any of this
  feature's code, docs, or task commits) is the **only** working branch.
  T01 through T04 — infra, pipeline config, the FastAPI app, and the Pi
  push client — are all developed, committed, and tested locally here.
  There is no separate `feat/azure-cd` branch developed in parallel; `git
  status`/`git log` on this repo will show everything landing on
  `feat/dashboard` alone until the step below.
- **`feat/azure-cd` is created once development on `feat/dashboard` is
  complete** — branched from that finished state, not built up
  independently from `feat/rpi`. At that point it holds the same Terraform
  + workflow + app code `feat/dashboard` does; it exists specifically
  because `.github/workflows/deploy.yml` (T01, ADR-0013) is hardcoded to
  trigger only on a `pull_request` targeting a branch literally named
  `feat/azure-cd` — the workflow doesn't care when that branch was created,
  only that it exists by the time a PR targets it.
- **The actual CD trigger** is whatever PR sequence the owner opens against
  `feat/azure-cd` after creating it (e.g. a further small PR from
  `feat/dashboard`, or from a short-lived branch cut after `feat/azure-cd`
  exists) — the mechanism (ADR-0013: PR opened/synchronize → build → push
  `ghcr.io` → `terraform apply`) is unchanged from the original design,
  only *when* `feat/azure-cd` comes into existence has moved from "day one"
  to "once `feat/dashboard` is done."
- Any `design.md` / `module-map.json` / `delivery-plan.json` edits a task's
  scope required travel in the same commit/PR as that task's code, same as
  before.

## Dependencies & Assumptions

- Azure for Students subscription exists and has Container Apps, Storage,
  and the Container Apps free scale-to-zero grant available (standard on
  every Azure subscription tier, including student ones).
- Azure CLI (`az`) and Terraform CLI available in the developer's
  environment for local validation before pushing; not assumed
  pre-installed — first T01 step includes verifying/installing them.
- A GitHub PAT with `read:packages` scope exists as a repo secret for the
  Container App to pull from `ghcr.io` (ADR-0012) — created once, manually,
  before T01's workflow can succeed; not itself automated.
- `agent/machine.py` changes in T03 must not alter the existing
  `Decision`/`PrefilterResult`/`VisionVerdict` contracts in `agent/types.py`
  — the new `email_sent` and `event_id` fields are additive to the logged
  record only, not new fields on those dataclasses (avoids touching
  `tests/test_machine.py`'s existing assertions beyond what T03 explicitly
  adds).
- T04's retrofit changes `cloud/app/schemas.py::EventIn` (adds required
  `event_id`) and `cloud/app/storage.py::write_event()` (create → upsert) —
  both already shipped by T02. T04's own tests must cover the upsert path
  explicitly; T02's existing `tests/test_ingest.py` needs its fixture
  payload updated to include `event_id` once T04 lands, or those tests will
  start failing on the new required field.
- No staging environment or deploy approval gate — the single PR's merge is
  the release (ADR-0013, accepted at single-owner hobby scale).
