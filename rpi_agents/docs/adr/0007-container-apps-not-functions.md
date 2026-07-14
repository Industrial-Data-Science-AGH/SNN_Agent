# ADR-0007: Azure Container Apps (Consumption, scale-to-zero) instead of Azure Functions

- Status: accepted
- Date: 2026-07-14
- Deciders: Wiktor
- Relates to: F01_ingest_api, F02_storage_infra
- Supersedes: the compute half of an earlier, now-removed ADR that had
  originally paired "Azure Functions Consumption" with the storage decision
  below as one bundled choice. The storage half of that reasoning is
  restated here rather than kept as a separate file, per the project's
  "keep only what's necessary to build" instruction.

## Context

The owner wants "the cheapest version possible" delivered as "a VM or
container," with a fixed username/password instead of Entra ID, deployed via
Terraform + GitHub Actions CD. Azure Functions (ADR-0002) is a serverless
compute model that doesn't map cleanly onto "a container" and — more
concretely — Azure Static Web Apps' auth model (paired with Functions in the
original design) has no clean way to plug in a fixed username/password; it's
built around federated identity providers. Reworking that pairing to support
Basic Auth is more awkward than just running a normal container.

## Decision

Replace the Function App with a single **Azure Container App** on the
**Consumption (serverless, scale-to-zero) plan**, running one FastAPI
application that serves both the API routes and the dashboard UI (see
ADR-0008). Terraform provisions it (ADR-0011).

## Alternatives Considered

### A plain Azure VM (e.g. B1s burstable)
- **Pros:** Simplest mental model ("it's just a computer"); the user
  explicitly named this as an option.
- **Cons:** No free tier at all — a VM is billed continuously whether it's
  serving a request or sitting idle, roughly $7–13/month minimum even on the
  smallest burstable size, indefinitely. It also needs the owner to patch
  the OS themselves. Since the dashboard's whole point is being reachable
  "any time," the VM can't be shut off between requests either — it's
  genuinely the *most* expensive option on the table, not the cheapest.
- **Why not:** Directly contradicts "cheapest possible" — concrete numbers
  beat the VM's simplicity here.

### Azure Container Instances (ACI), running continuously
- **Pros:** Also "just a container," slightly simpler than Container Apps'
  revision/environment model.
- **Cons:** No free tier, no scale-to-zero — billed per-second for as long
  as it's running, and "running continuously so the dashboard is always
  reachable" is exactly the failure mode that makes this as expensive as a
  small VM.
- **Why not:** Same cost problem as the VM option, for the same reason.

### Storage choice (carried forward unchanged, restated briefly)

Data still lives in **one Azure Storage Account** — a private Blob container
(`snapshots`, images) and a Table (`events`, metadata) — not Cosmos DB and
not two separate resources. Reasoning: at this project's volume (a handful
of events/day), Table Storage's simple partition/row-key query model is
sufficient (see F02 design, Data model), and Cosmos DB's free tier would
mean reasoning about RU capacity for no query-capability benefit this system
actually needs. One Storage Account also means one resource to secure
instead of two. This is unaffected by the compute change above.

### Keep Azure Functions, but bolt on custom Basic Auth
- **Pros:** Keeps the original ADR-0002 stack; Functions are also
  effectively free at this volume.
- **Cons:** Functions + Static Web Apps' auth integration doesn't offer a
  clean seam for "fixed username/password" without either building a custom
  auth provider (real effort, the thing SWA's built-in providers exist to
  avoid) or splitting API auth (function key) from a separately-hand-rolled
  UI login — more moving parts than one container with one auth middleware.
- **Why not:** Container Apps gets the same near-$0 cost *and* a
  straightforward place to put HTTP Basic Auth (one middleware, one
  container), which is what was actually asked for.

## Consequences

### Positive
- Genuinely near-$0/month: Container Apps Consumption plan includes a
  permanent free monthly grant (vCPU-seconds, memory-seconds, and requests)
  that comfortably covers this workload's volume, same free-forever
  character as Functions had.
- "A container" the owner can reason about directly — one Dockerfile, one
  image, one deployable unit — while still scaling to zero when idle.
- HTTP Basic Auth (ADR-0009) is trivial to add as FastAPI middleware; no
  federated-identity plumbing required.

### Negative
- Loses Functions' per-route granularity (e.g. independent scaling/auth per
  endpoint) — not needed at this system's size.
- Cold start after scale-to-zero (low single-digit seconds) — same accepted
  characteristic as the original Functions design; unchanged trade-off.

### Risks (with mitigation)
- **Risk:** a single container means a bug in the dashboard UI code could in
  principle affect the ingest API's availability (shared process/deployment
  unit). **Mitigation:** FastAPI's routing keeps them logically separate
  within the app; acceptable coupling for a single-owner hobby system, not
  acceptable if this ever needs independent scaling or blast-radius
  isolation — flagged as the trigger to reconsider.
