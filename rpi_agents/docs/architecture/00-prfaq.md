# PR/FAQ: Wake-Up AI Cloud Dashboard

*(Tenet 2 revised 2026-07-14 — the Pi now retains events it couldn't push
and retries them on later wake cycles, instead of dropping them
permanently. See ADR-0014, ADR-0015.)*

## Tenets

1. **The Pi's own contract comes first.** Nothing added by this feature may
   delay or block `agent/power.py`'s wake → decide → re-halt cycle beyond a
   short bounded timeout. This is an ultra-low-power device; the cloud push is
   an observer, never a dependency of the safety/alarm logic.
2. **Best-effort telemetry, with bounded local durability — still not a
   delivery guarantee.** The Pi has intermittent connectivity (hotspot
   today, home Wi-Fi later). A missed push is queued locally (capped at 20
   pending events) and retried on subsequent wake cycles — every wake is
   already a fresh boot on this hardware, so "retry on powerup" and "retry
   every cycle" are the same thing (ADR-0015). This is *bounded*
   eventually-consistent delivery, not a guarantee: a long enough offline
   stretch still drops the oldest queued events once the cap is reached,
   and there is no unbounded store-and-forward. `event.log` on the Pi
   remains the complete, uncapped system of record for what actually
   happened; the cloud copy is a convenience view that now catches up
   automatically instead of just going stale.
3. **Lowest sustainable cost, not lowest today-only cost.** Every service
   choice defaults to a tier that is still ~$0/month after the student credit
   grant expires (Azure Container Apps Consumption free monthly grant,
   Storage Account pay-as-you-go pennies, ghcr.io free registry) rather than
   a tier that merely fits inside the credit balance now — and specifically
   never a fixed-cost resource like a VM or a paid container registry.
4. **The data is sensitive by default.** Snapshot images and wake timestamps
   reveal whether a home is occupied. No component may expose them without
   authentication, even for a single-owner hobby deployment.

## Press Release *(written as if launched)*

**FOR RELEASE — Q3 2026**

### Check whether your window was really broken — from your phone, even while the Pi is powered off

**The problem.** Wake-Up AI's Raspberry Pi spends nearly all its life fully
halted to save power. Today the only record of what happened during a wake —
the captured frame, Gemini's read of the scene, whether the alert email went
out — lives in a local log file the owner can only see by SSHing into the Pi
while it happens to be awake. That's a multi-minute round trip (find the Pi on
the network, log in, `tail` a log) for something that should be a five-second
glance at a phone.

**The solution.** After every wake cycle, the Pi pushes a compact record of
its decision — the same fields already written to its local `event.log`, plus
the captured snapshot, Gemini's reasoning text, and whether the alert email
was actually delivered — to a small always-on Azure backend. A web dashboard,
reachable any time regardless of whether the Pi is awake or halted, shows the
event history, a running real-vs-false-alarm count, and the image + reasoning
behind each decision. It deliberately does **not** try to guarantee every
event arrives (the Pi's network is intermittent by design) and does **not**
upload full video clips (only the single decision snapshot) — both keep the
system cheap and simple.

**How the customer experiences it.** The window trigger fires. Ninety seconds
later, before the owner has even found their phone, a new entry with a photo
and a one-line Gemini explanation is already sitting in the dashboard —
whether or not the alarm actually sounded, whether or not the email went
through.

**Availability.** Single-owner deployment, Azure Free/Consumption tiers,
funded initially by Azure for Students credit and designed to stay near-$0
after it's gone.

## FAQ

### Customer FAQ

**Q: Can I see history from before the Pi last woke up?**
A: Yes — that's the whole point. The dashboard reads from the cloud store, not
the Pi, so it's available whenever the Pi is halted (which is nearly always).

**Q: What if the Pi's network is down when it wakes?**
A: The event stays in the local `event.log` as it does today, and is also
queued locally (`sync_queue.jsonl`, capped at 20 pending events); the cloud
push is attempted with a short timeout, fails silently, and the wake cycle
continues to re-halt on schedule. That event won't appear in the dashboard
*yet* — the next wake cycle (and every one after, up to 5 queued events per
cycle) retries it until it succeeds, gets dropped after 5 failed attempts,
or the queue cap evicts it to make room for newer events. This is a bounded
catch-up, not a guarantee (see Tenet 2, ADR-0015) — not treated as an
incident either way.

**Q: Who can see the dashboard?**
A: Anyone with the fixed credential (default `ids`/`ids`, changeable via an
environment variable without a code change). See ADR-0009 — deliberately
simple rather than federated identity, at the owner's explicit request.

**Q: Does this replace the email alert?**
A: No. Email remains the real-time alert path (already implemented in
`agent/notifier.py`). The dashboard is a history/audit view, not a
notification channel, in this phase.

### Stakeholder / Internal FAQ

**Q: Why push instead of having the dashboard poll the Pi?**
A: The Pi is unreachable (halted) the vast majority of the time by design —
polling a target that's usually off doesn't work. See ADR-0001.

**Q: Why not just extend the existing (now-source-missing) `dashboard/`
package to run on the Pi itself?**
A: A Pi-hosted dashboard is only reachable while the Pi is awake, which is a
few seconds to a couple of minutes per wake — the opposite of what's being
asked for. That package's approach is superseded by this design, not reused.

**Q: What does this cost per month once the student credit is gone?**
A: Target is $0–$2/month at hobby-project volume. Azure Container Apps'
Consumption plan has a permanent (not just trial) free monthly grant that
comfortably covers this workload (ADR-0007), the Storage Account costs
fractions of a cent at this volume (ADR-0007), and GitHub Container Registry
is free (ADR-0012) — no fixed-cost resource (no VM, no ACR) is in the design.

### Integration FAQ *(assumptions — invite correction)*

- Assume the existing GitHub repo (`Industrial-Data-Science-AGH/SNN_Agent`,
  branch `feat/rpi`) is where this new code lives, alongside `rpi_agents/`.
- Assume deployment is automated via a GitHub Actions workflow that builds
  the container, pushes it to `ghcr.io`, and runs `terraform apply`
  (ADR-0011, ADR-0013) — not a manual step.
- Assume the Pi already has outbound HTTPS egress on whatever network it's
  joined to (true on the hotspot and will remain true on home Wi-Fi).
- Assume "the user" (dashboard viewer) is a single person (Wiktor) who will
  be given the fixed credential directly — no identity provider involved
  (ADR-0009).
- Assume the whole build fits one day, three implementation tasks, across
  two branches: `feat/azure-cd` (Terraform infra + the GitHub Actions CD
  workflow) and `feat/dashboard` (the FastAPI app — API, dashboard UI, Pi
  push client). A PR from `feat/dashboard` into `feat/azure-cd` is what
  triggers the infra build + deploy. See `02-delivery.md` and
  `delivery-plan.json`.

## Assumptions to confirm

- Azure for Students subscription already exists / will be created before
  Phase 0 starts.
- One captured snapshot image per event is sufficient (not the full 10-frame
  burst or the saved `.mp4` clip) — clips stay local only, per Tenet 3
  (bandwidth/cost).
- English-only dashboard UI is acceptable for v1.

## Open questions

- Should the dashboard eventually push a browser/mobile notification instead
  of (or alongside) email? Deferred — out of scope for this phase.
- Should old events/images ever be purged (retention policy) to bound Storage
  cost as history grows? Deferred to a follow-up ADR once real volume is
  observed; flagged as a risk in `01-system-overview.md`.

## Out of scope (MVP)

- Video clip upload (image only).
- Multi-user accounts, federated identity, or sharing the dashboard with
  anyone but the holder of the one fixed credential (ADR-0009).
- **Unbounded** retry/store-and-forward for failed pushes — a *bounded*
  local sync queue (20 events, 5 retries/attempts each) is now in scope
  (ADR-0015, revised Tenet 2); indefinite retention until delivery is not.
- Push notifications to a phone (email already covers real-time alerting).
- Credential rotation automation, staging environment, or deploy approval
  gates (ADR-0013) — accepted trade-offs for a single-owner hobby project
  deployed in one day.
