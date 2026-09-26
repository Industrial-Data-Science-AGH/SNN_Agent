# SNN Agent backend (rpi side)

The service that the edge bridge talks to: it accepts spike batches, runs the SNN runtime, asks the Pi for a photo when
the runtime triggers, has a vision model look at the photo, applies the alarm policy, sends alarm commands back to the
Pi and e-mails the result. One Python package, one image, two roles (`api`, `worker`; `both` for local runs).

```
Pi bridge --batches--> API --trigger--> event + capture command (one transaction)
Pi bridge <--command-- API <--poll----- (Pi fetches, executes, acks)
Pi bridge --photo----> API --queue----> worker: vision -> policy -> alarm command / review -> e-mail job
```

## Design decisions that are not obvious from the code

- **One atomic step per event.** The event, its capture command and the outbox rows that publish them live in a single
  table partition and are written in one transaction. There is no state where an event exists without its command or
  the reverse. A publisher (inline, plus a reconciler in the worker) drains the outbox to the queues.
- **The runtime is stateful, so a crash is never papered over.** A batch is recorded as `processing` before the runtime
  steps. If the process dies in between, the retry ends the session (`SESSION_LOST`) and the device opens a new epoch:
  an explicit gap and a new warm-up, never a silently double-stepped state. The bridge handles this on its own.
- **Epochs fence stale devices.** Every command and batch carries the epoch; the device row is updated with
  compare-and-swap. A command for an old epoch is refused, not executed late.
- **Default policy is human review.** `manual-review-only-v1` never raises an alarm. `armed-glass-and-person-v1` is an
  explicit opt-in. Alarm downgrades are visible reasons (session inactive, event too old, cooldown) and the cooldown
  is claimed once per resolution.
- **A failed model never blocks an event.** Vision is retried a bounded number of times, then the job goes to a poison
  queue and the event still resolves to human review with an honest "vision unavailable".
- **E-mail is a separate job and never over-claims.** Outcomes: `sent`, `failed`, `delivery_unknown` (the connection
  died after the body went out), `not_configured`. Recipients come from configuration only; model text is data.
- **Auth is deliberately small.** One shared operator login (scrypt hash, lockout, CSRF token, Origin check, hashed
  session tokens, `__Host-` cookie) and per-device bearer tokens (`device_id~secret`, only the hash is stored).
  Logs pass a redaction filter.
- **Nothing unsafe by default.** Memory storage, the stand-in runtime, the emulator connection string, live sessions
  and plain-http cookies each need their own explicit flag (`backend_config.py` lists them).

## Run it locally

```
python -m venv .venv && .venv/bin/pip install --require-hashes -r requirements-backend.lock
python -m rpi_agents.cloud.app.admin hash-password           # prints the value for SNN_OPERATOR_PASSWORD_HASH
python -m rpi_agents.cloud.app.admin issue-device-token <device_id>   # needs the storage env below; prints the token once
python -m rpi_agents.cloud.app.server both --check-config    # validates the configuration, then exits
```

Minimal demo environment (no Azure, data is lost on restart):

```
SNN_STORAGE=memory SNN_ALLOW_MEMORY=1 SNN_RUNTIME=demo SNN_ALLOW_DEMO_RUNTIME=1
SNN_DEMO_VISION=glass_person SNN_ALLOW_DEMO_VISION=1 SNN_POLICY=armed-glass-and-person-v1
SNN_MANIFEST_PATH=contracts/fixtures/model-manifest.json SNN_OPERATOR_USERNAME=operator
SNN_OPERATOR_PASSWORD_HASH=<hash> SNN_INSECURE_DEV=1 SNN_ALLOW_LIVE=1
```

Real variables (all documented at the top of `app/backend_config.py`): `SNN_STORAGE=azure` with `SNN_AZURE_ACCOUNT`
(managed identity), `SNN_RUNTIME=package.module:factory`, `SNN_VISION_ENDPOINT` + `SNN_VISION_DEPLOYMENT`,
`SNN_ALLOWED_HOSTS`, `SNN_ALERT_RECIPIENTS`, `SNN_SMTP_*`.

Image: `docker build -f rpi_agents/cloud/infra/Dockerfile -t snn-backend .` (an allow-list `.dockerignore` keeps datasets
and models out of the context).

## Azure

`infra/deploy.sh` runs the whole rollout with the az CLI (read its header for the variables); `infra/main.bicep` is the
template. Phase 1 creates everything except the apps, then the image is built in the registry (`az acr build`) and the
secrets go into Key Vault, then phase 2 creates the two container apps. Run `az deployment group what-if` first.

| Resource | Notes |
| --- | --- |
| Storage account | Tables, blob container `images`, four queues. Shared-key access **off**; identity only. |
| Key Vault (RBAC) | `operator-password-hash`, `operator-password`, `smtp-user`, `smtp-password`, `smtp-from`, `alert-recipients`, `device-token-<id>`. The apps carry `keyvault:<name>` markers and read the values themselves at start-up with the managed identity (`app/keyvault.py`); nothing secret is in the template, the parameters or the app definition. |
| AI Services account + `gpt-5-mini` deployment | Local (key) auth **off**: the apps call it with their managed identity (`Cognitive Services OpenAI User`). |
| Container registry (Basic) | Admin user off; the apps pull with `AcrPull`. |
| Container Apps: `-api` (public https) and `-worker` (no ingress, scales on the vision queue) | 0.25 vCPU / 0.5 GiB each, min 1 replica each. |
| One managed identity | Table/Blob/Queue data contributor, vault secrets user, AcrPull, OpenAI user. |

**Express environments.** A new students subscription gets Container Apps "express" environments. They refuse the platform's
own Key Vault references (hence `keyvault.py`) and custom scale rules (hence one always-on worker; `scaleWorkerOnQueue=true`
turns KEDA queue scaling on for a standard environment), and have no log streaming (query Log Analytics instead:
`ContainerAppConsoleLogs_CL`). The platform's identity endpoint is plain http on a carrier-grade-NAT address
(`100.64.x.x`), which the identity client accepts and nothing else does. Health probes reach the pod by its internal address,
so `/healthz` is exempt from the host allow-list.

A subscription may be limited to **one** Container Apps environment (Azure for Students is). Then pass
`ENVIRONMENT_ID=<existing environment id>` (the region must match); the apps join it and nothing in it is modified.

Cost on a students subscription is small but not zero: the two always-on replicas, the registry and the vault are
the standing cost; gpt-5-mini at `detail: low` is a fraction of a cent per photo. `az group delete -n <rg>` removes
everything the template created (the shared environment, if joined, is not part of the group).

### Adding a dashboard or the real SNN runtime later

- **Dashboard**: another container app in the same environment and registry, its own identity with read roles only
  (`Storage Table Data Reader`, `Storage Blob Data Reader`). It calls the API for what the API already exposes
  (`/v1/events`, `/v1/events/{id}`, images, device status) or reads the tables. Serve it from the same host as the
  API, or behind the same reverse proxy: the operator cookie is `__Host-` scoped and the API refuses other origins. The
  vision run's `rationale` and token `usage` are in the `visionruns` table (also printed by `admin events`); exposing them
  through the API needs an additive change to the shared Event contract, which is a decision for the contract owners.
- **Real SNN runtime**: add the package to the image (a second build stage or a derived image), set
  `SNN_RUNTIME=package.module:factory` and `SNN_ALLOW_DEMO_RUNTIME=0`, deploy a new revision. Nothing else changes: the
  runtime contract (`load/reset/step/snapshot/checkpoint/restore`) is what `demo_runtime.py` implements.
- **Second device**: `admin issue-device-token <device_id>`; the token goes to that device's private credential file.

## Verified, and what is not

Verified by tests in `tests/w0` (every module mutation-tested; survivors added as tests or judged equivalent):
the whole chain in-process with the real bridge (serial bytes to alarm ack), backend restart, outage, session loss,
idempotent retries, poison messages, auth attacks (CSRF, lockout, token replay), log redaction. The Azure Table, Blob
and Queue adapters also run against the Azurite emulator (`SNN_TEST_AZURITE=1`), which caught four adapter bugs the fakes
could not. The container image builds (297 MB, non-root), passes `--check-config`, serves `/healthz` and the login.

On hardware: a Pi 5 with the stand-in Mega and the D415 ran the whole chain against this backend (API + worker on a
laptop, Azurite storage, tunnelled to the Pi's loopback): serial bursts, trigger, capture command, photo upload, scripted
vision, armed policy, alarm command, applied on the Pi with a mock GPIO factory, acknowledged. Three events resolved to
`alarm_confirmed`; outbox, queues and poison queues ended empty. The tunnel dropped for the last 30 s and the bridge kept
its batches in the outbox and shut down cleanly. Vision was the scripted stand-in and no real pin was driven.

**Verified on Azure** (students subscription, francecentral, gpt-5-mini): managed identity from the container to Storage, Key
Vault, the registry and the model; real TLS with the `__Host-` cookie, HSTS and the proxy-header count behind the ingress;
the Foundry request shape for the `reasoning` family against a real deployment (rationale and token usage come back; with
`reasoning_effort: minimal` the model spends no reasoning tokens, so the rationale field is the reasoning); the e-mail job
from the container (SMTP accepted, status `sent`; inbox arrival is for a human to confirm); the full chain from the Pi:
Mega stand-in wake, D415 photo, upload, vision, armed policy, alarm command applied on the Pi (mock GPIO), acks.
Two scenarios, same path: a real photo (nothing to see: `no_alarm` / `review_required`) and a synthetic intruder in a
`replay` session (labelled `synthetic`; glass and person found, `alarm_confirmed`, later events `ALARM_COOLDOWN`).
The model is not deterministic: the same desk scene came back once `person_visible: false` and once `unknown`, which the
policy turns into a human review. Never rely on a single answer.

**Not verified** (needs something only the team has):

| Item | Why open |
| --- | --- |
| Real LED/buzzer on the Pi | Ports are assumed (BCM 17 LED, 27 buzzer); a human must be present for the first activation. |
| The real SNN runtime | The `demo` runtime is a stand-in; the real one plugs in through `SNN_RUNTIME`. |
| Uno + microphone | Only the Mega stand-in has been used on hardware. |
| KEDA queue scaling with identity | Refused by express environments; needs a standard environment. |
| Model accuracy | The synthetic scene is a drawing and the real scene had nothing in it: the run proves the pipeline, not detection quality. |

## Decisions the project owner has to make

1. ~~Which Foundry model~~ Decided: gpt-5-mini, called with the managed identity (no API key). `SNN_VISION_FAMILY=reasoning` for gpt-5 / gpt-5-mini / gpt-5-nano (no temperature,
   `max_completion_tokens`, minimal reasoning effort, api-version 2025-04-01-preview), `chat` (default) for gpt-4o-mini /
   gpt-4.1-mini. The photo is sent at `detail: low`, so the cheapest tier is usually enough for glass + person.
2. Whether glass-only or person-only detections may ever raise an alarm; today only glass + person does, and only when armed.
3. Buzzer wiring and whether the first live alarm is LED only.
4. Retention of photos and events (nothing is deleted today).
5. How device tokens reach the Pi (issue-once via the admin tool, then copied by hand).
