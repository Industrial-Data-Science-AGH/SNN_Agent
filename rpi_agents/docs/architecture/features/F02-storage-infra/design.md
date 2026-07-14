# F02 storage_infra

## Context

Foundational feature — everything else (ingest API, dashboard reads) depends
on this existing. One Azure Storage Account provides both object storage
(snapshot images) and a lightweight table (event metadata), deliberately
kept as a single resource rather than a separate Blob account + Cosmos DB
account, to minimize both cost and the number of things to provision/secure
(see ADR-0007, "Storage choice" section).

Diagram: see the ingest flow in F01's design — this feature is the two
cylinders (`Blob Storage`, `Table Storage`) it writes to, provisioned here.

## Contracts

- **Blob container:** `snapshots` (private, no public read access).
  Object key: `{event_id}.jpg`.
- **Table:** `events`. PartitionKey: `YYYY-MM-DD` (the event's UTC date, from
  `ts_wall`) — bounds partition size and gives cheap date-range scans.
  RowKey: `{event_id}` (a ULID generated at push time — sortable, unique,
  avoids clock-skew collisions between Pi and Azure clocks).
- Provisioned via **Terraform** (`cloud/infra/*.tf`, `azurerm` provider,
  ADR-0011), on the `feat/azure-cd` branch (task T01), parameterized by
  environment (`dev`/`prod`) so a throwaway dev storage account can exist
  alongside the real one without duplicating modules.

## Data model

Table Storage entity schema (mirrors the local `event.log` JSON record, plus
the two cloud-only fields):

| Field | Type | Notes |
|---|---|---|
| `PartitionKey` | string | `YYYY-MM-DD` |
| `RowKey` | string | ULID, = `event_id` |
| `ts_wall` | double | epoch seconds, from the Pi |
| `woken_by_trigger` | bool | |
| `escalate` | bool | |
| `motion` | bool | |
| `person` | bool | |
| `score` | double | |
| `vision_source` | string | `"gemini"` \| `"failsafe"` \| null |
| `is_intrusion` | bool \| null | |
| `alarm` | bool | |
| `reason` | string | Gemini's reasoning text (or prefilter/failsafe reason) |
| `email_sent` | bool | added by F03 |
| `latency_s` | double | |
| `blob_name` | string | `{event_id}.jpg`, empty if no snapshot was pushed |
| `received_at` | double | epoch seconds, set by the app on ingest (for skew/debugging, not shown in UI) |

## Security

- **Blob container `snapshots` is private** — no public/anonymous read access
  at the container level. All reads happen through F01's short-lived
  (15-minute) SAS URLs, minted per authenticated dashboard request; there is
  no long-lived or public link to any image at any point.
- **Table `events`** is reachable only via the Storage Account connection
  string, which lives solely in the Container App's native secret store
  (ADR-0010) — never issued to the browser or the Pi. Neither the dashboard
  page nor the Pi push client ever holds a Storage credential; both go
  through F01's routes, which are the sole holder of the connection string.
- **Terraform state** (which will contain this connection string as a
  resource attribute) lives in a private Blob container in this same
  Storage Account — never local, never committed (ADR-0010, ADR-0011).
- **This resource has no internet-facing surface of its own** — its only
  attack surface is transitively through F01/F05 (already threat-modeled
  there) and through whoever holds the Storage connection string or
  Terraform state access. No additional STRIDE pass is needed at this layer.

## Risks

- **Table Storage has no server-side full-text/complex query support** —
  acceptable because F04 does client-side aggregation over a bounded recent
  window (see F04 design); would need to revisit (e.g. move to Cosmos DB) if
  query needs grow materially past "list recent, filter by date/alarm".
- **Unbounded growth over time** — flagged in `01-system-overview.md` Risks;
  no retention policy in this phase, deferred to a follow-up ADR once real
  volume is observed.

## Decisions

- ADR-0007 (Container Apps; storage choice — one Storage Account, Blob+Table,
  vs. Cosmos DB / two resources — restated in that same ADR).
- ADR-0011 (Terraform, not Bicep).

## Branch

`feat/azure-cd` (task T01)
