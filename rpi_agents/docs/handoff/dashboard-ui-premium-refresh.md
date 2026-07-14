# Handoff: premium dashboard UI refresh

For whoever (Claude Code) implements the frontend. **The backend is done —
you should not need to touch it.** This file is the complete brief: brand,
current state, API contracts, hard boundaries, and the chart requirements.

## What this project is

`wakeup-ai-cloud-dashboard` — a security-camera companion for a Raspberry
Pi 5 agent ("Wake-Up AI"). The Pi wakes, captures a frame, runs a
prefilter + Gemini vision call, decides real/false alarm, and pushes a
record (metadata + snapshot photo) to this Azure-hosted dashboard. The
owner (single user, fixed credential) checks the dashboard from a phone or
laptop to see event history — even while the Pi itself is fully powered
off, which is most of the time.

The whole app is one FastAPI service (`cloud/app/`) that serves both the
JSON API and server-rendered (Jinja2) HTML pages — no separate frontend
host, no build pipeline, no SPA framework in use today.

## The ask

The current UI (`cloud/app/templates/*.html`) is functional but plain: a
flat near-black background, basic bordered cards, an HTML table. The owner
wants it to feel **premium** — not just "dark mode with buttons" — with
real charts built from the metrics data (a trend of real/false/non-
escalating wakes over time, a vision-source breakdown, email delivery rate,
latency), and photos wired in cleanly. Brand accents should echo the
company's logo.

## Brand

Company: **Industrial Data Science**. The logo is a gear/cog icon with a
color gradient running from a deep forest/emerald green through near-black
to a deep maroon/oxblood red, with a small neural-network node-and-edge
graph inset inside the gear's cutout, next to the bold black wordmark
"INDUSTRIAL DATA SCIENCE."

Suggested palette (sampled by eye from the logo — treat as a starting
point, fine-tune with a color picker against the actual logo file if you
have it, and implement as CSS custom properties so they're easy to
retune):

```css
--bg-base: #0a0a0b;        /* near-black canvas */
--bg-elevated: #141518;     /* card/panel surface, one step up from base */
--bg-gradient-start: #0d1f16; /* dark emerald, for subtle gradient washes */
--bg-gradient-end: #1a0d12;    /* dark maroon, for subtle gradient washes */
--accent-emerald: #1b6b45;
--accent-emerald-bright: #2f9e64;  /* hover/active states, chart lines */
--accent-maroon: #7a1230;
--accent-maroon-bright: #a8274f;   /* hover/active states, chart lines */
--text-primary: #f2f2f0;
--text-muted: #9a9a9e;
--border-subtle: #232427;
```

Use the green/maroon accents deliberately and sparingly — for chart
series, active states, badges, focus rings, the header/nav — not as large
flat color blocks. The base UI should read as sophisticated dark
grey/graphite with a soft directional gradient (e.g. a faint radial or
diagonal gradient from graphite toward the emerald/maroon at the edges),
not a flat `#111` fill like today's `base.html`. Think "premium
industrial/technical instrumentation panel," not neon or sci-fi.

## Current state (what exists today)

```
cloud/app/templates/
  base.html          # <head>, inline <style>, header, {% block content %}
  event_list.html    # metrics band (4 cards) + event table with thumbnails
  event_detail.html  # single event: full image, reason, field table
cloud/app/static/     # empty — created for you, already mounted at /static
                       # (see main.py; no backend change needed to use it)
```

All styling today is a single inline `<style>` block in `base.html`. There
is no CSS framework, no JS bundler, no build step. Keep it that way unless
you have a strong reason not to — this is a small single-owner hobby
project (see `docs/architecture/00-prfaq.md`, Tenet 3: lowest sustainable
cost/complexity, not lowest effort today only). A CDN-loaded charting
library (e.g. Chart.js) plus vanilla JS/CSS in `cloud/app/static/` is very
much in scope and the intended way to add charts — do not introduce a
frontend build toolchain, a JS package manager, or a SPA framework for this.

## Hard boundaries

**You may freely edit or create:**
- `cloud/app/templates/*.html` (rewrite entirely if needed)
- `cloud/app/static/**` (new: CSS, JS, fonts, the logo image if you have
  a file for it — this directory is already mounted at `/static` in
  `main.py`, nothing else to wire)

**Do not touch:**
- Any `cloud/app/*.py` file (`main.py`, `routes_api.py`,
  `routes_dashboard.py`, `storage.py`, `schemas.py`, `metrics.py`,
  `auth.py`) — the backend, including the new analytics endpoint below, is
  already implemented and tested (183 passing tests). If you think you need
  a new/changed API shape, stop and say so rather than editing Python.
- `cloud/infra/`, `.github/`, `agent/`, `docs/architecture/`, `docs/adr/`,
  `docs/plans/` — outside this task's scope entirely.
- `cloud/app/static/` must only ever hold generic UI assets (CSS, JS,
  fonts, the brand logo). It is served **without** authentication (a
  deliberate, documented exception in `main.py` — FastAPI's global auth
  dependency doesn't cover `app.mount()`'d static file serving). Never put
  a snapshot photo, an event record, or anything user-specific there —
  those must only ever reach the browser through the existing SAS-URL path
  (see below).

## Auth — you don't need to build anything for this

Every route (dashboard pages and API alike) sits behind one shared HTTP
Basic Auth credential (default `ids`/`ids` locally, overridable via
`DASHBOARD_USER`/`DASHBOARD_PASSWORD` env vars). The browser prompts once
per session and then automatically resends the credential on every
same-origin request — including your `fetch()` calls to `/api/events` and
`/api/metrics` from JavaScript running on an already-authenticated page.
Don't build a login form, a token flow, or manual header-passing; a plain
`fetch("/api/metrics")` from a page the user already loaded will just work.

## API contracts available to you

### `GET /api/events?since=&limit=`

Returns the most recent events first. `since` is an ISO date
(`YYYY-MM-DD`), defaults to 30 days ago. `limit` defaults to 100, max 500.

```json
[
  {
    "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "ts_wall": 1784048796.83,
    "woken_by_trigger": false,
    "escalate": true,
    "motion": true,
    "person": false,
    "score": 0.1547,
    "vision_source": "gemini",
    "is_intrusion": false,
    "alarm": false,
    "reason": "vision: no person visible, likely a shadow",
    "email_sent": false,
    "latency_s": 10.67,
    "received_at": 1784048801.2,
    "image_url": "https://<account>.blob.core.windows.net/snapshots/<id>.jpg?<sas-token>"
  }
]
```

### `GET /api/events/{event_id}`

Same shape as one item above (full detail; today identical to the summary
shape, kept as its own contract since detail-only fields may land here
later).

### `GET /api/metrics?since=` — new, built for this refresh

No `limit` — always aggregates the full window. `since` same default as
above.

```json
{
  "since": "2026-06-14",
  "until": "2026-07-14",
  "summary": {
    "real_wakes": 12,
    "false_wakes": 34,
    "non_escalating_wakes": 210,
    "email_delivery_rate": 0.92
  },
  "daily": [
    {"date": "2026-06-14", "real_wakes": 0, "false_wakes": 2, "non_escalating_wakes": 18, "total": 20},
    {"date": "2026-06-15", "real_wakes": 1, "false_wakes": 1, "non_escalating_wakes": 22, "total": 24}
  ],
  "vision_source_breakdown": {
    "real_wakes": {"gemini": 10, "failsafe": 2, "none": 0},
    "false_wakes": {"gemini": 30, "failsafe": 0, "none": 4},
    "non_escalating_wakes": {"gemini": 0, "failsafe": 0, "none": 210}
  },
  "latency_s": {"avg": 8.2, "p50": 7.9, "p95": 14.1, "max": 22.0}
}
```

`daily` is sorted **oldest first** (ready to plot left-to-right as a trend
line — don't reverse it). `vision_source` is `"none"` for non-escalating
events, since those never call the vision model at all.

**Important framing for `vision_source_breakdown`:** this is not a
ground-truth confusion matrix. This system has no way to confirm whether an
event was actually a real intrusion — it only knows what its own pipeline
decided. Label this chart as something like "Decision breakdown by vision
path" or "Outcome vs. vision source" — not "accuracy," "confusion matrix,"
or "correct/incorrect." It's genuinely useful (e.g. a high `failsafe` share
means Gemini itself is failing/timing out often, which is worth surfacing),
just don't imply it's measuring correctness.

## Photos — already fully wired end-to-end, nothing to build

Every event that escalated (prefilter thought something was happening) has
a captured snapshot pushed from the Pi, stored in Azure Blob Storage, and
exposed as `image_url` in both `GET /api/events` and `GET
/api/events/{id}` — a fresh, short-lived (15-minute) SAS URL, minted per
request. Just use `image_url` directly as an `<img src>`. Two things to
know:

- `image_url` can be `null` (non-escalating events never captured a
  snapshot worth pushing, or the blob upload failed) — always handle the
  missing-image case gracefully (you already have a pattern for this in
  the current templates).
- SAS URLs expire in 15 minutes. If you build anything like a photo
  gallery/lightbox that could stay open longer than that, either refetch
  the event data periodically or accept that a stale tab needs a reload —
  don't try to cache or persist these URLs (matches the project's "no
  localStorage / no client persistence" convention).

## Suggested chart set (not a rigid spec — use judgment)

1. **Trend over time** — stacked or grouped chart from `daily`: real vs.
   false vs. non-escalating wakes per day.
2. **Summary cards** — the four `summary` values, styled as premium KPI
   cards rather than today's plain bordered boxes.
3. **Vision-source breakdown** — a heatmap/grid or grouped bar chart from
   `vision_source_breakdown`, labeled per the framing note above.
4. **Email delivery rate** — could be its own small donut/gauge, or folded
   into the summary cards.
5. **Latency** — `latency_s`'s avg/p50/p95/max, e.g. as a small stat block
   or a simple distribution indicator.
6. Event list stays, but restyle: better thumbnails, refined badges,
   premium table or card-grid layout — your call.
7. Event detail page: larger hero image, cleaner metadata layout.

## How to run this locally to check your work

Same setup as the automated testing guide the backend used
(`docs/plans/T04-testing-guide.md`, section 2) — Azurite (Docker) standing
in for real Azure Storage, no cloud account needed:

```bash
docker run -d --name azurite -p 10000:10000 -p 10002:10002 \
  mcr.microsoft.com/azure-storage/azurite \
  azurite --blobHost 0.0.0.0 --tableHost 0.0.0.0 --skipApiVersionCheck

# create the events table + snapshots container once (see T04-testing-guide.md 2.2)

export AZURE_STORAGE_CONNECTION_STRING="DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;TableEndpoint=http://127.0.0.1:10002/devstoreaccount1;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
export DASHBOARD_USER=ids
export DASHBOARD_PASSWORD=ids
uv run uvicorn cloud.app.main:app --reload --port 8000
```

Then POST a few sample events (see `T04-testing-guide.md` section 2.4 for
an example payload; vary `alarm`/`escalate`/`vision_source`/`ts_wall`
across a few days) so the charts have something to render, and open
`http://localhost:8000` (Basic Auth: `ids`/`ids`).

## Deliverables checklist

- [ ] `cloud/app/templates/base.html` — new visual system (colors,
      typography, layout shell), no other file's `{% block %}` contract
      changed unless you also update the templates that extend it.
- [ ] `cloud/app/templates/event_list.html` — premium metrics band + at
      least the trend and vision-source-breakdown charts, restyled table.
- [ ] `cloud/app/templates/event_detail.html` — restyled detail view.
- [ ] `cloud/app/static/` — CSS/JS (and logo asset if available) backing
      the above.
- [ ] Charts fetch from `GET /api/metrics` client-side (don't have the
      Jinja route pre-compute chart data — that defeats the point of the
      new endpoint).
- [ ] No changes to any `.py` file, `cloud/infra/`, or `docs/`.
- [ ] Confirmed working locally against Azurite per the steps above.
