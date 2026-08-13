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
  `auth.py`) — the backend, including the analytics endpoint and the manual
  review endpoint below, is already implemented and tested. If you think you
  need a new/changed API shape, stop and say so rather than editing Python.
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
    "window_broken": false,
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
    "email_delivery_rate": 0.92,
    "gemini_call_success_rate": 0.80,
    "window_break_confirmation_rate": 0.75
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
  "trigger_breakdown": {
    "triggered": {"real_wakes": 12, "false_wakes": 34, "non_escalating_wakes": 210},
    "not_triggered": {"real_wakes": 0, "false_wakes": 0, "non_escalating_wakes": 0}
  },
  "last_sync": {
    "event_id": "01ARZ3NDEKTSV4RRFFQ69G5FAV",
    "ts_wall": 1784048796.83,
    "received_at": 1784048801.2
  },
  "review_accuracy": {
    "reviewed_count": 18,
    "window_broken": {"tp": 10, "fp": 1, "tn": 6, "fn": 1, "accuracy": 0.889},
    "intrusion": {"tp": 8, "fp": 2, "tn": 7, "fn": 1, "accuracy": 0.833}
  },
  "latency_s": {"avg": 8.2, "p50": 7.9, "p95": 14.1, "max": 22.0}
}
```

**Four fields added after the first UI pass (2026-07-15) — new to this
version of the doc:**

- **`summary.gemini_call_success_rate`** — of every event that actually
  attempted a vision call (excludes non-escalating events, which never call
  vision), what fraction got a real Gemini response rather than falling
  back to failsafe (a timeout/exception). Render as a fifth KPI card,
  same style as the other four in `summary`.
- **`trigger_breakdown`** — outcome × `woken_by_trigger`, same shape as
  `vision_source_breakdown`. The deployed system only wakes via a genuine
  SNN hardware trigger, so `not_triggered` should normally be all zeros —
  if you build a chart for this, treat a nonzero `not_triggered` value as
  something to visually flag (e.g. a warning color), not just another data
  point.
- **`last_sync`** — the most recent event in the currently-selected window,
  by `received_at` (when the cloud actually received the push — use this
  one for "last synced," not `ts_wall`, which is when the Pi woke). `null`
  when the window has no events at all. Put this somewhere prominent and
  persistent — e.g. next to the page title/header, not buried in a card —
  formatted as relative time ("Last synced 4 minutes ago") with the exact
  timestamp available on hover/title attribute. If `last_sync` is `null`,
  show something like "No sync in the last 30 days" rather than hiding the
  element or leaving it blank — that's the state most worth surfacing
  clearly, since it likely means the Pi has been offline or broken for a
  while.
- **`summary.window_break_confirmation_rate`** — of every event Gemini
  actually classified (excludes failsafe fallbacks, which default this to
  `true` conservatively but aren't a real visual confirmation), what
  fraction did Gemini classify as showing a broken window. This is the
  metric that most directly validates the SNN's own detection target
  (glass breakage) rather than generic intrusion. Render as a sixth KPI
  card, or pair it visually with `gemini_call_success_rate` since they're
  both "how reliable/confirmatory is the vision stage" numbers. Each event
  in `GET /api/events` also now carries its own `window_broken` field
  (`true`/`false`/`null`) if you want to show it per-row or on the detail
  page (e.g. a small "window" badge next to the existing alarm badge).

**Fifth field added 2026-07-15, new to this version of the doc —
`review_accuracy`.** Unlike everything else on this endpoint, this one
genuinely **is** a confusion matrix — see the new "Manual review" section
below for the full picture (both the write endpoint that produces this data
and how to render it). Short version: `reviewed_count` tells you the sample
size (0 until the owner has reviewed at least one event — render the whole
block as an empty/placeholder state in that case, not 0% accuracy, which
would be misleading); `window_broken` and `intrusion` are each `{tp, fp, tn,
fn, accuracy}` scoring Gemini's own judgment against what the owner
confirmed actually happened.

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

## Manual review — the dashboard's first write action (new, 2026-07-15)

Everything above this point is read-only. This is different: `PATCH
/api/events/{event_id}` lets the owner confirm what actually happened for a
given event — was the window really broken, and was an intruder/person
actually present — closing the loop so `review_accuracy` (above) can be a
real accuracy measurement instead of another operational breakdown.

### `PATCH /api/events/{event_id}`

Request body — both fields required, submitted together:
```json
{"window_broken_confirmed": true, "intrusion_confirmed": false}
```

Response `200` — the updated event, same shape as `GET
/api/events/{event_id}` plus three new fields: `window_broken_confirmed`,
`intrusion_confirmed` (both `null` until reviewed), `reviewed_at` (epoch
seconds, server-assigned, `null` until reviewed). `404` if `event_id`
doesn't exist. Same Basic Auth as everything else — the already-authenticated
page's `fetch()` call just works, same as your `GET /api/metrics` calls.

A second `PATCH` of the same `event_id` **overwrites** the previous review —
this is the intended way for the owner to correct an earlier mistake, not an
error case to guard against in the UI.

### What to build

- **On the event detail page**, for events that have a real Gemini verdict
  (`vision_source == "gemini"` — don't offer review controls for
  `failsafe`/`none` events, since there's no real prediction to score
  against): two confirm controls, e.g. "Was the window really broken?" /
  "Was an intruder or person actually present?" — each a yes/no toggle or a
  pair of buttons, not a free-text field. Submitting both calls the `PATCH`
  endpoint.
- **Show the comparison, not just the input.** Once reviewed, display
  Gemini's original call (`window_broken`/`is_intrusion`) next to the
  owner's confirmed value — a simple "Gemini said X, you confirmed Y"
  layout, with a visual match/mismatch indicator (e.g. a checkmark when they
  agree, a distinct color/icon when they don't). This is the detail-page
  payoff of the whole feature — make disagreement obvious at a glance.
- **Allow re-review.** If `reviewed_at` is already set, still show the
  controls (pre-filled with the existing confirmed values), not a locked
  "already reviewed" state — the owner correcting themselves is normal.
- **Render `review_accuracy` as a real confusion matrix** — this is the one
  place on the dashboard where that framing is accurate (contrast with
  `vision_source_breakdown`/`trigger_breakdown`, which must keep the
  "operational breakdown, not accuracy" framing already established above).
  A 2×2 grid (or two side-by-side grids, one per judgment) showing TP/FP/
  TN/FN plus the accuracy percentage works well; show `reviewed_count`
  prominently next to it so a small sample doesn't read as a confident
  score. When `reviewed_count == 0`, show an empty/placeholder state with a
  short explanation ("Review an event to start tracking accuracy") rather
  than a bare 0%.

## Suggested chart set (not a rigid spec — use judgment)

1. **Trend over time** — stacked or grouped chart from `daily`: real vs.
   false vs. non-escalating wakes per day.
2. **Summary cards** — the five `summary` values (now including
   `gemini_call_success_rate`), styled as premium KPI cards rather than
   today's plain bordered boxes.
3. **Vision-source breakdown** — a heatmap/grid or grouped bar chart from
   `vision_source_breakdown`, labeled per the framing note above.
4. **Trigger breakdown** *(new)* — from `trigger_breakdown`. This can be
   small/secondary (e.g. a compact stat: "212/212 wakes were genuine SNN
   triggers") rather than a full chart, since `not_triggered` is expected
   to always be zero — its whole purpose is to make a nonzero value stand
   out immediately if it ever happens, not to visualize a normal
   distribution.
5. **Email delivery rate** — could be its own small donut/gauge, or folded
   into the summary cards.
6. **Latency** — `latency_s`'s avg/p50/p95/max, e.g. as a small stat block
   or a simple distribution indicator.
7. Event list stays, but restyle: better thumbnails, refined badges,
   premium table or card-grid layout — your call.
8. Event detail page: larger hero image, cleaner metadata layout.

## A framing note worth putting in the UI itself

The owner's first reaction to the "Non-escalating" count being much larger
than "Real"/"False" wakes was to wonder if something's wrong. It isn't —
this project's SNN hardware trigger is a deliberately cheap, sensitive
first-stage sensor (comparable to a smoke detector): it's *expected* to
latch on plenty of things that aren't real intrusions, and the local
prefilter + Gemini vision stages exist specifically to filter those out
before anything reaches the owner. A large non-escalating count is the
funnel working as intended. Consider a short subtitle/tooltip on that KPI
card or chart section along these lines, so the next person looking at the
dashboard doesn't have the same reaction — something like "Cleared by the
on-device prefilter before an expensive vision call was needed" rather than
leaving "Non-escalating" unexplained.

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
- [ ] `cloud/app/templates/event_detail.html` — restyled detail view,
      including the review controls and Gemini-vs-owner comparison (see
      "Manual review" above) for events with a real Gemini verdict.
- [ ] `cloud/app/static/` — CSS/JS (and logo asset if available) backing
      the above.
- [ ] Charts fetch from `GET /api/metrics` client-side (don't have the
      Jinja route pre-compute chart data — that defeats the point of the
      new endpoint).
- [ ] Review submission calls `PATCH /api/events/{event_id}` client-side and
      updates the page (either an optimistic UI update or a refetch) without
      a full reload.
- [ ] `review_accuracy` rendered as a real confusion matrix (see "Manual
      review" above), with an explicit empty state when `reviewed_count == 0`.
- [ ] No changes to any `.py` file, `cloud/infra/`, or `docs/`.
- [ ] Confirmed working locally against Azurite per the steps above.
