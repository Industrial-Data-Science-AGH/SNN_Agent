# Product

## Register

product

## Users

Single owner (fixed credential, HTTP Basic Auth) checking event history from
a phone or laptop, often while the Raspberry Pi 5 agent itself is fully
powered off. The workflow is quick, periodic status-checking — "did anything
happen while I was away, and was it real?" — not a multi-user admin tool.

## Product Purpose

`wakeup-ai-cloud-dashboard` is the cloud-hosted companion to a Raspberry Pi 5
security-camera agent ("Wake-Up AI"). The Pi wakes on motion, captures a
frame, runs a prefilter + Gemini vision call, decides real/false alarm, and
pushes a record (metadata + snapshot photo) to this dashboard. Success looks
like: the owner can glance at the dashboard and immediately understand recent
activity (real vs. false wakes, trends over time, delivery/latency health of
the pipeline itself) without digging through raw logs.

## Brand Personality

Premium industrial/technical instrumentation panel — sophisticated dark
graphite, not neon or sci-fi, not a flat "#111 dark mode with buttons."
Brand accents (deep emerald green through near-black to deep maroon/oxblood,
echoing the Industrial Data Science gear logo) are used deliberately and
sparingly for chart series, active states, badges, and focus rings — not as
large flat color blocks.

## Anti-references

- Flat `#111` fill with plain bordered cards (today's current state — the
  thing being replaced).
- Neon / cyberpunk / sci-fi dark mode.
- Generic SaaS hero-metric dashboard cliché (oversized number + small label
  + gradient-text accent).
- Overuse of the brand green/maroon as large solid fills rather than sparing
  accents.

## Design Principles

- Lowest sustainable cost/complexity: no JS build step, no framework, no
  package manager for the frontend — CDN-loaded charting library plus
  vanilla JS/CSS only, matching the project's minimal-dependency stance.
- Instrumentation-panel restraint: dark graphite base with a soft directional
  gradient, brand color reserved for meaningful signal (series, states,
  badges), never decoration for its own sake.
- Charts are load-bearing, not decorative: every chart must trace back to a
  real field in `GET /api/metrics` and be labeled honestly (e.g. the
  vision-source breakdown is a decision breakdown, not an accuracy/confusion
  matrix — this system has no ground truth).
- Glanceable first, detailed second: the owner should get the "is everything
  okay" answer at a glance, with drill-down (event detail, larger charts)
  available but not required.

## Accessibility & Inclusion

WCAG AA contrast ratios throughout (including muted/secondary text and
placeholder states). Respect `prefers-reduced-motion` for all animation. No
additional stated requirements beyond these defaults.
