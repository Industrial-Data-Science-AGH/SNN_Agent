# Wake-Up AI Dashboard UI

The directory contains the frontend shell for the Wake-Up AI dashboard. It is
currently a Vite + React + TypeScript starter app; it is not yet connected to
the SNN pipeline, FastAPI backend, host agent, or live hardware trigger events.

## Setup

```bash
cd software/web_dashboard/ui
npm ci
```

## Development

```bash
npm run dev
```

The Vite dev server normally runs on `http://localhost:5173`.

## Build And Lint

```bash
npm run build
npm run lint
```

`npm run build` runs TypeScript first and then Vite. If it reports missing
`vite/client` or `node` type definitions, dependencies have not been installed
with `npm ci`.

## Current State

- `src/App.tsx` is still the default Vite counter/logo page.
- The backend API currently exposes only `GET /health`.
- There is no frontend API client yet.
- There is no visualization of SNN spikes, exported weights, HIL results, or
  hardware-trigger telemetry yet.

## Expected Integration Direction

The dashboard should eventually consume:

- SNN pipeline outputs from `output/weights.json`, `weights.csv`, and generated
  reports.
- Hardware trigger events from the Raspberry Pi / host bridge.
- Backend health/status APIs from `software/web_dashboard/api`.
- Agent/RAG responses from `software/host_agent` once that layer is real.
