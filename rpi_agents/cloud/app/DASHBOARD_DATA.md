# Dashboard — dane wejściowe/wyjściowe (in-out)

Zadanie **T03 (Karolina)** — angielski dashboard SNN Lab z płytkami Lu.i i pomiarami.
Ten dokument opisuje **skąd bierze się każda dana** i **co przekazujemy komu**.

Dashboard żyje w `rpi_agents/cloud/app/`:
`routes_dashboard.py`, `templates/index.html`, `static/{css,js,img,demo}`, `tests/dashboard/`.

---

## Zasada: dwa źródła, jeden interfejs

Warstwa danych (`static/js/data.js`, `runtime.js`) ma **dwie implementacje za jednym interfejsem**:

- **DEMO** — działa bez sprzętu, backendu i logowania („Explore demo"). Dane z fixtures.
  Wszystko oznaczone badge **`DEMO · SAMPLE DATA`** / `Demo data`.
- **LIVE** — po zalogowaniu; dane z API Wiktora po HTTP/OpenAPI. Brak endpointu → `Not available`.

Przełączenie demo→live **nie wymaga przepisywania UI** — podmienia się tylko źródło.

> **Ważne:** liczby w plikach `static/demo/*` (poza kontraktowymi fixtures) to **wartości przykładowe, które wymyśliłam na potrzeby UI — NIE są to realne pomiary.** Realne wartości dostarczają właściciele metryk (Marcel/Andrzej) i backend (Wiktor).

---

## IN — dane wejściowe (skąd co pochodzi)

| Panel / element | DEMO — źródło | LIVE — endpoint (API Wiktora) | Kontrakt | Właściciel danych |
|---|---|---|---|---|
| **Device** (heartbeat, serial, gap, kamera, bufor/outbox) | `contracts/fixtures/device-status.json` przez `/dashboard/fixtures/device-status` | `GET /v1/devices/{id}/status` | `DeviceStatus` | W0 / Wiktor |
| **Network — model** (hash, topology) | `contracts/fixtures/model-manifest.json` | *(gdy sesja/model załadowany)* | `ModelManifest` | W0 / Patryk / Marcel |
| **Network — LED potencjału, spike, raster, wykres Vmem** | `static/demo/neuron-frames.json` *(wygenerowany deterministycznie, model LIF)* | SSE `GET /v1/sessions/{id}/telemetry` | `NeuronFrame` | **placeholder** / Patryk (P3) |
| **Events** (lista + detale: zdjęcie, timeline, glass/person/authorization, ACK, error) | `static/demo/events.json` *(napisane ręcznie, zgodne z kontraktem)* | `GET /v1/events`, `GET /v1/events/{id}`, zdjęcie `GET /v1/events/{id}/images/0` | `Event`, `VisionResult` | **placeholder** / Wiktor |
| **Experiments** (split, background h, seed, model_hash, encoder_hash, FA/h+CI, recall) | `static/demo/experiments.json` *(liczby wymyślone)* | *(brak endpointu w v1)* → `Not available` | *(brak w v1)* | **placeholder** / Marcel |
| **Energy** (source measured/estimated, boundary, W, J/Wh) | `static/demo/energy.json` *(liczby wymyślone)* | *(brak endpointu w v1)* → `Not available` | *(brak w v1)* | **placeholder** / Andrzej |
| **Login** (Username/Password/Sign in/Logout) | demo stub w `routes_dashboard.py` (`operator`/`demo`) | `POST /auth/login`, `POST /auth/logout`, `GET /auth/session` | — | Wiktor (`app/auth.py`, `app/api.py`) |

### Pliki źródłowe danych
- **Kontraktowe (wspólne, nie moje):** `contracts/fixtures/*.json` — serwowane read-only przez
  `GET /dashboard/fixtures/{name}` (jedno źródło prawdy, oznaczone `demo: true`).
- **Demo dashboardu (moje, do zastąpienia):**
  - `static/demo/neuron-frames.json` — golden replay: 8 neuronów, 101 klatek (deterministyczny LIF). LED i raster czytają to samo → zgodność neuron/czas.
  - `static/demo/events.json` — 4 przykładowe zdarzenia (alarm/review/no_alarm/failed).
  - `static/demo/experiments.json` — 2 runy (różne modele/datasety) — **liczby przykładowe**.
  - `static/demo/energy.json` — 3 źródła (measured/estimated) — **liczby przykładowe**.

### Zasady dot. danych (odbiór taska)
- Dane przykładowe są **oznaczone** (`DEMO`, `Demo data`).
- Brak pomiaru → **`Not available`**, nigdy `0`.
- Filtr wyników pokazuje **jeden run** — nie miesza modeli/datasetów; energia measured vs estimated
  i różne granice pomiaru są **osobno**, nie sumowane.
- Hasło **nie jest** trzymane w JS/localStorage; sesja to httponly cookie; CSRF tylko w pamięci.

---

## OUT — co przekazujemy komu

| Odbiorca | Co dostaje | Gdzie w kodzie |
|---|---|---|
| **Wiktor** (integracja) | Lista route/asset dashboardu; kontrakt formularza logowania; lista konsumowanych endpointów `/v1/*` i `/auth/*` | `routes_dashboard.py`, `data.js`, `auth.js`, `runtime.js` |
| **Patryk** (sygnały) | Payload draft topologii `editor.getDraft()` = `{boards:[{id,label,x,y}], connections:[{id,source,target,kind}]}`; mapowanie sygnałów: `neuron_id`→płytka, `v_mem`→jasność LED, `spiked`→błysk + tick rastera | `network.js`, `runtime.js` |
| **Marcel** (metryki SNN) | Format metryk: `experiments.json.runs[].snn.{fa_per_h{value,ci_low,ci_high}, recall{...}}` + meta (split/seed/hash/background) | `static/demo/experiments.json`, `c4.js` |
| **Andrzej** (energia) | Format energii: `energy.json.sources[].{scope, source, boundary, power_w, energy_j, energy_wh, window_s}`; brak = `null` | `static/demo/energy.json`, `c4.js` |

### Route i assety dashboardu (dla Wiktora)
- `GET /` — shell (HTML, renderowany po stronie klienta)
- `GET /dashboard/fixtures` — lista nazw fixtures
- `GET /dashboard/fixtures/{name}` — jeden fixture kontraktu (oznaczony demo)
- `GET /static/**` — css/js/img/demo
- Konsumowane (LIVE): `POST /auth/login`, `POST /auth/logout`, `GET /auth/session`,
  `GET /v1/devices/{id}/status`, `GET /v1/events`, `GET /v1/events/{id}`,
  `GET /v1/events/{id}/images/0`, SSE `GET /v1/sessions/{id}/telemetry`.

### Kontrakt formularza logowania (dla Wiktora)
- `POST /auth/login` body `{username, password}` → cookie sesji (httponly) + `{csrf_token, expires_at, actor}`;
  błędy `INVALID_CREDENTIALS` (401), `TOO_MANY_ATTEMPTS` (429, `Retry-After`).
- `POST /auth/logout` — cookie + nagłówek `X-CSRF-Token` → 204.
- `Pause view` **nie** woła `stop` sesji.

---

## Uruchomienie

```sh
# z korzenia repo (venv W0)
./.venv-w0/Scripts/python.exe -m rpi_agents.cloud.app.routes_dashboard   # http://127.0.0.1:8080
./.venv-w0/Scripts/python.exe -m pytest tests/dashboard -q
```

Demo: „Explore demo" lub logowanie `operator` / `demo` (stub deweloperski, nie produkcja).

---

## TODO — dane do zastąpienia realnymi

- [ ] `static/demo/experiments.json` — realne metryki od **Marcela** (endpoint + wartości).
- [ ] `static/demo/energy.json` — realne pomiary od **Andrzeja** (endpoint + wartości).
- [ ] `static/demo/neuron-frames.json` — realny stream runtime od **Patryka** (P3, SSE).
- [ ] `static/demo/events.json` — realne zdarzenia z backendu **Wiktora** (`/v1/events`).
