# SNN Lab — Dashboard (T03)

Angielski dashboard operatora dla detektora stłuczenia szkła na analogowych
neuronach **Lu.i**. Renderowany po stronie klienta (bez frameworka), serwowany
przez FastAPI; integruje się z backendem Wiktora po HTTP/OpenAPI.

- **Wersja UI:** `0.9.0` (widoczna w nagłówku)
- **Baza SHA:** `d5c817be` — bieżące: `git rev-parse --short HEAD`
- **Dane wejściowe/wyjściowe i przekazania:** [DASHBOARD_DATA.md](DASHBOARD_DATA.md)

---

## Co to jest / co potrafi

Pięć sekcji (wszystkie etykiety produktu po **angielsku**):

| Sekcja | Zawartość |
|---|---|
| **Network** | edytor topologii płytek Lu.i (0–50), zoom/pan/Fit, drag, wybór neuronu; **na żywo**: LED potencjału (`v_mem`) i osobny błysk spike (`spiked`); inspektor neuronu (Signals/Parameters/Connections/Notes) z wykresem Vmem (próg, jednostki, kalibracja); raster impulsów; Live/Replay/Edit/Pause/Start |
| **Events** | lista + detale zdarzenia: zdjęcie, timeline SNN/capture/vision/alarm, osobno glass/person/authorization, decyzja SNN, ACK i error state; brak danych → `Not available` |
| **Experiments** | metryki per run: split, godziny tła, seed, model_hash, encoder_hash, FA/h z 95% CI i recall; **osobno** metryki SNN i całego systemu; filtr pokazuje jeden run (nie miesza modeli/datasetów) |
| **Energy** | źródła measured/estimated z granicą pomiaru, moc W i energia J/Wh; osobno, nigdy sumowane; brak pomiaru → `Not available` (nie zero) |
| **Device** | heartbeat, stan, serial (frames/gaps/stalls/reconnects), kamera, bufor/outbox |

Tryby danych:
- **DEMO** (`Explore demo`) — działa bez sprzętu/backendu; dane z fixtures; oznaczone `DEMO · SAMPLE DATA`.
- **LIVE** (po zalogowaniu) — dane z API Wiktora; brak endpointu → `Not available`.

---

## Jak uruchomić

### 1. Dev harness (samodzielny, do pracy nad UI)

Z korzenia repo, w środowisku W0:

```sh
py -3.13 -m venv .venv-w0                                        # jednorazowo
./.venv-w0/Scripts/python.exe -m pip install --require-hashes -r requirements-w0.lock
./.venv-w0/Scripts/python.exe -m rpi_agents.cloud.app.routes_dashboard
```

→ **http://127.0.0.1:8080**. Logowanie demo: `operator` / `demo`, albo „Explore demo".
Harness ma stub `/auth/*` (mówi tym samym kontraktem co produkcja) i serwuje fixtures —
**nie jest to produkcja**.

> Uwaga: maszyna deweloperska ma Python 3.13 (nie 3.12), lock instaluje się na 3.13 poprawnie.

### 2. Mock API W0 + Swagger (kontrakty i fixtures)

```sh
./.venv-w0/Scripts/python.exe -m rpi_agents.cloud.app.mock_api        # http://127.0.0.1:8000
```

- **Swagger UI:** http://127.0.0.1:8000/docs — pełna specyfikacja endpointów `/v1/*` i fixtures.
- Fixtures: `GET /demo/fixtures/{name}` (np. `neuron-frame`, `model-manifest`, `device-status`).

### 3. Produkcja (backend Wiktora)

Router dashboardu (`routes_dashboard.py:router`) montuje się w aplikacji
`rpi_agents.cloud.app.api`, która dostarcza realne `/auth/*` i `/v1/*`.
UI nie wymaga zmian przy przejściu demo→live (patrz warstwa `data.js`/`runtime.js`).

---

## Struktura plików

```
rpi_agents/cloud/app/
  routes_dashboard.py        # router (shell + /dashboard/fixtures) + dev harness
  templates/index.html       # shell (login + 5 zakładek), renderowany po stronie klienta
  static/
    css/dashboard.css        # ciemny motyw SNN Lab (v2/v3)
    img/neuron.svg           # płytka Lu.i (warstwy: porty, LED potencjału, spike, zaznaczenie)
    js/
      app.js                 # bootstrap: login, tryb demo, nawigacja, sterowanie runtime
      data.js                # warstwa danych: DemoSource / LiveSource (jeden interfejs)
      auth.js                # login/logout/session; CSRF w pamięci; hasło nietrzymane
      ui.js                  # helpery DOM + stany loading/empty/error
      network.js             # edytor sieci (SVG): zoom/pan/Fit, drag, LED z klatek runtime
      runtime.js             # klatki: DemoRuntime (golden) / LiveRuntime (SSE), replay/stale/reconnect
      inspector.js           # inspektor neuronu + wykres Vmem
      raster.js              # raster impulsów
      c4.js                  # panele Events / Experiments / Energy
    demo/                    # dane DEMO (do zastąpienia realnymi — patrz DASHBOARD_DATA.md)
      neuron-frames.json     # golden replay (8 neuronów, 101 klatek, deterministyczny LIF)
      events.json, experiments.json, energy.json
tests/dashboard/test_dashboard.py   # testy (shell, fixtures, auth, warstwy, C4)
```

---

## Testy

```sh
./.venv-w0/Scripts/python.exe -m pytest tests/dashboard -q
```

Uwaga o CI: `pr-gate` lintuje `rpi_agents` (ruff) i uruchamia `tests/w0` — `tests/dashboard`
**nie jest** w bramce CI (odpalane lokalnie). Testy pokrywają: sekcje po angielsku, brak
polskich etykiet, fixtures demo + 404, kontrakt logowania (invalid/session/logout),
warstwy C2/C3 i dane C4 (rozdzielenie metryk, brak≠zero).

---

## Wydajność i dostępność (C5)

- **Ograniczony bufor:** LiveRuntime trzyma maks. 2000 klatek (starsze odrzucane).
- **Ograniczone punkty wykresu:** wykres Vmem ≤ 400 punktów (downsampling, spike'i zachowane);
  raster próbkuje klatki powyżej 1500 (kursor/tiki pozostają czytelne).
- **50 płytek:** sprawdzone; Fit dopasowuje widok; etykiety się nie nakładają.
- **Klawiatura:** nawigacja i formularz logowania dostępne z klawiatury; aktywna zakładka ma
  `aria-current`; focus widoczny.
- **Wąski ekran:** górny pasek i toolbar zawijają się; nawigacja staje się poziomym paskiem.
- **Reduced motion:** animacje wyłączane przy `prefers-reduced-motion`.

---

## Bezpieczeństwo

- Hasło **nie** trafia do JS/localStorage; sesja to httponly cookie ustawiane przez serwer.
- CSRF token trzymany tylko w pamięci; wysyłany w `X-CSRF-Token` przy logout.
- **Dane blokowane po logout niezależnie od UI:** realny backend wymaga sesji operatora na
  każdym `/v1/*` (401 bez cookie) — ukrycie UI to nie jedyna bariera.
- Import JSON topologii: `JSON.parse` + walidacja, **żadnego** wykonywania kodu; import
  wpływa tylko na `draft` — **nie nadpisuje** załadowanego/uruchomionego modelu (championa).

---

## Odbiór C5 — scenariusze (checklist)

- [x] login → logout → ponowne wejście
- [x] jedna sesja live + replay (seek odtwarza spójny stan)
- [x] błąd vision (evt-002: `Error state: VISION_TIMEOUT`, ACK failed)
- [x] utrata łącza → `Stale data` → reconnect/restore
- [x] puste dane → `Not available` (nie zero)
- [x] próba zmiany aktywnego modelu przez import → tylko draft, champion nietknięty
- [x] desktop + wąski ekran
- [x] nawigacja klawiaturą
- [x] 50 płytek
- [x] brak polskich etykiet produktu, brak wiszących spinnerów, brak wymyślonych danych w live

Zrzuty referencyjne (before/after) dołączane do PR: jawne demo oraz jedna oznaczona sesja.

---

## Przekazanie (Wiktor)

Gotowe zasoby do tego samego kontenera co backend:
- montaż: `from rpi_agents.cloud.app.routes_dashboard import router` → `app.include_router(router)`
- statyki: `app.mount("/static", StaticFiles(directory=.../static))`
- konsumowane endpointy i kontrakt formularza logowania: [DASHBOARD_DATA.md](DASHBOARD_DATA.md)

Dokumentacja niesie wersję UI (`0.9.0`) i SHA bazy; przy wydaniu zaktualizować SHA
(`git rev-parse --short HEAD`).
