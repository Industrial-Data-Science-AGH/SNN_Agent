# W0 — wspólny punkt startu zespołu

W0 dostarcza kontrakty v1.0, fixtures, lokalne mock API, lekkie granice adapterów
oraz root CI `pr-gate`. Kod inferencji/treningu i adaptery sprzętowe są dalszymi
zadaniami. **Nie trzeba czekać na dostęp do Azure ani sprzętu, żeby rozpocząć UI
lub implementację przeciwko kontraktom.**

## Pobranie aktualnej bazy

Najpierw zachowaj własną niezacommitowaną pracę. Nie wykonuj reset --hard.
Po scaleniu W0:

```sh
git fetch origin
git switch -c feat/twoj-zakres origin/master
```

`master` pozostaje bazą integracyjną. Każda zmiana przez PR + zielony `pr-gate`
+ squash. Wymóg akceptacji innej osoby został tymczasowo wyłączony na wyraźne
polecenie Wiktora; PR-only, zakaz force push/usunięcia pozostają. Starsze DOCX
opisują docelowe review; ten komunikat opisuje aktualny wyjątek. Nie omijać CI.
Po squash następny przyrost zaczynaj z aktualnego origin/master.

| Osoba | Branch / pierwszy krok |
|---|---|
| Wiktor | feat/wiktor-edge-cloud: W1/W2, adaptery z rpi_agents/agent/ports.py; produkcyjne API oddzielnie od mock_api.py |
| Patryk | feat/patryk-lui-runtime: P1, implementacja SNNRuntime z rpi_agents/runtime/ports.py i walidacja ModelManifest |
| Karolina | feat/karolina-dashboard: C1, nowe UI na mock API, schematy JSON i fixtures; napisy po angielsku |
| Marcel | istniejący feat/master-pipeline: aktualizacja z master, M0/M2; eksporter zgodny z ModelManifest |
| Kacper | istniejące feat/encoder-features i feat/continuous-dataset: aktualizacja z master, K1 mapowanie kanałów, K2 parity i serial |
| Andrzej | feat/andrzej-hardware-energy: A1/A2, pomiary i jawne jednostki; bez wymyślonych wartości kalibracji |

Nowe branche osób nie są tworzone automatycznie. W istniejącym feature branchu
po zabezpieczeniu lokalnej pracy można `git merge origin/master`, rozwiązać
konflikty i uruchomić testy. To aktualizacja feature, nie direct merge do master.
Nie przepisywać opublikowanej historii innym osobom. Otwarte PR #47/#48/#50
właściciele przekierowują na master dopiero po sprawdzeniu różnicy i konfliktów.

## Uruchomienie i odbiór

[Instrukcja kontraktów i mock API](../../contracts/README.md) zawiera dokładne
polecenia, formaty i ograniczenia. Z korzenia repo:

```sh
python3.12 -m venv .venv-w0
.venv-w0/bin/python -m pip install --require-hashes -r requirements-w0.lock
.venv-w0/bin/python -m ruff check --config ruff-w0.toml contracts rpi_agents tests/w0
.venv-w0/bin/python -m pytest tests/w0 -q
.venv-w0/bin/python -m rpi_agents.cloud.app.mock_api
```

W drugim terminalu: `.venv-w0/bin/python -m contracts.demo`.
Swagger: http://127.0.0.1:8000/docs. Frontend: localhost:5173 albo :3000.
Karolina może również pobrać GET /demo/fixtures/neuron-frame i model-manifest,
a następnie POST /v1/sessions oraz odtwarzać fixtures w porządku podanym w demo.

Kryteria: prawidłowe fixtures przechodzą, nieznany kanał/wersja są odrzucane,
retry nie duplikuje zdarzenia, luka nie jest ciszą, niedostępne vision nie
włącza alarmu. Testy nie wymagają urządzeń, torch, datasetu, kluczy ani Azure.
CI wykonuje właśnie te testy; nie potwierdza gotowości produkcyjnej całości.

## Granice plików i konfliktów

- `contracts/v1`, walidator i wspólne fixtures: zmiany skoordynowane przez Wiktora
  z producentem/konsumentem. Nigdy równoległe, niezgodne kopie schematu.
- `rpi_agents/agent`: Wiktor; Kacper dostarcza protokół i enkoder w swoim katalogu.
- `rpi_agents/runtime`: Patryk. MockStore jest skryptem testowym, nie jego modelem.
- `rpi_agents/cloud/app`: Wiktor. Karolina integruje się po HTTP/OpenAPI.
- `rpi_agents/dashboard`: Karolina może utworzyć tutaj nowy frontend.
- trening/master_pipeline i encoder/dataset: dotychczasowi właściciele.
- `.github/workflows/pr.yml`: Wiktor; nazwa checka `pr-gate` jest stabilna.

Kontrakty są zamrożone jako baza implementacji W0, a nie deklaracja, że Patryk
lub Kacper je już osobiście zweryfikowali. Wybrany fizyczny encoder_config,
parametry Lu.i i reguła automatycznego alarmu nadal mają odbiór w K1/P1/A2/W3.
Zmiana tych wartości nie wymaga przerabiania transportu, jeśli profil/manifest
zostanie zgodnie z kontraktem zwersjonowany i otrzyma nowy hash.

## Selektywny import starego agenta

[Lista pochodzenia i odłożonych komponentów](../../rpi_agents/MIGRATION.md).
Nie skopiowano .env, danych, modeli, starej polityki fail-open, halt ani starego
pipeline deploymentu. Mock nigdy nie uruchamia gpio.py.

## Trening na M5 Max 128 GB

W0 nie zmienia obecnego GA ani jego środowiska. M0 mierzy CPU/MPS i workers na
realnej pracy, M2 naprawia selekcję/export, a dopiero potem profil trafia do
manifestu eksperymentu. Lock W0 jest tylko do integracji; nie jest lockiem torch.
