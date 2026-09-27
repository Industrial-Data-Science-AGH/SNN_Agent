# Telemetria runtime: jak czytać `NeuronFrame` (zadanie P3)

Ten dokument jest przekazaniem do dashboardu (T03) i do backendu (T01). Opisuje,
co runtime wysyła, czego **nie** wolno z tego wywnioskować i jak narysować z tego
płytkę, raster i wykres potencjału.

Kontrakt: `contracts/v1/NeuronFrame.schema.json`. Przykład: `contracts/fixtures/neuron-frame.json`.
Producent: `LuiRuntime.snapshot()` (`snn_runtime/runtime.py`), budowa klatki:
`snn_runtime/telemetry.py`.

## 1. Skąd bierze się klatka

`snapshot()` czyta stan, który już powstał, i niczego nie liczy od nowa. Wynika
z tego jedna rzecz ważna dla UI: **czytanie telemetrii nie zmienia decyzji ani
nie przesuwa symulacji**. „Pause view" po stronie przeglądarki oznacza tylko, że
nikt nie prosi o klatki; sesja i strumień lecą dalej.

Klatka opisuje **ostatnią przetworzoną ramkę** paczki (`dt_us`, u nas 10 ms), a
`source_time_us` to czas źródłowy końca tej paczki, nie czas ściany.

## 2. Mapowanie na widok sieci

| Pole klatki | Co z nim zrobić | Czego nie robić |
|---|---|---|
| `neurons[].neuron_id` | klucz płytki w edytorze; kolejność jest kolejnością z manifestu | nie zakładać, że to indeks tablicy |
| `neurons[].v_mem` | jasność LED potencjału: `(v_mem − v_reset) / (v_threshold − v_reset)`, **obciąć do `0..1`** — hamowanie potrafi zepchnąć potencjał daleko poniżej `v_reset` (w naszej sieci bywa −80) | nie skalować do maksimum z ostatnich klatek — LED przestanie znaczyć to samo między sesjami |
| `neurons[].v_mem == null` | „brak odczytu" (pakiet bez fizyki) | **nie** rysować jako 0 |
| `neurons[].spiked` | błysk płytki + kropka w rasterze na `source_time_us` | nie wyprowadzać spike'a z `v_mem ≥ v_threshold` — to jest ta sama informacja policzona gorzej |
| `neurons[].v_threshold`, `v_reset` | linie odniesienia na wykresie `Vmem` | nie trzymać globalnie jednej wartości: każda płytka ma własną |
| `potential_unit` | podpis osi; `a.u.` znaczy „jednostki umowne" | nie pisać „V", dopóki nie przyjdzie `V` |
| `provenance` | badge: `demo` / `simulated` / `measured` | `measured` pojawia się dopiero po kalibracji płytek (A2) — do tego czasu UI nie ma prawa twierdzić, że to pomiar |
| `status` | `warmup` / `running` / `gap` / `stopped` (patrz niżej) | nie renderować `gap` jako ciszy |
| `frame_seq` | numer **wysłanej** klatki; rosnący, ciągły w obrębie sesji | nie traktować jako numeru ramki symulacji |
| `epoch`, `session_id`, `device_id` | odrzucić klatkę z innej sesji/epoki zamiast dokleić ją do wykresu | — |
| `model_hash`, `topology_version` | pokazać w panelu modelu; zmiana = nowa sesja | — |

## 3. Statusy

- `warmup` — stan spoczynku jest założeniem, nie obserwacją: sesja dopiero
  wystartowała albo właśnie przepadł kawałek wejścia i linie opóźniające nie
  niosą jeszcze prawdziwej historii. Decyzje w tym czasie są wstrzymane.
- `running` — normalna praca.
- `gap` — **w wejściu była dziura**. Urządzenie wyprodukowało ramki, które nigdy
  nie dotarły. Pierwsza klatka po dziurze niesie ten status dokładnie raz, więc
  UI musi ją pokazać (przerwa w rasterze, znacznik na osi), a nie przemilczeć.
- `stopped` — sesja zamknięta; stan jest zamrożony i dalej czytelny.

## 4. Przerzedzanie strumienia

Runtime chodzi na 100 ramkach na sekundę. `TelemetryFeed` z `snn_runtime/telemetry.py`
przerzedza strumień **po czasie źródłowym**, nie po zegarze ściany, więc ten sam
materiał odtworzony szybciej daje te same klatki. Dwie reguły są nienaruszalne:

1. klatka ze zmienionym `status` przechodzi zawsze (dziura nie ginie między
   dwoma próbkowaniami),
2. `frame_seq` liczy klatki wysłane, a `source_time_us` mówi, z którego momentu
   pochodzą — po tych dwóch polach widać, że strumień jest przerzedzony.

```python
feed = TelemetryFeed(min_interval_us=100_000)   # 10 klatek na sekundę do UI
frame = feed.offer(runtime.snapshot())          # None = nie wysyłaj
```

## 5. Przykłady: 8 i 50 neuronów

Klatka dla 8 płytek (nasza sieć 7→4→3→1) ma 8 wierszy w `neurons`, po jednym na
`H0..H3, G0..G2, D`. Klatka dla 50 płytek ma 50 wierszy i **nic więcej się nie
zmienia**: ani rozmiar pola, ani semantyka, ani częstotliwość. Limit 50 jest
twardy w kontrakcie (`maxItems`), więc UI nie musi się bronić przed 51.

Golden replay do podmiany danych demo:

```sh
.venv-w0/bin/python -m snn_runtime.tools.make_demo_frames \
    --manifest tests/runtime/fixtures/lui8-v2-manifest.json \
    --frames 200 --every-us 50000 --out neuron-frames.json
```

Plik ma kształt `{"schema_version": "1.0", "frames": [NeuronFrame, ...]}`.
Każda klatka przechodzi `validate("NeuronFrame", frame, manifest=...)`.

## 6. Szkic topologii z edytora

Edytor sieci produkuje **draft**, nie model. Kontrakt:
`contracts/v1/TopologyDraft.schema.json`, przykład `contracts/fixtures/topology-draft.json`,
przegląd: `snn_runtime.topology.review_draft`.

Draft niesie `boards` (0–50) i `connections` z `target_port ∈ {1,2,3}`, bo płytka
Lu.i ma trzy fizyczne wejścia synaptyczne i **jedno wejście przyjmuje jeden
przewód**. Rysunek z czterema przewodami do jednej płytki albo z dwoma do tego
samego portu jest odrzucany w walidacji, a nie dopiero przy lutowaniu.

Draft nigdy nie jest uruchamialny: nie ma wag, stałych czasowych, dekodera,
profilu enkodera ani checkpointu. `review_draft` zwraca `runnable=False` i mówi,
czego brakuje; `LuiRuntime.load(draft)` odmawia z kodem `DRAFT_NOT_A_MODEL`.
Zmiana liczby płytek w edytorze **nie przebudowuje championa** — nowa sesja
wymaga nowego, ważnego `ModelManifest`.

## 7. Dla backendu (T01)

`snapshot()` zwraca komplet pól, łącznie z `device_id` i `session_id`. Runtime
poznaje je z pierwszej paczki albo z `reset(..., device_id=..., session_id=...)`;
paczka z innej sesji kończy się błędem `SESSION_MISMATCH`, a snapshot bez
tożsamości błędem `NO_STREAM_IDENTITY`. Trasa SSE (`GET /v1/sessions/{id}/telemetry`,
zdarzenie `snapshot`) należy do backendu — runtime nie ma własnego API.
