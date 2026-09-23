# SNN wire contracts v1.0

**Źródło prawdy dla W0:** schematy `v1/*.schema.json` i reguły semantyczne
`validation.py`. Dokument architektury zawierał przykłady; tutaj format jest
wykonywalny i testowany. Wersja 1.0 jest wspólnym punktem startu zespołu.
Zmiana znaczenia pola/kanału wymaga nowej wersji i skoordynowanego PR.

## Szybki start

Z korzenia repozytorium, w nowym venv Python 3.12:

```sh
python3.12 -m venv .venv-w0
.venv-w0/bin/python -m pip install --require-hashes -r requirements-w0.lock
.venv-w0/bin/python -m pytest tests/w0 -q
.venv-w0/bin/python -m rpi_agents.cloud.app.mock_api
```

Jeśli port 8000 jest zajęty, dodaj `--port 18765` do polecenia mocka i
`--base-url http://127.0.0.1:18765` do klienta demo. Host pozostaje loopback.

Drugi terminal:

```sh
.venv-w0/bin/python -m contracts.demo
```

Dokumentacja API: http://127.0.0.1:8000/docs ; OpenAPI:
http://127.0.0.1:8000/openapi.json. Nie instalować zależności W0 do środowiska
trwającego GA ani na Pi. Pi docelowo używa tylko lekkich adapterów W1.

## Typy i właściciele

| Schemat | Producent → konsument |
|---|---|
| EncoderProfile | Kacper → Wiktor, Marcel, Patryk |
| ModelManifest | Marcel → Patryk, backend, Karolina |
| SessionCreate / SessionState | Wiktor backend/edge → Patryk, Karolina |
| SpikeBatch | Wiktor z enkodera Kacpra → Patryk |
| SNNDecision | Patryk → backend Wiktora i Karolina |
| NeuronFrame | Patryk → Karolina |
| CaptureCommand / AlarmCommand | Backend Wiktora → edge Wiktora |
| CommandAck | Edge → backend |
| VisionResult | Worker Wiktora → polityka i Karolina |
| BatchAck / StreamGap | Backend → edge i obserwacja sesji |
| Event | Backend → Karolina |
| SessionControl / Error | Wspólne kontrolowanie sesji i błędy |

JSON Schema Draft 2020-12 jest niezależny od języka. Schematy są samodzielne,
bez zdalnych `$ref`. Python używa dodatkowo `validate(name, payload,
manifest=...)`. **Sam JSON Schema nie sprawdza** mapowania kanałów do konkretnego
modelu, porządku czasu, hashy i powiązań topologii. Inny język musi odtworzyć te
reguły oraz przejść pozytywne i negatywne testy zgodności. Wszystkie wartości
liczbowe muszą być skończone, a liczniki mieszczą się w bezpiecznych integerach JS.
Dodatkowe pola są odrzucane; 409 oznacza niezgodną wersję, 422 błędną strukturę.

## Kanały, profile i hashe

`channel_map` definiuje stabilny `channel`, indeks od zera oraz prawdziwe
znaczenie cechy `feature`. Indeksy muszą być unikalne i ciągłe. Nie wolno
wywnioskować nazwy cechy z pozycji, nazwy CSV ani liczby wejść modelu.

W repo `feat/encoder-features` wariant base ma kanały:
`peak, peak_cnt, cv, zcr, flux, hf_lo, hf_hi`. W swap pozycje 1 i 2 oznaczają
`hjorth_mobility` i `autocorr_lag1`. K1 wybierze rzeczywisty profil i potwierdzi
zgodność firmware/twin. **W0 nie wybiera zwycięskiego enkodera**. Fixture
`demo-base-v1` ma `provenance=demo`; częstotliwość i szerokość impulsu są danymi
demo, nie pomiarem ani nastawami zaakceptowanymi dla fizycznego Lu.i.

- `encoder_hash = sha256(JCS(EncoderProfile))`.
- `model_hash = sha256(JCS(ModelManifest))`; manifest nie zawiera własnego hasha.
- `JCS` to RFC 8785, UTF-8. Prefiks każdego hasha: `sha256:` i 64 małe znaki hex.
- Hash pliku (artefaktu/checkpointu/configu) jest SHA-256 **surowych bajtów**.
- `implementation_sha` wskazuje pełny commit implementacji enkodera;
  `config_sha256` obejmuje pełną jego konfigurację, w tym normalizację/progi.
- SHA nie uwierzytelnia klienta i nie zastępuje podpisu ani poświadczenia urządzenia.

Fixture `demo-model.txt` to jawny znacznik demo, bez wytrenowanych wag.
Produkcja musi zweryfikować rzeczywiste pliki manifestu przed load. Nie używać
pickle ani wykonywania kodu przy imporcie modeli. Champion wymaga provenance
treningu/danych/oceny i nie może zawierać profilu demo. Hashe i status opisują
artefakt; sam napis `evaluated_champion` nie dowodzi jakości eksperymentu.

## Czas i sesje

- `source_*_us`: mikrosekundy monotonicznego, rozszerzonego licznika źródła.
  Pi rozszerza rollover Uno i mapuje serial do tej osi. Reset to nowy `boot_id`
  i nowa sesja/epoka; nie udajemy ciągłości. CRC/format binarny Uno należy do K2.
- Batch obejmuje `[source_start_us, source_end_us)`; dodatnia długość do 1 s.
  `dt_us` to offset od startu. Zdarzenia są niemalejące; równoczesne są dozwolone.
- Pusty batch oznacza zaobserwowaną ciszę i przesuwa stan SNN. Nieotrzymany batch
  nie oznacza ciszy. `batch_seq` zaczyna się od 0 i rośnie o 1 w obrębie epoki.
- Luka zakresu czasu/sekwencji jest jawna. Nakładanie/stara sekwencja poza
  dokładnym retry daje 409. Clipping/utrata zdarzeń oznacza zdegradowaną porcję.
- `epoch` jest tokenem nadanym przez backend, nie liczbą samodzielnie zwiększaną
  przez urządzenie. W0 tworzy nową sesję z epoch=1; W2 wdraża trwałe fencing/lease.
- UTC (`issued_at`, `expires_at`, `completed_at`) to RFC3339 z `Z`. Zegary
  muszą być synchronizowane dla TTL. W1 musi stosować dodatkowy lokalny zegar
  monotoniczny i limit czasu alarmu; opóźnień między zegarami nie raportujemy bez
  informacji o synchronizacji.
- `received_seq`, `processed_seq`, `durable_seq` mają różne znaczenia.
  Mock zawsze zwraca `durable_seq=null`, bo nie zapisuje nic trwale.

## Idempotencja i błędy

Każdy POST JSON zawiera `request_id`. Nagłówek `Idempotency-Key` musi być tą samą
wartością, stabilną przy retry. Klucz jest związany z urządzeniem i operacją.
Identyczny retry zwraca identyczny wynik; inna treść daje 409. Zmiana scenariusza
mocka też jest zmianą treści. Retry starego ACK może zawierać pierwotną komendę:
odbiorca zawsze sprawdza TTL, tryb, epokę i własny dziennik wykonania.

413: body >64 KiB; 415: nie-JSON; 422: schemat/semantyka; 409: wersja,
konflikt/idempotencja/kolejność; 404: brak obiektu; 429: limit mocka.
Produkcja W2/W4 doda 401/403 i Retry-After/backoff. Backend nigdy nie interpretuje
błędu vision jako potwierdzenia włamania.

## Komendy, vision i UI

Komenda capture ma do 3 klatek po maksymalnie 1 MiB. Alarm trwa maksymalnie
30 s według kontraktu; lokalny limit może być mniejszy. TTL komendy do 30 s.
`accepted` nie znaczy `completed`; terminalne wyniki to completed/failed/expired.
W0 nie generuje alarmów. Fixture AlarmCommand służy wyłącznie walidacji.
Każdy rzeczywisty adapter musi odrzucać `mode=demo`; replay nie wydaje komend.

`VisionResult` rozdziela obserwacje szkła/osoby, jakość, błędy i pochodzenie.
`authorization=unknown`; polityka uprawnień pozostaje decyzją W3. Brak obrazu lub
usługi zwraca unknown i Review required. To nie jest rozpoznawanie domowników.

`NeuronFrame` jest pełnym snapshotem 0–50 neuronów. `v_mem=null` oznacza brak
wartości; zero jest rzeczywistą liczbą. `spiked` jest informacją demonstracyjną w
mocku; prawdziwy raster wymaga strumienia zdarzeń Patryka w P3. W0 zamraża
snapshot; format delty/raster jest rozszerzeniem wersjonowanym P3. `a.u.` i `V`
nie są wymienne. Topologia ma jawne porty 1–3 i jednostki wag. Diagram i viewport
nie nadają automatycznie sieci zgodności fizycznej ani kalibracji.

## Mock: zakres i ograniczenia

Mock nasłuchuje tylko na 127.0.0.1:8000. Sprawdza Host/Origin; CORS dopuszcza
localhost/127.0.0.1 na portach 3000, 5173 i 8000. Bez cookies i bez logowania.
To wyjątek **lokalnego środowiska demonstracyjnego**, nie publiczny portal W4.
Nie wdrażać go w Azure ani nie podpinać do sterowania prawdziwym urządzeniem.

Scenariusz wybiera się parametrem query `scenario` przy POST /v1/sessions:
`silence`, `spike`, `trigger`, `vision_unavailable`. Trigger jest skryptem demo,
nie klasyfikacją SNN. Przy trigger/unavailable dopiero porcja z impulsami bez
luki tworzy zdarzenie. Fixture `gap` opuszcza seq=2 i zakres 500000–750000 us.

| Endpoint | Zachowanie W0 |
|---|---|
| GET /healthz | demo=true |
| GET /demo/fixtures/{name} | tylko nazwy z fixtures/index.json |
| POST /v1/sessions | demo/replay, znany demo manifest; live odrzucone |
| GET /v1/sessions/{session_id} | aktualny stan |
| POST /v1/sessions/{session_id}/batches | walidacja, sekwencja, gap, idempotencja |
| POST /v1/sessions/{session_id}/stop | stop i wycofanie oczekujących komend |
| GET /v1/devices/{device_id}/commands | wyłącznie ważne demo capture |
| POST /v1/commands/{command_id}/ack | rejestruje wynik mocka, bez side effect |
| GET /v1/sessions/{session_id}/telemetry | SSE snapshot, retry:1000, następnie EOF |
| GET /v1/events?limit=20&offset=0 | stronicowana historia demo |
| GET /v1/events/{event_id} | szczegół wraz z vision unavailable |

SSE mocka wysyła jeden snapshot; klient może użyć natywnego EventSource, który
ponownie się połączy. Zawsze pełny snapshot; brak obietnicy odtwarzania delty
przez Last-Event-ID. P3/W2 wdrożą długie połączenie, raster i recovery.

NIEzaimplementowane w W0: upload JPEG, auth, baza, worker, prawdziwe vision,
trening, inferencja, metryki badawcze, sterowanie GPIO i deployment.
Planowane ścieżki image/experiments/auth z architektury nadal wymagają W2/W4;
nie zwracają fikcyjnego powodzenia. Mock ma limity 16 sesji, 256 batchy/sesję,
256 zdarzeń i 1024 idempotentnych mutacji; wyczerpanie daje 429, restart czyści RAM.
Osiągniętych metryk demo nie wolno używać w artykule.

## Źródła techniczne

- [JSON Schema 2020-12](https://json-schema.org/draft/2020-12): format może być
  adnotacją, dlatego walidator W0 jawnie sprawdza date-time oraz zależności pól.
- [RFC 8785](https://www.rfc-editor.org/rfc/rfc8785): wspólna kanonizacja JSON.
- [GitHub: required checks](https://docs.github.com/en/enterprise-cloud%40latest/pull-requests/how-tos/merge-and-close-pull-requests/troubleshooting-required-status-checks):
  wymagany pr-gate nie ma filtra paths; agregator zawsze sprawdza wynik zależnego joba.
