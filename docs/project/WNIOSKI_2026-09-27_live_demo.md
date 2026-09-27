# Wnioski — sesja 27.09.2026: naprawa treningu, firmware i pierwszy live E2E

Ten dokument podsumowuje jedną sesję pracy (27.09.2026, Wiktor + Claude Code) od
kontynuacji M0 do działającego demo na żywym sprzęcie: prawdziwy dźwięk
tłuczonego szkła → mikrofon → Arduino Mega → Raspberry Pi → wytrenowany model →
LED. Zebrane tu ustalenia dotyczą kodu Marcela (`master_pipeline`,
`ga_neuron_search`), Kacpra (`encoder_v2.ino`) i integracji Wiktora
(`rpi_agents`) — każdy PR jest osobny, ale wnioski są wspólne.

## Co się udało

1. **M0 dokończone na M5 Pro 64GB** (nie M5 Max 128GB — patrz zastrzeżenie
   niżej). Pełne wyniki i logi przekazane Marcelowi wcześniej tego dnia.
2. **Realny bug treningowy naprawiony i zweryfikowany dwa razy**: kwantyzacja
   (QAT) zeruje wagi poniżej rozdzielczości trymera (`W_DEADZONE=0.05` w
   `snn_hw_pipeline.py`) — poprawnie odwzorowuje sprzęt. Problem: trening float
   (HAT) nie ma presji, żeby trzymać wagi z dala od tej strefy, więc ok. 60%
   seedów w tej sesji zapadało się do zera w chwili włączenia kwantyzacji i
   już nigdy nie wracało. Naprawa: wykrycie płaskiego zera po 8 epokach QAT +
   do 2 prób ponownych tego samego seeda z innym punktem startu, zanim wynik
   wejdzie do puli mediany. `ga_neuron_search/winner.py`, PR #65 (zmergowany).
3. **`decoder_k` przestał być zahardkodowany na 2.** Ten sam checkpoint,
   oceniony bez retreningu przy k=1/2/3: k=1 dał TEST recall 0,828 (cel
   zespołu: ≥80%), k=2 tylko 0,698. `master_pipeline/ga_runner.py`, PR #66.
4. **Firmware `encoder_v2.ino`: trzy niezależne, realne błędy**, wszystkie
   zweryfikowane na fizycznym sprzęcie:
   - Sterowanie pinami D2–D8 przez surowe `PORTD`/`PORTB` działało tylko na
     Uno — te rejestry mapują się inaczej na Mega (ATmega2560). Zamienione na
     przenośne `digitalWrite()`.
   - **Firmware nigdy nie mówił protokołem, którego most na Pi oczekuje**
     (`$B`/`$F` z `rpi_agents/agent/serial_protocol.py`) — wysyłał tylko
     czytelny dla człowieka CSV do kalibracji z poradnika. To dotyczyłoby też
     prawdziwego Uno, nie tylko Mega — zadanie K2 zostało po cichu
     niedokończone. Dodano poprawną emisję `$B`/`$F` z CRC8, zweryfikowaną
     najpierw przez prawdziwy parser Pythona, potem na żywo (0 luk, 0
     resetów w wielotysięcznej sesji ramek).
   - **ADC próbkował 2× za szybko**: prescaler 32 dawał zmierzone 38 440–38 479
     Hz zamiast zakładanych 19 231 Hz (błędne założenie liczby cykli konwersji
     w komentarzu firmware — 13 cykli w trybie ciągłym, nie 25–26; ten sam
     fakt stoi za popularnym "~9,6 kHz max" dla `analogRead()`). Naprawione
     przez prescaler 64, zmierzone po poprawce: 19 220–19 239 Hz. **To dotyczy
     każdego klasycznego AVR z tą konfiguracją, nie tylko Mega — jeśli
     prawdziwe Uno było kiedyś mierzone z prescalerem 32, ma ten sam błąd.**
     Efekt przed naprawą: pętla `loop()` nie nadążała (~17 000 próbek na
     ramkę zamiast 192, ramki co ~650ms zamiast co 10ms).
   `architecture_14_neurons_patryk_09_07/encoder_v2.ino`, PR #67.
5. **Pierwszy prawdziwy end-to-end test na żywym sprzęcie, potwierdzony
   wizualnie przez Wiktora**: powtarzalne detekcje na dźwięk szkła, zero
   fałszywych alarmów w ciszy, zero luk transmisji. Most demo (poza repo,
   patrz niżej) czyta realne ramki z Mega, liczy inferencję lokalnie i steruje
   diodą LED na GPIO17 przez SSH.

## Kluczowe metryki (champion `run_20260927_205133`, k=1)

| Zbiór | recall | precision | clip_f1 | FA (odsetek negatywów z fałszywym alarmem) |
|---|---|---|---|---|
| VAL | 87,6% | 44,2% | 0,588 | 0,348 |
| **TEST** (trzymany w tajemnicy) | **82,8%** | 61,9% | 0,709 | 0,458 |

Świadomy kompromis pod wymaganie zespołu "złap prawie każde szkło" — wysoki
recall kosztem częstszych fałszywych alarmów. To metryka na zbiorze
testowym (VOICe+ESC-50), **nie** formalny pomiar recall na żywym mikrofonie —
do tego potrzeba systematycznej serii nagrań i policzenia trafień na żywo.

## Czego NIE zrobiono i dlaczego

- **Formalny pakiet M5** (`package_champion.py`) nie powstał dla żadnego z
  dzisiejszych championów. Wymaga budżetu FA/h ze strumienia ciągłego (M4,
  dane K3 Kacpra), których nie ma na tej maszynie — próba automatycznego
  odtworzenia reguły dekodera zwróciła jawnie `infeasible`. Skrypt słusznie
  odmówił zgadywania; nie obchodziliśmy tego na sztucznych danych.
- **Demo na żywo to świadomy skrót, nie produkcyjna ścieżka.** Osobny skrypt
  (`live_glass_demo.py`, na razie poza repo — w scratchpadzie sesji) ładuje
  surowy checkpoint bezpośrednio przez `net.GenomeNet`, z pominięciem
  `snn_runtime`/backendu/Azure. Inferencja liczy się na Macu (Pi celowo
  zostaje "lekkim mostem", bez torch — zgodnie z architekturą), a LED
  odpalana zdalnie przez SSH. Wystarcza do dzisiejszego testu; docelowa
  ścieżka produkcyjna nadal idzie przez `snn_runtime` + Azure.
- **Buzzer nadal niepodłączony** — tylko LED na GPIO17.
- **Nie zrobiono formalnego pomiaru recall na żywym mikrofonie** — dzisiejsze
  detekcje potwierdzają, że łańcuch działa, ale to nie jest systematyczny
  test z policzonymi trafieniami/pudłami.

## Rozbieżności do wyjaśnienia z zespołem (niezablokowane, ale realne)

1. **Niespójność nazw kanałów.** `encoder_twin.py` (budował dane treningowe:
   `peak, peak_cnt, cv, zcr, flux, hf_lo, hf_hi`) i `snn_hw_pipeline.py`
   (używany przez `winner.py`/eksport na płytki: `peak, hjorth_mobility,
   autocorr_lag1, zcr, flux, hf_lo, hf_hi`) mają różne listy `CHANNELS`. To
   wpływa tylko na **etykiety** w tabeli eksportu sprzętowego (np. źle
   nazwane gniazdo przy montażu fizycznych płytek Lu.i), nie na trening ani
   inferencję (te działają po indeksach 0–6, nie po nazwach). Fixture K1
   Kacpra (`contracts/fixtures/encoder-profile.json`, profil "swap") również
   nie zgadza się z wariantem "base", którego realnie używa nasz firmware i
   model — do potwierdzenia z Kacprem, który wariant jest docelowy.
2. **`neurons_range: [4, 6, 8]` w configu jest de facto ignorowane** —
   `run_ga_stage` (ga_runner.py) bierze tylko `max(neurons_range)` jako
   sztywny sufit liczby neuronów w jednym przebiegu GA, nie zamiata po trzech
   wartościach osobno. Możliwe, że to celowe uproszczenie (GA i tak dobiera
   mniej neuronów w ramach tego sufitu), ale nie jest to udokumentowane —
   warto potwierdzić z Marcelem.
3. **Zmienność między biegami jest realna.** Powtórzenie treningu z tą samą
   konfiguracją (k=1) dało TEST recall 0,641 zamiast 0,828 — mediana z 5
   losowych seedów nie gwarantuje powtarzalności między przebiegami. Do
   rozważenia: więcej seedów, albo jawne zapisywanie/porównywanie wielu
   przebiegów przez `champion.py` zamiast polegania na jednym.

## Linki

- PR #65 — poprawka QAT-collapse (`winner.py`), zmergowany.
- PR #66 — `decoder_k=1` zamiast zahardkodowanego 2 (`ga_runner.py`).
- PR #67 — Mega + protokół `$B/$F` + poprawka ADC (`encoder_v2.ino`).
