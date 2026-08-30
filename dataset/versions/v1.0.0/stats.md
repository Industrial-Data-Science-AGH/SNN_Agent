# Statystyki zbioru `v1.0.0`

Wygenerowane 2026-08-26 18:29 UTC, ziarno podziału `42`.

Razem **14114 nagrań**, **4878 grup źródłowych**, **23.1 h** audio.

## Podział

| split | nagrania | grupy | czas [h] | pozytywne | grupy pozytywne |
|---|---:|---:|---:|---:|---:|
| train | 9777 | 3412 | 15.96 | 2337 (23.9%) | **231** |
| val | 2419 | 732 | 3.61 | 640 (26.5%) | **62** |
| test | 1918 | 734 | 3.51 | 445 (23.2%) | **59** |

Kolumna *grupy pozytywne* jest ważniejsza niż liczba plików: to ona mówi, na ilu **niezależnych nagraniach** liczona jest metryka.

## Rodzaj dźwięku (`kind`) — podstawa raportu fałszywych alarmów

| kind | nagrania | czas [h] | train | val | test |
|---|---:|---:|---:|---:|---:|
| `positive` | 3422 | 2.85 | 2337 | 640 | 445 |
| `stationary` | 853 | 3.55 | 614 | 109 | 130 |
| `loud_event` | 4540 | 11.13 | 3155 | 770 | 615 |
| `speech` | 4551 | 3.30 | 3193 | 781 | 577 |
| `animal` | 748 | 2.25 | 478 | 119 | 151 |

## Źródła i licencje

| źródło | nagrania | grupy | czas [h] | licencja |
|---|---:|---:|---:|---|
| datasec | 3226 | 3207 | 15.27 | CC BY 4.0 |
| esc50 | 2000 | 1528 | 2.78 | CC BY-NC 3.0 |
| voice | 8888 | 207 | 5.04 | Other (Attribution) |

> **Uwaga licencyjna:** 2000 nagrań jest na licencji niekomercyjnej. Do zbioru treningowego modelu przeznaczonego do produktu trzeba je odfiltrować (`license != 'CC BY-NC 3.0'`).

## Parametry audio

- częstotliwości: {44100: 14114}
- kanały: {1: 14114}
- format próbki: {'PCM_16': 14114}
- długość [s]: min 0.20, p50 2.40, p95 27.82, max 481.48

## Znane ograniczenia

- Klasa pozytywna jest zdominowana przez wycinki VOICe pochodzące z 207 miksów; liczba **niezależnych** nagrań szkła jest o rząd wielkości mniejsza niż liczba plików.
- Tło stacjonarne pochodzi z ciągłych klas DataSEC i ESC-50, nie z nagrań z docelowego pomieszczenia. Fałszywe alarmy *w ciszy* są więc mierzone na zastępniku, nie na realnym tle instalacji.
- Audio nie jest transkodowane ani normalizowane — parametry są tylko mierzone i walidowane. Konwersję robi enkoder przy odczycie.
- ESC-50 jest na licencji niekomercyjnej.
