# Statystyki zbioru `v2.0.0`

Wygenerowane 2026-08-27 17:46 UTC, ziarno podziału `42`.

Razem **10853 nagrań**, **4878 grup źródłowych**, **20.8 h** audio.

## Podział

| split | nagrania | grupy | czas [h] | pozytywne | grupy pozytywne |
|---|---:|---:|---:|---:|---:|
| train | 7120 | 3337 | 14.17 | 2846 (40.0%) | **156** |
| val | 1866 | 770 | 3.22 | 863 (46.2%) | **100** |
| test | 1867 | 771 | 3.41 | 884 (47.3%) | **96** |

Kolumna *grupy pozytywne* jest ważniejsza niż liczba plików: to ona mówi, na ilu **niezależnych nagraniach** liczona jest metryka.

## Rodzaj dźwięku (`kind`) — podstawa raportu fałszywych alarmów

| kind | nagrania | czas [h] | train | val | test |
|---|---:|---:|---:|---:|---:|
| `positive` | 4593 | 3.33 | 2846 | 863 | 884 |
| `stationary` | 853 | 3.55 | 614 | 109 | 130 |
| `loud_event` | 2871 | 10.16 | 1964 | 475 | 432 |
| `speech` | 1788 | 1.51 | 1218 | 300 | 270 |
| `animal` | 748 | 2.25 | 478 | 119 | 151 |

## Źródła i licencje

| źródło | nagrania | grupy | czas [h] | licencja |
|---|---:|---:|---:|---|
| datasec | 3226 | 3207 | 15.27 | CC BY 4.0 |
| esc50 | 2000 | 1528 | 2.78 | CC BY-NC 3.0 |
| voice | 5627 | 207 | 2.75 | Other (Attribution) |

> **Uwaga licencyjna:** 2000 nagrań jest na licencji niekomercyjnej. Do zbioru treningowego modelu przeznaczonego do produktu trzeba je odfiltrować (`license != 'CC BY-NC 3.0'`).

## Parametry audio

- częstotliwości: {44100: 10853}
- kanały: {1: 10853}
- format próbki: {'PCM_16': 10853}
- długość [s]: min 0.20, p50 2.40, p95 30.01, max 481.48

## Zmiany względem v1.0.0

**Naprawiony błąd etykietowania VOICe (issue #34).** W v1.0.0 `collect_voice` wyprowadzało etykietę regułą maksymalnego pokrycia po WSZYSTKICH adnotacjach miksu i ignorowało katalog ekstrakcji. Interwał w nazwie pliku JEST interwałem jednej adnotacji, więc każde dłuższe zdarzenie, które go zawiera, remisowało lub wygrywało. Skutek: 975 babycry + 437 gunshot z katalogu `glass/` miało etykietę `negative` (1412 = 31.8% wyciętego szkła), a 241 klipów z `hard_negative/` etykietę `positive`. Teraz polaryzacja klasy pochodzi z katalogu ekstrakcji, a reguła pokrycia rozstrzyga wyłącznie gunshot vs babycry.

**Polifonia.** VOICe to miksy nakładających się zdarzeń: przy pad 0.30 s 3961 z 4444 zdarzeń glassbreak nachodzi na gunshot/babycry, a 3261 z 4444 wycinków hard_negative nachodzi na glassbreak. Przyjęto strażnika **asymetrycznego** (jak w `build_combined_dataset.py`, zgubionego przy przepisywaniu): odrzucamy skażony negatyw, bo jego etykieta byłaby fałszywa i uczyłaby sieć tłumienia na szkle; zachowujemy skażony pozytyw, bo on nadal ZAWIERA szkło — jest trudniejszy, nie błędny. Symetryczny strażnik zostawiłby 483 pozytywy zamiast 4444. Twardych negatywów VOICe: 1183 zamiast 4444.

**Podział VOICe** wzięty z opublikowanych list `dataset/clean/source/*.txt` (69/69/69 miksów, parami rozłączne). Podnosi liczbę niezależnych nagrań pozytywnych w teście z 59 do 96, kosztem proporcji: VOICe dzieli się 33/33/33, a pozostałe źródła 70/15/15, więc udział pozytywów w val/test jest wyższy niż w train. Metryką wdrożeniową jest FA/h liczone na godzinach tła, nie odsetek klipów, więc ta nierównowaga nie zniekształca celu.

## Znane ograniczenia

- Klasa pozytywna jest zdominowana przez wycinki VOICe pochodzące z 207 miksów; liczba **niezależnych** nagrań szkła jest o rząd wielkości mniejsza niż liczba plików.
- Udział pozytywów różni się między splitami (train ~40%, val/test ~46-47%) przez różne proporcje podziału VOICe i pozostałych źródeł. Progi kalibrowane na val trzeba przenosić na deployment ostrożnie.
- Tło stacjonarne pochodzi z ciągłych klas DataSEC i ESC-50, nie z nagrań z docelowego pomieszczenia. Fałszywe alarmy *w ciszy* są więc mierzone na zastępniku, nie na realnym tle instalacji.
- Audio nie jest transkodowane ani normalizowane — parametry są tylko mierzone i walidowane. Konwersję robi enkoder przy odczycie.
- ESC-50 jest na licencji niekomercyjnej.
