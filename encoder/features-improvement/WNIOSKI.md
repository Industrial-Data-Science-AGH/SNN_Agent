# Analiza logów z treningu sieci neuronowej

Poniżej znajdują się wnioski z przeprowadzonych 10 cykli treningowych dla 5 różnych wartości początkowych (seed: s0-s4) w dwóch wariantach architektury (`base` oraz `swap`). Zastosowano prosty, codzienny język, aby ułatwić zrozumienie wyników technicznych.

## 1. Zestawienie wyników (Tabela)

Metryka **F1** to wskaźnik łączący precyzję (ile wykryć było trafnych) z czułością (ile faktycznych zdarzeń wykryto). Im wyżej, tym lepiej (max to 1.0). **Walidacja** to testy podczas nauki, **Test** to ostateczny sprawdzian na nowych danych. **Odporność** pokazuje, jak model radzi sobie z fizycznymi niedoskonałościami układu scalonego.

| Seed | Architektura | F1 Walidacja | F1 Test | Odporność sprzętowa (Średnia) |
| --- | --- | --- | --- | --- |
| **s0** | base | 0.618 | 0.552 | 0.611 |
| **s0** | swap | 0.701 | 0.638 | 0.720 |
| **s1** | base | 0.618 | 0.555 | 0.605 |
| **s1** | swap | 0.702 | 0.623 | 0.679 |
| **s2** | base | 0.718 | 0.648 | 0.699 |
| **s2** | swap | **0.763** | **0.683** | **0.755** |
| **s3** | base | 0.636 | 0.561 | 0.638 |
| **s3** | swap | 0.604 | 0.537 | 0.626 |
| **s4** | base | 0.742 | 0.648 | 0.719 |
| **s4** | swap | 0.743 | 0.663 | 0.718 |

## 2. Główne wnioski z treningu

* **Wariant `swap` jest wyraźnie lepszy:** W 4 na 5 przypadków architektura `swap` osiągnęła lepsze wyniki niż wersja `base`. Zapewnia wyższy wskaźnik F1 zarówno na zbiorze walidacyjnym, jak i testowym. Jedynym wyjątkiem był seed 3, gdzie wariant `base` poradził sobie nieznacznie lepiej.
* **Najlepszy model:** Zdecydowanym zwycięzcą całego zestawienia jest model **`swap_s2`**, który osiągnął najwyższy wynik na zbiorze testowym (F1: 0.683) i najwyższą odporność na szumy (F1: 0.755).
* **Spadek skuteczności (Generalizacja):** Zauważalny jest spadek jakości (F1) pomiędzy danymi walidacyjnymi a testowymi we wszystkich eksperymentach (średnio o ok. 0.08). Oznacza to, że model delikatnie "uczy się na pamięć" danych treningowych, ale w granicach normy.
* **Odporność na wahania sprzętowe:** Trening QAT (trening uwzględniający kwantyzację, czyli sztuczne ograniczanie precyzji obliczeń, by symulować realny sprzęt) zadziałał znakomicie. Modele utrzymują swoją skuteczność pod wpływem symulowanego szumu elektroniki (wahania napięć itp.). Dodatkowo, system potrójnego głosowania (zespół 3 układów, "głos 2-z-3") lekko poprawia tę stabilność, choć różnice nie są drastyczne.

## 3. Problem fałszywych alarmów (FA/h)

Założony przez projekt rygorystyczny budżet fałszywych alarmów na poziomie **1 FA/h oraz 6 FA/h (fałszywych alarmów na godzinę) jest całkowicie nieosiągalny** dla obecnych modeli. Logi w każdym eksperymencie jasno komunikują "budżet nieosiągalny".

Rzeczywista liczba fałszywych alarmów w testach (przy klasyfikacji klipów jako "tło") waha się od 78 do nawet 260 na godzinę, w zależności od przyjętej reguły.

**Wpływ reguł decyzyjnych (k):**
Reguła `k` oznacza, ile sygnałów (skoków napięcia, ang. *spikes*) system musi odnotować w oknie czasowym (np. 2500 ms), by podnieść alarm.

* `k=1` (wystarczy 1 sygnał): Model wykrywa bardzo dużo prawdziwych zdarzeń zbijanego szkła, ale generuje gigantyczną liczbę fałszywych alarmów z tła (nawet ponad 40% pomyłek).
* `k=3` (wymagane 3 sygnały): Drastycznie redukuje liczbę fałszywych alarmów (spadają o połowę), ale odbywa się to kosztem przegapiania prawdziwych zdarzeń (czułość na zbijane szkło spada do okolic 50-60%). Niestety, nawet przy `k=3` fałszywych alarmów jest wciąż zbyt dużo na standardy komercyjne.

## 4. Niepotrzebne cechy dźwięku (Zjawisko przycinania synaps)

Systematycznie w każdym cyklu sprzętowym pewne "synapsy" (połączenia w sztucznej sieci) są odcinane (ich waga jest ustawiana na zero, ponieważ sygnał jest poniżej rozdzielczości/czułości sprzętu).

Cechy, które najczęściej okazywały się nieprzydatne do wykrywania zbijanego szkła:

* **`cv`** (Coefficient of Variation – współczynnik zmienności): Odłączany w niemal każdym logu na płytce H1 (szczególnie synapsa J1).
* **`hf_hi` / `hf_lo**` (wysokie i niskie częstotliwości): Odłączane sporadycznie, co sugeruje, że nie każdy kanał częstotliwości niesie ważne informacje.
* **`H2` do innych modułów**: Często odłączane są powiązania między blokami G, np. `G0.J3 (H2)`.

System automatycznie pozbywa się szumu z wejść, które nie poprawiają precyzji wykrywania. Warto w przyszłości rozważyć całkowite usunięcie tych cech z etapu przetwarzania wstępnego (co oszczędzi energię i moc obliczeniową układu).

## 5. Wymiana kanałów przyniosła radykalną zmianę w tym, po jakie informacje sięga sieć:

   * `autocorr` (widoczne jako `cv`): Aktywne w 5 na 5 eksperymentów. Sieć nie wyłączyła tej cechy ani razu. W seedzie 0 i 2 otrzymała ona maksymalne możliwe wagi układu (-100% oraz +100%), co świadczy o jej krytycznym znaczeniu decyzyjnym.

* `mobility` (widoczne jako `peak_cnt`): Aktywne w 5 na 5 eksperymentów. Również ani razu nie odrzucone.

*    W tym wariancie sieć odrzucała jedynie słabsze kanały częstotliwościowe (np. `hf_hi` w seedzie 0, hf_lo w seedzie 1), aby zwolnić zasoby na mocniejszy sygnał z nowych kanałów.

Obliczanie każdej cechy w pętli mikrokontrolera kosztuje czas procesora i energię z baterii. Jeśli układ oblicza cechę cv, a sieć sprzętowa i tak w 80% przypadków mnoży ten wynik przez zero (pot 0.0%), to zasoby mikrokontrolera są marnowane. Wyniki z treningu jasno udowadniają, że usunięte cechy były ślepym zaułkiem, a ich zamienniki (`autocorr` i `mobility`) stanowią teraz główny filar decyzyjny sprzętu.


Oto zestawienie cech wchodzących do obu architektur:

| Kanał SNN | Architektura `base` (Baseline) | Architektura `swap` (Nowa) | Rola / Opis zmiany |
| --- | --- | --- | --- |
| **Kanał 0 (`s0`)** | `peak` | `peak` | Amplituda szczytowa (bez zmian) |
| **Kanał 1 (`s1`)** | **`peak_cnt`** | **`hjorth_mobility`** | **Wymiana:** Liczba pików zastąpiona mobilnością Hjortha |
| **Kanał 2 (`s2`)** | **`cv`** | **`autocorr_lag1`** | **Wymiana:** Współczynnik zmienności zastąpiony autokorelacją |
| **Kanał 3 (`s3`)** | `zcr` | `zcr` | Przejścia przez zero (bez zmian) |
| **Kanał 4 (`s4`)** | `flux` | `flux` | Strumień widmowy / dynamika energii (bez zmian) |
| **Kanał 5 (`s5`)** | `hf_lo` | `hf_lo` | Pasmo dolne wysokich częstotliwości (bez zmian) |
| **Kanał 6 (`s6`)** | `hf_hi` | `hf_hi` | Pasmo górne wysokich częstotliwości (bez zmian) |

* **Źródło zamieszania w logach:** W logach z eksportu `hw_swap_sX.json` dla ramienia `swap` pojawiały się jeszcze stare nazwy `peak_cnt` i `cv`. Była to wyłącznie nieaktualna tablica etykiet tekstowych w skrypcie `snn_hw_pipeline.py`. Fizycznie i matematycznie twin enkodera generował w tych miejscach nowe cechy: **`hjorth_mobility`** oraz **`autocorr_lag1`**.
