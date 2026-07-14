# Dziennik decyzji — pełny zapis sesji (Wake-Up AI / Delta Spike)

Kompletny, chronologiczny zapis KAŻDEJ decyzji projektowej podjętej w tej sesji, w
formacie: **Decyzja → Alternatywy → Dlaczego → Dowód**. Pod publikację i do samodzielnego
zrozumienia. Kolejność mniej więcej chronologiczna, pogrupowana w fazy.

Kontekst: budujemy detektor tłuczonego szkła na analogowych płytkach neuronów LIF „Lu.i"
(3 wejścia J1/J2/J3, 1 wyjście J4, próg V_th = VDD/2 sprzętowo nieruchomy, waga = trymer,
znak = przełącznik ±). Enkoder na Arduino zamienia audio na spiki, sieć SNN 4→3→1 na
płytkach podejmuje decyzję. Ponieważ nie da się ręcznie nagrać tłuczenia szkła, cały tor
uczenia jest programowy: audio → cyfrowy bliźniak enkodera → CSV spików → trening →
`hw_config.json` (nastawy trymerów/paska LED per płytka).

---

## FAZA A — Cyfrowy bliźniak enkodera i budowa datasetu

### A1. Resampling audio do 19231 Hz zamiast przeliczania długości ramki
- **Decyzja:** wczytywać każdy plik z `sr=19231 Hz` (rzeczywiste fs ADC Arduino przy
  prescalerze 32), tak by `HOP_SAMPLES=192` nadal znaczyło dokładnie 10 ms.
- **Alternatywa:** zostawić 44.1 kHz i przeliczać długość ramki w próbkach.
- **Dlaczego:** przy resamplingu WSZYSTKIE stałe firmware (THR_Z, A_UP/A_DN/A_MAD,
  spike_thr, HOP_SAMPLES) przenoszą się 1:1 bez przeliczeń na inne jednostki. Bliźniak
  liczy dokładnie to, co policzy płytka.
- **Dowód:** decyzja delegowana przez użytkownika („wybierz sam"). Wierność potwierdzona
  później zgodnością formatu i sensownymi rozkładami cech.

### A2. Rekwantyzacja do 10-bitowych kodów ADC (bias 511.5)
- **Decyzja:** mapować sygnał na 0..1023 wyśrodkowany na 511.5 (mikrofon na Vcc/2).
- **Dlaczego:** stałe `spike_thr` (clamp 8..1023) i start DC (512<<4) mają sens tylko w
  tych samych jednostkach co ADC.

### A3. Arytmetyka float w bliźniaku (nie Q4/int16 z ATmega)
- **Decyzja:** liczyć cechy w floatach.
- **Dlaczego:** stałoprzecinkowość ATmega to ograniczenie MCU (implementacja), nie model
  DSP. Świadomie NIE odtwarzamy jej, bo nie zmienia logiki cech.

### A4. JEDEN ciągły stan enkodera z rozgrzewką 30 s — BEZ resetu per plik ★ kluczowa
- **Decyzja:** utrzymywać jeden `EncoderState` przez cały zbiór, rozgrzany na ~30 s
  realnego tła; NIE resetować floorów między nagraniami.
- **Alternatywa:** resetować stan przy każdym klipie (jak power-on-reset Arduino).
- **Dlaczego:** stała czasowa wzrostu floora (A_UP) odpowiada ~6.7 s, a klipy mają ~3 s —
  przy resecie floor nigdy nie dogania prawdziwego tła i mechanizm „zamrożenia floora w
  zdarzeniu" utrwala błędny, zbyt niski floor. W realnym wdrożeniu urządzenie działa
  ciągle, floor kalibruje się raz przez minuty.
- **Dowód (pomiar):** reset per plik dawał niemal identyczny odsetek ramek ze spikiem —
  glass ~25% vs negative ~27% (enkoder NIE rozróżniał klas). Ciągły stan + warmup: negative
  spadło do ~12%, wyraźna separacja per kanał (peak_cnt 8×, cv 5×, peak 3× wyższe dla szkła).
- **Koszt:** 1–2 ramki na granicy plików dziedziczą trochę stanu (np. rms_prev) — dużo
  mniejsze zniekształcenie niż reset co 3 s.

### A5. VOICe (gunshot/babycry) jako „trudne negatywy"
- **Decyzja:** dokopać do zbioru głośne, ostre zdarzenia NIE-szklane jako negatywy.
- **Dlaczego:** pierwszy trening miał ~28% precyzji, bo kanały peak/crest reagują na
  DOWOLNY ostry transient. Trudne negatywy zmuszają sieć do cech specyficznych dla szkła.
- **Dowód:** rozbicie fałszywych alarmów po źródle pokazało później, że to głośne zdarzenia
  (nie cisza) są problemem — potwierdza sens tego zabiegu.

---

## FAZA B — Przebudowa pipeline treningowego (`snn_hw_pipeline.py`)

### B1. Checkpoint po F1 zamiast `recall + 0.3·precision` ★
- **Decyzja:** wybierać najlepszą epokę po F1.
- **Alternatywa:** poprzednia formuła `recall + 0.3·precision`.
- **Dlaczego:** waga precyzji była za mała, by przebić zdegenerowane wczesne epoki typu
  „zgłaszaj wszystko". F1 karze to wprost.
- **Dowód:** pierwszy trening z formułą złapał epokę z recall 0.93 / precision 0.05 / F1 0.09.
  Po zmianie na F1: recall 0.82 / precision 0.28 / F1 0.42.

### B2. Prawdziwe fazy HAT → QAT (nie kwantyzacja od startu) ★
- **Decyzja:** najpierw ~50% epok w pełnej precyzji z wstrzykiwanym szumem sprzętowym
  (HAT), potem kwantyzacja STE do 20 działek trymera (QAT), z RESETEM najlepszego wyniku
  przy przejściu do QAT.
- **Dlaczego:** każda waga trafia na fizyczny potencjometr, więc finalny model MUSI być
  skwantyzowany; ale najpierw trzeba znaleźć dobre rozwiązanie w pełnej precyzji.
  Szum sprzętowy (±½ działki trymera, ±10% τ, ±2% V_leak) w HAT czyni rozwiązanie odpornym
  na ręczną kalibrację. Reset best przy QAT, żeby skwantyzowany checkpoint nie przegrywał
  z lepszym, ale nierealizowalnym pełnoprecyzyjnym.
- **Dowód:** odporność F1 (Monte Carlo) skoczyła w wersji v3 do min 0.55 (vs 0.36 wcześniej).

### B3. Podział walidacji PO PLIKACH, nie po oknach ★
- **Decyzja:** dzielić train/val po plikach źródłowych.
- **Alternatywa:** losowy podział po oknach (jak było).
- **Dlaczego:** okna z jednego klipu zachodzą na siebie (stride 50 < T 200), więc podział
  po oknach przecieka — te same dane w train i val, wynik zawyżony.
- **Dowód:** po naprawie test held-out ≈ walidacja (0.45 vs 0.44), brak przepaści = brak
  przeuczenia; wcześniejsze 0.42 było zawyżone.

### B4. Stały budżet okien na epokę (12000 przez WeightedRandomSampler)
- **Decyzja:** losować stałą liczbę zbalansowanych okien na epokę, niezależnie od rozmiaru
  zbioru.
- **Dlaczego:** 10× więcej danych ma dawać różnorodność MIĘDZY epokami, nie 10× dłuższą
  epokę. Sampler balansuje też klasy 50/50 (dlatego realny stosunek pos:neg w zbiorze jest
  bez znaczenia dla treningu).

### B5. Test odporności Monte Carlo po treningu
- **Decyzja:** po treningu policzyć F1 przy kilku losowaniach rozrzutu trymerów/τ/V_leak.
- **Dlaczego:** przewiduje, jak model przetrwa ręczną kalibrację; ostrzega, jeśli wynik
  jest kruchy. Trafia do `hw_config.json`.

### B6. Cache datasetu (.npz), early stopping, batch 128, jedno GEMM/sekwencję
- Cache: parsowanie ~11k CSV trwa minuty, wczytanie sekundy.
- Early stopping z patience osobnym per faza (w HAT skraca fazę, w QAT kończy).
- Jedno `F.linear` na całą sekwencję zamiast T małych + batch 128 → szybkość.

### B7. Metryki na poziomie KLIPÓW + reguła dekodera k
- **Decyzja:** liczyć „ile % nagrań szkła budzi system i ile klipów tła daje fałszywy alarm"
  przy regule „k spików neuronu D", nie tylko F1 na oknie 2 s.
- **Dlaczego:** okno 2 s to nie jest to, co widać na demie; liczy się zdarzenie/klip.
  k > 1 może egzekwować dekoder (zliczanie impulsów na J4) bez zmian w analogu.

---

## FAZA C — Powiększony zbiór z manifestu

### C1. Użycie `dataset/combined/manifest.csv` (13.5k plików, gotowy split)
- **Decyzja:** wykorzystać zbiór złożony w innej sesji (VOICe + ESC-50 + datasec
  PT_DATASET + notebooks) z gotowym podziałem train/val/test.
- **Dowód:** 3724 glass / 7094 negative (train), 463/874 (val i test).

### C2. Tryb `build-manifest` w bliźniaku
- **Decyzja:** dodać tryb kodujący wg manifestu, zachowujący split jako podkatalogi
  wyjścia, z jednym ciągłym stanem enkodera, rozgrzanym na STACJONARNYM tle (notebooks).
- **Dlaczego:** warmup floora musi iść na prawdziwym tle — krótkie wycinki zdarzeń
  (gunshot/dzwony) się nie nadają, bo same są zdarzeniami.

---

## FAZA D — Sweep 14-neuronowy (6 kanałów), diagnoza sufitu

### D1. Sweep 6 konfiguracji (pos_weight × seed × strata)
- **Decyzja:** puścić równolegle warianty pos_weight {1.0, 1.75, 3.0} i strat, po kilka
  seedów.
- **Dlaczego:** mała sieć ma dużą wariancję od inicjalizacji; pos_weight to główna dźwignia
  balansu recall↔precision.
- **Dowód:** C pw3.0 → F1 0.383 (za duży recall-bias); A pw1.0 → F1 0.487 (najlepszy).

### D2. Strata zliczania spików (spk_w) — testowana i ODRZUCONA
- **Decyzja:** dodać opcjonalny człon straty „szkło ≥2 spiki D, tło 0".
- **Dlaczego (hipoteza):** reguła dekodera k≥2 działa tylko, gdy szkło daje serię spików.
- **Dowód (negatywny):** wymusiła więcej spików na szkle I tle jednakowo (2.33 vs 2.29) —
  nie rozdzieliła klas. Odrzucona.

### D3. Reguła dekodera k=1 (nie k≥2)
- **Decyzja:** rekomendować k=1 (alarm gdy D strzeli choć raz).
- **Dowód:** k=1 to jedyny punkt pracy z użytecznym recall (72%); k≥2 spadał do ~3%, bo
  szkło nie dawało serii ≥2 spików. Testowano też reguły serii (k w krótkim oknie) i
  regułę liniową na warstwie G (`g_tap_eval.py`) — nie biły k=1 (G0/G1 nie rozdzielały klas).

### D4. Rozbicie fałszywych alarmów PO ŹRÓDLE (`eval_stream.py`) ★ diagnoza
- **Decyzja:** rozbić fałszywe alarmy na ciche tło vs głośne zdarzenia (datasec/esc50/voice).
- **Dowód:** na cichym tle ~0% FA przy k=3, ale na głośnych zdarzeniach 65–71%. WNIOSEK:
  sufit to ENKODER (6 cech pełnopasmowych), nie trening — model nie odróżnia szkła od
  innych głośnych transientów.

### D5. Dźwignia progu D w terenie (`--d-leak-delta`)
- **Decyzja:** dodać możliwość podniesienia progu D (fizycznie: niższy pasek LED), bez
  wracania do treningu — jedyne pokrętło czułość↔fałszywe alarmy na płytce.

### D6. Eksport wersji 14-neuronowej (`hw_config.json`, `kalibracja_sciaga.md`)
- Winner A wyeksportowany jako gotowy-na-jutro model 14-neuronowy, z regułą k=1 i ramowaniem
  „brama always-on, reaktor LLM weryfikuje".

---

## FAZA E — Enkoder v3 widmowy (sieć 15-neuronowa)

### E1. Diagnoza: głównym brakiem jest informacja WIDMOWA
- **Decyzja/wniosek:** wszystkie 6 cech liczone na pełnym paśmie; szkło jest rozpoznawalne
  widmowo (energia 4–10 kHz), a głośne mylące zdarzenia są szerokopasmowe/niskopasmowe.

### E2. Pytania do użytkownika (AskUserQuestion)
- **Decyzja:** dopytać o znaczenie „15 neuronów" i apetyt na zmianę firmware.
- **Odpowiedź użytkownika:** Arduino Uno, mikrofon typu WM60 (płaska charakterystyka —
  stosunki pasm miarodajne), 15 płytek Lu.i; wybrał TANI filtr 1-biegunowy (nie biquad).

### E3. Topologia 7→4→3→1 = 15 neuronów; maski projektuje asystent
- **Decyzja:** 7 kanałów (dodać widmowe, usunąć martwą `crest`), warstwa H nadal 4 płytki,
  liczba płytek Lu.i bez zmian (8), +1 pin Arduino (D8). `MASK_H` zaprojektowana tak, że
  każda płytka H dostaje jeden kanał widmowy zmieszany z czasowymi.
- **Dlaczego:** sieć może od pierwszej epoki uczyć się koniunkcji „głośne ORAZ HF".

### E4. Bramka decyzyjna: czy hf_ratio W OGÓLE rozdziela klasy?
- **Decyzja:** przed jakąkolwiek zmianą modelu policzyć separację cechy na surowym audio.
- **Dowód:** AUC szkło vs trudne negatywy = 0.73 przy cutoff k=1 (~2.2 kHz). Bramka zdana.

### E5. Pojedynczy 1-pole, nie kaskada
- **Decyzja:** zostać przy jednym stopniu filtra.
- **Dowód:** kaskada dwóch 1-poli podniosła AUC tylko 0.730 → 0.739 — nie warta kosztu.

### E6. ★★ Próg BEZWZGLĘDNY dla hf_ratio, nie z-score — kluczowe odkrycie
- **Problem:** pierwsza wersja kodowała hf_ratio adaptacyjnym z-score jak resztę. Trening
  NIE poprawił się. Diagnoza: hf_ratio-z-score strzelał RZADZIEJ dla szkła (1.05%) niż dla
  głośnych negatywów (datasec 3.39%) — SYGNAŁ ODWRÓCONY.
- **Dlaczego:** adaptacyjny floor reaguje na ZMIANĘ względem tła kanału, a hf_ratio to
  POZIOM (kształt widma), nie transient. Floor adaptuje się do trwale wysokiego HF szkła i
  kanał milknie; głośne negatywy z niskim tłem HF dają szpilki z-score.
- **Decyzja:** kodować hf_ratio progiem BEZWZGLĘDNYM (stałym), nie z-score.
- **Dowód (dobór progu):** próg 0.35 → szkło 41% głośnych ramek, esc50/quiet/voice ~0–7%,
  datasec 23%. Próg 0.28 → szkło 58%. Wybrano OBA jako kod termometrowy.
- **Lekcja publikacyjna:** kodowanie z-score (change-detector) jest dobre dla transientów
  (peak, flux), ale NIEWŁAŚCIWE dla stacjonarnych deskryptorów widma. To dlatego walidujemy
  w symulacji przed sprzętem.

### E7. Dwa kanały widmowe (hf_lo 0.28, hf_hi 0.35), usunięcie hf_flux
- **Decyzja:** zamiast słabego hf_flux dać drugi próg hf_ratio → termometr „umiarkowany /
  mocny HF".
- **Dowód:** hf_flux nie różnicował (szkło 5.76% vs cisza 7.11%); termometr daje sieci
  stopniowany sygnał, hf_hi jest mocnym dowodem na szkło.

### E8. Bramka zdarzenia dla kanałów widmowych (peak > 1.5×floor)
- **Decyzja:** hf_lo/hf_hi strzelają tylko, gdy ramka ma transient.
- **Dlaczego:** hf_ratio = energia_HF/energia_całkowita jest źle określona na ciszy
  (dzielenie przez ~0) — bez bramki strzelałaby losowo na szum tła.
- **Dowód po naprawie:** hf_lo szkło 25.9% vs negatywy 2.6–4.3%; hf_hi szkło 18.5% vs
  0.6–2.1% — silna, poprawnie skierowana separacja.

### E9. Naprawa kolizji cache przy równoległych treningach
- **Problem:** trzy równoległe treningi zapisywały wspólny `.npz` naraz → EOFError
  (ucięty plik).
- **Decyzja:** zapis atomowy (temp per-proces + `os.replace`) + odporny odczyt (rebuild przy
  wyjątku). Dodatkowo cache budowany raz szeregowo przed treningami.

### E10. Wybór zwycięzcy: seed 2, pos_weight 1.0
- **Decyzja:** spośród 3 seedów wybrać seed 2.
- **Dowód:** najlepszy pod KAŻDYM względem — test F1 0.609, odporność min 0.55, k=1 szkło
  72% / FA łącznie 20% (D strzela 15.5×/szkło vs 3.6×/tło, najlepszy stosunek).
  ```
  seed 2 pw1.0: F1 0.609, rob 0.55, k=1 72%/FA20%  (WYBRANY)
  seed 1 pw1.0: F1 0.571, rob 0.53, k=1 79%/FA32%  (wyższy recall)
  seed 3 pw1.5: F1 0.592, rob 0.53, k=1 58%/FA18%  (niższe FA)
  ```

### E11. Port do firmware `encoder_v2.ino` (Faza 3 planu)
- **Decyzja:** dodać 1-pole HF w ISR, `hf_ratio`, progi bezwzględne + bramkę, 7. kanał na
  D8 (PORTB), wyjście dwuinstrukcyjne PORTD+PORTB.
- **Dlaczego dopiero teraz:** plan zakładał firmware TYLKO po potwierdzeniu zysku w symulacji.
- **Budżet ISR:** ~6 µs przy okresie próbki ~52 µs — z zapasem, ale wymaga potwierdzenia na
  płytce (jedyne realne ryzyko HW).

### E12. Rozdzielenie configów 14 vs 15 neuronów
- **Decyzja:** NIE nadpisywać `hw_config.json` (14-neur., gotowy bez firmware) configiem
  15-neuronowym `hw7_config.json` — bo firmware 6-kanałowy jeszcze nie produkuje 7 kanałów.
  Decyzja 14-vs-15 na jutro należy do użytkownika (v3 lepszy, ale wymaga wgrania firmware +
  D8 + weryfikacji timingu).

---

## Wynik końcowy (dowód liczbowy)

| metryka (zbiór TESTOWY) | 14-neur. (v2) | 15-neur. widmowy (v3) |
|---|---|---|
| F1 okienkowe | 0.487 | **0.609** |
| odporność F1 min (rozrzut trymerów) | 0.355 | **0.550** |
| recall szkła (k=1) | 72% | 72% |
| fałszywe alarmy łącznie (k=1) | 63% | **20%** |
| FA datasec / esc50 / voice | 71/65/66% | 23/19/8% |
| FA cisza (notebooks) | 30% | 46% |

Ten sam recall szkła, 3× mniej fałszywych alarmów, znacznie stabilniejszy pod rozrzutem
sprzętu. Jedyny regres (cisza) z nawiązką zbity spadkiem na głośnych zdarzeniach.
