# Wymiana kanałów enkodera — instrukcja wykonania

Zadanie: dodać `hjorth_mobility` i `autocorr_lag1`, usunąć `peak_cnt` i `cv` (wymiana **pozycyjna**: kanał 1 i 2,
więc maski sieci `MASK_H…` i piny D2..D8 zostają bez zmian), zmieścić się w ISR na ATmega328P @16 MHz, pokazać A/B
na tym samym podziale i zgodność twin ↔ firmware.

**Oryginały nie są ruszane.** Wszystko jest w plikach `*_swap.*` i sterowane flagami. Przy flagach = 0 firmware jest
**bajt w bajt identyczny** z Twoim `encoder_v2.ino` (sprawdza `tools/run_predictions.sh`), a twin w trybie `baseline`
daje **bit w bit** te same spike'i co `encoder_twin.py` (`tests/test_twin_patch.py`).

---

## 0. Co już wiadomo (SYMULACJA cyklowa simavr + analiza asemblera) — a co musi potwierdzić płytka

Prawdziwy firmware (avr-gcc 7.3, `-Os`, rdzeń Arduino) uruchomiony w symulatorze, próbki ADC wstrzykiwane co 832 cykle.
Analiza statyczna i symulator zgadzają się w granicach 1%. **To nie jest pomiar na sprzęcie.**

| wariant firmware | ISR cykle min–max | µs max @16 MHz | % okresu 52 µs | Δ vs baseline |
|---|---:|---:|---:|---:|
| `baseline` (Twój `.ino`) | 585–624 | 39,0 | 75,0% | — |
| `swap` (2 nowe akumulatory, 64-bit) | 716–754 | 47,1 | 90,6% | +130 |
| `acc32` (akumulatory 32-bit) | 454–493 | 30,8 | 59,3% | −131 |
| `parity` (DC + HF round + EPS_FLOOR) | 500–539 | 33,7 | 64,8% | −85 |
| **`swap_full`** (swap + parity + acc32) | **501–539** | **33,7** | **64,8%** | **−85** |

Co z tego wynika (do potwierdzenia w krokach 1–2):

1. **Baseline już zajmuje ~75% budżetu.** Kompilator zapisuje ~33 rejestry w ISR (woła `__mulhisi3` i dwa razy `__adddi3`
   dla 64-bitowych `acc_sq`/`acc_hf_sq`). Samo dołożenie dwóch akumulatorów (+130 cykli) daje 90,6% — formalnie
   mieści się w 52 µs, ale bez zapasu.
2. **`acc32` odzyskuje dokładnie tyle, ile kosztują nowe akumulatory** (`192·1023² = 2,0·10⁸ < 2³²`, więc 64 bity
   są zbędne). `swap_full` jest tańszy od baseline o 85 cykli.
3. **Poprawki zgodności z twinem są niezależne od wymiany kanałów** (`parity`, `acc32`) — mają sens, nawet jeśli
   krok 3 pokaże, że nowe cechy nic nie dają.
4. **fs:** przy prescalerze 32 (linia 102 `.ino`) zegar ADC = 500 kHz, a jedna konwersja to 13 taktów ⇒ **~38,5 kHz**,
   nie 19 231 Hz (to daje prescaler 64). W symulacji przy okresie 416 cykli **żaden wariant nie działa** (ISR ≈ 98% CPU,
   20–31% próbek zgubionych, 0 przetworzonych ramek). To wynik z datasheetu + symulacji, **do pomiaru** (krok 1).
5. **Ramki „spóźnione":** w symulacji baseline ma 21% ramek z 193 próbkami (loop nie zdąża zabrać migawki przed
   następną próbką ⇒ siatka ramek dryfuje względem twina); `swap_full` ma 0%.
6. **Usuwanie DC w `.ino` ma martwą strefę ~32 LSB** (`>>9` na Q4): przy tle o kilku LSB estymata DC utyka, sygnał ma
   stały offset (zcr→0, peak zawyżony). To największe źródło rozjazdu z twinem.

---

## 1. Zawartość paczki

```
RUNBOOK.md                       ten plik
measurements.json                SZABLON na wyniki fizyczne (null = do zmierzenia) + sim_reference (symulacja)
predictions.json                 przewidywane cykle ISR wszystkich wariantów (generuje tools/run_predictions.sh)
esos_time_analysis_avr.py        zamiennik esos_time_analysis.py dla ATmega328P (model + pomiar + werdykt)
phase0_analysis.py               analiza kandydatów na prawdziwych nagraniach (Ty uruchamiasz)
firmware/encoder_v2_swap.ino     firmware z flagami (domyślnie = oryginał)
twin/encoder_twin_swap.py        twin z zestawem kanałów baseline|swap (zmienna ENCODER_CHANNEL_SET)
patches/*.patch                  te same zmiany jako łatki: `patch encoder_v2.ino < patches/encoder_v2_swap.patch`
tools/                           make_swap_*.py (generatory), isr_cycles.py, simharness.c, parity_test.py,
                                 capture_bench.py, build_fw.sh, run_predictions.sh, run_checks.sh, synth_audio.py
tests/                           test_twin_patch.py, make_synth_dataset.py
```

Wymagania: Python 3.10+ (numpy, scipy, pandas, librosa, soundfile, scikit-learn, pyserial). Narzędzia „bez płytki"
(`build_fw.sh`, `simharness`, `run_*.sh`) potrzebują Linuksa/WSL: `apt install gcc-avr avr-libc binutils-avr simavr
libsimavr-dev libelf-dev` oraz `git clone https://github.com/arduino/ArduinoCore-avr` (ścieżka: `ARDUINO_CORE=...`).
Odtworzenie moich weryfikacji jednym poleceniem: `tools/run_checks.sh encoder_v2.ino encoder_twin.py`.

### Flagi firmware (wszystkie domyślnie 0 / oryginał)

| flaga | działanie |
|---|---|
| `ENC_SET_SWAP=1` | kanał 1 = hjorth_mobility, kanał 2 = autocorr_lag1 (usuwa `acc_pk`, `spike_thr`, cv) |
| `ENC_PARITY=1` | włącza trzy poniższe naraz |
| `ENC_DC_FIX=1` | usuwanie DC bez martwej strefy (Q9, przesunięcia o bajt — tańsze niż oryginał) |
| `ENC_HF_ROUND=1` | zaokrąglanie w LP pasma górnego |
| `ENC_EPS_FLOOR=1` | `EPS` per kanał w z-score jak `EPS_FLOOR` w twinie (zamiast `1e-6`) |
| `ENC_ACC32=1` | `acc_sq`, `acc_hf_sq` jako `uint32` |
| `MOB_THR`, `AC_THR`, `MOB_FIRE_BELOW`, `AC_FIRE_BELOW` | progi bezwzględne nowych kanałów. **Domyślnie kanały MILCZĄ** — wstaw wartości z kroku 3 |
| `ENC_ADC_PRESCALER=64` | prescaler ADC 64 (fs = 19 231 Hz); domyślnie 32 |
| `ENC_BENCH=1` | tryb pomiarowy, polecenie `B` przez Serial |
| `ENC_ISR_PIN=1` | D9 wysoko na czas ISR (oscyloskop; +4 cykle; puls nie obejmuje prologu/epilogu) |
| `ENC_DEBUG_FEAT=1`, `ENC_BAUD` | wypisywanie wartości cech / prędkość UART (do testów parytetu) |

**Arduino IDE:** flag nie trzeba podawać w opcjach kompilatora — wpisz `#define ENC_BENCH 1` (itd.) w **pierwszych liniach**
`encoder_v2_swap.ino`, przed pierwszym `#include`. (`arduino-cli`: `--build-property "compiler.cpp.extra_flags=-DENC_BENCH=1"`.)

---

## 2. Krok po kroku

### KROK 1 — realne fs i budżet (PŁYTKA, ~10 min) — **najpierw to**

1. `encoder_v2_swap.ino` + `#define ENC_BENCH 1`, wgraj na Uno/Nano, **zamknij Monitor portu** (port musi być wolny).
2. `python3 tools/capture_bench.py --port COMx --variant baseline` (~13 s). Zapisuje do `measurements.json → board.baseline`.
3. Powtórz z dodatkowym `#define ENC_ADC_PRESCALER 64` → `--variant baseline_ps64`.

| | fs [Hz] (`fs_hz`) |
|---|---|
| prescaler 32 (oryginał) | ______ |
| prescaler 64 | ______ |

Interpretacja: **≈ 19 231** przy 32 ⇒ moja hipoteza o prescalerze fałszywa, idziemy dalej. **≈ 38 000** albo wartość
mniejsza, ale ≠ 19 231 (np. 25–27 kHz = ADC szybszy niż ISR nadąża) ⇒ prescaler 32 jest błędem: ustaw 64 na stałe
(jedna linia) i **wszystkie dalsze pomiary rób z 64**. Wpisz też `fs_check.wniosek` w `measurements.json`.

### KROK 2 — ISR na płytce (PŁYTKA, ~30 min)

Trzy buildy (z prescalerem wybranym w kroku 1), po jednym `capture_bench.py` na każdy:

| wariant | `#define` na górze pliku | `--variant` |
|---|---|---|
| baseline | `ENC_BENCH 1` | `baseline` |
| parity | `ENC_BENCH 1`, `ENC_PARITY 1` | `parity` |
| swap_full | `ENC_BENCH 1`, `ENC_SET_SWAP 1`, `ENC_PARITY 1`, `ENC_ACC32 1` | `swap_full` |

Metoda „wolnych cykli": puste iteracje pętli przy wyłączonym i włączonym przerwaniu ADC ⇒ ułamek CPU zabrany przez
ISR (wraz z prologiem/epilogiem). W symulatorze błąd tej metody wyszedł < 1%. Wypełnij (skrypt robi to sam):

| wariant | fs [Hz] | ISR CPU % | ISR µs śr. | cykle śr. | p1 ramek / spóźn. | p1 proc śr./max [ms] | p2 proc śr./max [ms] |
|---|---|---|---|---|---|---|---|
| baseline | ___ | ___ | ___ | ___ | ___ / ___ | ___ / ___ | ___ / ___ |
| parity | ___ | ___ | ___ | ___ | ___ / ___ | ___ / ___ | ___ / ___ |
| swap_full | ___ | ___ | ___ | ___ | ___ / ___ | ___ / ___ | ___ / ___ |
| *sim: baseline* | 19 230 | 72,4 | 37,7 | 602,5 | 448 / 94 | 6,76 / 6,92 | 4,25 / 4,46 |
| *sim: swap_full* | 19 230 | 62,3 | 32,4 | 518,5 | 448 / 0 | 4,79 / 4,93 | 2,85 / 2,97 |

`p1` = normalna praca z wypisywaniem linii (115200), `p2` = bez wypisywania; „spóźn." = ramki z > 192 próbek.

Potem: `python3 esos_time_analysis_avr.py` — tabela przewidywanie vs pomiar + werdykt. *(Opcjonalnie: oscyloskop/analizator
z `ENC_ISR_PIN 1` na D9 — patrz uwaga o prologu w tabeli skryptu.)*

**Bramki:** (a) przewidywane max ≤ 52 µs (przy zmierzonym fs: ≤ 1/fs); (b) zmierzone cykle śr. w przedziale przewidywań
±3% — jeśli nie, model do poprawy, nie ufaj przewidywaniom; (c) `late = 0`; (d) `proc max` < 80% okresu ramki.
**Kryterium 1 z zadania** („ISR ≤ 52 µs zmierzone na ATmega328P") = wiersz `swap_full`, kolumna „ISR µs śr." oraz max z analizy.

### KROK 3 — faza 0: czy nowe cechy coś wnoszą (DANE, ~15–30 min)

```bash
python3 phase0_analysis.py --manifest dataset/versions/v2.0.0/manifest.csv --root . \
    --twin twin/encoder_twin_swap.py --jobs 8 --max-per-cell 400 --out phase0_results.json
# najlepiej z --gain <produkcyjne globalne wzmocnienie>; bez niego liczone z 300 plików train
```

Skrypt liczy na **train** (front-end twina: 19 231 Hz, kody ADC, rozgrzany stan): Cohen's d (zastępuje liczby z README,
które pochodziły z banku bez resamplingu), korelacje Spearmana, wartość przyrostowa (LR i GBM, zestawy A/E/B/C/D),
**sparowany bootstrap po `group_id`** dla ΔAUC klipowego oraz progi bezwzględne z siatką kwantyli. Ocena na **val**.

Jak czytać (reguły, nie automat):

| obserwacja | wniosek |
|---|---|
| `\|ρ(mobility, autocorr)\|` > 0,9 | to prawie ta sama informacja (`mobility² = 2(1−ρ₁)(1+cv²)/cv²`) — rozważ tylko jeden kanał (wtedy zestaw C lub D) |
| ΔAUC(B−A) i ΔAUC(B−E) z CI obejmującym 0 (lub < 0) | brak dowodu na zysk z wymiany; zostaje `parity` + `acc32` (zysk czasowy i zgodność) |
| ΔAUC(B−A) z CI nad zerem | wymiana ma sens → krok 5 |
| neg. rate na val ≫ na train | progi przeuczone na train, zmniejsz cel `--neg-target` |

Wpisz `recommended_thresholds` do `measurements.json → phase0` oraz: twin — zmienne środowiskowe `ENCODER_MOB_THR`,
`ENCODER_AC_THR` (albo stałe w pliku); firmware — `#define MOB_THR <x>f` / `AC_THR <y>f`. Jeśli kierunek wyszedł inny
niż domyślny (mobility: powyżej progu; autocorr: poniżej), ustaw `MOB_FIRE_BELOW` / `AC_FIRE_BELOW` (twin: stałe
`MOB_FIRE_BELOW`, `AC_FIRE_BELOW`).

| wynik fazy 0 | wartość |
|---|---|
| d(mobility), d(autocorr) | ___ , ___ |
| ρ(mobility, autocorr), ρ(mobility, hf_ratio) | ___ , ___ |
| ΔAUC klipowy B−A (LR / GBM), CI95 | ___ [___, ___] / ___ [___, ___] |
| mob_thr, ac_thr (kierunek) | ___ , ___ |
| decyzja (oba / jeden / żaden) | ______ |

### KROK 4 — zgodność twin ↔ firmware (kryterium 3, PC, bez płytki)

```bash
python3 tools/parity_test.py --wav <plik_testowy.wav> --gain <G> --variant all --mob-thr <x> --ac-thr <y> --json parity.json
```

Firmware działa w symulatorze cyklowym; **te same całkowite kody ADC** trafiają do firmware i do twina (twin nie
zaokrągla kodów — w moim teście zmienia to spike'i w 0,2–1,4% ramek, więc podajemy obu stronom zaokrąglone).
Porównanie per ramka: wartości cech (mediana błędu względnego) i bity s0..s6. „Zdarzeniowe" = rms ≥ 20 LSB
(od nich zależą spike'i przez bramkę `hf_gated`).

Wynik na audio **syntetycznym** (14 s, twin/firmware, symulator) — punkt odniesienia, nie wynik na Twoich danych:

| wariant | zgodność spike'ów (7 kanałów naraz), wszystkie / zdarzeniowe | uwaga |
|---|---|---|
| baseline vs twin | 86,7% / 57,8% | martwa strefa DC, `EPS`, zaokrąglenia |
| parity vs twin | 97,7% / 98,7% | |
| swap_full vs twin (tło 2 LSB) | 98,2% / 100% | mediana błędu `mobility` 0,10%, `autocorr` 0,02% |
| swap_full vs twin (tło 10 LSB) | 98,6% / 100% | s1, s2: 0 rozjazdów |
| swap_full vs twin (tło 26 LSB) | 99,85% / 99,8% | |

Reszta rozjazdów to ramki tuż przy progu (float32 vs float64, całkowite `x` vs ułamkowe). „Zgodne z dokładnością do
rozsądnego ε" proponuję zdefiniować tak: **≥ 99% zgodnych ramek zdarzeniowych i mediana błędu względnego cech na
ramkach zdarzeniowych ≤ 1%** (na własnym pliku — wpisz wynik):

| plik | wariant | zgodność zdarz. [%] | mediana rel. błędu: peak / zcr / hf / mobility / autocorr |
|---|---|---|---|
| ______ | parity | ___ | ___ / ___ / ___ / – / – |
| ______ | swap_full | ___ | ___ / ___ / ___ / ___ / ___ |

Uwaga: test jest „wyrównany" (ramki dokładnie po 192 próbki). Na sprzęcie loop reaguje z opóźnieniem — stąd metryka
`late` z kroku 2; dla `swap_full` w symulacji była 0.

### KROK 5 — A/B na `spikes_v2` (PC + trening; wymaga decyzji z kroku 3)

Oba ramiona z **tego samego** pliku twina, różni je tylko zmienna środowiskowa. Bazowy zbiór ważny to v2.0.0
(`spikes_manifest7` nie używamy — patrz notatka o jego wadach w przewodniku).

```bash
# ramię baseline
ENCODER_CHANNEL_SET=baseline python3 twin/encoder_twin_swap.py build-manifest --manifest dataset/versions/v2.0.0/manifest.csv \
    --root . --out architecture_14_neurons_patryk_09_07/spikes_v2_base --warmup-seconds 30
# ramię swap (progi z kroku 3)
ENCODER_CHANNEL_SET=swap ENCODER_MOB_THR=<x> ENCODER_AC_THR=<y> python3 twin/encoder_twin_swap.py build-manifest \
    --manifest dataset/versions/v2.0.0/manifest.csv --root . --out architecture_14_neurons_patryk_09_07/spikes_v2_swap --warmup-seconds 30
# (PowerShell: $env:ENCODER_CHANNEL_SET="swap"; $env:ENCODER_MOB_THR="..." )

# trening: ten sam zestaw seedów w obu ramionach (np. 0..4), MASK_H bez zmian
for S in 0 1 2 3 4; do for ARM in base swap; do
  python3 snn_hw_pipeline.py train --data spikes_v2_$ARM/train --val-data spikes_v2_$ARM/val --test-data spikes_v2_$ARM/test \
      --epochs 100 --patience 15 --hat-frac 0.5 --seed $S --pos-weight 1.0 --out hw_${ARM}_s$S.json --ckpt ${ARM}_s$S.pt
  python3 eval_stream.py --ckpt ${ARM}_s$S.pt --data spikes_v2_$ARM/test
done; done
```

Uwagi: w `snn_hw_pipeline.py` (linia 60) dla ramienia swap zmień **nazwy** w `CHANNELS` na
`["peak","hjorth_mobility","autocorr_lag1","zcr","flux","hf_lo","hf_hi"]` (dotyczy tylko eksportu nastaw; `CH_IN`=7 i maski
bez zmian). Nie wybieraj „najlepszego seeda" per ramię — raportuj wszystkie.

| seed | baseline: recall@1 / @6 FA/h, clip-F1 | swap: recall@1 / @6 FA/h, clip-F1 |
|---|---|---|
| 0 | ___ / ___ , ___ | ___ / ___ , ___ |
| 1 | ___ / ___ , ___ | ___ / ___ , ___ |
| 2 | ___ / ___ , ___ | ___ / ___ , ___ |
| 3 | ___ / ___ , ___ | ___ / ___ , ___ |
| 4 | ___ / ___ , ___ | ___ / ___ , ___ |
| mediana (zakres) | ___ | ___ |
| Δ (swap − baseline), sparowany bootstrap po `group_id`, CI95 | ______ |

Skala szumu: test ma **96 niezależnych grup pozytywnych** (884 pliki), a 2,76 h negatywów daje ok. 0,36 FA/h na jeden
fałszywy alarm. Różnica poniżej kilku punktów recallu przy jednym seedzie jest nieodróżnialna od szumu. Sparowanego
bootstrapu nie napisałem, bo wymaga per-klipowych wyników z `snn_pipeline/stream_eval.py`, którego nie widziałem
(patrz sekcja 4).

### KROK 6 — domknięcie

1. Podmień `esos_time_analysis.py` na `esos_time_analysis_avr.py`; w README dopisz, że „1,7% CPU" dotyczy Cortex-M4F @64 MHz,
   a kanały FFT nie są wycenione dla ATmega328P.
2. Firmware docelowy = wariant z kroku 2, który spełnia bramki (kolejność preferencji: `swap_full` jeśli krok 3/5 wspierają
   wymianę, inaczej `parity` + `acc32` bez `ENC_SET_SWAP`). Przed wgraniem na Lu.i usuń `ENC_BENCH`.
3. Wpisz progi i wyniki do `measurements.json`; zachowaj `bench_*.log`.

---

## 3. Mapowanie na kryteria akceptacji

| kryterium | jak spełnione | status |
|---|---|---|
| ISR ≤ 52 µs, zmierzone na ATmega328P | krok 2, `swap_full` | przewidywanie 33,7 µs max; **pomiar: ___** |
| clip-F1 / FA-h przed i po, ten sam podział | krok 5 (`spikes_v2_base` vs `_swap`, te same seedy) | **czeka na trening** |
| twin ↔ firmware na ustalonym pliku | krok 4 | symulacja: 99,8–100% ramek zdarzeniowych; **własny plik: ___** |

## 4. Ograniczenia i czego brakuje

* **Symulator to nie sprzęt.** ADC jest podmieniony (próbki wstrzykiwane co 832 cykle), nie modeluję szumu zasilania ani
  jitteru. Cykle procesora są dokładne; czasu ADC nie testowałem — od tego jest krok 1.
* **Toolchain:** użyłem avr-gcc 7.3.0 (Atmel 3.7.0) z `-Os`. Arduino IDE ma zbliżoną, ale nie identyczną wersję (może
  dokładać `-flto`); kod ISR może różnić się o kilka procent — dlatego krok 2 porównuje pomiar z przedziałem ±3%.
* **Testy na audio syntetycznym.** Progi z `--mob-thr 1.95 --ac-thr 0.31` w moich testach to mediany syntetycznych cech,
  bez znaczenia dla Twoich danych. `phase0_analysis.py` przetestowałem tylko na sztucznym zbiorze w formacie manifestu.
* **Brakuje mi `snn_pipeline/stream_eval.py`** (i `stream_eval_torch.py`), żeby napisać sparowane porównanie A/B po
  `group_id` na wynikach `eval_stream.py`. Po jego wgraniu dopiszę `ab_compare.py`.
* **Ramię z GA** nie jest objęte. `run_search.py` ma błąd (`args.stream_boot` bez definicji w argparse — tryb `real`
  się wywróci), a wyniki dotychczasowego GA są na nieaktualnym zbiorze.
* **Twin nie zaokrągla kodów ADC** (wykryte; wpływ ~1% ramek). Nie zmieniałem — zmiana przebudowałaby każdy zbiór.
* **Decyzje projektowe, które podjąłem:** wymiana pozycyjna; nowe kanały z progiem bezwzględnym + bramką `hf_gated`
  jak hf_lo/hf_hi (z-score odwracał sygnał dla cech poziomowych); mobility wg `feature_metrics.py` (przez var(|x|));
  nowe cechy ciągłe po granicach ramek (różnica względem definicji w banku: wyraz brzegowy rzędu 1/192).
* **Odłożone na Twoją prośbę:** ustalenia z Karoliną oraz definicja „sieć wykorzystuje wejścia".
