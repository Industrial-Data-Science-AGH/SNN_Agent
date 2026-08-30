# WYNIKI — dziennik przebiegów GA (dobór topologii SNN Lu.i)

Miejsce na podsumowania wyników z **różnych wersji kodu**. Dopisuj nowe przebiegi
na górze sekcji „Dziennik" (najnowsze pierwsze). Trzymaj jeden wiersz w tabeli
zbiorczej + opcjonalnie blok szczegółów pod spodem.

> **Jak czytać liczby — uważaj na spójność:** wynik zależy od **(dataset, k,
> metryka)**. Nie porównuj `AP@k=2` z `clip_f1@k=1` ani `spikes_manifest7` ze
> `spikes_ext`. Zawsze notuj te trzy rzeczy w wierszu. Patrz „Definicje" na dole.

> ⚠️ **Wszystkie wiersze z kolumną `spikes_manifest7` są WYCOFANE (issue #36):**
> ten artefakt ma przeciek między splitami (194/194 miksów VOICe z testu jest
> w treningu) i `clock_tick` zamiast `glass_breaking` w klasie pozytywnej ESC-50.
> Nowe przebiegi notuj z artefaktem `spikes_v2` i wersją zbioru `v2.0.0`.

---

## Punkt odniesienia — sieć oryginalna (ręczna, `snn_hw_pipeline.py`)

> ⚠️ **WYCOFANE (issue #36).** Liczby w tej sekcji pochodzą z artefaktu
> `spikes_manifest7`, który ma dwie niezależne wady: 194 z 194 miksów VOICe
> obecnych w teście są też w treningu (VOICe daje 95.6% klipów szkła, więc wynik
> jest mierzony niemal wyłącznie na nagraniach, na których model się uczył), oraz
> wszystkie 40 plików ESC-50 w klasie pozytywnej to target 38 = `clock_tick`,
> nie 39 = `glass_breaking`. Nie cytuj ich. Przeliczenie na `spikes_v2`
> (zbudowanym z `dataset/versions/v2.0.0`, podział grupowy, 0/69 wspólnych
> miksów) jest w `models/WYNIKI_v2.md`.

Z `architecture_14_neurons_patryk_09_07/hw_config.json` (model 7→4/8→3→1,
`spikes_manifest7`, pełny trening HAT→QAT, dekoder **k=1**):

| metryka | wartość |
|---|---|
| val F1 (okienkowe, `evaluate`, k≈1) | **0.512** (rec 0.630 / prec 0.431) |
| test F1 (okienkowe) | 0.487 |
| val clip **k=1** | recall **0.726** / FA 0.636 |
| val clip **k=2** | recall **0.039** (praktycznie martwe) |
| val clip k=3 | recall 0.024 |

**Wniosek bazowy:** to detektor **k=1** — seria ≥2 spików D NIE rozdziela szkła od
tła dla tej topologii. Dlatego w GA trzymamy **k=1** spójnie (strata/eval/fitness).

---

## Definicje i konwencje (żeby wiersze były porównywalne)

- **k** — reguła dekodera: alarm, gdy neuron D da **≥ k** spików w oknie. Trzymamy **k=1**.
- **metryka fitness** — `clip_f1` (spójna z k) albo `ap` (bezprogowa, mniej szumna).
  Uwaga: `fitness` w JSON/CSV = metryka − `fanout_penalty·śr_fanout` − `feature_penalty·#cech`.
- **fitness ≠ czysta metryka** — zawiera drobne kary parsymonii. Do porównań podawaj też surową metrykę, jeśli ją masz z logu `[eval]`.
- **dataset**:
  - `spikes_manifest7` — 7 kanałów HW, oryginalny zbiór (ESC-50/notebooks/VOICe). **Brak źródłowego audio w repo.**
  - `spikes_ext` — 14 kanałów (7 HW + 7 z banku), zbudowany z `voice_extracted` (VOICe: glass vs hard_negative). Osobny zbiór — **nie porównywać 1:1** z oryginałem.
- **topologia** — zapis `layer_sizes`, np. `7-3-2-1` = 7 wejść → 3 → 2 → 1 (decyzyjny).
- **cechy użyte** — które kanały encodera GA faktycznie wpiął (wynik selekcji cech; w `spikes_ext` po nazwach).
- **Znany błąd w starym kodzie:** `run_search` drukował „clip-F1", a wartość była **AP** (przy `--metric ap`). Wiersze sprzed poprawki oznaczaj `metryka=ap(!label)`.

---

## Tabela zbiorcza

| data | wersja / branch | dataset | k | metryka | seedy | N | fitness | topologia | cechy użyte | uwagi |
|---|---|---|---|---|---|---|---|---|---|---|
| 2026-08-24 | feat/testing-ideas (stary) | spikes_manifest7 | 2 | ap(!label) | 1 | 6 | 0.553 | 7-3-2-1 | 7 HW (wszystkie) | demo; krzywa płaska (`wyniki_demo.json`) |
| 2026-08-24 | feat/testing-ideas (stary) | spikes_manifest7 | 2 | ap(!label) | 1 | 10 | 0.563 | 7-7-2-1 | 7 HW (wszystkie) | demo; +0.01 vs N=6 → parsymonia woli N=6 |
| 2026-08-24 | feat/testing-ideas (stary) | spikes_manifest7 | 2 | ? | 1 | 6 | 0.495 | 7-5-1 | 7 HW | `wyniki_test.json`, config nieznany |
| 2026-08-24 | feat/testing-ideas (stary) | spikes_manifest7 | 2 | ? | 1 | 8 | 0.486 | 7-7-1 | 7 HW | `wyniki_test.json`, config nieznany |
| — | (synth, nie-real) | — | — | synth | — | 8 | 0.988 | 7-4-3-1 | — | tylko test mechaniki GA (`wyniki_synth.json`), NIE jakość |
| 2026-08-24 | k1+selekcja (probe) | spikes_ext | 1 | clip_f1 | 1 | 8 | **0.611** | 7-5-2-1 | peak,peak_cnt,flux,hf_lo,hf_hi | A/B baseline, tylko 7 HW (`--channels-head 7`); tani budżet 8ep/pop6/gens3 |
| 2026-08-24 | k1+selekcja (probe) | spikes_ext | 1 | clip_f1 | 1 | 8 | **0.665** | 14-2-2-3-1 | peak,hf_hi,**crest**,**spectral_flatness** | A/B pełne 14 kan.; **wygrywa +0.054**, wpiął 2 NOWE cechy |
| _(wklej nowy wiersz tu)_ | | | | | | | | | | |

---

## Kalibracja budżetu epok (referencja)

`calibrate.py`, topologia testowa 7-3-2-2-1, `spikes_manifest7`, metryka=ap, k=2
(stary kod). Z `calibrate-results.txt`:

| epoki | loss_end | AP śr±std | clipF1 śr±std | wniosek |
|---|---|---|---|---|
| 2–8 | 3.3→3.1 | 0.42–0.57 **±0.31** | 0.34–0.36 ±0.25 | szum (loss stoi, ogromna wariancja) |
| **12** | **2.29** | 0.346 **±0.018** | 0.501 ±0.060 | **wybór** — uczy się, mała wariancja |
| 16–20 | 1.84→1.78 | 0.34 ±0.03 | 0.50 ±0.06 | loss spada, metryka stoi = overfitting |

**Budżet: 12 epok proxy-treningu.** (Do przeliczenia dla `spikes_ext` — inny zbiór.)

---

## Dataset `spikes_ext` — separacja kanałów (firing-rate glass − tło)

Build 2026-08-24, 8876 plików, autokalibracja progów `--bg-rate 0.08`. Z `build_ext.log`:

| kanał | Δ (pkt proc.) | | kanał | Δ |
|---|---|---|---|---|
| spectral_centroid | **+8.8** | | hf_lo (najlepszy HW) | +5.6 |
| hjorth_mobility | **+8.7** | | hf_hi | +3.1 |
| autocorr_lag1 | **+8.5** | | crest / curve_length | +1.7 / +1.3 |
| spectral_flatness | **+8.5** | | peak/peak_cnt/cv/zcr/flux | ~0 |
| band_energy_low | +7.2 | | | |

Wniosek: 5 nowych cech separuje ~2× lepiej niż najlepszy kanał HW; 4 kanały HW
(peak, peak_cnt, cv, flux) to praktycznie szum — kandydaci do porzucenia przez GA.

---

## Dziennik przebiegów (najnowsze na górze)

<!-- SZABLON — skopiuj blok poniżej dla każdego nowego przebiegu -->
<!--
### YYYY-MM-DD — <krótki tytuł / wersja kodu / commit>

- **Komenda:** `...`
- **Dataset / k / metryka / seedy / epoki / pop / gens:** ...
- **Wyniki per N:**
  | N | fitness | surowa metryka | topologia | cechy użyte | krzywa (gen 0→ost.) |
  |---|---|---|---|---|---|
  | 6 |  |  |  |  |  |
  | 8 |  |  |  |  |  |
- **Wybór (parsymonia eps=…):** N=…
- **Obserwacje / wnioski:** ...
-->

### (tu wklejaj nowe przebiegi)

### 2026-08-24 — A/B: czy mocniejsze cechy pomagają? (7 HW vs 14 kan., spikes_ext, k=1)

- **Cel:** ten sam zbiór/val/k, różnica **tylko w puli cech** → izoluje wpływ bogatszego encodera.
- **Config (tani PROBE):** `spikes_ext`, k=1, metryka=clip_f1, epoki 8 (nie 12!),
  num-samples 1500, pop 6, gens 3, seedy 1, N=8, feature-penalty 0.005.
- **Wyniki:**
  | wariant | fitness | najlepszy surowy clipF1 | topologia | cechy użyte | krzywa | ocen | czas |
  |---|---|---|---|---|---|---|---|
  | 7 HW (`--channels-head 7`) | 0.611 | ~0.654 | 7-5-2-1 | peak, peak_cnt, flux, hf_lo, hf_hi (5) | 0.60→0.611 | 15 | 725 s |
  | **14 kanałów** | **0.665** | **~0.703** | 14-2-2-3-1 | peak, hf_hi, **crest**, **spectral_flatness** (4) | 0.651→0.665 | 10 | 541 s |
- **Wniosek:** pełne 14 kanałów **wygrywa o +0.054 fitness (~+0.05 clipF1)**, a zwycięzca
  **wpiął 2 nowe cechy** (`crest`, `spectral_flatness`) i **porzucił większość kanałów HW**
  (peak_cnt, cv, zcr, flux). To bezpośredni dowód, że mocniejsze cechy pomagają, a stare HW
  są w dużej mierze zbędne — dokładnie jak sugerował Cohen's d z banku.
- **Zastrzeżenia:** tani, zaszumiony budżet (8 ep zamiast 12, 1 seed, jedno N) → wynik
  KIERUNKOWY. Do potwierdzenia: pełny bieg 12 ep, N∈{6,8,10}, fitness-seeds≥3.
  Dodatkowo: `spikes_ext` to VOICe (glass vs hard_negative), NIE porównywać z oryginałem 0.51.
- **Pliki:** `wyniki_ext_hw7.json/csv`, `wyniki_ext_14.json/csv`, log `ab_run.log`.
- **Jak odpalić PEŁNE potwierdzenie** (12 epok, N∈{6,8,10}, 3 seedy; kilka godzin
  na CPU — najlepiej GPU). Z katalogu `ga_neuron_search` (venv = `SNN`; jeśli masz
  `.venv`, podmień ścieżkę):
  ```
  :: BASELINE — 7 HW
  ..\SNN\Scripts\python.exe run_search.py --neurons 6 8 10 --mode real ^
    --arch-dir ..\architecture_14_neurons_patryk_09_07 ^
    --data spikes_ext\train --val-data spikes_ext\val ^
    --channels-head 7 --k 1 --metric clip_f1 --epochs 12 --pop 12 --gens 8 ^
    --fitness-seeds 3 --feature-penalty 0.005 --out wyniki_ext_hw7_full

  :: NOWE — 14 kanałów (GA sam wybiera)
  ..\SNN\Scripts\python.exe run_search.py --neurons 6 8 10 --mode real ^
    --arch-dir ..\architecture_14_neurons_patryk_09_07 ^
    --data spikes_ext\train --val-data spikes_ext\val ^
    --k 1 --metric clip_f1 --epochs 12 --pop 12 --gens 8 ^
    --fitness-seeds 3 --feature-penalty 0.005 --out wyniki_ext_14_full
  ```
  Porównaj najlepsze `clip_f1@k=1` z obu + `features_used` zwycięzcy 14-kan.
  W tle (PowerShell): dopisz `*> wyniki_ext_full.log 2>&1` i odpal przez
  `Start-Process`. Krzywe: `..\SNN\Scripts\python.exe plot_history.py wyniki_ext_14_full.json`.


---

## Historia zmian kodu (kontekst do interpretacji wierszy)

- **2026-08-24 — v-k1+selekcja cech:** dekoder ujednolicony na **k=1** (strata/eval/
  fitness/CLI); poprawka mylnej etykiety „clip-F1"→faktyczna metryka; **selekcja
  cech przez GA** (koniec wymuszania pokrycia, mutacja `mut_drop_feature`, kara
  `--feature-penalty`); pula kanałów z danych (`configure_features`); naprawa
  ślepego zaułka w `repair` (round-robin). Nowy builder `build_ext_dataset.py`
  → zbiór 14-kanałowy `spikes_ext`.
- **wcześniej (stary kod):** metryka domyślnie `ap` przy k=2; `run_search`
  drukował ją jako „clip-F1" (mylące). Wyniki `wyniki_demo/test.json` pochodzą stąd.
