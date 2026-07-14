# Ściąga kalibracyjna v3 — sieć 15-neuronowa 7→4→3→1 (enkoder widmowy)

Źródło prawdy: `hw7_config.json` (model seed 2). To ludzko-czytelna wersja + procedura.
Wersja 14-neuronowa (6 kanałów) jest w `hw_config.json` / `kalibracja_sciaga.md` — patrz
sekcja "Którą wersję składać jutro".

## 0. Co się zmieniło względem v2 (14 neuronów) i po co

Martwa cecha `crest` wymieniona na dwie cechy **widmowe** `hf_lo`/`hf_hi` (udział energii
pasma górnego >~2.2 kHz, próg BEZWZGLĘDNY). Szkło ma trwale wysoki udział HF (4–10 kHz),
głośne nie-szkła (łomot/strzał/dzwon) nie. Efekt na **zbiorze testowym** (nigdy niewidzianym):

| metryka | v2 (14 neur.) | **v3 (15 neur.)** |
|---|---|---|
| test F1 (okna) | 0.487 | **0.609** |
| odporność F1 min (rozrzut trymerów) | 0.355 | **0.550** |
| recall szkła (reguła k=1) | 72% | 72% |
| **fałszywe alarmy łącznie (k=1)** | **63%** | **20%** |
| FA na głośnych: datasec / esc50 / voice | 71 / 65 / 66% | 23 / 19 / 8% |
| FA na cichym tle (notebooks) | 30% | 46% |

**Ten sam recall szkła, 3× mniej fałszywych alarmów, i dużo stabilniejszy pod rozrzutem
sprzętu** (min F1 0.55 vs 0.36 — kalibracja ręką jest znacznie wybaczliwsza, to kluczowe).
Jedyny regres: cisza (46% vs 30%), z nawiązką zbity spadkiem na głośnych zdarzeniach.
Uczciwie: to nadal zgrubna brama always-on; reaktor (LLM) weryfikuje i odrzuca fałszywki.

## 1. Zmiany sprzętowe względem v2

- **7. kanał**: dochodzi jedno wyjście Arduino **D8** (kanały 0..5 na D2..D7 jak dawniej,
  kanał 6 `hf_hi` na D8). Liczba płytek Lu.i bez zmian — **8** (H0–H3, G0–G2, D).
- **Firmware** `encoder_v2.ino` zaktualizowany: 1-pole highpass w ISR, `hf_ratio`,
  progi bezwzględne `hf_lo`>0.28 / `hf_hi`>0.35 z bramką zdarzenia. Kanały: nagłówek
  CSV `frame,s0..s6`. Budżet ISR: ~6 µs przy okresie próbki ~52 µs — z zapasem, ale
  **potwierdź timing na płytce** (toggle wolnego pinu na wejściu ISR + oscyloskop/logic).

## 2. Kanały wejściowe (v3)

| # | Pin | Kanał | Kodowanie | Co wykrywa |
|---|---|---|---|---|
| 0 | D2 | `peak` | z-score | obwiednia — ostry transient |
| 1 | D3 | `peak_cnt` | z-score | liczba mikro-szpilek w ramce |
| 2 | D4 | `cv` | z-score | współczynnik zmienności |
| 3 | D5 | `zcr` | z-score | zero-crossing rate — charakter widma |
| 4 | D6 | `flux` | z-score | dodatni przyrost log-RMS — atak |
| 5 | D7 | `hf_lo` | **próg bezwzgl. 0.28** | udział energii HF, czuły |
| 6 | **D8** | `hf_hi` | **próg bezwzgl. 0.35** | udział energii HF, glass-specyficzny |

`hf_lo`/`hf_hi` NIE mają adaptacyjnego floora — strzelają, gdy hf_ratio przekroczy próg
bezwzględny ORAZ ramka jest zdarzeniem (peak > 1.5×floor). Powód w `encoder_twin.py`.

## 3. Tabela nastaw (8 płytek, model seed 2)

Format synapsy: `wejście znak trymer% (test: n* impulsów do odpalenia)`.

### Warstwa H (kanały enkodera → H)
| płytka | LED (V_leak) | J1 | J2 | J3 |
|---|---|---|---|---|
| **H0** | 10.0% | peak **−** 13% (n*=3) | peak_cnt — **POMIŃ** (0%) | hf_lo **+** 57% (n*=1) |
| **H1** | 10.0% | cv **+** 49% (n*=2) | zcr **+** 35% (n*=2) | hf_hi **−** 42% (n*=2) |
| **H2** | 15.5% | peak **+** 100% (n*=1) | flux **+** 12% (n*=3) | hf_lo **+** 12% (n*=3) |
| **H3** | 22.4% | peak_cnt **−** 40% (n*=1) | flux **+** 15% (n*=2) | hf_hi **−** 100% (n*=1) |

τ: H0 (syn 27 / mem 113 ms), H1 (78/779), H2 (34/270), H3 (98/431).
Uwaga: `hf_hi` wchodzi HAMUJĄCO (znak −) do H1 i H3 — "mocny HF" wygasza te neurony;
`hf_lo` excytująco do H0/H2. To wyuczony podział ról, nie błąd.

### Warstwa G (wyjścia H → G)
| płytka | LED | J1 | J2 | J3 |
|---|---|---|---|---|
| **G0** | 17.1% | H0 **+** 75% (n*=1) | H1 **−** 100% (n*=1) | H2 — **POMIŃ** (0%) |
| **G1** | 10.0% | H1 **−** 14% (n*=3) | H2 **−** 9% (n*=4) | H3 **+** 63% (n*=1) |
| **G2** | 10.0% | H0 **+** 35% (n*=2) | H2 **−** 40% (n*=2) | H3 **+** 25% (n*=2) |

τ: G0 (19/418), G1 (51/158), G2 (79/667).

### Neuron decyzyjny D (G0/G1/G2 → D)
| płytka | LED | J1 | J2 | J3 |
|---|---|---|---|---|
| **D** | 26.4% | G0 **+** 29% (n*=2) | G1 **−** 100% (n*=1) | G2 **−** 21% (n*=2) |

τ: D (syn 20 / mem 328 ms). Pełne 3 wejścia z G0/G1/G2.

Dwie synapsy martwe (POMIŃ, zostaw na zerze): **H0.J2 (peak_cnt)** i **G0.J3 (H2)**.

## 4. Reguła dekodera

**Zalecane: k=1** (alarm gdy D strzeli choć raz) — recall 72%, FA łącznie 20%.
Jeśli tło zbyt gadatliwe: **k=2 w oknie 2.5 s** → recall 63%, FA 16%. Albo obniż pasek
LED płytki D o 1–2 działki (twardszy próg), bez wracania do treningu. D strzela na szkle
~15×/klip, na tle ~3.6× — reguła k rozróżnia po gęstości serii.

## 5. Procedura kalibracji, walidacja sim↔hw

Bez zmian względem `kalibracja_sciaga.md` sekcje 1/3/5 (kolejność H→G→D, test binarny
`pulses_to_fire`, `compare` sim↔hw). Test binarny synapsy: podaj n* impulsów 100 Hz
(tryb CALIB: `C <pin> <n> 100`), płytka ma odpalić na n*-tym.

## 6. Którą wersję składać jutro — 14 czy 15 neuronów

- **v3 (15 neur.) jest wyraźnie lepszy** (3× mniej FA, stabilniejszy), ALE wymaga:
  wgrania nowego `encoder_v2.ino`, podłączenia **D8** jako 7. kanału, i **potwierdzenia
  timingu ISR na płytce**. Płytek Lu.i tyle samo (8), dochodzi jedno wyjście MCU.
- **v2 (14 neur.) jest gotowy bez ryzyka firmware** (`hw_config.json`, 6 kanałów D2..D7).
- **Rekomendacja:** jeśli jest 30 min na wgranie firmware i sprawdzenie D8 przed składaniem
  — bierz v3, różnica w jakości jest duża. Jeśli chcesz zero niespodzianek sprzętowych —
  składaj v2, a v3 wgraj po zweryfikowaniu timingu. Model i tabela v3 są gotowe niezależnie.

## Reprodukcja
```
# enkoder v3 (7 kanałów) już w encoder_twin.py; przekoduj:
python encoder_twin.py build-manifest --manifest ../dataset/combined/manifest.csv \
    --root .. --out spikes_manifest7
# trening (zwycięzca: seed 2, pos_weight 1.0):
python snn_hw_pipeline.py train --data spikes_manifest7/train \
    --val-data spikes_manifest7/val --test-data spikes_manifest7/test \
    --epochs 100 --patience 15 --hat-frac 0.5 --seed 2 --pos-weight 1.0 \
    --out hw7_config.json --ckpt best7.pt
python eval_stream.py --ckpt best7.pt --data spikes_manifest7/test   # metryki klipowe
```
Checkpoint zwycięzcy: `best7.pt` (=hw7_s2.pt). Pozostałe seedy: `hw7_s1/s3.*`.
```
seed 2 pw1.0: test F1 0.609, rob min 0.55, k=1 glass 72% / FA 20%  (WYBRANY)
seed 1 pw1.0: test F1 0.571, rob min 0.53, k=1 glass 79% / FA 32%  (wyższy recall)
seed 3 pw1.5: test F1 0.592, rob min 0.53, k=1 glass 58% / FA 18%  (niższe FA)
```
