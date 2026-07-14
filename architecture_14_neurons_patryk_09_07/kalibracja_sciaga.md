# Ściąga kalibracyjna — Delta Spike 6→4→3→1 (składanie + strojenie płytek Lu.i)

Źródło prawdy: `hw_config.json` (model **A** ze sweepu). Ten plik to jego ludzko-czytelna
wersja + procedura. Jeśli coś się rozjeżdża, wierz `hw_config.json`.

## 0. Co masz w ręku (bądź szczery na demie)

Model wytrenowany na 13.5 tys. plików (VOICe + ESC-50 + PT_DATASET + notebooks), z gotowym
podziałem train/val/test **po plikach** (bez przecieku). Wynik na **zbiorze testowym, którego
model nigdy nie widział**:

| poziom | metryka | wartość |
|---|---|---|
| okna 2 s | recall / precision / F1 | 0.56 / 0.43 / **0.487** |
| klipy (reguła k=1) | wykryte szkło | **72 %** |
| klipy (reguła k=1) | fałszywy alarm na **cichym** tle | ~30 % klipów |
| klipy (reguła k=1) | fałszywy alarm na **głośnych** zdarzeniach (strzały, dzwony, syreny) | 65–71 % |
| odporność | F1 pod rozrzutem trymerów (±½ działki) | 0.48 śr., 0.36 min |

**Uczciwa interpretacja:** to działająca **zgrubna brama always-on**, wyraźnie lepsza od losu
(precyzja 0.43 przy 10 % pozytywów to ~4× powyżej przypadku). Dobrze oddziela szkło od **ciszy**,
słabo od **innych głośnych transientów** — bo ogranicza ją enkoder (6 ręcznych cech), nie trening.
W docelowej architekturze to nie problem: analog jest tanią bramą, a reaktor (LLM) weryfikuje audio
i odrzuca fałszywe wybudzenia. **Kosztowny błąd to przeoczone szkło (28 %), nie fałszywy alarm.**

Cel jutra nie jest „model 99 %" — celem jest **zweryfikować cały pipeline sim↔sprzęt** (Faza C/D):
że wyeksportowane trymery dają na płytce te same spiki co symulacja. Jakość modelu podbijemy potem,
poprawiając enkoder (patrz sekcja 6).

## 1. Kolejność strojenia — ZAWSZE od dołu do góry

Nie strojysz D, dopóki G nie działa; nie strojysz G, dopóki H nie działa. Inaczej gonisz błąd,
który propaguje się z poprzedniej warstwy.

1. **Warstwa H** (H0–H3): wejścia to kanały enkodera (peak, peak_cnt, crest, cv, zcr, flux).
2. **Warstwa G** (G0–G2): wejścia to wyjścia H.
3. **Neuron D**: wejścia to G0/G1/G2.

Dla każdej płytki: ustaw **pasek LED (V_leak)** na podaną wartość, potem **każdy trymer** na %,
potem **przełącznik znaku** (+/−). Test poprawności: sekcja 3.

## 2. Tabela nastaw (8 płytek)

Trymer % = pozycja potencjometru (0–100 % skali). "test: N imp." = ile impulsów 100 Hz na TĘ jedną
synapsę ma odpalić neuron ze spoczynku — mierzalne kryterium z Fazy C (patrz sekcja 3).

### Warstwa H (kanały enkodera → H)

| płytka | pasek LED | J1 | J2 | J3 |
|---|---|---|---|---|
| **H0** | 10 % | peak, **−**, 51.9 %, (1 imp.) | peak_cnt, **+**, 36.3 %, (2 imp.) | crest, **−**, 15.6 %, (3 imp.) |
| **H1** | 25.1 % | cv, **+**, 100 %, (1 imp.) | zcr, **−**, 85.7 %, (1 imp.) | flux, **+**, 14.3 %, (2 imp.) |
| **H2** | 10 % | peak, **+**, 10.7 %, (3 imp.) | cv, **−**, 53.6 %, (1 imp.) | flux, **+**, 21.4 %, (2 imp.) |
| **H3** | 10 % | peak_cnt, **+**, 22.2 %, (2 imp.) | crest — **POMIŃ** (0 %, nieaktywna) | zcr, **−**, 72.2 %, (1 imp.) |

τ: H0 (syn 49.9 / mem 74.5 ms), H1 (46.4 / 419.3), H2 (34.8 / 340.3), H3 (16.4 / 229.1).

### Warstwa G (wyjścia H → G)

| płytka | pasek LED | J1 | J2 | J3 |
|---|---|---|---|---|
| **G0** | 10 % | H0, **+**, 90.8 %, (1 imp.) | H1, **−**, 18.2 %, (3 imp.) | H2, **+**, 40.8 %, (2 imp.) |
| **G1** | 10 % | H1, **−**, 43.3 %, (2 imp.) | H2, **+**, 34.7 %, (2 imp.) | H3, **+**, 13.0 %, (3 imp.) |
| **G2** | 39.8 % | H0, **−**, 100 %, (1 imp.) | H2, **−**, 18.2 %, (1 imp.) | H3, **+**, 54.5 %, (1 imp.) |

τ: G0 (14.5 / 106.3 ms), G1 (34.0 / 159.2), G2 (22.1 / 360.1).

### Neuron decyzyjny D (G0/G1/G2 → D)

| płytka | pasek LED | J1 | J2 | J3 |
|---|---|---|---|---|
| **D** | 35.7 % | G0, **−**, 12.5 %, (2 imp.) | G1, **−**, 100 %, (1 imp.) | G2, **+**, 50 %, (1 imp.) |

τ: D (syn 43.5 / mem 57.6 ms). D jest celowo **szybki** (mem 57 ms) — sygnał decyzji nie ma zanikać.
Ma pełne 3 wejścia z G0/G1/G2, zgodnie z założeniem „przedostatnia warstwa = 3 neurony".

## 3. Test poprawności każdej płytki (Faza C, binarny — nie „na oko")

Kolumna "N imp." w tabeli to gotowy test. Dla synapsy o teście „2 imp.":

1. Odłącz pozostałe wejścia (albo ustaw ich trymery na 0).
2. Podaj na TĘ synapsę serię impulsów TTL 100 Hz (co 10 ms) z generatora/Arduino.
3. Neuron ma odpalić (dioda J4 / 7. dioda paska) **dokładnie na N-tym impulsie**, nie wcześniej,
   nie później. Jeśli odpala za wcześnie → trymer za wysoko lub pasek LED za wysoko. Za późno /
   wcale → odwrotnie.
4. „nie odpala sam" (G-brak / H3.J2) = synapsa ma zostać na zerze, jest nieaktywna w modelu.

Zapisuj zmierzone N do porównania z symulacją (Faza D).

## 4. Reguła dekodera (firmware na J4 płytki D)

**Zalecane: k = 1** — alarm, gdy neuron D strzeli choć raz. To jedyny punkt pracy z użytecznym
recall (72 %). Serie ≥2 spików D **nie** rozdzielają szkła od tła (sprawdzone: szkło 1.9 vs tło
1.1 spika/klip — za blisko), więc k≥2 zabija recall do ~3 %. Nie komplikuj dekodera.

Jeśli na demie tło okaże się zbyt „gadatliwe", **nie zmieniaj wag** — obniż pasek LED (V_leak)
płytki D o 1–2 działki (twardszy próg). To jedyne pokrętło, którym w terenie handlujesz
czułość ↔ fałszywe alarmy, bez powrotu do treningu.

## 5. Walidacja sim↔sprzęt (Faza D) — czy kalibracja się udała

Po złożeniu warstwy zbierz z płytek realne spiki (dekoder → CSV `frame,neuron0,...`) na tym samym
sygnale, który podasz symulacji, i porównaj:

```
python snn_hw_pipeline.py compare --sim sim.npz --hw hw_spikes.csv --layer H
```

Zgodność <85 % dla którejś płytki → wróć do Fazy C dla niej (zła nastawa trymera/paska).
`compare` liczy zgodność spike-po-spiku + dystans van Rossuma (toleruje przesunięcie w czasie).

## 6. Znane ograniczenia i co poprawić PO jutrze

- **Enkoder to sufit.** Kanały crest/zcr/flux słabo różnicują; H3.J2 (crest) wyszła martwa.
  Największy zysk da dodanie cechy pasmowej (szkło ma charakterystyczne wysokie częstotliwości
  5–10 kHz) — to zmiana w `encoder_v2.ino` **i** `encoder_twin.py` naraz.
- **Głośne transienty** (strzały, dzwony) to główne źródło fałszywych alarmów. Więcej takich
  trudnych negatywów w treningu pomoże, ale bez lepszych cech nie przeskoczy sufitu.
- **Odporność min 0.36** oznacza, że pojedyncza pechowa kombinacja rozrzutu trymerów potrafi
  zepsuć F1 — dlatego test z sekcji 3 rób dokładnie, nie „na oko".

## Reprodukcja / pełny sweep

Zwycięzca (model A): `sweep_pw10_s1.pt` → skopiowany do `best.pt`, wyeksportowany do `hw_config.json`.

```
python snn_hw_pipeline.py train --data spikes_manifest/train \
    --val-data spikes_manifest/val --test-data spikes_manifest/test \
    --epochs 100 --patience 15 --hat-frac 0.5 --seed 1 --pos-weight 1.0 \
    --out hw_config.json --ckpt best.pt
python eval_stream.py --ckpt best.pt --data spikes_manifest/test   # metryki klipowe
```

Pozostałe biegi sweepu (`sweep_*.pt/json/csv`) trzymam dla porównania — A wygrał każdą metrykę.
```
A pw1.0/s1 : val F1 0.512  test 0.487  (WYBRANY, najlepsza precyzja)
D pw0.75/s4: val F1 0.495  test 0.474  (najczystsze k≥2, ale niski recall)
B pw1.75/s2: val F1 0.431  test 0.408
E/F spk-loss: val F1 ~0.39  (strata zliczania spików nie rozdzieliła klas)
C pw3.0/s3 : val F1 0.383  test 0.372  (za duży przechył w recall)
```
