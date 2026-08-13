# Output z ```build_feature_bank.py```

```
[OK] Zapisano feature bank -> feature_bank.npz
     ramki: 925435, kanały: 18, plików: 13401

=== 1. Sanity strukturalny ===
kształt X: (925435, 18)  NaN: False  Inf: False
balans klas: positive = 297008  negative = 628427
brak stałych kolumn - OK

=== 2. eparowalność per kanał (Cohen's d, |d|>0.2 = coś widać) ===
  peak                 d=-0.011
  peak_cnt             d=-0.051
  crest                d=+0.410  <-- ma sygnał
  cv                   d=+0.146
  zcr                  d=+1.154  <-- ma sygnał
  flux                 d=+0.115
  hjorth_mobility      d=+1.140  <-- ma sygnał
  tkeo_mean            d=+0.098
  curve_length         d=+0.335  <-- ma sygnał
  autocorr_lag1        d=-1.041  <-- ma sygnał
  kurtosis             d=+0.003
  spectral_centroid    d=+1.269  <-- ma sygnał
  dominant_freq        d=+0.672  <-- ma sygnał
  band_energy_low      d=-1.245  <-- ma sygnał
  band_energy_mid      d=+1.047  <-- ma sygnał
  band_energy_high     d=+1.009  <-- ma sygnał
  spectral_flatness    d=+1.305  <-- ma sygnał
  spectral_flux        d=+0.275  <-- ma sygnał

=== 3. Baseline: regresja logistyczna, split PO PLIKACH ===
accuracy:            0.835  (naiwny baseline klasy większościowej: 0.698)
balanced_accuracy:   0.766
f1:                  0.684
roc_auc:             0.858
jest sygnał ponad przypadek - bank nadaje się jako wejście dla GA.
```

# Wnioski

### Kluczowe wnioski z wygenerowanych statystyk

*   **Skuteczność nowych cech czasowych:** Wprowadzenie `hjorth_mobility` (d=+1.140) oraz `autocorr_lag1` (d=-1.041) było wysoce uzasadnione. Mają one siłę dyskryminacyjną dorównującą najlepszym cechom z domeny FFT. Oznacza to, że algorytm ma dostęp do precyzyjnych informacji o rozkładzie częstotliwości (estymator pasma) bez konieczności wykonywania kosztownej sprzętowo transformaty Fouriera.
*   **Niska przydatność kurtozy i TKEO:** `kurtosis` (d=+0.003) oraz `tkeo_mean` (d=+0.098) wykazują znikomy sygnał w ujęciu liniowym. Podobnie reagują najprostsze metryki amplitudowe (`peak`, `peak_cnt`, `flux`). W kontekście sieci neuronowej można rozważyć ich usunięcie w celu redukcji liczby wejść (odciążenie MCU), o ile eksperymenty z nieliniowymi modelami nie wykażą ich ukrytej użyteczności.
*   **Dominacja sygnałów częstotliwościowych:** Zjawisko fizyczne (prawdopodobnie pękanie szkła) jest najsilniej separowalne przez wysoką częstotliwość. Potwierdzają to najwyższe wartości Cohena dla `spectral_flatness` (+1.305), `spectral_centroid` (+1.269) oraz silny spadek energii w niskim paśmie `band_energy_low` (-1.245). Sprzętowy `zcr` (+1.154) doskonale odzwierciedla to samo zjawisko.
---

# Output z ```esos_time_analysis.py```

```
=== Analiza czasu na Arduino (Cortex-M4F @ 64 MHz) ===
Baza (RMS, DC, zcr_pre): 27.44 us (1756 cykli)
Baza FFT (Okno + Trans.): 107.25 us (6864 cykli)
-----------------------------------------------------------------
[Czas] peak                :   3.00 us (  192 cykli) [Sugestia usunięcia: Nieopłacalny stosunek czasu do siły d]
[Czas] peak_cnt            :   3.00 us (  192 cykli) [Sugestia usunięcia: Nieopłacalny stosunek czasu do siły d]
[Czas] crest               :   0.22 us (   14 cykli)
[Czas] cv                  :   0.44 us (   28 cykli)
[Czas] zcr                 :   0.22 us (   14 cykli)
[Czas] flux                :   0.22 us (   14 cykli)
[Czas] hjorth_mobility     :   0.44 us (   28 cykli)
[Czas] tkeo_mean           :   9.22 us (  590 cykli) [Sugestia usunięcia: Nieopłacalny stosunek czasu do siły d]
[Czas] curve_length        :   6.00 us (  384 cykli)
[Czas] autocorr_lag1       :   0.22 us (   14 cykli)
[Czas] kurtosis            :   0.25 us (   16 cykli) [Sugestia usunięcia: Nieopłacalny stosunek czasu do siły d]
[FFT]  spectral_centroid   :   3.25 us (  208 cykli)
[FFT]  dominant_freq       :   1.52 us (   97 cykli)
[FFT]  band_energy_low     :   1.73 us (  111 cykli)
[FFT]  band_energy_mid     :   1.73 us (  111 cykli)
[FFT]  band_energy_high    :   1.73 us (  111 cykli)
[FFT]  spectral_flatness   :   1.73 us (  111 cykli)
[FFT]  spectral_flux       :   3.03 us (  194 cykli)
-----------------------------------------------------------------
CAŁKOWITY CZAS RAMKI: 172.64 us / 10000.00 us budżetu (10ms)
CPU Load: 1.7%
```


