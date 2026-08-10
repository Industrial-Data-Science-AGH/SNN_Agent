# Output z ```build_feature_bank.py```

```
[OK] Zapisano feature bank -> feature_bank.npz
     ramki: 925435, kanały: 18, plików: 13401

=== 1. Sanity strukturalny ===
kształt X: (925435, 18)  NaN: False  Inf: False
balans klas: positive = 297008  negative = 628427
brak stałych kolumn - OK

=== 2. Separowalność per kanał (Cohen's d, |d|>0.2 = coś widać) ===
  peak                 d=-0.011
  peak_cnt             d=-0.051
  crest                d=+0.410  <-- ma sygnał
  cv                   d=+0.146
  zcr                  d=+1.154  <-- ma sygnał
  flux                 d=+0.115
  hjorth_mobility      d=+1.140  <-- ma sygnał
  tkeo_mean            d=+0.098
  curve_length         d=+0.335  <-- ma sygnał
  autocorr_lag1        d=-1.041  <-- ma sygnał
  kurtosis             d=+0.003
  spectral_centroid    d=+1.269  <-- ma sygnał
  dominant_freq        d=+0.672  <-- ma sygnał
  band_energy_low      d=-1.245  <-- ma sygnał
  band_energy_mid      d=+1.047  <-- ma sygnał
  band_energy_high     d=+1.009  <-- ma sygnał
  spectral_flatness    d=+1.305  <-- ma sygnał
  spectral_flux        d=+0.275  <-- ma sygnał

=== 3. Baseline: mała regresja logistyczna, split PO PLIKACH (bez wycieku) ===
accuracy na plikach nie widzianych w treningu: 0.835
jest sygnał ponad przypadek - bank nadaje się jako wejście dla GA.
```

# Wnioski

### Kluczowe wnioski z wygenerowanych statystyk

*   **Skuteczność nowych cech czasowych:** Wprowadzenie `hjorth_mobility` (d=+1.140) oraz `autocorr_lag1` (d=-1.041) było wysoce uzasadnione. Mają one siłę dyskryminacyjną dorównującą najlepszym cechom z domeny FFT. Oznacza to, że algorytm ma dostęp do precyzyjnych informacji o rozkładzie częstotliwości (estymator pasma) bez konieczności wykonywania kosztownej sprzętowo transformaty Fouriera.
*   **Niska przydatność kurtozy i TKEO:** `kurtosis` (d=+0.003) oraz `tkeo_mean` (d=+0.098) wykazują znikomy sygnał w ujęciu liniowym. Podobnie reagują najprostsze metryki amplitudowe (`peak`, `peak_cnt`, `flux`). W kontekście sieci neuronowej można rozważyć ich usunięcie w celu redukcji liczby wejść (odciążenie MCU), o ile eksperymenty z nieliniowymi modelami nie wykażą ich ukrytej użyteczności.
*   **Dominacja sygnałów częstotliwościowych:** Zjawisko fizyczne (prawdopodobnie pękanie szkła) jest najsilniej separowalne przez wysoką częstotliwość. Potwierdzają to najwyższe wartości Cohena dla `spectral_flatness` (+1.305), `spectral_centroid` (+1.269) oraz silny spadek energii w niskim paśmie `band_energy_low` (-1.245). Sprzętowy `zcr` (+1.154) doskonale odzwierciedla to samo zjawisko.
*   **Znaczenie niezbalansowania klas w ewaluacji:** Zbiór posiada naturalny narzut klasy negatywnej względem pozytywnej w proporcji ~2.1:1. Zwykły model przewidujący zawsze "0" osiągnąłby skuteczność na poziomie 67.9%.
*   **Solidny Baseline:** Wynik regresji logistycznej wynoszący 83.5% na danych testowych (podzielonych prawidłowo, po plikach) udowadnia, że wyekstrahowane cechy zawierają mocny, łatwy do wyodrębnienia wzorzec, przebijający wynik losowy (67.9%) o ponad 15 punktów procentowych.

---

**Wskazówka do konfiguracji algorytmu genetycznego (GA):**
Ze względu na asymetrię (297008 vs 628427), funkcja celu (fitness) oceniająca osobniki w algorytmie genetycznym nie może polegać wyłącznie na metryce `accuracy`. Należy zastosować `F1-score`, `Balanced Accuracy` lub `ROC AUC`, aby uniknąć ewolucji sieci faworyzującej przewidywanie wyłącznie klasy negatywnej.
