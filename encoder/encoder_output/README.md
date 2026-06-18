# Analiza Cech Audio dla Enkodera SNN (Detekcja Rozbijanego Szkła)

---

## 1. Zero Crossing Rate (ZCR) — Częstotliwość przejść przez zero
To absolutny klasyk analizy audio bez użycia FFT. Polega na zliczaniu, ile razy sygnał zmienia znak (z plusa na minus lub odwrotnie) w obrębie jednej ramki (20 ms).

* **Jak to pomaga SNN:** ZCR działa jak uproszczony miernik dominującej częstotliwości. Tłuczone szkło generuje dźwięki o bardzo wysokiej częstotliwości, więc ZCR będzie ekstremalnie wysokie. Z kolei niskie tąpnięcia (kroki, uderzenie pięścią w stół, zamknięcie drzwi) dadzą bardzo niskie ZCR.
* **Plusy (+):** Ekstremalnie tanie obliczeniowo (zwykły licznik `if (current_sample > 0 && prev_sample <= 0)`). Doskonale separuje wysokie tony od basowych hałasów.
* **Minusy (-):** Wrażliwe na szum tła o wysokiej częstotliwości (np. syczenie z głośników). Skoro jednak masz sprzętowy filtr LPF/HPF, ten problem w zasadzie znika.

---

## 2. Delta Energii / Średniej (Temporal Derivative)
Zamiast patrzeć tylko na energię w danej ramce, liczymy różnicę między średnią wartością z obecnej ramki a ramki poprzedniej:

$$Mean_{t} - Mean_{t-1}$$

* **Jak to pomaga SNN:** Sieci neuronowe z natury (a SNN w szczególności!) uwielbiają wykrywać zmiany. Tłuczenie szkła zaczyna się od potężnego, natychmiastowego uderzenia (tzw. *onset*). Taka „pochodna” energii wygeneruje natychmiastową, zsynchronizowaną paczkę impulsów (*Spike Burst*) w trybie TTFS lub nagły skok częstotliwości w Rate Coding.
* **Plusy (+):** Praktycznie zerowy koszt obliczeniowy (pamiętasz tylko jedną wartość z poprzedniej ramki). Świetnie „budzi” sieć w momencie uderzenia.
* **Minusy (-):** Sama delta nie odróżni klaśnięcia w dłonie od pęknięcia szyby (oba mają szybki start). Musi współdziałać z ZCR.

---

## 3. Współczynnik Szczytu (Crest Factor)
To stosunek wartości maksymalnej do wartości średniej w ramce:

$$\frac{Peak}{Mean}$$

Mierzy, jak bardzo „szpiczasty” jest sygnał.

* **Jak to pomaga SNN:** Pozwala odróżnić głośny sygnał ciągły (np. pracujący odkurzacz, suszarkę, głośną rozmowę) od sygnału impulsowego. Odkurzacz da wysoki *Mean* i umiarkowany *Peak* (niski Crest Factor). Rozbicie szkła da gigantyczny *Peak* przy stosunkowo niskiej średniej z całych 20 ms (bardzo wysoki Crest Factor).
* **Plusy (+):** Masz już w kodzie zmienne `frameMax` oraz `mean_val`, więc wyliczenie tego to tylko jedno dzielenie na koniec ramki. Czysty zysk informacyjny dla enkodera.
* **Minusy (-):** Wymaga operacji dzielenia zmiennoprzecinkowego (choć raz na 20 ms to dla Arduino żaden wydatek).

---

## 4. Licznik Lokalnych Maksimów (Peak Counting Rate)
Zamiast szukać tylko jednego, globalnego `frameMax`, zliczasz ile „lokalnych miniaturowych szczytów” (lokalnych ekstremów) pojawiło się wewnątrz ramki. Rejestrujesz punkt, w którym sygnał rósł, a nagle zaczął spadać.

* **Jak to pomaga SNN:** Dźwięk tłuczonego szkła to chaos mikropęknięć i sypiących się odłamków. Sygnał wewnątrz ramki nie jest gładką sinusoidą, ale przypomina „szczotkę” najeżoną dziesiątkami mikro-szpilek. Liczba tych lokalnych szczytów drastycznie rośnie podczas fazy wtórnej (szuranie i sypanie się szkła).
* **Plusy (+):** Bardzo wysoka jakość danych pod kątem detekcji „tekstury” dźwięku szkła. Pozwala odróżnić czysty ton (np. gwizdek, alarm) od bogatego w anomalie trzasku szkła.
* **Minusy (-):** Wymaga porównywania trzech kolejnych próbek w locie `(sample[t-1] > sample[t-2] && sample[t-1] > sample[t])`, co lekko obciąża główną funkcję zbierania audio.

---

## Podsumowanie — Rekomendowana konfiguracja enkodera

Gdybyśmy mieli zbudować idealny, **5-kanałowy zestaw cech pod SNN dla szkła**, wyrzucając cechy zbyt podobne do siebie, optymalna konfiguracja prezentuje się następująco: