# Projekt Backplane dla 3-5 Neuronów Analogowych (SNN)

## 1. Topologia
Zalecana topologia to **Star / Crossbar** z centralną szyną sygnałową, umożliwiająca elastyczne kierowanie sygnałami spike'ów pomiędzy wyjściami i wejściami poszczególnych neuronów układu bez konieczności robienia nowego layoutu.

## 2. Złącza
**Pin Header 2.54mm (2x8, 16 pin) dla modułów wejściowych/wyjściowych:**
Piny łatwodostępne, proste w lutowaniu i wytrzymałe.
- 1-2: VCC (zalecane przefiltrowane)
- 3-4: GND, AGND
- 5: SPI_MOSI
- 6: SPI_MISO
- 7: SPI_SCK
- ... (Piny sygnałowe - wejścia neuronu, wejście hamujące)

## 3. Wytyczne rutingowe (Layout na PCB)
1. **Zasilanie i masy (KRYTYCZNE):**
   - Absolutny podział na masę sygnałową (AGND) i cyfrową (DGND) łączoną w jednym punkcie (Star Ground).
   - Unikaj nakładania sygnałów SPI na masę analogową, co prowadzi do sztucznego wyzwalania neuronu (cyfrowe interferencje na oscyloskopie).
   - Na wejściu zasilania kondensatory odsprzęgające (100nF i 10uF).
2. **Sygnały Spike:**
   - Ścieżki sygnałów wyjściowych maksymalnie krótkie, do 50mm.
   - Użyj szeregowych rezystorów tłumiących (ok. 100 Ohm) dla zmniejszenia zjawisk odbić przy szybkich zboczach napięć.
   - Prowadzenie pod kątem 45 stopni.
3. **Punkty testowe (Test Points):**
   - Wyprowadź dedykowane pady / szpilki testowe dla V_in oraz V_out (np. TP_N1_IN, TP_N1_OUT) dla ułatwionego pomiaru sondami oscyloskopowymi.
