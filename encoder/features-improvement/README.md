## Krok 1: Weryfikacja rzeczywistej częstotliwości próbkowania (fs)

* Pomiary wariantu `baseline` z domyślnym preskalerem 32 wykazały częstotliwość `fs_hz` na poziomie ok. 27050 Hz, a przerwania pochłaniały 99,73% czasu procesora, co skutkowało brakiem możliwości przetwarzania ramek w głównej pętli.
* Ustawienie preskalera sprzętowego na 64 ustabilizowało częstotliwość próbkowania na oczekiwanym poziomie 19230,5 Hz i obniżyło obciążenie układu (ISR CPU) do 70,97%, co umożliwiło poprawne procesowanie sygnału zgodnie z wytycznymi.



## Krok 2: Pomiary budżetu czasowego (ISR) na mikrokontrolerze

* **Wariant baseline:** Przetwarzanie przerwania zajmowało średnio 36,91 µs (590 cykli) przy łącznym obciążeniu CPU rzędu 71,0% oraz maksymalnym czasie przetwarzania pętli 4,11 ms.
* **Wariant parity:** Optymalizacje skróciły czas przerwania do 31,56 µs (505 cykli), obniżając użycie CPU do 60,7% (maksymalny czas pętli: 2,06 ms).
* **Wariant swap_full:** Docelowa konfiguracja z nowymi kanałami wygenerowała średni czas ISR na poziomie 35,54 µs (569 cykli) przy 68,3% wykorzystania CPU i najkrótszym maksymalnym czasie pętli 1,71 ms.
* Zmierzone obciążenie wariantu docelowego bez problemu mieści się w krytycznym limicie 52 µs, spełniając pierwsze kryterium akceptacji budżetu czasowego. Pętla główna we wszystkich badanych przypadkach pracowała bez opóźnień, nie generując żadnych zgubionych ramek (`late=0`).



## Krok 3: Faza 0 – Analiza wartości nowych cech na danych

* Nowe funkcjonalności wykazały bardzo wysoką zdolność separacji sygnału tła od docelowego szkła, osiągając wskaźnik Cohen's d równy +1,046 dla kanału `mobility` oraz -1,125 dla kanału `autocorr`.
* Zmierzona korelacja Spearmana między nowymi cechami wyniosła -0,87, co jednoznacznie spełnia regułę decyzyjną o zachowaniu obu kanałów (wartość poniżej 0,9 potwierdza, że nie są one nadmiernie redundantne).


* Badanie przyrostu użyteczności przy pomocy nieliniowego modelu GBM pokazało statystycznie istotny wzrost skuteczności rozróżniania klas (ΔAUC) o +0,028 względem wariantu bazowego. Przedział ufności CI95 wynoszący [+0,016, +0,041] uplasował się w całości powyżej zera, co uzasadniło sens analityczny wprowadzonych zmian.


* Na podstawie rozkładów wyznaczono bezwzględne progi odcięcia dla mikrokontrolera: `mob_thr` = 2,4973 (sygnał wyzwalany powyżej progu) oraz `ac_thr` = -0,2336 (sygnał wyzwalany poniżej progu).



## Krok 4: Test zgodności (Parity Test) między modelem a firmware

* Weryfikacja zgodności kodu C++ z Pythonem wymagała dostosowania środowiska Fedora (ręczna kompilacja symulatora `simavr`, instalacja pakietów OpenGL) oraz usunięcia błędów implementacyjnych w skryptach testowych (naprawa rzutowania typów logicznych i indeksowania macierzy).
* Proces testowania został skonfigurowany do uruchomienia wyłącznie na maszynie PC w cykl-dokładnym symulatorze, do którego przekazano identyczne, całkowite kody ADC z uwzględnieniem docelowych progów odcięcia. Skrypt jest w pełni gotowy do wygenerowania końcowego procentowego wskaźnika zgodności zdarzeń na wybranym pliku audio.