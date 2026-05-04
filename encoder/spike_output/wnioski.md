### Początek: dog 10, door 26, glass 345
1. zmiana: **rc_noise_floor** *z 40 na 100*: dog 8, door 5, glass 350, 
2. na 150: glass 284, dog 7, door 8
3. 300: door 0, dog 4, glass 224
4. 500: dog 1, door 0, glass 5
5. 400: glass 147, dog 3, door 0
6. 10: door 45, dog 12, glass 362, silence 0

a) noise floor ma wykrywać faktyczny dźwięk, nie tylko szkło
b) gdy nic sie nie dzieje, powinno nie być spike'ow
c) prosta logika
d) oblicz parametry: 
```
z wejściowej obwiedni energii audio (RMS energy per frame) wylicza statystyki i podaje do sieci 3 parametry kodowane w postaci 3-kanałowego tensora wejściowego (spikes). Są to:

    Peak energy (maksymalna wartość energii) — podawana na Kanał 0 (wykorzystywana m.in. do detekcji nagłych impulsów).
    Mean energy (średnia wartość energii) — podawana na Kanał 1 (służy jako baseline ogólnego poziomu sygnału).
    Std energy (znormalizowane odchylenie standardowe energii) — podawane na Kanał 2 (wskazuje na zmienność sygnału, przydatne do burst detection).

Te wartości numeryczne, jako tablica [peak, mean, std], są następnie konwertowane za pomocą wybranego algorytmu na odpowiednie ciągi impulsów binarnego tensora o wymiarach (3, n_timesteps) wykorzystując schemat Rate Coding lub TTFS (Time-To-First-Spike).
```
