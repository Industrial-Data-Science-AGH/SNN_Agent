# Raport z charakteryzacji neuronu SNN

**Data:** 2026-03-22
**Osoba przeprowadzająca:** 
**Cel:** Określenie stabilności działania układu analogowego i jego parametrów wyzwalania.

## 1. Parametry środowiskowe
- Temperatura otoczenia: 
- Napięcie zasilania (VCC): 5.0V / 3.3V

## 2. Charakteryzacja jednorazowa
- Pomiary progu wyzwalania `V_th`
  (Zwiększać powoli `V_in` i zapisać napięcie pierwszego spike'a)
- Zmierzony próg wyzwalania V_th: [ ] mV
- Amplituda wyjściowa spike'a: [ ] mV
- Czas trwania spike'a: [ ] µs

## 3. Testy stabilności (Drift)
- Co 10-15 minut mierzyć próg wyzwalania przy stałym zasilaniu:
  - T=0 min: [ ] mV
  - T=15 min: [ ] mV
  - T=30 min: [ ] mV
  - T=45 min: [ ] mV
  - T=60 min: [ ] mV
- Maksymalny wyliczony drift (% CV): [ ] %

## 4. Odtwarzalność (Reset zasilania)
- Odłącz zasilanie, odczekaj 30 sek, włącz i zmierz próg wyzwalania:
  - Próba 1: [ ] mV
  - Próba 2: [ ] mV
  - Próba 3: [ ] mV
  - Próba 4: [ ] mV
  - Próba 5: [ ] mV
- Współczynnik zmienności (CV) < 1% oznacza gotowość do skalowania hardware'u (Faza 2).
