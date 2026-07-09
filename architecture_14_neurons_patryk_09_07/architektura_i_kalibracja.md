# Delta Spike — architektura 6→4→3→1 na Lu.i + kalibracja

## 0. Co się zmieniło i dlaczego

| Było | Jest | Powód |
|---|---|---|
| 3→4→7→1, fan-in do 7, **pruning po treningu** | **6→4→3→1**, fan-in ≤ 3 wszędzie z definicji | Neuron decyzyjny ma 3 wejścia (J1/J2/J3). Przycinanie 7 wag do 3 po treningu wyrzuca to, czego sieć się nauczyła. Tu maska jest *w grafie od pierwszej epoki* — sieć uczy się w przestrzeni, którą sprzęt realnie potrafi zrealizować. |
| Impulsy 500 µs, rate coding 5–200 Hz | Impulsy 6 ms, **1 impuls na ramkę**, ramka 10 ms | Ładunek wpompowany w synapsę ∝ szerokość impulsu. Stała szerokość = stała, znana waga bazowa. A skoro max 1 impuls/ramkę, to `dt_symulacji = 10 ms` i trening liczy dokładnie to samo, co robi sprzęt. |
| Progi cech zaszyte na sztywno | Adaptacyjny floor + MAD, progi w jednostkach z-score | Encoder nie rozjeżdża się przy zmianie mikrofonu / poziomu tła. |
| `snn.Leaky` (tylko τ_mem) | Własny LIF z **τ_syn + τ_mem + V_leak + reset-do-zera** | Lu.i ma synapsę prądową z własną stałą czasową i membranę relaksującą do V_leak, a nie do 0. `Leaky` modeluje inny neuron niż ten, który masz na stole. |
| „Ciągłość" jako ręcznie dolutowany neuron hamujący | Hamowanie **wyuczone** (znak wagi = przełącznik +/−) | Nie musisz zgadywać, który kanał hamuje. Trening ustawi ujemne wagi tam, gdzie trzeba — Ty tylko przestawiasz przełącznik zgodnie z tabelą eksportu. |

**Bilans płytek: 8 użytych (4 + 3 + 1), 7 zapasowych.**

## 1. Topologia

```
Arduino D2..D7 (6 kanałów cech, 1 impuls/ramkę, 6 ms)
        │
        ├──► H0  H1  H2  H3      warstwa ukryta, 4 płytki, fan-in = 3
        │      │   │   │   │
        │      └───┴───┴───┴──► G0  G1  G2   przedostatnia, 3 płytki, fan-in = 3
        │                         │   │   │
        │                         └───┴───┴──► D   decyzyjny, fan-in = 3 (dokładnie)
        │                                       │
        └────────────────────────────────────►  Arduino D8 / RPi GPIO17
```

Kluczowe: **|G| = 3 = liczba wejść płytki D**. Neuron decyzyjny dostaje całą przedostatnią warstwę, bez odrzucania czegokolwiek.

## 2. Kanały wejściowe

| # | Pin | Kanał | Co wykrywa |
|---|---|---|---|
| 0 | D2 | `peak` | obwiednia — ostry transient |
| 1 | D3 | `peak_cnt` | liczba mikro-szpilek w ramce |
| 2 | D4 | `crest` | crest factor = peak/RMS |
| 3 | D5 | `cv` | współczynnik zmienności (Welford) |
| 4 | D6 | `zcr` | zero-crossing rate — charakter widma |
| 5 | D7 | `flux` | dodatni przyrost log-RMS — atak |

`mean` wyleciał (nie różnicuje klas — Wasza własna analiza).

## 3. Tabela połączeń (kabel = 2 żyły: sygnał + GND)

Każda płytka: bateria CR2032. **Wszystkie GND (Arduino + 8 płytek) na jednej listwie — pierwszy krok, przed czymkolwiek.**

| Skąd | Dokąd | Port | Znak |
|---|---|---|---|
| D2 `peak` | H0 | J1 | z eksportu |
| D3 `peak_cnt` | H0 | J2 | z eksportu |
| D4 `crest` | H0 | J3 | z eksportu |
| D5 `cv` | H1 | J1 | z eksportu |
| D6 `zcr` | H1 | J2 | z eksportu |
| D7 `flux` | H1 | J3 | z eksportu |
| D2 `peak` | H2 | J1 | z eksportu |
| D5 `cv` | H2 | J2 | z eksportu |
| D7 `flux` | H2 | J3 | z eksportu |
| D3 `peak_cnt` | H3 | J1 | z eksportu |
| D4 `crest` | H3 | J2 | z eksportu |
| D6 `zcr` | H3 | J3 | z eksportu |
| H0 J4 | G0 | J1 | z eksportu |
| H1 J4 | G0 | J2 | z eksportu |
| H2 J4 | G0 | J3 | z eksportu |
| H1 J4 | G1 | J1 | z eksportu |
| H2 J4 | G1 | J2 | z eksportu |
| H3 J4 | G1 | J3 | z eksportu |
| H2 J4 | G2 | J1 | z eksportu |
| H3 J4 | G2 | J2 | z eksportu |
| H0 J4 | G2 | J3 | z eksportu |
| G0 J4 | **D** | J1 | z eksportu |
| G1 J4 | **D** | J2 | z eksportu |
| G2 J4 | **D** | J3 | z eksportu |
| D J4 | Arduino D8 / RPi GPIO17 | — | — |

Fan-out: `peak` → 2 płytki, `H2` → 3 płytki. Wejście synaptyczne Lu.i jest wysokoimpedancyjne, jeden pin cyfrowy uciągnie 3 równolegle.

Kolumna „Znak" wypełnia się z `hw_config.json` po treningu (przełącznik `+` / `−` przy W1/W2/W3).

## 4. Model neuronu, który trenujemy (i dlaczego taki)

Krok czasowy `dt = 10 ms` (= hop ramki encodera).

```
α = exp(−dt/τ_syn)          τ_syn ∈ [5 ms, 220 ms]   (C_syn = 10 µF)
β = exp(−dt/τ_mem)          τ_mem ∈ [20 ms, 2200 ms] (C_mem = 22 µF)

I[t] = α·I[t−1] + Σ_j w_j · s_j[t]
V[t] = β·V[t−1] + (1−β)·V_leak + I[t]
s[t] = 1  gdy V[t] ≥ V_th          (V_th = 1.0, sprzętowo VDD/2 — nieruchome)
V[t] ← 0  po spiku                  (reset do zera, tak jak na oscylogramie Lu.i)
```

Uwaga na człon prądowy: `I[t]` wchodzi **bez** czynnika `(1−β)`. Fizycznie `dV/dt = (V_leak − V)/τ_mem + I/C_mem` — prąd synaptyczny nie skaluje się przewodnością upływu. Jeśli go tam wsadzisz (łatwa pomyłka), to przy `τ_mem = 1 s` masz `1−β ≈ 0.01` i gradient przez trzy warstwy znika; sieć nigdy nie odpala. Sprawdzone empirycznie na tym pipeline.

Konsekwencja, z której wynika cała kalibracja. Podstaw `U = V − V_leak`; między spikami `U[t] = β·U[t−1] + I[t]`, a neuron odpala przy `U ≥ V_th − V_leak`. `I` jest liniowe w wagach, więc:

> Przemnożenie **wszystkich wag neuronu przez k** i zmiana `V_leak' = V_th − k·(V_th − V_leak)`
> **nie zmienia momentu pierwszego spiku ze stanu spoczynku** — a to jest dokładnie zdarzenie, które wykrywamy.

Dlatego można wykręcić najsilniejszą wagę na pełną skalę trymera i odrobić to zapasem do progu:

```
k        = W_pot_fullscale / max_j |w_j|
V_leak'  = V_th − k · (V_th − V_leak)
pasek%   = 50 · V_leak' / V_th          (próg = 3. z 6 diod = 50% paska)
```

Zastrzeżenie: reset-do-zera łamie tę niezmienniczość dla **kolejnych** spików w serii (bo `U` po resecie skacze do `−V_leak`, a to nie skaluje się przez `k`). Dlatego skrypt trzyma `V_leak' ≥ 0.20·V_th` (pasek ≥ 10%) i raczej zjedzie z potencjometrem poniżej pełnej skali, niż zepchnie V_leak na dno.

## 5. Kalibracja — właściwa kolejność

### Faza A — charakteryzacja płytek (raz, 8 płytek, ~2 h)

Nie kalibrujesz „pod rolę". Najpierw **mierzysz, co dana płytka realnie robi**, bo trymery mają rozrzut i dwie płytki przy tym samym kącie pokrętła nie mają tej samej τ.

**A1. τ_mem.** Wywołaj spike (seria impulsów z trybu `CALIB`), membrana leci do 0. Mierzysz czas powrotu paska LED do 63% wysokości docelowej (V_leak). Bez oscyloskopu: nagranie slow-motion telefonem (120–240 kl/s → rozdzielczość 4–8 ms) albo stoper, jeśli τ > 0.5 s.

**A2. τ_syn.** Jeden impuls na spoczynkową membranę, slow-mo na pasek, liczysz klatki do zaniku PSP do 37% szczytu. Zakres 0–220 ms, ręczny stoper odpada.

**A3. Krzywa wagi.** To najważniejszy i najczęściej pomijany pomiar. Dla pozycji potencjometru 25 / 50 / 75 / 100%:

1. Ustaw V_leak tak, by pasek stał na 25% (znany zapas do progu: `V_th − V_leak = 0.5`).
2. Tryb `CALIB`: `C 2 <n> 100` — wyślij `n` impulsów po 100 Hz na J1.
3. Znajdź najmniejsze `n*`, przy którym 7. dioda odpala.
4. Zapisz `(pot%, n*)`.

Z `n*`, zmierzonych τ i znanego zapasu skrypt (`--fit-weight-curve`) odwraca model i podaje wagę `w` w tych samych jednostkach, w których trenuje sieć. Dostajesz per-płytkę funkcję `pot% → w`. **Pomiar multimetrem tylko pomocniczo** — in-circuit zaniża odczyt przez ścieżki równoległe; wartość rezystancji i tak nie mówi Ci, jaki PSP wyjdzie.

Wynik fazy A zapisujesz do `hw_params.json`:

```json
{
  "H0": {"tau_syn": 0.031, "tau_mem": 0.180, "w_fullscale": 1.42},
  "H1": {"tau_syn": 0.028, "tau_mem": 1.900, "w_fullscale": 1.51},
  "...": {}
}
```

### Faza B — trening pod TWÓJ sprzęt

```bash
python snn_hw_pipeline.py train \
    --data ./spikes_csv --hw-params hw_params.json \
    --epochs 120 --pos-weight 3.0 --out hw_config.json
```

Z `--hw-params` skrypt **zamraża zmierzone τ i skalę wag każdej konkretnej płytki** i uczy tylko wag i V_leak. Sieć kompensuje rozrzut sprzętu zamiast go ignorować. Bez tego flagi trenuje τ i V_leak też i podaje wartości docelowe do wykręcenia.

### Faza C — ustawianie trymerów, od dołu do góry

Kolejność: **H0..H3 → G0..G2 → D**. Nigdy odwrotnie, nigdy wszystko naraz.

Dla każdej płytki, **bez podłączonych wejść**:
1. τ_syn i τ_mem na wartości z `hw_config.json` (metoda A1/A2 lub kąt z krzywej z fazy A).
2. V_leak: kręć RV5, aż pasek osiądzie na `pasek%` z tabeli. Sprawdź, że 7. dioda **nie miga** w spoczynku. Jeśli miga — schodzisz niżej niezależnie od tabeli, ten warunek jest nadrzędny.

Potem wejścia, jedna płytka na raz:
3. Podłącz wejścia, ustaw przełączniki `+`/`−` wg kolumny „znak".
4. Ustaw W1/W2/W3 na `pot%` z tabeli.
5. **Weryfikacja liczbowa, nie „na oko":** `hw_config.json` zawiera dla każdej płytki pole `pulses_to_fire` — ile impulsów po 100 Hz na dane wejście (przy pozostałych milczących) ma wystarczyć do odpalenia. Tryb `CALIB` podaje dokładnie tyle impulsów. Odpala przy `n*`, nie odpala przy `n*−1` → płytka gotowa. Rozjazd o ±1 impuls jest OK, o ±3 znaczy, że waga albo V_leak są nie tam, gdzie myślisz.

To zamienia „kręć aż zamiga" w test binarny z jednoznacznym wynikiem.

### Faza D — integracja i domykanie pętli

1. Podłącz warstwę H do Arduino, odtwórz zbiór walidacyjny, zaloguj J4 każdej H na wolne piny Arduino.
2. `python snn_hw_pipeline.py compare --sim sim_spikes.npz --hw hw_spikes.csv --layer H`
   → dostajesz zgodność spike-trainów per neuron (dopasowanie ramka-po-ramce + van Rossum).
3. Zgadza się H → dopinasz G, powtarzasz. Zgadza się G → dopinasz D.
4. Rozjazd na jednej płytce → wracasz do tej jednej płytki (Faza C krok 5), a nie do całości.

Cała sensowność tej kolejności polega na tym, że przy 8 płytkach błąd jednego trymera ginie w szumie systemu, ale w izolacji jest oczywisty.

### Faza E — test końcowy

Cisza / mowa / muzyka / szkło na przemian, patrzysz na 7. diodę płytki D albo na `decoder.py`. Metryki liczysz zdarzeniowo (recall na szkle, FP/godzinę na tle), nie per-ramka — jedna ramka to 10 ms, per-ramkowa accuracy będzie zawsze ~99% i nic nie znaczy.

## 6. Co zrobić z 7 zapasowymi płytkami

- 3 sztuki: ensemble `peak` (Peak-A/B/C z lekko różnymi W1 → uśrednienie rozrzutu trymerów), wpięte w wolny H.
- 1 sztuka: rezerwa na uszkodzenie przy lutowaniu.
- 3 sztuki: drugi tor detekcji / drugi mikrofon (kierunkowość), model Jeffressa na sygnaturę dwuetapową szkła.

## 7. Kolejność uruchomienia — checklista

- [ ] Wspólna masa: Arduino GND + 8× GND na listwie
- [ ] `encoder_v2.ino` wgrany, tryb `DEBUG`, zbiór nagrany do CSV
- [ ] Faza A: `hw_params.json` uzupełniony dla 8 płytek
- [ ] Faza B: `hw_config.json` wygenerowany
- [ ] Faza C: każda płytka przechodzi test `pulses_to_fire`
- [ ] Faza D: zgodność sim↔hw dla H, potem G, potem D
- [ ] Faza E: metryki zdarzeniowe na zbiorze testowym
