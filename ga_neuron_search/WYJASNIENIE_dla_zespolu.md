# SNN na płytkach Lu.i — dobór topologii algorytmem genetycznym

Dokument do przekazania zespołowi. Wyjaśnia: po co to jest, jak działa cały
potok, co z niego wychodzi, co i jak zostało poprawione względem pierwszej
wersji, oraz jak wynik przenieść na fizyczne neurony. Na końcu — interpretacja
realnych wyników i uczciwe ograniczenia.

---

## 1. Po co to jest — dwa podejścia

Zadanie: wykryć zdarzenie (np. tłuczone szkło) siecią spikującą złożoną z płytek
Lu.i. Każda płytka = jeden neuron LIF z **3 wejściami (fan-in) i 3 wyjściami
(fan-out)** — to twardy limit sprzętu.

**Podejście A (dotychczasowe)** — topologia projektowana ręcznie. W
`snn_hw_pipeline.py` architektura jest wpisana na sztywno (maski
`MASK_H/MASK_G/MASK_O`, 14–15 neuronów 7→4→3→1). Człowiek zgaduje liczbę
neuronów i połączenia; trening dobiera tylko wagi/τ/V_leak.

**Podejście B (nowe, `ga_neuron_search/`)** — algorytm genetyczny (GA) **sam
szuka najlepszej topologii i liczby neuronów**, respektując limity sprzętu.
Nie zastępuje treningu — podaje mu lepszy punkt startowy (którą płytkę z czym
połączyć i ile ich użyć).

---

## 2. Jak to działa — potok krok po kroku

1. **Genom = topologia.** Sieć warstwowa, ale liczba warstw też ewoluuje.
   Warstwa 0 = 7 kanałów encodera (`peak, peak_cnt, cv, zcr, flux, hf_lo, hf_hi`),
   dalej neurony ukryte, na końcu 1 neuron decyzyjny. Neuron = lista 1–3 wejść z
   warstwy poprzedniej. To jest uogólnienie masek z podejścia A.
2. **GenomeNet.** Z genomu budujemy sieć na tej samej fizyce płytki Lu.i
   (`LuiLayer` z pipeline'u) — GA zmienia tylko okablowanie, nie model neuronu.
3. **Proxy-trening.** Każdą topologię trenujemy krótko (12 epok) i oceniamy na
   walidacji. To „szybka próbka jakości" do rankingu, nie finalny trening.
4. **Fitness.** Ocena ZDARZENIOWA: dekoder „≥ k spików neuronu decyzyjnego w
   oknie", agregowany do poziomu klipu (jak na żywym demie). Metryka: **AP**
   (Average Precision) — bezprogowa, mało szumna.
5. **GA.** Selekcja turniejowa + elityzm + mutacje (przepnij wejście, dodaj/usuń,
   przenieś neuron, podziel/scal warstwę) + krzyżowanie. Po każdej operacji
   „repair" pilnuje wszystkich ograniczeń sprzętu.
6. **Sweep po N.** Osobny run GA dla każdej liczby neuronów (np. 6, 8, 10) →
   odpowiedź „ile płytek się opłaca".
7. **Zwycięzca → sprzęt.** Najlepszą topologię można dotrenować pełnym cyklem
   (HAT→QAT) i wyeksportować nastawy płytek. (Do samego demo ten krok pomijamy.)

---

## 3. Co z tego wychodzi

- **`wyniki_*.json` / `.csv`** — dla każdego N: najlepsza topologia, jej fitness,
  krzywa poprawy (fitness w kolejnych pokoleniach).
- **`wyniki_*.png`** (z `plot_history.py`) — wykres „fitness vs pokolenie":
  wizualny dowód, że GA się uczy.
- **tabela w konsoli** — porównanie N (fitness + układ warstw).
- **`wyniki_*_hw_config_N{n}.json`** (tylko przy pełnym `--train-winner`) —
  nastawy każdej płytki: trymer %, znak +/−, styk J1/J2/J3, pasek LED (V_leak),
  τ_syn/τ_mem, `pulses_to_fire` do weryfikacji sim↔hw.

Zapis topologii, np. `7 -> 5 -> 1` z `H1: [0,1,5],[2,3,6],...` czyta się:
7 kanałów → 5 płytek ukrytych → 1 decyzyjna; listy to wejścia (indeksy kanałów
lub wcześniejszych płytek).

---

## 4. Co poprawiliśmy względem pierwszej wersji i jak

Pierwsza wersja miała cztery słabe punkty. Wszystkie zaadresowane:

**#1 — ocena i strata w czasie.** Wcześniej decyzja szła po `vmax` (maksimum
membrany w oknie) — to zwijało całe 2-sekundowe okno do jednej liczby i
tolerowało spóźnione/przypadkowe spiki. Teraz strata ma **człon czasowy**
(neuron decyzyjny ma dać ≥ k spików na zdarzeniu i 0 na tle), a metryka liczy
**spiki w czasie i agreguje do klipu** — dokładnie jak dekoder na sprzęcie.

**#2 — stabilny sygnał fitness + kalibracja budżetu.** F1 po kilku epokach był
bardzo szumny (GA optymalizowałby przypadek). Zmiany: metryka **AP** (mniej
szumna), możliwość uśredniania po seedach, log pokazujący `loss start→end` (widać,
że sieć się uczy), oraz `calibrate.py` do doboru liczby epok. Kalibracja
pokazała to wprost (topologia testowa 7-3-2-2-1):

| epoki | loss_end | AP śr±std | wniosek |
|---|---|---|---|
| 2–8 | ~3.3→3.1 | 0.42–0.57 **±0.31** | szum — loss stoi, wariancja ogromna |
| 12 | 2.29 | 0.346 **±0.018** | uczy się, wariancja mała → **wybór** |
| 16–20 | 1.84→1.78 | 0.34 ±0.03 | loss spada, ale metryka stoi = overfitting |

Stąd decyzja: **12 epok** — najmniejszy budżet, przy którym fitness jest
wiarygodny (mała wariancja), a nie szum.

**#3 — koniec z „łańcuchem" zamiast sieci.** Wcześniej GA produkował neurony z
jednym wejściem i warstwy szerokości 1 (równoległe pojedyncze linie, nie sieć).
Teraz neurony ukryte mają **fan-in ≥ 2** (realne mieszanie sygnałów), a warstwy
ukryte **nie mają szerokości 1** — wymuszone przy losowaniu, mutacjach i w
„repair". Efekt: topologie to teraz sieci z fan-in 2–3, nie łańcuchy.

**#4 — higiena algorytmu.** Fitness nie może być NaN (psuł sortowanie) — teraz
każdy NaN/inf → −∞. Dodany wybór zwycięzcy z **parsymonią** (najmniejsze N w
zasięgu eps od najlepszego — mniej płytek przy podobnej jakości) oraz opcjonalny
successive-halving (na tym zbiorze wyłączony, bo tani budżet = szum).

---

## 5. Interpretacja realnych wyników (demo, proxy)

Demo (metryka clip-F1, dekoder k=2, 12 epok, pop 8, gens 5, N=6 i N=10) dało:

| N | najlepsza topologia | clip-F1 | krzywa (pokolenia 0→5) |
|---|---|---|---|
| 6 | 7 → 3 → 2 → 1 | **0.553** | 0.553 płasko |
| 10 | 7 → 7 → 2 → 1 | **0.563** | 0.553 → skok w gen 2 → 0.563 |

Jak to czytać:
- **Obie sieci to realne detektory** clip-F1 ~0.55–0.56 (na ~10% klasie
  pozytywnej), poprawnie sprzętowe (fan-in/out ≤ 3) i **rozgałęzione, nie
  łańcuchowe** — np. N=10 to 7 kanałów → 7 płytek → 2 płytki → 1 decyzyjna.
- **N=10 wygrał minimalnie** (0.563 vs 0.553). Dokładanie płytek z 6 do 10 dało
  tylko +0.01 — czyli parsymonia (`--parsimony-eps`) wskazałaby raczej **N=6**:
  praktycznie ta sama jakość, mniej płytek.
- **Krzywa jest płaska** (N=6 wcale, N=10 jeden skok w gen 2). To uczciwy efekt
  malutkiego budżetu: przy pop 8 / gens 5 losowa populacja startowa już
  zawierała rozwiązanie bliskie najlepszemu, więc GA nie miał się z czego mocno
  poprawić. Żeby pokazać WYRAŹNĄ krzywą poprawy, trzeba więcej populacji i
  pokoleń — a to realnie wymaga GPU (na CPU jedna ocena to ~3 min).

Guard anty-miraż zadziałał: rozwiązania, które przy progu nic nie wykrywają
(clipF1=0, a AP fałszywie 1.0), dostają w logu `[MARTWY->0]` i fitness 0 — nie
zatruwają wyniku.

**Co to udowadnia zespołowi:** potok działa end-to-end na realnych danych, sam
znajduje poprawne sprzętowo, rozgałęzione topologie, które faktycznie wykrywają
zdarzenie, i porównuje liczbę płytek. To dowód, że **podejście ma sens**; do
finalnej jakości brakuje mocy obliczeniowej (więcej pokoleń + pełny trening
zwycięzcy na GPU). Wykres krzywej: `wyniki_demo.png` (z `plot_history.py`).

---

## 6. Koszt i ograniczenia (uczciwie)

- Jedna ocena topologii = ~**3 min na CPU** (12 epok; wąskie gardło to pętla po
  200 krokach czasowych neuronu). Pełny sweep (pop 30 × gens 20 × 4 wartości N)
  to dziesiątki godzin — **na CPU nierealny**.
- **GPU zmienia wszystko** (10–50×) — wtedy pełny sweep + trening zwycięzcy jest
  w zasięgu.
- Dlatego to, co pokazujemy teraz, to **dowód koncepcji**: mechanizm działa,
  metryka jest stabilna, topologie są poprawne sprzętowo i sieć się uczy. Finalny
  wynik wymaga GPU albo dłuższego liczenia.

---

## 7. Przeniesienie na fizyczne neurony

1. Wybór topologii (zwycięzca sweepu, najlepiej potwierdzony na kilku seedach).
2. Pełny trening `--train-winner` (HAT→QAT, kwantyzacja do 20 działek trymera).
3. Eksport `hw_config.json` — nastawy każdej płytki.
4. Kalibracja płytek (kolejność H→G→D): τ_syn/τ_mem, pasek LED = V_leak, znak
   +/−, trymer wagi (`pot_pct`), wpięcie wejść w J1/J2/J3.
5. Weryfikacja: test binarny `pulses_to_fire` (tryb CALIB), potem `compare` sim↔hw.
6. Połączenie płytek drutami wg okablowania (fan-in/out ≤ 3 gwarantowane przez GA).

Uwaga: dotychczasowa kalibracja pod tabelkę starego modelu 14/15 neuronów jest
przypisana do TAMTEJ topologii — nowa topologia z GA wymaga dokręcenia od nowa
wg jej własnego `hw_config.json`.

---

## 8. Podsumowanie jednym zdaniem

Zamiast zgadywać architekturę, **GA sam znajduje najlepszą topologię i liczbę
płytek** na realnych danych, respektując limity sprzętu; kalibracja pokazała, że
przy 12 epokach ocena jest wiarygodna, a demo potwierdza, że sieć się uczy — to
działa i ma sens; do finalnego wyniku brakuje tylko mocy obliczeniowej (GPU).

---

## 9. Jak co odpalać (instrukcja komend)

Wszystko odpalamy pythonem z `.venv` repo, z katalogu `ga_neuron_search`.
W PowerShell łamanie linii to backtick `` ` `` (w cmd `^`); można też wpisać
całość w jednej linii. Zamiast `..\.venv\Scripts\python.exe` można aktywować
środowisko (`..\.venv\Scripts\Activate.ps1`) i wołać `python`.

### 9.0. Test poprawności kodu (bez torcha, kilka sekund)
```
..\.venv\Scripts\python.exe test_genome.py
```
Sprawdza niezmienniki genomu (fan-in/out ≤ 3, brak łańcuchów, stałe N). Ma
wypisać „WSZYSTKIE TESTY PRZESZŁY".

### 9.1. Kalibracja budżetu epok (raz, ~kilkanaście min)
```
..\.venv\Scripts\python.exe calibrate.py --n 8 `
  --arch-dir ..\architecture_14_neurons_patryk_09_07 `
  --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train `
  --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val `
  --epochs-grid 12 16 20 --seeds 3
```
Trenuje jedną losową topologię dla różnej liczby epok × seedów i wypisuje
`loss_end | AP śr±std`. Wybierasz najmniejsze `epoki`, gdzie std AP jest małe, a
loss wyraźnie spadł. **Wynik dla naszych danych: 12 epok.**

Parametry: `--n` liczba neuronów topologii testowej; `--epochs-grid` lista
wartości epok do porównania; `--seeds` ile powtórzeń na każdą (do liczenia
wariancji); `--data/--val-data` katalogi spike-CSV (7-kanałowe = `spikes_manifest7`);
`--arch-dir` folder z `snn_hw_pipeline.py`; `--k` próg dekodera (≥k spików = alarm).

### 9.2. Sweep GA — znajdź topologię (długie na CPU)
```
..\.venv\Scripts\python.exe run_search.py --neurons 6 8 10 --mode real `
  --arch-dir ..\architecture_14_neurons_patryk_09_07 `
  --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train `
  --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val `
  --epochs 12 --pop 12 --gens 8 --screen-mult 1 `
  --metric ap --fitness-seeds 1 --parsimony-eps 0.02 --out wyniki_demo
```
Wynik: `wyniki_demo.json` / `.csv` + tabela + wybór N. **Bez `--train-winner`
nie powstaje `hw_config.json`** (do samego demo nie jest potrzebny).

Parametry `run_search.py`:
- `--neurons 6 8 10` — które liczby neuronów porównać (osobny run GA na każdą).
- `--mode real` — trening na danych (jest też `synth` = szybki test mechaniki bez treningu).
- `--arch-dir` — folder z `snn_hw_pipeline.py` (fizyka neuronu Lu.i).
- `--data` / `--val-data` — katalogi spike-CSV treningowe / walidacyjne (7 kanałów!).
- `--epochs 12` — epoki proxy-treningu na jedną ocenę (z kalibracji: 12).
- `--pop 12` — rozmiar populacji GA (ile topologii na pokolenie).
- `--gens 8` — liczba pokoleń (ile razy GA ulepsza populację).
- `--screen-mult 1` — successive-halving wyłączony (1). >1 = wstępny przesiew
  większej puli tanim budżetem (u nas bez sensu, bo tani budżet = szum).
- `--metric ap` — metryka fitness: `ap` (Average Precision, bezprogowa, mało
  szumna) lub `clip_f1`.
- `--fitness-seeds 1` — ile seedów uśrednić na ocenę (więcej = mniejsza wariancja,
  proporcjonalnie dłużej).
- `--parsimony-eps 0.02` — wybór zwycięzcy: najmniejsze N w zasięgu 0.02 fitness
  od najlepszego (mniej płytek przy podobnej jakości).
- `--out wyniki_demo` — prefiks plików wyjściowych.
- (opcjonalnie) `--limit` — max plików/klasę; ZOSTAW PUSTE, inaczej próbuje
  przebudować cache w katalogu danych (WinError 5 na read-only).
- (opcjonalnie) `--seed 0` — ziarno losowości; różne seedy = różne przebiegi.
- (opcjonalnie) `--quiet` — bez logu per-kandydat.

Szybki wariant na próbę (~30–40 min): `--neurons 8 --pop 6 --gens 4`.

### 9.3. Wykres krzywej uczenia (do slajdu, sekundy)
```
..\.venv\Scripts\python.exe plot_history.py wyniki_demo.json
```
Zapisuje `wyniki_demo.png` — best-fitness w kolejnych pokoleniach dla każdego N.

### 9.4. Pełny trening zwycięzcy + nastawy płytek (najlepiej na GPU)
```
..\.venv\Scripts\python.exe export_winner.py --winner-json wyniki_demo.json --n 8 --epochs 60 `
  --arch-dir ..\architecture_14_neurons_patryk_09_07 `
  --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train `
  --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val `
  --out winner8
```
Bierze topologię z `wyniki_demo.json`, robi pełny trening HAT→QAT i zapisuje
`winner8_hw_config.json` (nastawy płytek) + `winner8_winner.pt` (checkpoint).
To jest most „topologia znaleziona tutaj → trening gdzie indziej": na maszynę z
GPU wystarczy skopiować ten folder z kodem, dane spike-CSV i `wyniki_demo.json`.

Parametry `export_winner.py`:
- `--winner-json wyniki_demo.json` — plik z wynikiem sweepu (zawiera topologie).
- `--n 8` — którą liczbę neuronów wziąć; bez tego bierze topologię o najlepszym
  fitness w JSON-ie.
- `--epochs 60` — epoki pełnego cyklu HAT→QAT (dużo więcej niż proxy, bo to
  finalny trening pod sprzęt).
- `--arch-dir`, `--data`, `--val-data` — jak wyżej.
- `--k 2` — próg dekodera (≥k spików D = alarm), spójny ze sweepem.
- `--out winner8` — prefiks plików wyjściowych.

### Skróty (.bat)
`run.bat <plik.py> <argumenty>` odpala dowolny skrypt w `.venv`.
`run_real_sweep.bat` to gotowy sweep. W PowerShell wołaj z `.\`, np.
`.\run.bat test_genome.py`.

### Słowniczek pojęć z logu
- `[eval] 7-3-2-1 loss 4.9->1.9 AP 0.32 clipF1 0.47` — jedna oceniona topologia:
  7 kanałów→3→2→1 neuron, loss spadł (uczy się), AP i clipF1 na walidacji.
- `[N=6] gen 3 best=0.36` — koniec pokolenia 3 dla N=6; `best` ma rosnąć.
- `[parsymonia] ... wybór N=...` — którą liczbę neuronów rekomenduje sweep.
