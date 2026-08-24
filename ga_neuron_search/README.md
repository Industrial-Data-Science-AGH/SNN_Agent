# GA neuron search — algorytm genetyczny topologii SNN dla płytek Lu.i

Poszukiwanie **najlepszej topologii** spikującej sieci neuronowej pod pipeline
sprzętowy Lu.i, przy **zmiennej liczbie neuronów** (sweep po N). Zamiast trenować
jedną, z góry ustaloną architekturę (jak `snn_hw_pipeline.py`, gdzie maski
`MASK_H/MASK_G/MASK_O` są wpisane ręcznie), GA przeszukuje przestrzeń połączeń
i układów warstw, a osobny run dla każdego N odpowiada na pytanie **„ile płytek
naprawdę się opłaca”**.

Wagi, τ i V_leak dalej uczy backprop (faza HAT z `snn_hw_pipeline`). GA ewoluuje
**wyłącznie strukturę** — genom to uogólnienie masek łączności.

## Ograniczenia sprzętowe (twarde niezmienniki)

Każdy genom, na każdym etapie (losowanie, mutacja, krzyżowanie), spełnia:

| Ograniczenie | Skąd | Gdzie pilnowane |
|---|---|---|
| **fan-in ≤ 3** | 3 styki wejściowe płytki | `Genome.violations`, `repair` |
| **fan-out ≤ 3** | 3 styki wyjściowe płytki | `fanout_counts`, `_enforce_fanout` |
| **DAG** | sygnał płynie warstwa→warstwa | z definicji reprezentacji |
| **n\_{k+1} ≤ 3·n\_k** | warstwa s płytek ma 3s wyjść | `sizes_feasible`, repair-fallback |
| **N neuronów = const** | budżet płytek na dany run | mutacje/krzyżowanie zachowują N |
| **neuron decyzyjny: 1, fan-in = min(3, n\_ost)** | jedna decyzja | `_fix_decision` |
| **każda cecha użyta ≥ 1×** (miękko) | nie marnuj kanału encodera | `_cover_features` |

## Reprezentacja genomu

Sieć warstwowa, ale **liczba warstw też ewoluuje**. Warstwa 0 = 7 cech encodera
(`peak, peak_cnt, cv, zcr, flux, hf_lo, hf_hi`), nie liczy się do N. Warstwy
1..L-1 = neurony ukryte Lu.i, warstwa L = 1 neuron decyzyjny.

```python
Genome.layers = [
    [[0,1,5],[2,3,6],[0,4,5],[1,4,6]],  # H1: 4 neurony, wejścia = indeksy cech
    [[0,1,2],[1,2,3],[2,3,0]],          # H2: 3 neurony, wejścia = neurony H1
    [[0,1,2]],                          # D:  1 neuron,  wejścia = neurony H2
]
```
Neuron = lista 1–3 indeksów wejść w warstwie **poprzedniej**. To jest 1:1
uogólnienie `MASK_H/MASK_G/MASK_O` z `snn_hw_pipeline.py` (wariant wąski powyżej).

## Fitness

`fitness(genome) -> float` (więcej = lepiej). Dwa tryby:

- **real** (`RealFitness`): buduje `GenomeNet` z genomu na bazie `LuiLayer`,
  robi **krótki proxy-trening** (sama faza HAT, mały `--limit`, kilka epok),
  zwraca `F1` z walidacji minus drobna kara za średni fan-out (przy równym F1
  wybierz oszczędniejsze okablowanie). Dane ładowane raz (`SpikeClips`),
  wynik cache'owany po `Genome.key()`.
- **synth** (`synth_fitness`): tania heurystyka bez torcha (pokrycie cech,
  fan-in ~3, 1–2 warstwy) — tylko do testów mechaniki GA i preselekcji.

Kara za liczbę neuronów jest **zbędna** — N jest stałe w obrębie runu.
Porównanie między N robi `run_search.py`.

## Operatory GA

- **mutacje**: `rewire` (zmień wejście), `fanin` (dodaj/usuń wejście),
  `move_neuron` (przenieś neuron między warstwami), `split_layer`
  (rozbij warstwę → +głębokość), `merge_layers` (scal → −głębokość).
- **krzyżowanie**: struktura warstw z rodzica A, okablowanie mieszane A/B.
- po każdej operacji **`repair`** przywraca wszystkie niezmienniki.
- selekcja turniejowa + elityzm, cache fitnessu po hashu topologii.

## Pliki

```
genome.py            reprezentacja, walidacja, repair, mutacje, krzyżowanie (bez torcha)
ga.py                silnik GA (jeden run = jedno N; ewaluacja hurtem przez fitness.batch)
net.py               GenomeNet z genomu na LuiLayer + loss/eval (torch)
fitness.py           synth_fitness + RealFitness + ParallelFitness (proxy-trening, pula procesów)
run_search.py        CLI: sweep po N, tabela F1 vs liczba neuronów, zapis JSON/CSV
winner.py            train_full (HAT→QAT) + export_genome_config + tune_k
validate_hw_config.py  round-trip: odtwórz sieć z hw_config_*.json i sprawdź metryki
test_genome.py       testy niezmienników i zbieżności GA (bez torcha)
```

## Uruchomienie

```bash
# test mechaniki (bez torcha) — musi przejść
python test_genome.py

# szybki sweep na fitnessie syntetycznym
python run_search.py --neurons 4 6 8 10 --mode synth --out wyniki_synth

# realny sweep z proxy-treningiem na spike-CSV (równolegle przez --workers)
python run_search.py --neurons 4 6 8 10 \
    --mode real --data ../data/spikes_csv \
    --arch-dir ../architecture_14_neurons_patryk_09_07 \
    --limit 120 --epochs 4 --pop 24 --gens 15 --out wyniki_real
```

Wynik: dla każdego N najlepsza topologia + jej F1, tabela porównawcza i pliki
`*.json` / `*.csv`. Najlepszą topologię przenosisz do treningu docelowego,
podmieniając `topo()` / maski w `snn_hw_pipeline.py` (albo trenując pełny cykl
HAT→QAT bezpośrednio przez `GenomeNet`).

## Równoległa ewaluacja (`--workers`)

W trybie `real` z `--workers > 1` ocena osobników idzie przez `ParallelFitness`
(pula procesów, jeden `RealFitness` + `torch.set_num_threads(1)` na workera).
GA ocenia każdą generację **hurtem** (`fitness.batch`), zachowując kolejność
i dedup przez cache topologii — wyniki są **bit-for-bit identyczne** z wersją
sekwencyjną (zweryfikowane w smoke teście). Przykład na 18-rdzeniowej maszynie:

```bash
python run_search.py --neurons 4 6 8 10 --mode real ... --workers 18 --out wyniki_real
```

- `--workers 1` = stara ścieżka sekwencyjna.
- `--workers` domyślnie = `os.cpu_count()`.
- W procesie głównym `torch.set_num_threads(1)` jest wymuszone — małe SNN nie
  zyskują na wielowątkowym matmul, a jeden wątek w głównym == jeden wątek w
  workerach, więc porównania i determinizm są spójne.
- `--device cpu|cuda` (domyślnie cuda-jak-dostępne, inaczej cpu). Na Macu trzymaj
  **cpu** — MPS zmierzone ~3× wolniejsze dla tego rozmiaru sieci.

## Status i dalsze kroki

- `genome.py`, `ga.py`, `run_search.py`, `test_genome.py` — przetestowane
  (niezmienniki + sweep synth przechodzą).
- `net.py`, `fitness.py`, `winner.py`, `validate_hw_config.py` — napisane pod
  istniejące `LuiLayer`/`SpikeClips`; wymagają środowiska z torch i katalogu
  spike-CSV (proxy-trening).
- **Zrobione (2026-08-24)**: równoległa ewaluacja (`--workers`, `ParallelFitness`),
  stabilszy/tańszy search domyślnie (`--screen-mult 3`, `--fitness-seeds 3`),
  strojenie wdrożenia (`--pos-weight-grid`, `--tune-k`), round-trip walidacja
  eksportu (`validate_hw_config.py`).
- **Najlepsze parametry (2026-08-24)**: topologia **`7→4→3→1`** (8 płytek, N=8),
  `pos_weight=1.4`, `tuned_k=3` → **clip-F1 0.658, FA 0.158, rec 0.80, prec 0.56** —
  `wyniki_fine_tuned_hw_config_N8.json` (nastawy płytek). Kampania pokazała, że
  ranking searchu po `--metric clip_f1` NIE przebija ranking po `ap` dla tego
  zbioru (mniejsze sieci gorzej trenują w pełnym HAT→QAT) — topologia z oryginalnego
  sweepu pozostaje zwycięska, zysk jest w strojeniu `pos_weight` (1.4 zamiast 1.5).
- **TODO opcjonalnie**: front Pareto zamiast osobnych runów per N, eksport
  wygranej topologii wprost do `hw_config.json`, walidacja `compare` na prawdziwych
  płytkach (nagrania tej konfiguracji).

## Środowisko (Windows)

Repo ma już gotowy `.venv` (Python 3.12, uv) z **torch 2.12.0, numpy 2.4.2,
snntorch 0.9.4** — nic nie trzeba instalować. Uruchamiaj przez dołączone skrypty:

```bat
run.bat test_genome.py                 :: testy (bez torcha, szybkie)
run.bat run_search.py --neurons 4 6 8 10 --mode synth --out wyniki_synth
run_real_sweep.bat                     :: realny sweep z proxy-treningiem (torch)
```

`run_real_sweep.bat` używa danych 7-kanałowych `spikes_manifest7\train` i
`\val` (pipeline wymaga 7 kanałów = `CH_IN`). Odpowiednik ręcznie:

```bat
..\.venv\Scripts\python.exe run_search.py --neurons 4 6 8 10 --mode real ^
  --arch-dir ..\architecture_14_neurons_patryk_09_07 ^
  --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train ^
  --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val ^
  --limit 120 --epochs 4 --pop 24 --gens 15 --out wyniki_real
```

Regulacja kosztu: `--limit` (plików/klasę), `--epochs`, `--pop`, `--gens`.
Start małe (`--limit 60 --epochs 3 --pop 12 --gens 6`), potem zwiększaj.

Uwaga o domyślnych (stabilniejsze, ale droższe): `--screen-mult` domyślnie **3**
(successive-halving: oceń 3·pop kandydatów tanim budżetem, zatrzymaj pop
najlepszych) i `--fitness-seeds` domyślnie **3** (średnia fitness po 3 seedach —
mniejsza wariancja selekcji). Dla szybkiego eksperymentu: `--screen-mult 1
--fitness-seeds 1`.

## Dotrenowanie zwycięzcy i nastawy płytek (`--train-winner`)

Sweep robi tylko krótki proxy-trening (do rankingu topologii) — te wagi są za
surowe na trymery. Żeby dostać **realne nastawy sprzętowe**, dodaj `--train-winner`:
po sweepie bierze najlepszy genom, trenuje go **pełnym cyklem HAT→QAT**
(kwantyzacja do 20 działek trymera, zamrożenie znaków) i eksportuje.

```bat
run.bat run_search.py --neurons 4 6 8 10 --mode real ^
  --arch-dir ..\architecture_14_neurons_patryk_09_07 ^
  --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train ^
  --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val ^
  --epochs 4 --pop 24 --gens 15 ^
  --train-winner --winner-epochs 60 --out wyniki_real
```

Wynik dla zwycięzcy:
- `wyniki_real_hw_config_N{n}.json` — **nastawy każdej płytki**: `synapses[].pot_pct`
  (trymer %), `sign` (+/−), `from`+`port` (które wejście do J1/J2/J3),
  `led_bar_pct` (pasek LED = V_leak), `tau_syn_ms`/`tau_mem_ms`,
  `pulses_to_fire_100Hz` (kontrola sim↔hw).
- `wyniki_real_winner_N{n}.pt` — checkpoint modelu.
- tabela nastaw wypisana w konsoli (jak `kalibracja_sciaga_v3.md`).

Flagi: `--winner-per-n` dotrenowuje zwycięzcę **każdego N** (nie tylko globalnego),
`--winner-epochs` steruje długością pełnego treningu.

`winner.py` zawiera `train_full()` (pełny harmonogram HAT→QAT),
`export_genome_config()` (uogólnienie `export_config` z pipeline'u na dowolną
liczbę warstw `GenomeNet`) i `tune_k()` (przebieg progu dekodera).

### Strojenie do wdrożenia: `--pos-weight`, `--pos-weight-grid`, `--tune-k`

FA-rate (fałszywe alarmy) zwycięzcy można obniżyć bez zmiany architektury:

- `--pos-weight X` — waga klasy pozytywnej w BCE proxy-treningu **i** w
  pełnym treningu (`train_full`). Domyślnie 3.0.
- `--pos-weight-grid 1.5 2.0 3.0` (razem z `--train-winner`) — dotrenuj zwycięzcę
  dla każdej wartości, zatrzymaj najlepszy **clip-F1**, zapisz
  `{out}_tuned_winner_N{n}.pt` + `{out}_tuned_hw_config_N{n}.json`.
- `--tune-k 1 2 3 4 5 6` — przebieg progu dekodera **k** (≥ k spików D = alarm)
  na wytrenowanym modelu; wybiera k o najlepszym clip-F1 i dopisuje `tuned_k` +
  tabelę `tune_k_table` do eksportu.

### Walidacja eksportu (`validate_hw_config.py`)

Round-trip bez płytek: odtwarza sieć z `hw_config_*.json` (trymer %, znak, port,
τ, V_leak → parametry modelu, `quantize=True`) i porównuje clip-F1 na zbiorze
walidacyjnym z `winner_val_metrics` zapisanym w configu. Łapie odwrócone znaki,
zgubione synapsy i trymery poniżej rozdzielczości (zerowane przy `pot_pct < 5`).

```bash
python validate_hw_config.py \
    --config wyniki_real_hw_config_N8.json \
    --arch-dir ../architecture_14_neurons_patryk_09_07 \
    --data  ../architecture_14_neurons_patryk_09_07/spikes_manifest7/train \
    --val-data ../architecture_14_neurons_patryk_09_07/spikes_manifest7/val \
    --k 6 --tol 0.02
```

To jest **programowa namiastka** walidacji sprzętowej: realne porównanie to
`snn_hw_pipeline.py compare` z nagraniami z płytek (wymaga nagrań tej konfiguracji).
`calibrate.py` kalibruje budżet epok proxy-treningu, nie płytki.

## Uruchamianie bez plików .bat

Skrypty `.bat`/`.ps1` to tylko skróty. To samo ręcznie — najpierw wejdź do folderu:

```powershell
cd D:\agh\IDS\SNN_Agent\ga_neuron_search
```

Opcja A — wywołaj python z venv bezpośrednio (bez aktywacji):

```powershell
..\.venv\Scripts\python.exe test_genome.py
..\.venv\Scripts\python.exe run_search.py --neurons 4 6 8 10 --mode synth --out wyniki_synth
```

Opcja B — aktywuj venv, potem zwykłe `python`:

```powershell
..\.venv\Scripts\Activate.ps1     # PowerShell  (w cmd: ..\.venv\Scripts\activate.bat)
python test_genome.py
python run_search.py --neurons 4 6 8 10 --mode synth --out wyniki_synth
deactivate                        # gdy skończysz
```

Realny sweep + strojenie zwycięzcy (jedna linia; w PowerShell łamanie wiersza to backtick `` ` ``, w cmd `^`):

```powershell
..\.venv\Scripts\python.exe run_search.py --neurons 4 6 8 10 --mode real `
  --arch-dir ..\architecture_14_neurons_patryk_09_07 `
  --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train `
  --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val `
  --epochs 4 --pop 24 --gens 15 --train-winner --winner-epochs 60 --out wyniki_real
```

Albo wszystko w jednej linii bez znaków łamania:

```powershell
..\.venv\Scripts\python.exe run_search.py --neurons 4 6 8 10 --mode real --arch-dir ..\architecture_14_neurons_patryk_09_07 --data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\train --val-data ..\architecture_14_neurons_patryk_09_07\spikes_manifest7\val --epochs 4 --pop 24 --gens 15 --train-winner --winner-epochs 60 --out wyniki_real
```

Jeśli `Activate.ps1` zablokuje polityka wykonywania, na bieżącą sesję:
`Set-ExecutionPolicy -Scope Process -Bypass`.
