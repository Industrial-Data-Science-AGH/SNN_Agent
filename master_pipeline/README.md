# Instrukcja treningu SNN dla Wiktora

Sep 27, 2026

## 1. Zakres i cel

Ta instrukcja prowadzi przez **cały** proces od uruchomienia treningu (M2) do przekazania gotowego pakietu championa (M5): GA → ewaluacja rozszerzona → ewaluacja testowa/ciągła → eksport pod hardware → wybór championa spośród wielu przebiegów → spakowanie go do przekazania.

**Zależność, o której nie wiem** — M5 zależy też od "P1", ale nie znam treści tego kroku (nie było mi przekazane). Jeśli P1 narzuca coś dodatkowego (np. inny format pakietu, dodatkowy podpis, inny storage), dopytaj Marcela przed końcowym przekazaniem pakietu.

Odbiorcy końcowego pakietu (M5): Ty i Patryk ładujecie dokładnie oceniony model; Karolina i Andrzej dostają CSV/JSON metryk z ciągłej ewaluacji (M4).

## 2. Wymagania wstępne

- Środowisko: aktywny venv `edge_env` (widoczny w promptach Marcela), Python 3.10, `torch`, `numpy`, `scipy`, `librosa`, `soundfile` zainstalowane.
- Repo `SNN_Agent` sklonowane, pracujesz z katalogu `master_pipeline/`.
- Dane treningowe/walidacyjne/testowe (`arch/spikes_v2/{train,val,test}`) obecne — bez nich GA i ewaluacje nie ruszają.
- **Ciągły dataset ewaluacyjny (`dataset/continuous/out/*.manifest.json` + audio) NIE jest w gicie.** Trzeba go odtworzyć osobno komendą z `pipeline_config.py` (pole `DataConfig.continuous_eval`, sekcja komentarza tam wyjaśnia jak). Bez tego Etap 5 (ciągła ewaluacja, M4) zostanie pominięty z jasnym komunikatem — to nie błąd, tylko brakujące dane.
- `config.json` w `master_pipeline/` wskazuje poprawne ścieżki do wszystkich powyższych (`data.train`, `data.val`, `data.test`, `data.continuous_eval`).

## 3. Krok 1: uruchomienie pełnego treningu

```bash
cd ~/IDS/SNN/SNN_Agent/master_pipeline
python3 pipeline.py --config config.json run-all
```

**Ważne o kolejności flag:** `--config`, `--device`, `--workers`, `--resume` i inne globalne flagi MUSZą stać PRZED nazwą subkomendy (`run-all`/`train-ga`/`evaluate`/`continuous-eval`). To składnia argparse, nie błąd:

```bash
# DOBRZE:
python3 pipeline.py --config config.json --workers 8 run-all
# ŹLE (rzuci "unrecognized arguments"):
python3 pipeline.py run-all --config config.json
```

**Benchmark liczby workerów (`--workers auto`, domyślne):** na starcie pipeline odpala krótkie, ale PRAWDZIWE przebiegi GA dla różnych liczby workerów (1,2,4,8,12,16), żeby wybrać najszybszą konfigurację. To potrafi trwać długo — jeśli chcesz to pominąć, podaj liczbę jawnie: `--workers 8` (dopasuj do liczby rdzeni).

**Co się dzieje w `run-all` (kolejno, każdy etap można pominąć przy `--resume` jeśli już policzony):**

1. Etap 1 — GA (poszukiwanie topologii)
2. Etap 2 — ewaluacja na `spikes_ext`
3. Etap 3 — ewaluacja testowa/ciągła (`continuous_test`), tu powstaje checkpoint
4. Etap 4 — eksport pod hardware
5. Etap 5 — ciągła ewaluacja 600s (M4, **opcjonalna**: pomijana z komunikatem, jeśli brak checkpointu z Etapu 3 albo brak `dataset/continuous/out/*.manifest.json`)

Jeśli proces się przerwie (Ctrl+C, awaria zasilania itp.), wznów go z tego samego katalogu runu zamiast zaczynać od zera:

```bash
python3 pipeline.py --config config.json --resume runs/run_XXXXXXXX_XXXXXX run-all
```

## 4. Krok 2: co sprawdzić po zakończeniu biegu

Otwórz `runs/run_XXXXXXXX_XXXXXX/manifest.json` i sprawdź:

- `status == "COMPLETED"` — jeśli nie, bieg się przerwał albo jeszcze trwa.
- `metrics.hardware_export.encoder_hash` i `metrics.hardware_export.dataset_manifest_hash` są obecne (niepuste) — bez nich `champion.py` później ODRZUCI ten bieg jako niezgodny protokołem, nawet jeśli reszta wygląda dobrze.
- `metrics.continuous_eval.recall_mean` — bliski 1.0 to dobry znak, wartość bardzo niska albo `None` oznacza problem (albo Etap 5 został pominięty — sprawdź log pipeline'u wyżej, powinien być jasny komunikat dlaczego).
- `metrics.continuous_eval.fa_per_hour_mean` i `.fa_per_hour_ci_pooled` — fałszywe alarmy na godzinę w rozsądnym zakresie; górna granica CI powinna być skończona i dodatnia nawet przy zerowym FA (to celowe, nie błąd).
- `metrics.continuous_eval.decoder_rule` — para (k, w) użyta do decyzji o alarmie; zapisz sobie te wartości, przydadzą się przy pakowaniu.
- `execution_times_sec` — jeśli któryś etap trwał podejrzanie krótko/długo względem pozostałych, to sygnał, że mógł czegoś nie policzyć poprawnie (np. zakodować tylko część strumienia).

## 5. Krok 3: powtórzenie biegów i wybór championa

`champion.py` porównuje przebiegi MIĘDZY sobą — jeden `run_XXXX` to za mało, żeby było z czego wybierać. Zrób co najmniej 2–3 pełne przebiegi (różne seedy i/lub zakresy architektur w `config.json`), każdy zakończony `status=="COMPLETED"`.

Potem, z głównego katalogu repo:

```bash
cd ~/IDS/SNN/SNN_Agent
python3 champion.py --runs-dir runs
```

Sprawdź `runs/champion.json`:

- `status == "ok"` — wybrano zwycięzcę.
- `status == "infeasible"` — źDEN kandydat zgodny protokołem nie osiąga budżetu FA/h przy żadnym progu decyzji. To czytelny sygnał o architekturze/danych, nie błąd do zignorowania — nie wybieraj “najmniej złego zera” na siłę, zgłoś to Marcelowi.
- `status == "no_candidates"` — żaden bieg nie ma kompletu wymaganych pól (patrz Krok 2) — wróć i dopełnij brakujące etapy.
- Pole `excluded` / `dropped_other_protocol` — pokazuje, które przebiegi zostały odrzucone i dlaczego (np. inny `encoder_hash`, różny commit źródłowy) — warto przejrzeć, jeśli wynik jest zaskakujący.

## 6. Krok 4: pakowanie championa

```bash
cd master_pipeline
python3 package_champion.py --champion-json ../runs/champion.json --out-dir ../packages/champion_$(date +%Y%m%d)
```

Skrypt sam zbierze checkpoint, manifest i config biegu, topologię, listę wag z sha256, regułę dekodera (k, w), status kalibracji i tzw. golden replay (deterministyczne wejście/wyjście do weryfikacji). **Automatycznie** na końcu odpala też weryfikację w OSOBNYM procesie — szukaj w logu linii:

```
[VERIFY] WSZYSTKO OK.
```

Jeśli zamiast tego zobaczysz `[BŁĄD]` przy weryfikacji — **nie przekazuj tego pakietu dalej**, dopóki się to nie wyjaśni (możliwe przyczyny: uszkodzony plik, niedeterministyczna kwantyzacja, zła wersja `net.py`).

Po udanym spakowaniu, w `packages/champion_.../package_manifest.json` znajdziesz podział plików na:

- `small_files_for_git` — JSON-y, można commitować normalnie,
- `large_files_for_storage` — zwykle `champion_checkpoint.pt` — **NIE commituj tego do gita**, przekaż przez uzgodniony storage (spytaj Marcela, jaki dokładnie).

Proces PR: kieruj PR (#47 lub kolejny) do `master`. Po jego squashu, kolejny etap zaczynaj z **aktualnego** `master`, żeby nie powielać starej historii.

## 7. Znane ograniczenia — na co uważać

- **Brak kalibracji na fizycznym sprzęcie.** `calibration_status.json` w pakiecie mówi to wprost: wzmocnienie enkodera (`gain`) jest liczone na cyfrowym bliźniaku (`encoder_twin.py`), NIE na prawdziwym ADC płytki Lu.i. Model oceniony w ten sposób nie jest automatycznie gotowy pod produkcję na sprzęcie.
- **`decoder_k=2` zahardkodowane w 4 miejscach** (`ga_runner.py`, `hardware.py`), niezależnie od faktycznej reguły operacyjnej (k, w) używanej przy wyborze championa (`recall_fa`). To dwie różne ścieżki decyzyjne — nie powinny ze sobą kolidować, ale warto o tym wiedzieć czytając kod.
- **`recall_tolerance_s` domyślnie 0.0** w ciągłej ewaluacji — to decyzja domyślna, formalnie niepotwierdzona przez Marcela. Jeśli wyniki recall wyglądają dziwnie nisko, to jedno z miejsc do sprawdzenia.
- **Zależność “P1” w M5 nieznana** — patrz sekcja 1.
- Reguła dekodera (k, w) użyta przy wyborze championa NIE jest nigdzie trwale zapisana przez GA — `continuous_eval.py`/`package_champion.py` odtwarzają ją na żywo przez ponowne wywołanie `RealFitness.stream_recall`. Jeśli to wywołanie da inny wynik przy innym seedzie środowiska, możesz zobaczyć inną regułę niż oczekiwana — zgłoś, jeśli się to zdarzy.

## 8. Rozwiązywanie typowych problemów

**`pipeline.py: error: unrecognized arguments: --config config.json`** Globalne flagi (`--config`, `--device`, `--workers`, `--resume` itd.) stały PO subkomendzie. Przenieś je przed `run-all`/`train-ga`/`evaluate`/`continuous-eval` (patrz sekcja 3).

**`KeyboardInterrupt` na etapie “benchmark liczby workerów”** To nie błąd — ktoś przerwał proces (Ctrl+C) w trakcie fazy `--workers auto`. Uruchom ponownie i poczekaj, albo pomiń benchmark podając `--workers N` jawnie.

**`[ETAP 5/5] Pomijam — brak checkpointu z Etapu 3`** Etap 3 (`run_final_evaluation_stage`) się nie zakończył w tym runie. Uruchom go najpierw (`run-all` od początku albo `--resume` na tym samym runie).

**`[ETAP 5/5] Pomijam — brak *.manifest.json`** Dataset ciągły (`dataset/continuous/out`) nie został odtworzony na tej maszynie — audio nie jest w gicie. Odtwórz go komendą opisaną w `pipeline_config.py` (`DataConfig.continuous_eval`), albo odpal ręcznie:

```bash
python3 continuous_eval.py --checkpoint <ścieżka> --continuous-dir <ścieżka> --dataset-manifest-csv <ścieżka>
```

**`champion.py` zwraca `status: "infeasible"` albo `"no_candidates"`** Patrz sekcja 5 — to nie awaria skryptu, tylko czytelny komunikat o stanie Twoich przebiegów/architektury.

**Cokolwiek innego (prawdziwy traceback, nie `KeyboardInterrupt` ani komunikat `[POMIJAM]`)** Skopiuj pełny traceback i wyślij Marcelowi — większość celowych “pominięć” w tym pipeline jest jawnie opisana w logu, więc nieopisany traceback to zwykle realny błąd, nie zaprojektowane zachowanie.
