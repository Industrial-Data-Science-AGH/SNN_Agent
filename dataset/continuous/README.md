# continuous_eval — generator ciągłego datasetu ewaluacyjnego

Generator deterministycznej **pary** strumieni audio, `continuous-val` i
`continuous-test`, każdy z dokładnie **5 zdarzeniami rozbicia szkła** w
losowych, nienachodzących pozycjach. Etap 3 master pipeline'u Marcela.

Jeden `--seed` (K3) daje jedną parę: `val` jest budowany pierwszy, a jego
wybory (pliki tła ESC-50 po `group_id`, pliki źródłowe szkła VOICe po
`source_stem`) są wykluczone z puli `test` — patrz sekcja 5 i 9 niżej.

## Szybki start

```bash
# 3 pary val+test z różnymi seedami nadrzędnymi
# root: SNN_Agent
python -m dataset.continuous.eval.cli \
    --glass-annotation-dir dataset/clean/clean/annotation \
    --glass-audio-root     dataset/clean/clean/audio \
    --glass-allowed-stems  dataset/clean/clean/target/synthetic_target_test.txt \
    --train-stems-files    dataset/clean/clean/source/synthetic_source_training.txt \
                           dataset/clean/clean/source/synthetic_source_validation.txt \
    --train-manifest       dataset/versions/v2.0.0/manifest.csv \
    --background-dir       data/ESC-50-master/audio \
    --seeds 42 43 44 \
    --out-dir dataset/continuous/out
```

Wynik dla każdego seeda N: `continuous_eval_seedN_val.wav` +
`continuous_eval_seedN_val.manifest.json` oraz analogicznie `..._test.wav` /
`..._test.manifest.json`. `val` i `test` mają własne, wyprowadzone deterministycznie
pod-seedy (`derive_seed(N, "val")` / `derive_seed(N, "test")`) — manifest
każdego z nich zapisuje to w polach `role` i `parent_seed`.

## Testy automatyczne

```bash
python -m pytest dataset/continuous/tests/ -v
# 33 passed — bez torcha, bez plików produkcyjnych, uruchamialne w CI
```

---

## Skąd pochodzą dźwięki

| rola | źródło | ścieżka |
|---|---|---|
| **tło** | ESC-50 | `data/ESC-50-master/audio/*.wav` |
| **szkło** | VOICe | `dataset/clean/clean/audio/synthetic_XXX.wav` |

**Tło (ESC-50):** losowe nagrania z 50 kategorii (psy, deszcz, silniki itd.).
Kategoria 39 (`glass_breaking`) jest automatycznie wykluczona na podstawie
`meta/esc50.csv`. Każdy segment tła dostaje `kind` z metadanych ESC-50
(`animal`, `stationary`, `speech`, `loud_event`) — potrzebne do liczenia FA/h
per kind przez Marcela.

**Szkło (VOICe):** pliki `synthetic_XXX.wav` to wielominutowe miksy wielu
zdarzeń naraz. Z pliku adnotacji (`annotation/synthetic_XXX.txt`) wiemy
dokładnie w których sekundach brzmi szkło i wycinamy tylko te fragmenty.
`--glass-allowed-stems` ogranicza, z których miksów wolno korzystać.

---

## Decyzje implementacyjne

### 1. `--glassbreak-mode clean` jako default

VOICe to miksy — glassbreak prawie zawsze nachodzi na gunshot lub babycry
(stats.md: 3961/4444 zdarzeń). Dwa tryby:

| tryb | co zwraca | kiedy używać |
|---|---|---|
| `clean` (domyślny) | tylko glassbreak bez nakładki na inne klasy | mierzysz odpowiedź sieci czysto na szkło |
| `background` | wszystkie glassbreak, niezależnie od nakładek | trudniejszy, bardziej realistyczny wariant |

W trybie `background` manifest odnotowuje `is_contaminated: true` i listę
`overlapping_labels` — konsument może filtrować wyniki po zdarzeniu.

### 2. Warm-up 30 sekund

Pierwsze 30 sekund strumienia to samo tło — brak zdarzeń szkła. Cel: dać
enkoderowi czas na ustabilizowanie się przed pierwszym zdarzeniem, analogicznie
do realnego deploymentu. Parametr `--warmup-s` (domyślnie 30).

Warmup jest zapisany w manifeście (`config.warmup_s`) i **wyłączony z liczenia
FA/h** (`warmup_excluded_from_fa: true`). Marcel liczy FA/h na odcinku
`[warmup_s, duration_s]`, nie na całym pliku.

### 3. Brak skoku amplitudy w tle

Tło budowane jest z losowego offsetu wewnątrz każdego pliku ESC-50, ale
**bez zawijania** (nie sklejamy końca z początkiem). Zawijanie powodowało skok
amplitudy, który enkoder rozpoznawał jako zdarzenie uderzeniowe i generował
fałszywe alarmy.

### 4. Kind w segmentach tła

Każdy segment tła ma pole `kind` z metadanych ESC-50 (`animal`, `stationary`,
`speech`, `loud_event`). Bez tego nie dało się policzyć FA/h per kind —
nie wiadomo ile godzin każdego rodzaju tła jest w strumieniu.

### 5. Sprawdzanie rozłączności eval/train

**Szkło (VOICe)** — `--train-stems-files`:
Porównuje stemmy plików szkła w puli eval z każdą podaną listą treningową.
Jeśli cokolwiek się pokrywa — `ValueError`. Blokuje generację.

**Tło (ESC-50)** — `--train-manifest`:
`collect_background_pool` czyta `group_id` wierszy `split == "train"` z
`manifest.csv` i **aktywnie wyklucza je z puli tła** — te pliki nigdy nie są
losowane, nie tylko odnotowywane. (Wcześniejsza wersja tego README opisywała
to jako "raportowane, nie blokujące" — to nie jest już zgodne z kodem; K3
zaostrzył to do twardego wykluczenia, żeby continuous-eval było naprawdę
niewidziane przez model.) Overlap, gdyby mimo to wystąpił, i tak trafia do
`config.overlap_check.background` w manifeście dla śladu.

Wynik obu sprawdzeń trafia do `config.overlap_check` w manifeście.
### 6. Standard audio: 44100 Hz / mono / PCM_16

Przyjęty z `dataset/versions/v2.0.0/stats.md`. Każdy plik źródłowy
resamplowany przez `scipy.signal.resample_poly` niezależnie od natywnego SR.
**Wymaga potwierdzenia przez Patryka.**

### 7. Zabezpieczenie przed clippingiem

Po zmiksowaniu, jeśli szczyt > 0.97, **cały miks skalowany proporcjonalnie
w dół**. Nie per-sample clip — to zniekształciłoby kształt fali dokładnie
w oknach zdarzeń szkła.

### 8. Deterministyczność

Cała losowość przez jeden `random.Random(seed)` w ustalonej kolejności:
wybór klipów → pozycje → skale głośności → kolejność tła → offsety w plikach.
Ten sam seed + te same pliki = identyczny WAV.

### 9. Para continuous-val / continuous-test z jednego seeda (K3)

Jeden `--seed`/element `--seeds` generuje **dwa** strumienie, nie jeden:

1. `val` budowany jako pierwszy, z pod-seedem `derive_seed(seed, "val")`
   (deterministyczny hash `sha256(f"{seed}:val")`, nie zwykła arytmetyka —
   inaczej `val` i `test` z tym samym `seed` dałyby identyczny strumień,
   bo `build_stream` zużywa `random.Random(seed)` w ustalonej kolejności).
2. Z `val.background_segments` zbierane są `group_id`, a z `val.events` —
   `source_stem` plików VOICe. Oba zbiory są **wykluczone** przy budowie
   `test`: tło przez rozszerzony `collect_background_pool(..., extra_excluded_group_ids=...)`,
   szkło przez odfiltrowanie kandydatów po całym pliku źródłowym (nie tylko
   konkretnym interwale czasowym — bezpieczniej: `test` nie zawiera żadnego
   fragmentu pliku widzianego w `val`).
3. `validate_val_test_disjoint` sprawdza to niezależnie, od zewnątrz, po
   zbudowaniu obu strumieni — łapie regresję, nawet jeśli ktoś kiedyś zepsuje
   logikę wykluczania wewnątrz `build_val_test_pair`.
4. Wczesna walidacja marginesu: zaraz po zbudowaniu `val`, liczba pozostałych
   (niewykluczonych) `group_id` tła jest porównywana z liczbą, której użył
   `val` (ten sam `duration_s` = podobne zapotrzebowanie) — jeśli zostaje
   mniej, generacja przerywa się od razu z czytelnym komunikatem, zamiast
   budować cały `test` i dopiero wtedy wywalić się głęboko w
   `collect_background_pool`.

Manifest każdego z dwóch plików ma pole `role` (`"val"`/`"test"`) i
`parent_seed` (wspólny `--seed`), a `seed` to faktyczny pod-seed użyty do
zbudowania TEGO konkretnego strumienia.

---

## Kontrakt manifestu (schema 1.1.0) — do akceptacji przez Marcela i Patryka

```json
{
  "manifest_schema_version": "1.1.0",
  "role": "val",
  "parent_seed": 42,
  "seed": 1789234561,
  "audio": {
    "path": "continuous_eval_seed42_val.wav",
    "sha256": "abc123...",
    "sample_rate": 44100,
    "channels": 1,
    "subtype": "PCM_16",
    "duration_s": 600.0
  },
  "config": {
    "glassbreak_mode": "clean",
    "warmup_s": 30.0,
    "warmup_excluded_from_fa": true,
    "overlap_check": {
      "train_stems_files": ["...source_training.txt"],
      "result": {"source_training.txt": []}
    }
  },
  "events": [
    {
      "index": 0,
      "start_s": 47.23,
      "end_s": 48.09,
      "duration_s": 0.86,
      "source_stem": "synthetic_014",
      "is_contaminated": false,
      "overlapping_labels": [],
      "gain_db": 1.2
    }
  ],
  "background_segments": [
    {
      "path": "data/ESC-50-master/audio/1-100032-A-0.wav",
      "source": "ESC-50",
      "kind": "animal",
      "stream_start_s": 0.0,
      "stream_end_s": 5.02
    }
  ]
}
```

**Jak Marcel liczy metryki z manifestu:**

| metryka | jak liczyć                                                                                                                                                    |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| detected / 5 | dla każdego `[start_s, end_s]` — czy detektor podniósł alarm (+tolerancja)                                                                                    |
| event recall | `detected / 5`                                                                                                                                                |
| false alarms/h | alarmy poza wszystkimi oknami [start_s, end_s], liczone NA ODCINKU [warmup_s, duration_s] (warmup wyłączony), podzielone przez (duration_s - warmup_s) / 3600 |
| FA/h per kind | jak wyżej, ale sumując tylko czas segmentów tła z danym "kind" (z background_segments); group_id pozwala powiązać segment z rekordem w manifeście treningowym |
| latency | config.overlap_check.background zawiera listę group_id ESC-50 obecnych w obu datasetach — oczekiwany, raportowany, nie błąd                                   |

---
