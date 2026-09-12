# continuous_eval — generator ciągłego datasetu ewaluacyjnego

Generator deterministycznego strumienia audio z **dokładnie 5 zdarzeniami
rozbicia szkła** w losowych, nienachodzących pozycjach. Etap 3 master
pipeline'u Marcela.

## Szybki start

```bash
# 3 warianty z różnymi seedami
python -m dataset.continuous.eval.cli \
    --glass-annotation-dir dataset/clean/clean/annotation \
    --glass-audio-root     dataset/clean/clean/audio \
    --glass-allowed-stems  dataset/clean/clean/target/synthetic_target_test.txt \
    --train-stems-files    dataset/clean/clean/source/synthetic_source_training.txt \
                           dataset/clean/clean/source/synthetic_source_validation.txt \
    --background-dir       data/ESC-50-master/audio \
    --seeds 42 43 44 \
    --out-dir dataset/continuous/out
```

Wynik dla każdego seeda: `continuous_eval_seedXX.wav` + `continuous_eval_seedXX.manifest.json`.

## Testy automatyczne

```bash
python -m pytest dataset/continuous/tests/ -v
# 23 passed — bez torcha, bez plików produkcyjnych, uruchamialne w CI
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

```bash
--train-stems-files dataset/clean/clean/source/synthetic_source_training.txt \
                    dataset/clean/clean/source/synthetic_source_validation.txt
```

Przed wygenerowaniem skrypt porównuje stemmy plików szkła w puli eval
z każdą podaną listą treningową. Jeśli cokolwiek się pokrywa — `ValueError`
z listą nakładających się plików. Wynik (pusta lista = brak overlap) trafia
do manifestu w `config.overlap_check`.

ESC-50 (tło) **nie jest sprawdzane** — model trenował na ESC-50 jako
negatywach, więc jego obecność w tle eval jest oczekiwana i poprawna.

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

---

## Kontrakt manifestu (schema 1.1.0) — do akceptacji przez Marcela i Patryka

```json
{
  "manifest_schema_version": "1.1.0",
  "seed": 42,
  "audio": {
    "path": "continuous_eval_seed42.wav",
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

| metryka | jak liczyć |
|---|---|
| detected / 5 | dla każdego `[start_s, end_s]` — czy detektor podniósł alarm (+tolerancja) |
| event recall | `detected / 5` |
| false alarms/h | alarmy poza oknami zdarzeń, na odcinku `[warmup_s, duration_s]` / `(duration_s - warmup_s)` × 3600 |
| FA/h per kind | jak wyżej, ale tylko segmenty tła z danym `kind` |
| latency | czas pierwszego alarmu w oknie zdarzenia minus `start_s` |

---

## Zależności — bez torcha

```
numpy
scipy        # resampling audio
soundfile    # WAV I/O
```

```bash
pip install numpy scipy soundfile
```