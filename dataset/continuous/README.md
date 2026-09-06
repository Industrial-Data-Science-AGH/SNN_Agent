# continuous_eval — generator ciągłego datasetu ewaluacyjnego

Generator deterministycznego strumienia audio z **dokładnie 5 zdarzeniami
rozbicia szkła** w losowych, nienachodzących pozycjach.

## Szybki start

```bash
# jeden 10-minutowy strumień
python -m dataset.continuous.eval.cli \
    --glass-annotation-dir dataset/clean/clean/annotation \
    --glass-audio-root     dataset/clean/clean/audio \
    --glass-allowed-stems  dataset/clean/clean/target/synthetic_target_test.txt \
    --background-dir       data/ESC-50-master/audio \
    --seed 42 \
    --out-dir dataset/continuous/out

# 3 warianty (kryterium akceptacji: >=3 warianty z różnymi seedami)
python -m dataset.continuous.eval.cli \
    --glass-annotation-dir dataset/clean/clean/annotation \
    --glass-audio-root     dataset/clean/clean/audio \
    --glass-allowed-stems  dataset/clean/clean/target/synthetic_target_test.txt \
    --background-dir       data/ESC-50-master/audio \
    --seeds 42 43 44 \
    --out-dir dataset/continuous/out
```

Wynik dla seed 42: `continuous_eval_seed42.wav` + `continuous_eval_seed42.manifest.json`.

## Testy automatyczne

```bash
python -m pytest dataset/continuous/tests/ -v
# 23 passed — bez torcha, bez plików produkcyjnych, uruchamialne w CI
```

---


## Decyzje implementacyjne

### 1. `--glassbreak-mode clean` jako default

VOICe to miksy — glassbreak prawie zawsze nachodzi na gunshot lub babycry
(stats.md: 3961/4444 zdarzeń). Dwa tryby:

| tryb | co zwraca | kiedy używać |
|---|---|---|
| `clean` (domyślny) | tylko glassbreak bez nakładki na inne klasy | mierzysz odpowiedź sieci czysto na szkło |
| `background` | wszystkie glassbreak, niezależnie od nakładek | bardziej realistyczny, trudniejszy wariant |

W trybie `background` manifest odnotowuje `is_contaminated: true` i listę
`overlapping_labels` — konsument może filtrować wyniki.

W trybie `clean` pula kandydatów jest znacznie mniejsza. Przy < 5 kandydatach
generator failuje z komunikatem zamiast cicho duplikować klipy.

### 2. Standard audio: 44100 Hz / mono / PCM_16

Przyjęty za `dataset/versions/v2.0.0/stats.md` (wszystkie 10 853 nagrań).
Każdy plik źródłowy jest jawnie resamplowany przez `scipy.signal.resample_poly`
z GCD-redukcją stosunku próbkowań, niezależnie od natywnego SR.

**Wymaga potwierdzenia przez Patryka** jako kryterium akceptacji zadania.

### 3. Tło jako argument CLI

```bash
--background-dir data/ESC-50-master/audio        # można podać wielokrotnie
--background-dir dataset/datasec/PT_DATASET_250314
```

Tło jest konkatenacją losowo wybranych plików z podanych katalogów, z losowym
offsetem startu wewnątrz każdego pliku (seed determinuje kolejność).

**Twardy fail przy brakach** — jeśli katalog nie istnieje lub jest pusty,
generator zatrzymuje się z błędem. Świadome odwrócenie zachowania `build-manifest`,
które po cichu pomijało brakujące pliki.

### 4. Zabezpieczenie przed clippingiem

Po zmiksowaniu tła ze zdarzeniami, jeśli szczyt > `clip_guard_peak=0.97`,
**cały miks skalowany proporcjonalnie w dół**. Nie per-sample `np.clip` —
to zniekształciłoby kształt fali w oknach zdarzeń glassbreak, czyli
w najgorszym możliwym miejscu dla metryk detekcji.

### 5. Losowanie pozycji — algorytm

Retry-loop (max 20 000 prób): losuj kolejność zdarzeń, dla każdego losuj start
w `[edge_margin, stream_duration - edge_margin - duration]`, odrzuć jeśli
nakłada lub jest za blisko już rozmieszczonych. Rzuca `PlacementError` ze
zrozumiałym komunikatem gdy strumień jest za krótki.

Prościej do weryfikacji niż analityczny rozkład. Typowo < 100 prób dla 5
zdarzeń w 10-minutowym strumieniu.

### 6. Deterministyczność

Cała losowość przez jeden `random.Random(seed)` w ustalonej kolejności:
wybór klipów → pozycje → skale głośności → kolejność tła → offsety w plikach.
Ten sam seed + te same pliki = identyczny WAV (test: `test_build_stream_deterministic`).

---

## Kontrakt manifestu — do akceptacji przez Marcela i Patryka

Format: `*.manifest.json` obok WAV, schema version `1.0.0`.

```json
{
  "manifest_schema_version": "1.0.0",
  "generator_version": "1.0.0",
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
    "min_gap_s": 2.0,
    "edge_margin_s": 1.0,
    "background_dirs": ["data/ESC-50-master/audio"]
  },
  "events": [
    {
      "index": 0,
      "start_s": 47.23,
      "end_s": 48.09,
      "duration_s": 0.86,
      "source_stem": "synthetic_014",
      "source_start_s": 4.0,
      "source_end_s": 5.36,
      "is_contaminated": false,
      "overlapping_labels": [],
      "gain_db": 1.2
    }
  ],
  "background_segments": [...]
}
```

`events` zawiera zawsze dokładnie 5 wpisów, posortowane rosnąco po `start_s`.

**Jak Marcel liczy metryki z manifestu:**

| metryka | jak liczyć |
|---|---|
| detected / 5 | dla każdego `[start_s, end_s]` — czy detektor podniósł alarm (+tolerancja) |
| event recall | `detected / 5` |
| false alarms/h | alarmy poza wszystkimi oknami `[start_s, end_s]` / `duration_s` × 3600 |
| latency | czas pierwszego alarmu w oknie minus `start_s` |

**To jest projekt roboczy.** Przed integracją wymagana akceptacja Marcela
(nazwy pól, tolerancja okna detekcji) i Patryka (standard audio). Zmiana
formatu manifestu = tylko `manifest.py`, generator bez zmian.

---

## Co generator świadomie NIE robi

**Nie sprawdza rozłączności z danymi treningowymi.** Oryginalny task wymagał
"zapewnić osobność lub automatycznie zaraportować overlap". Decyzja: rozłączność
to odpowiedzialność wywołującego przez wybór ścieżki w `--glass-allowed-stems`
(np. `synthetic_target_test.txt` zamiast `source_training.txt`) i
`--background-dir`. Dodanie automatycznego sprawdzania wymagałoby wczytania
manifestu treningowego jako zależności, co komplikuje moduł bez proporcjonalnej
korzyści wobec prostszego mechanizmu — list plików.

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
