# Transition Pipeline Mockup - SNN Agent

Branch: `integration/ci-cd-test` - stan aktualny + propozycja
Branch ten powstał w wyniku scalenia dev z feat/ci-cd-test oraz następnie z main.

---

## 1. Co już istnieje na tym branchu (stan faktyczny)

```text
snn_pipeline/          ← CORE - pełny pipeline software ML
snn_simulator/         ← osobna symulacja LIF (Python, torch), można wyrzucić
experiments/           ← runtime: decoder.py, encoder_sim.py, encoder.ino - do testowania, może być na dev, ale nie na main??
encoder/               ← Arduino: snn_encoder.ino, snn_decoder.ino
decoder/               ← pusty plik 02_decoder.py
software/              ← host_agent (LLM RAG), web_dashboard (React+FastAPI), infra (Docker)
tests/                 ← test_decoder.py (46 testów), test_ci_basics.py (3 testy)
output/                ← wygenerowane artefakty z pipeline'u
checkpoints/           ← zapisane modele PyTorch (.pt)
data/ESC-50-master/    ← dataset audio (2000 plików .wav, 50 klas), która powstaje po odpalniu pipeline'u

```

### Co z architekturą obecną?

Struktura jest dobra, ale kilka rzeczy trzeba uporządkować.

- `snn_pipeline/` zostaje jako core, cały pipeline ML.
  - **SUGEROWANA ZMIANA**: dodać contracts.py - jedn plik ze wspólnymi strukturami danych, żeby każdy moduł importował typy stamtąd, a nie definiował je po swojemu.

- Dataset zostaje w `data/` - ESC-50 już tam jest. W CI używamy --data-limit 50 żeby nie pobierać 600MB na każdym runie.

- `output/` i `checkpoints/` - też nie do repo, tylko jako GitHub Actions artifacts. Czyli pipeline generuje pliki, CI je uploaduje, każdy może pobrać bez uruchamiania całości lokalnie.

- `decoder/` to teraz pusty plik 02_decoder.py - należy ściągnąć tam aktualny decoder z brancha

- `experiments/` zostawiamy jako symulację i playground - encoder_sim.py, decoder.py. To nie jest produkcyjny kod, tylko narzędzie do testowania bez sprzętu. Raczej na dev niż na main.

- `tests/` rozbudowujemy.
  - **SUGEROWANA ZMIANA**: dodać test_pipeline_contracts.py, który testuje  kontrakt między modułami ML: czy encoder zwraca właściwy kształt, czy SNN przyjmuje output encodera, czy metryki mają wymagane klucze.
  - **SUGEROWANA ZMIANA**: dodać testy encodera, już są na dev te dotyczące kodu do Arduino, ale jeszcze test spikeów
  - **SUGEROWANA ZMIANA**: dodać test_snn_model.py, który sprawdza forward pass i kształty
  - **SUGEROWANA ZMIANA**: dodać test_data_pipeline.py - załadowanie audio, augmentacja
  - **SUGEROWANA ZMIANA**: dodać test_metrics.py - testy precision/recall/F1
  - **SUGEROWANA ZMIANA**: dodać test_e24_quantizer.py - testy mapowania wag na E24
  - Obecne testy w test_decoder.py testują runtime SPI.

```text
tests/
├── test_decoder.py          ← już istnieje, 46 testów runtime
├── test_ci_basics.py        ← już istnieje, 3 testy
├── test_encoder.py          ← nowy, testy spike encoderów
├── test_snn_model.py        ← nowy, testy forward pass, kształty tensorów
├── test_data_pipeline.py    ← nowy, testy loadingu audio, augmentacji
├── test_metrics.py          ← nowy, testy precision/recall/F1
├── test_e24_quantizer.py    ← nowy, testy mapowania wag na E24
└── test_pipeline_contracts.py ← nowy, testy kontraktów między etapami
```

- `snn_simulator/` - to jest osobna rzecz od snn_pipeline/, trochę duplikuje funkcjonalność. Trzeba zdecydować czy to zostaje jako eksperymentalna gałąź czy scalamy z snn_pipeline/. **SUGEROWANA ZMIANA**: USUNĄĆ.

CI dostaje trzy rzeczy: smoke test pipeline'u w trybie --quick, testy kontraktów, i quality gate który blokuje merge do main jeśli recall spada poniżej 0.85.

### Co robi `snn_pipeline/` - już działający

Pipeline ma **5 faz** (`run_pipeline.py --phase A/B/C/D/E/all`):

| Faza | Co robi | Output |
|------|---------|--------|
| A | Pobiera ESC-50, preprocessuje audio, baseline metrics | `checkpoints/phase_a.pt` |
| B | HAT - Hardware-Aware Training z Gumbel-softmax | `checkpoints/phase_b.pt`, `output/hat_weight_table.txt`, `output/hat_learning_curves.png` |
| C | QAT + Sensitivity Analysis | `checkpoints/phase_c.pt`, `output/qat_learning_curves.png`, `output/sensitivity_heatmap.png` |
| D | Benchmark Baseline vs HAT vs HAT+QAT, thermal drift, power estimate | `output/benchmark_table.txt`, `output/weight_error_histogram.png` |
| E | Export do JSON/CSV/Arduino `.h`, HIL validation, MCP4151 table | `output/weights.json`, `output/weights.csv`, `output/snn_config.h`, `output/mcp4151_table.csv` |

---

## 2. Architektura SNN - ile neuronów i dlaczego są tam 4?

Model `GlassBreakSNN` w `snn_pipeline/snn_model.py` ma **4 neurony LIF**:

```text
Wejście (spike train)
    ├──[w_n1]──→ N1 (Excitatory) -
 detektor >2kHz, "czy szkło?"
    ├──[w_n2]──→ N2 (Excitatory) -
 detektor burst pattern temporal
    └──[w_inh]─→ N_inh (Inhibitory) -
 tłumi N1/N2 gdy HVAC/szum <500Hz
                                        ↓ [w_inh_to_n3, ujemna]
         N1 ──[w_n3_from_n1]──→ N3 (Decision/Output) -
         trigger
         N2 ──[w_n3_from_n2]──→ ↗
```

**Aktualne wagi bazowe (z `config.py`):**

| Neuron | Wagi wejściowe | Próg V_th | Rola |
|--------|---------------|-----------|------|
| N1 | `w_n1 = 0.70` | `0.60` | HF detektor (>2kHz) |
| N2 | `w_n2 = 0.50` | `0.50` | Temporal burst |
| N3 | `w_n3_from_n1 = 0.72`, `w_n3_from_n2 = 0.44` | `0.80` | Output/trigger |
| N_inh | `w_inh = 0.33`, `w_inh_to_n3 = -0.50` | `0.40` | Hamujący (HVAC) |

**Pytanie:** Jeśli chcemy rozszerzyć do 5–10 neuronów, `BASELINE_NEURONS` w `config.py` to jedno miejsce do zmiany. Obecna architektura jest świadomą decyzją - 4 neurony mapują się 1:1 na rezystory E24 na PCB.

---

## 3. Pełny pipeline - etap po etapie

### ETAP 1 - Dataset Loader
**Plik:** `snn_pipeline/data_pipeline.py` → `build_dataset()`

**Wejście:**
```
data/ESC-50-master/audio/*.wav  - 2000 plików, 50 klas, 5 foldów
Klasa 38 = glass_breaking (40 próbek pozytywnych)
```

**Co robi:**
1. Pobiera ESC-50 automatycznie (jeśli brak)
2. Wczytuje `.wav` przez `librosa` @ 22050 Hz
3. Filtr pasmowy Butterworth 500–8000 Hz (zakres glass break)
4. Oblicza RMS energy w oknach 10ms/5ms hop
5. Normalizacja log-dB: `(rms_db + 50) / 40` → zakres [0.0, 1.0]
6. Augmentacja glass_break ×10: time stretch, pitch shift, HVAC noise, RIR

**Podział datasetu:**
```
Test:  10 pos + ~100 neg  (LOCKED - nie dotykamy podczas treningu)
Val:   10 pos + ~100 neg
Train: reszta + augmented (ok. 400+ pos, 1700+ neg)
```

**Co zwraca:** `Dict[str, GlassBreakDataset]`

```python
{
    "train": GlassBreakDataset,   # lista krotek (spike_tensor, label)
    "val":   GlassBreakDataset,
    "test":  GlassBreakDataset,
}
```

Każdy element datasetu: `(spikes: Tensor[n_channels, n_timesteps], label: Tensor[1])`

---

### ETAP 2 - Encoder (Spike Encoding)
**Plik:** `snn_pipeline/spike_encoders.py` → `TTFSEncoder` lub `RateCodingEncoder`
**Wywołanie:** wewnątrz `data_pipeline.py::process_files()`

**Wejście:**
```python
energy_normalized: np.ndarray  # kształt (n_frames,), wartości [0.0, 1.0]
n_timesteps: int = 100          # kroków czasowych (domyślnie 100)
```

**Dwa tryby enkodowania:**

| Tryb | Opis | Kiedy |
|------|------|-------|
| `ttfs` (domyślny) | Time-To-First-Spike: wysoka energia → burst na początku okna | lepsza dla impulsowych sygnałów (szkło) |
| `rate` | Rate Coding: energia → częstotliwość 5–200 Hz, Poisson | ciągłe sygnały |

**Co zwraca:**
```python
torch.Tensor  # kształt (n_channels, n_timesteps), dtype=float32, wartości {0.0, 1.0}
# Aktualnie n_channels = 1 (jeden kanał RMS energy)
# n_timesteps = 100 (domyślnie)
```

**Ważna informacja o TTFS:** energia < 0.05 → 0 spike'ów (filtr szumu tła). Energia = 1.0 → burst 5 spike'ów od timestep=0.

**Odpowiednik sprzętowy:** `encoder/snn_encoder/snn_encoder.ino` - robi to samo na Arduino: RMS okna 256 próbek ADC, próg energii, SPI packet.

---

### ETAP 3 - Model SNN (GlassBreakSNN)
**Plik:** `snn_pipeline/snn_model.py` → `GlassBreakSNN`

**Wejście:**
```python
spike_input: torch.Tensor
# kształt: (batch_size, n_channels, n_timesteps)
# np. (32, 1, 100)
# wartości: float32, typowo {0.0, 1.0} (spike train)
```

**Co robi (forward pass):**
1. Jeśli `n_channels > 1` → uśrednia po kanałach → `(batch, timesteps)`
2. Kwantyzuje wagi według trybu: `none` / `hat` (E24 STE) / `gumbel` / `qat`
3. Opcjonalny mismatch: ±1% wag, ±5mV V_th (trening)
4. Pętla przez timesteps: N1, N2, N_inh równolegle → N3 zbiera

**Co zwraca:**
```python
trigger: torch.Tensor         # kształt (batch, 1), float32
                              # = spike rate N3 = ułamek timestepów z N3 active
                              # 0.0 = brak triggera, ~1.0 = silny trigger

neuron_spikes: Dict[str, torch.Tensor]  # spike train każdego neuronu
# {
#   "N1":    Tensor(batch, n_timesteps),
#   "N2":    Tensor(batch, n_timesteps),
#   "N3":    Tensor(batch, n_timesteps),
#   "N_inh": Tensor(batch, n_timesteps),
# }
```

**Decyzja binarna** (w `evaluation.py`): `trigger >= 0.5` → zdarzenie wykryte

---

### ETAP 4 - Trainer (HAT → QAT)
**Pliki:** `snn_pipeline/hat_trainer.py`, `snn_pipeline/qat_trainer.py`

**HAT (Faza B):**
- Wejście: DataLoader train/val, model z wagami baseline
- Trenuje z Gumbel-softmax quantization (temperatura annealing 5.0→0.1)
- Loss = BCE + `lambda_recall=2.0` × recall penalty + `lambda_hw=0.5` × E24 regularyzacja
- Po każdej epoce: `model.clamp_weights()` - chroni inhibitor
- Output: wagi zmapowane bliżej wartości E24

**QAT (Faza C):**
- Wejście: model po HAT, 50 próbek kalibracyjnych glass_break
- Mixed precision: N3 ma 6 bitów, N1/N2 mają 5 bitów, N_inh ma 4 bity
- Sensitivity analysis → identyfikuje które synapsy wymagają MCP4151
- Output: finalne wagi gotowe do eksportu

---

### ETAP 5 - Evaluation
**Plik:** `snn_pipeline/evaluation.py` → `evaluate_model()`

**Wejście:** `(model, DataLoader test, label: str)`

**Co zwraca:**
```python
{
    "precision":       float,   # TP / (TP + FP)
    "recall":          float,   # TP / (TP + FN)  ← KLUCZOWE
    "f1":              float,
    "accuracy":        float,
    "fnr":             float,   # False Negative Rate = 1 - recall
    "confusion_matrix": np.ndarray,  # [[TN, FP], [FN, TP]]
    "avg_latency_ms":  float,   # czas do pierwszego spike'a N3
    "weights":         dict,    # aktualne wagi modelu
    "thresholds":      dict,    # aktualne progi V_th
    "label":           str,     # "Baseline" / "Po HAT" / "Po HAT+QAT"
}
```

**Dodatkowe funkcje w evaluation.py:**
- `thermal_drift_simulation()` - 100 uruchomień z ±2% szumem wag → mean/std recall
- `weight_error_histogram()` → `output/weight_error_histogram.png`
- `power_estimate()` → szacowanie mW per synaps
- `benchmark_table()` → `output/benchmark_table.txt` (Baseline vs HAT vs HAT+QAT)

---

### ETAP 6 - Hardware Mapping (E24 + MCP4151)
**Pliki:** `snn_pipeline/e24_quantizer.py`, `snn_pipeline/hil_validation.py`

**Wejście:** wagi z modelu, zakres E24: 10kΩ–470kΩ

**Co robi:**
- Mapuje wagę `w` → rezystancja: `R = R_in × (1 - w) / w`
- Szuka najbliższej wartości E24 w zakresie synaps PCB
- Identyfikuje synapsy wrażliwe (>5% zmiana recall na ±1% wagi) → kandydaci na MCP4151

**HIL Simulation:**
- 100 scenariuszy z ±1% szumem rezystorów (symulacja tolerancji)
- Raportuje: `P(recall ≥ 85%)`, `P(recall ≥ 80%)`, percentyl 5-ty

**Co zwraca:**
```python
{
    "e24_valid": bool,
    "mapped_weights": {"w_n1": 47000, "w_n2": 68000, ...},  # Ω lub "open"
    "open_weights": int,        # liczba połączeń niezmapowanych
    "hardware_ready": bool,     # True jeśli e24_valid
    "prob_recall_ge_85": float, # główny quality gate hardware
}
```

---

### ETAP 7 - Export
**Plik:** `snn_pipeline/export.py`

Generuje **wszystkie pliki produkcyjne**:

| Plik | Format | Zawartość | Odbiorca |
|------|--------|-----------|----------|
| `output/weights.json` | JSON | wagi, rezystancje E24, progi V_th, metryki | hardware, PCB |
| `output/weights.csv` | CSV | tabela: Neuron, Synapsa, Waga, R_exact, R_E24, Błąd% | hardware |
| `output/snn_config.h` | C header | `#define W_N1_INPUT`, `#define VTH_N1_MV`, ... | Arduino |
| `output/mcp4151_table.csv` | CSV | wiper_value, SPI bytes dla każdej krytycznej synapsy | firmware |
| `output/benchmark_table.txt` | TXT | Baseline vs HAT vs HAT+QAT | dokumentacja |
| `output/hat_learning_curves.png` | PNG | loss/recall per epoka | raport |
| `output/sensitivity_heatmap.png` | PNG | wrażliwość synaps | raport |
| `output/weight_error_histogram.png` | PNG | błąd kwantyzacji E24 per synaps | raport |

---

### ETAP 8 - Runtime (hardware path)

To osobny tor - nie jest częścią `snn_pipeline/`, ale musi z nim rozmawiać przez `snn_config.h`:

```
Arduino snn_encoder.ino (plik: encoder/snn_encoder/snn_encoder.ino)
  → [SPI 4MHz]
RPi5 experiments/decoder.py
  → AnomalyDetector (≥3 spike'ów, energy>150, w 0.5s)
  → LLMAgent (tinyllama @ Ollama)
  → Action (stub: _trigger_alarm)
```

Symulacja tego toru bez sprzętu: `experiments/encoder_sim.py` + `SNN_READER=tcp`.

---

## 4. Diagram przepływu danych

```text
data/ESC-50-master/*.wav
         │  2000 plików .wav, 22050Hz mono
         ▼
data_pipeline.py::build_dataset()
  ├─ load_audio()           -
 librosa, mono, resampling
  ├─ bandpass_filter()      -
 Butterworth 500–8000 Hz
  ├─ extract_rms_energy()   -
 okna 10ms/5ms
  ├─ normalize log-dB       -
 (rms_db + 50) / 40 → [0,1]
  └─ augment_glass_break()  -
 ×10 augmentacji (tylko train)
         │
         │  energy_normalized: np.ndarray (n_frames,)
         ▼
spike_encoders.py::TTFSEncoder.encode()
  lub RateCodingEncoder.encode()
         │
         │  torch.Tensor (1, 100)  ← 1 kanał, 100 timestepów, wartości {0,1}
         ▼
GlassBreakDataset[(spikes_tensor, label)]
  → DataLoader (batch=32, shuffle=True for train)
         │
         │  batch_spikes: Tensor (32, 1, 100)
         │  batch_labels: Tensor (32, 1)
         ▼
GlassBreakSNN.forward(spike_input)
  ├─ N1, N2, N_inh równolegle (timestep loop)
  └─ N3 zbiera z N1+N2, hamowany przez N_inh
         │
         │  trigger:       Tensor (32, 1)   ← spike rate N3, [0.0, 1.0]
         │  neuron_spikes: Dict {N1,N2,N3,N_inh} → Tensor(32, 100)
         ▼
evaluation.py::evaluate_model()
         │
         │  Dict: precision, recall, f1, fnr, confusion_matrix, avg_latency_ms, weights
         ▼
┌─────────────────────────────────────┐
│  HAT training (phase B)             │
│  → wagi zmapowane do E24            │
│  → checkpoints/phase_b.pt           │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  QAT + Sensitivity (phase C)        │
│  → mixed precision (4–6 bit)        │
│  → sensitivity_heatmap.png          │
│  → checkpoints/phase_c.pt           │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────┐
│  Benchmark + Thermal drift (phase D)│
│  → benchmark_table.txt              │
│  → weight_error_histogram.png       │
└──────────────┬──────────────────────┘
               ▼
┌─────────────────────────────────────────────────────────┐
│  Export + HIL validation (phase E)                      │
│  → weights.json    (dla PCB / hardware team)            │
│  → weights.csv     (tabela rezystancji)                 │
│  → snn_config.h    (dla Arduino encoder)                │
│  → mcp4151_table.csv (komendy SPI dla potencjometrów)   │
│  → HIL: 100 scenariuszy ±1% szum → P(recall≥85%)       │
└─────────────────────────────────────────────────────────┘
```

---

## 5. Kontrakt danych - tabela

| Etap | Producent | Konsument | Format | Wymiary | Warunek kontraktu |
|------|-----------|-----------|--------|---------|-------------------|
| .wav → features | `load_audio` + `extract_features` | `TTFSEncoder` | `np.ndarray float32` | `(1, n_frames)` | wartości [0.0, 1.0] |
| features → spikes | `TTFSEncoder` / `RateCodingEncoder` | `GlassBreakDataset` | `torch.Tensor float32` | `(1, 100)` | wartości ∈ {0.0, 1.0} |
| spikes → batch | `DataLoader` | `GlassBreakSNN.forward` | `torch.Tensor float32` | `(batch, 1, 100)` | - |
| forward → trigger | `GlassBreakSNN` | `evaluate_model` | `torch.Tensor float32` | `(batch, 1)` | wartości [0.0, 1.0] |
| trigger → metrics | `evaluate_model` | `benchmark_table`, HAT trainer | `Dict[str, float]` | - | recall ≥ 0.85 (gate) |
| metrics → export | `export.py` | hardware, Arduino | JSON / CSV / `.h` | - | `hardware_ready == True` |
| `snn_config.h` → Arduino | `export.py` | `snn_encoder.ino` | C header `#define` | - | plik istnieje i jest parsowany |

---

## 6. Gdzie co leży i kto odpowiada

```text
snn_pipeline/
├── config.py           ← JEDEN PLIK do zmiany: liczba neuronów, progi, zakres E24
│                          Zmieniasz tu N_NEURONS → zmienia się wszystko
├── data_pipeline.py    ← Dataset: pobieranie ESC-50, preprocessing, augmentacja
├── spike_encoders.py   ← Encoder: TTFS i Rate Coding, format Tensor(1, 100)
├── snn_model.py        ← SNN: architektura 4 neuronów, forward pass
├── hat_trainer.py      ← SNN: HAT trening, Gumbel-softmax
├── qat_trainer.py      ← SNN: QAT, mixed precision
├── e24_quantizer.py    ← Hardware: mapowanie wag → rezystancje E24
├── evaluation.py       ← CI/CD: benchmark, thermal drift, power estimate
├── metrics.py          ← CI/CD: precision, recall, F1, confusion matrix
├── sensitivity.py      ← Hardware: analiza wrażliwości synaps, MCP4151 kandydaci
├── hil_validation.py   ← Hardware: HIL 100 scenariuszy, tabela MCP4151
├── export.py           ← All: JSON/CSV/Arduino header → output/
└── run_pipeline.py     ← ENTRY POINT: python snn_pipeline/run_pipeline.py --phase all

encoder/
├── snn_encoder/snn_encoder.ino         ← Arduino: czyta ADC, enkoduje TTFS/Rate, SPI TX
└── snn_decoder/snn_decoder.ino         ← Arduino: odczytuje SPI z RPi (jeśli potrzebne)

experiments/
├── decoder.py      ← RPi5 runtime: SpiReader/TcpReader → AnomalyDetector → LLMAgent
├── encoder_sim.py  ← Symulacja encodera (bez hardware), TCP
└── encoder.ino     ← Stara wersja (zastąpiona przez encoder/)

software/
├── host_agent/run.py        ← LLM agent stub (RAG: ingest.py, query.py)
├── web_dashboard/api/app.py ← FastAPI backend (stub)
├── web_dashboard/ui/        ← React frontend (stub, Vite+TS)
└── infra/                   ← Docker Compose, Dockerfiles

output/                      ← GENEROWANE ARTEFAKTY (nie commitujemy do repo)
checkpoints/                 ← CHECKPOINTY PyTorch (nie commitujemy)
```

---

## 7. CI/CD - stan aktualny i co dodać

### Co już działa (`.github/workflows/ci.yml`)

```yaml
# Triggeruje na: push do dev, feat/**, integration/**
# PR do: dev
jobs:
  lint:  ruff check + ruff format --check
  test:  pytest tests/ -v
```

**Aktualnie testowane:**
- `tests/test_ci_basics.py` - 3 testy (podstawowe)
- `tests/test_decoder.py` - 46 testów (SpiPacket, AnomalyDetector, LLMAgent, TCP transport)

**Problem:** CI nie testuje `snn_pipeline/` w ogóle. Można by uruchomić cały pipeline w --quick mode.

### Co dodać do CI

**Krok 1 - Smoke test pipeline (bez GPU, bez pełnego datasetu):**
```yaml
- name: Pipeline smoke test
  run: uv run python snn_pipeline/run_pipeline.py --phase all --quick
  # --quick: data-limit=50, epochs=5, qat-epochs=3
```

**Krok 2 - Testy kontraktów (nowy plik `tests/test_pipeline_contracts.py`):**

```python
# test: encoder zwraca poprawny kształt
def test_encoder_output_shape():
    encoder = TTFSEncoder()
    energy = np.array([0.8, 0.5, 0.2])
    spikes = encoder.encode(energy, n_timesteps=100)
    assert spikes.shape == (3, 100)
    assert spikes.dtype == torch.float32
    assert set(spikes.unique().tolist()).issubset({0.0, 1.0})

# test: SNN przyjmuje output encodera
def test_snn_accepts_encoder_output():
    model = GlassBreakSNN(quantize_mode="none")
    batch = torch.zeros(4, 1, 100)
    trigger, neuron_spikes = model(batch)
    assert trigger.shape == (4, 1)
    assert set(neuron_spikes.keys()) == {"N1", "N2", "N3", "N_inh"}

# test: metryki mają wymagane klucze
def test_metrics_keys():
    result = all_metrics(torch.tensor([0.8, 0.2]), torch.tensor([1.0, 0.0]))
    assert all(k in result for k in ["precision", "recall", "f1", "accuracy", "fnr"])

# test: export tworzy pliki
def test_export_creates_files(tmp_path):
    model = GlassBreakSNN()
    path = export_weights_json(model, {"precision": 1.0, "recall": 0.9}, save_path=str(tmp_path/"w.json"))
    assert Path(path).exists()
```

**Krok 3 - Quality gate (blokuje merge do main):**
```yaml
- name: Quality gate check
  run: |
    python -c "
    import json; d = json.load(open('output/weights.json'))
    recall = d['metrics']['recall']
    assert recall >= 0.85, f'recall {recall} < 0.85'
    print(f'PASS: recall={recall}')
    "
```

**Krok 4 - Upload artifacts:**
```yaml
- uses: actions/upload-artifact@v4
  with:
    name: snn-pipeline-output
    path: output/
    if-no-files-found: warn
```

---

## 8. Strategia branchy i kolejność mergowania

```text
feat/encoder           → dev   (musi dostarczyć encoder.ino kompatybilny z snn_config.h)
feat/decoder           → dev   (testy 46/46, runtime RPi)
feat/neuron-architecture → dev (zmiany w config.py: BASELINE_NEURONS, N_NEURONS)
integration/ci-cd-test → dev   (ten branch: pełny snn_pipeline, CI rozbudowane)
feat/snn-pipeline      → dev   (gdy ww. są już na dev - integracja end-to-end)
```

**Zasada:** każdy PR do `dev` musi mieć:
1. Lint przechodzący (ruff)
2. Testy unit przechodzące
3. Opis co moduł **przyjmuje** i co **zwraca** (format z tabeli sekcji 5)

**Merge do `main`:** tylko gdy recall ≥ 0.85 na zbiorze testowym ESC-50.

---

## 9. Otwarte pytania do ustalenia

| Pytanie | Gdzie w kodzie | Wpływ |
|---------|---------------|-------|
| Ile neuronów finalnie? (4 jest teraz) | `config.py::BASELINE_NEURONS` | kształt danych w całym pipeline |
| Rate czy TTFS jako główny enkoder? | `run_pipeline.py --encoding` | charakterystyka spikes |
| Próg decyzyjny trigger: 0.5 czy inny? | `evaluation.py`, `metrics.py` | precision/recall trade-off |
| Które synapsy dostają MCP4151? | `sensitivity.py::identify_mcp4151_candidates` | koszt PCB |
| n_timesteps: 100 czy inne? | `run_pipeline.py --n-timesteps` | latencja vs dokładność |

---

## 10. Propozycja na spotkanie

> Mamy już działający pełny pipeline od `.wav` do `snn_config.h` na branchu `integration/ci-cd-test` i `main`.  
> Architektura to 4 neurony LIF (N1, N2, N3, N_inh) - każdy mapuje się 1:1 na rezystor E24.  
> Żeby dodać więcej neuronów trzeba rozszerzyć `BASELINE_NEURONS` w `config.py`.  
> Kluczowe do ustalenia na spotkaniu: **ile neuronów**.  
> CI/CD wymaga trzech kroków: (1) smoke test `--quick`, (2) testy kontraktów, (3) quality gate recall ≥ 0.85.