# Testing & Simulation Guide

## What we have now

```
experiments/
├── encoder.ino        # Arduino firmware (reads mic, sends SPI spikes)
├── encoder_sim.py     # Python TCP encoder that mimics encoder.ino
├── decoder.py         # Decoder: SPI/TCP/Mock reader → AnomalyDetector → LLM
└── __init__.py

tests/
└── test_decoder.py    # 46 tests (unit + integration + TCP transport)
```

### Components

| Component | File | Purpose |
|---|---|---|
| SpiPacket | decoder.py | Parses 6-byte `[0xAE][TS_HI][TS_LO][ENERGY][ID][CHECKSUM]` frames |
| SpiReader | decoder.py | Reads real SPI on Raspberry Pi (requires `spidev`) |
| MockSpiReader | decoder.py | Generates random spikes in-process (no hardware) |
| TcpReader | decoder.py | Listens on TCP, receives 6-byte frames from encoder_sim |
| AnomalyDetector | decoder.py | Detects spike clusters (≥3 spikes above threshold within 0.5s) |
| LLMAgent | decoder.py | Calls Ollama (tinyllama) to classify events |
| GPIOController | decoder.py | Pulses LED on RPi GPIO pin 17 |
| EncoderSim | encoder_sim.py | TCP client: builds & sends packets matching encoder.ino format |
| build_packet() | encoder_sim.py | Constructs a valid 6-byte frame with XOR checksum |

### Reader selection

| Env variable | Value | Reader used |
|---|---|---|
| (none, non-Linux) | — | MockSpiReader |
| `SNN_USE_MOCK_SPI` | `1` | MockSpiReader |
| `SNN_READER` | `tcp` | TcpReader |
| (Linux, spidev available) | — | SpiReader |

TCP-specific config: `SNN_TCP_HOST` (default `127.0.0.1`), `SNN_TCP_PORT` (default `9999`).

---

## Running the simulation (no hardware)

### 1. Unit tests

```bash
.venv/bin/pytest tests/test_decoder.py -v
```

All 46 tests run without hardware. SPI and GPIO are mocked at import time.

### 2. TCP simulation (two terminals)

Terminal 1 — start the decoder with TCP reader:

```bash
SNN_READER=tcp SNN_TCP_PORT=9999 python -m experiments.decoder
```

Terminal 2 — start the encoder simulator:

```bash
python -m experiments.encoder_sim --port 9999 --rate 5
```

`--rate` controls packets/second. `--duration 10` stops after 10 seconds (omit for infinite).

You will see the full pipeline in Terminal 1:

```
[SPIKE] Spike(T=00256, E=220, ID=0) | via TCP
[EVENT] Spike cluster: 3 spikes in 0.41s | avg energy: 207
[LLM] WAKING UP AGENT
[LLM RESPONSE] ALARM: suspicious acoustic pattern
```

### 3. Mock simulation (single terminal)

```bash
python -m experiments.decoder
```

Uses MockSpiReader by default on macOS/Windows. Generates random spikes internally.

### 4. Packet format reference

```
Byte 0: 0xAE          (header)
Byte 1: timestamp >> 8 (high byte)
Byte 2: timestamp & FF (low byte)
Byte 3: energy         (0–255)
Byte 4: spike_id       (0–255)
Byte 5: XOR(bytes 0–4) (checksum)
```

Same layout in encoder.ino (`SpikePacket` struct) and decoder.py (`SpiPacket` class).

---

## What's needed for hardware deployment

### Signal chain

```
Microphone → Analog LIF neurons (3-4 on breadboard) → Arduino A0 → encoder.ino → SPI → RPi5 decoder → LLM agent
```

The analog LIF (Leaky Integrate-and-Fire) neuron layer sits between the microphone and the Arduino. These neurons act as a hardware bandpass/pattern filter tuned to glass-breaking frequency signatures. Only spikes that survive the LIF cascade reach the Arduino ADC — the Arduino doesn't do raw audio processing, it reads the neuron output and packages it into SPI frames.

### Analog LIF neuron board

1. **Build 3–4 LIF neurons** on breadboard/perfboard (RC integrate + comparator fire + reset)
2. **Tune RC time constants** to match glass-breaking spectral bands (~1–5 kHz burst energy)
3. **Cascade or fan-in** neurons so only correlated high-frequency bursts produce output spikes
4. **Output** → Arduino A0 (spike train as analog voltage)

### Arduino (encoder)

1. **Upload firmware**: Flash `experiments/encoder.ino` via Arduino IDE
2. **Wire LIF output → A0**: Connect the final LIF neuron output to Arduino analog pin A0
3. **Calibrate threshold**: Run in silence, read baseline energy from Serial Monitor, set `ENERGY_THRESHOLD` to ~3× baseline
4. **SPI wiring to RPi5**: Pin 10→CS, 11→MOSI, 12→MISO, 13→SCLK, GND→GND

See `encoder/ENCODER_PLAN.md` for pinout diagrams.

### Raspberry Pi 5 (decoder)

1. **Enable SPI**: `sudo raspi-config` → Interface Options → SPI → Enable
2. **Install dependencies**:
   ```bash
   pip install spidev numpy requests
   ```
3. **Install Ollama** (for LLM classification):
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   ollama pull tinyllama
   ```
4. **Run decoder**:
   ```bash
   python -m experiments.decoder
   ```
   On Linux with SPI available, it auto-selects `SpiReader`. Force mock with `SNN_USE_MOCK_SPI=1`.

5. **LED feedback**: GPIO 17 pulses on each spike (optional, connect LED + 330Ω resistor)

### Board-to-board testing checklist

- [ ] LIF neurons produce spikes on glass-breaking sounds, stay quiet on speech/music
- [ ] Arduino Serial Monitor shows `[ENCODER] System ready. Sampling at 8kHz.`
- [ ] Baseline energy from LIF output in silence is stable (±10)
- [ ] Glass-breaking / sharp impact triggers spikes on Serial Monitor (`[SPI TX] Energy: ...`)
- [ ] RPi5 decoder logs `[SPIKE] ... | Energy rise detected!`
- [ ] Spike cluster triggers `[EVENT] Spike cluster: ...`
- [ ] Ollama responds with ALARM / WARNING / NORMAL
- [ ] LED on GPIO 17 pulses on spikes
- [ ] `ENERGY_THRESHOLD` tuned: silence = no spikes, glass-like sounds = spikes

### Remaining work

| Task | Status |
|---|---|
| TCP simulation (encoder_sim ↔ TcpReader) | Done |
| Mock simulation (MockSpiReader) | Done |
| Unit & integration tests (46 tests) | Done |
| Arduino firmware (encoder.ino) | Done, needs upload & calibration |
| Analog LIF neuron board (3–4 neurons) | Not started |
| LIF tuning for glass-breaking pattern | Not started |
| SPI wiring & board-to-board test | Not started |
| Ollama setup on RPi5 | Not started |
| Alarm actions (siren/email/push) | Stub only (`_trigger_alarm`) |
| Web dashboard integration | Not started |
| RAG agent integration | Not started |
