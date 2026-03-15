"""
decoder.py — Raspberry Pi: odbiera spiki z enkodera, klasyfikuje, wywołuje agenta.

Architektura "Wake-Up":
  ┌──────────┐  UDP/SPI  ┌──────────────────┐  zdarzenie  ┌─────────┐
  │ Arduino  │ ────────► │   Decoder (RPi)  │ ──────────► │  Agent  │
  │ Enkoder  │           │  spike counter   │             │  (LLM)  │
  └──────────┘           │  LIF classifier  │             └─────────┘
                         └──────────────────┘

RPi ŚPI (CPU ~0%) dopóki nie przyjdzie wystarczająco dużo spików.
Po klasyfikacji → agent działa przez 10s, potem RPi wraca do "idle".

Użycie:
  python decoder.py                        # nasłuchuje UDP :9000
  python decoder.py --spi                  # odbiera przez SPI (fizyczne Arduino)
  python decoder.py --port 9001 --verbose  # debug
"""

import argparse
import socket
import struct
import time
import threading
import json
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable
from enum import Enum, auto

# ============================================================
#  Stałe
# ============================================================
SPIKE_HEADER  = 0xAE
PACKET_SIZE   = 6
N_CHANNELS    = 3

class SystemState(Enum):
    IDLE    = auto()  # RPi śpi — minimalne zużycie CPU
    ACTIVE  = auto()  # Przetwarzanie — agent działa

# ============================================================
#  Konfiguracja
# ============================================================
@dataclass
class DecoderConfig:
    udp_port:         int   = 9000
    use_spi:          bool  = False
    verbose:          bool  = False

    # Okno czasowe zliczania spików [s]
    window_sec:       float = 0.5

    # Próg spików w oknie → zdarzenie
    spike_threshold:  int   = 5

    # Ile razy z rzędu musi być przekroczony próg → alarm
    confirm_count:    int   = 2

    # Po ilu sekundach bez spików → powrót do IDLE
    idle_timeout_sec: float = 10.0

    # Ścieżka do wag SNN (opcjonalnie)
    weights_path:     str   = "hardware_weights.json"

    # GPIO pin diody (BCM) — ustaw -1 żeby wyłączyć
    led_gpio_pin:     int   = 17

# ============================================================
#  Dekodowanie pakietu
# ============================================================
def decode_packet(raw: bytes) -> Optional[dict]:
    if len(raw) < PACKET_SIZE or raw[0] != SPIKE_HEADER:
        return None
    header, ts_h, ts_l, energy, channel, checksum = raw[:6]
    if checksum != (header ^ ts_h ^ ts_l ^ energy ^ channel) & 0xFF:
        return None
    return {
        "channel":      channel,
        "energy":       energy / 255.0,        # normalizuj 0..1
        "timestamp_ms": (ts_h << 8) | ts_l,
        "received_at":  time.time(),
    }

# ============================================================
#  Prosty klasyfikator oparty na wagach z SNN (hardware_weights.json)
# ============================================================
class SpikeClassifier:
    """
    Zlicza spiki w oknie czasowym per kanał,
    opcjonalnie aplicuje wagi z treningu SNN.
    Zwraca: 0 = tło, 1 = anomalia, -1 = niewystarczająco spików.
    """
    def __init__(self, cfg: DecoderConfig):
        self.cfg     = cfg
        self.weights = self._load_weights()

    def _load_weights(self) -> Optional[dict]:
        if os.path.exists(self.cfg.weights_path):
            with open(self.cfg.weights_path) as f:
                w = json.load(f)
            print(f"[CLASSIFIER] Wagi załadowane z {self.cfg.weights_path}")
            print(f"[CLASSIFIER] Test acc podczas treningu: {w.get('final_test_acc', '?')}")
            return w
        else:
            print("[CLASSIFIER] Brak hardware_weights.json — używam prostego licznika spików")
            return None

    def classify(self, spike_counts: list[int], spike_energies: list[float]) -> int:
        total = sum(spike_counts)

        if total < self.cfg.spike_threshold:
            return -1  # za mało spików — brak decyzji

        if self.weights is None:
            # Prosty heurystyk: dużo spików na wszystkich kanałach = anomalia
            active_channels = sum(1 for c in spike_counts if c > 0)
            return 1 if (active_channels >= 2 and total >= self.cfg.spike_threshold) else 0

        # Weighted voting z wagami FC2 z SNN
        # fc2_resistors_kohm: shape (2, 8) — tu upraszczamy do (2, 3)
        fc2 = self.weights.get("fc2_resistors_kohm", None)
        fc2_sign = self.weights.get("fc2_polarity", None)

        if fc2 and len(fc2) >= 2:
            # Użyj pierwszych 3 wagowych wartości (odpowiadają N_CHANNELS)
            votes = []
            for cls in range(2):
                w_row   = fc2[cls][:N_CHANNELS]
                sgn_row = fc2_sign[cls][:N_CHANNELS] if fc2_sign else [1]*N_CHANNELS
                # Konduktancja = 1/R, przeskalowana przez energię spika
                score = sum(
                    (s / (r + 1e-6)) * e
                    for r, s, e in zip(w_row, sgn_row, spike_energies)
                )
                votes.append(score)

            return int(votes[1] > votes[0])  # klasa z wyższym score

        return 1 if total >= self.cfg.spike_threshold else 0

# ============================================================
#  GPIO (opcjonalne — tylko jeśli dostępne na RPi)
# ============================================================
class GPIOController:
    def __init__(self, pin: int):
        self.pin     = pin
        self.enabled = False
        if pin < 0:
            return
        try:
            import RPi.GPIO as GPIO
            self.GPIO    = GPIO
            self.enabled = True
            GPIO.setmode(GPIO.BCM)
            GPIO.setwarnings(False)
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
            print(f"[GPIO] Dioda na pinie BCM {pin}")
        except ImportError:
            print("[GPIO] RPi.GPIO niedostępne — dioda wyłączona")

    def set(self, state: bool):
        if self.enabled:
            self.GPIO.output(self.pin, self.GPIO.HIGH if state else self.GPIO.LOW)

    def cleanup(self):
        if self.enabled:
            self.GPIO.cleanup()

# ============================================================
#  Mock Agent (energooszczędny)
# ============================================================
class MockAgent:
    """
    Placeholder dla prawdziwego agenta LLM/VLM.
    Wywoływany TYLKO gdy SNN wykryje anomalię — Wake-Up pattern.

    Podmień metodę `analyze` na prawdziwe wywołanie Ollama/LLM API.
    """

    def analyze(self, event: dict) -> dict:
        """Analizuje zdarzenie i zwraca akcję."""
        ts       = event.get("timestamp", time.time())
        energy   = event.get("mean_energy", 0.0)
        channels = event.get("active_channels", [])
        count    = event.get("spike_count", 0)

        # === PODMIEŃ NA PRAWDZIWY LLM ===
        # import requests
        # resp = requests.post("http://localhost:11434/api/generate", json={
        #     "model": "llama3",
        #     "prompt": f"Wykryto anomalię audio: {count} spików, energia={energy:.2f}. Co to może być?",
        #     "stream": False
        # })
        # decision = resp.json()["response"]
        # ================================

        # Mock decyzja
        if energy > 0.6:
            decision = "HIGH_CONFIDENCE_ANOMALY"
            action   = "TRIGGER_ALERT"
        elif count > 10:
            decision = "POSSIBLE_ANOMALY"
            action   = "LOG_AND_MONITOR"
        else:
            decision = "FALSE_POSITIVE"
            action   = "IGNORE"

        result = {
            "decision":   decision,
            "action":     action,
            "confidence": min(1.0, energy * count / 20.0),
            "timestamp":  ts,
            "raw_event":  event,
        }
        return result

    def execute_action(self, action: str, result: dict):
        """Wykonaj akcję: dioda, MQTT, webhook, etc."""
        print(f"\n{'='*50}")
        print(f"[AGENT] Decyzja:    {result['decision']}")
        print(f"[AGENT] Akcja:      {action}")
        print(f"[AGENT] Pewność:    {result['confidence']:.2f}")
        print(f"[AGENT] Spiki:      {result['raw_event'].get('spike_count', '?')}")
        print(f"[AGENT] Energia:    {result['raw_event'].get('mean_energy', 0):.3f}")
        print(f"{'='*50}\n")

        if action == "TRIGGER_ALERT":
            # Tutaj: GPIO dioda, buzzer, MQTT, HTTP webhook
            pass
        elif action == "LOG_AND_MONITOR":
            with open("anomaly_log.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")

# ============================================================
#  Główna pętla dekodera — Wake-Up pattern
# ============================================================
class Decoder:
    def __init__(self, cfg: DecoderConfig):
        self.cfg        = cfg
        self.classifier = SpikeClassifier(cfg)
        self.agent      = MockAgent()
        self.gpio       = GPIOController(cfg.led_gpio_pin)

        # Okno spików (deque z timestampem)
        self.spike_window: deque = deque()
        self.window_lock  = threading.Lock()

        # Stan systemu
        self.state              = SystemState.IDLE
        self.last_spike_time    = 0.0
        self.confirm_counter    = 0

        # Statystyki
        self.stats = {"total_spikes": 0, "events": 0, "false_positives": 0}

    # ----------------------------------------------------------
    #  Odbiór spików (UDP)
    # ----------------------------------------------------------
    def _udp_listener(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.cfg.udp_port))
        sock.settimeout(1.0)
        print(f"[DECODER] Nasłuchuję UDP :{self.cfg.udp_port}")

        while self._running:
            try:
                data, addr = sock.recvfrom(64)
            except socket.timeout:
                continue

            for i in range(0, len(data) - PACKET_SIZE + 1, PACKET_SIZE):
                pkt = decode_packet(data[i:i+PACKET_SIZE])
                if pkt:
                    self._on_spike(pkt)

        sock.close()

    # ----------------------------------------------------------
    #  Odbiór spików (SPI — fizyczne Arduino)
    # ----------------------------------------------------------
    def _spi_listener(self):
        try:
            import spidev
        except ImportError:
            print("[SPI] spidev niedostępne — zainstaluj: pip install spidev")
            return

        spi = spidev.SpiDev()
        spi.open(0, 0)         # bus 0, device 0 (CE0)
        spi.max_speed_hz = 1_000_000
        spi.mode = 0
        print("[DECODER] Nasłuchuję SPI (bus 0, CE0)")

        buf = bytearray()
        while self._running:
            chunk = spi.readbytes(PACKET_SIZE)
            buf.extend(chunk)
            # Szukaj nagłówka
            while len(buf) >= PACKET_SIZE:
                idx = buf.find(SPIKE_HEADER)
                if idx < 0:
                    buf.clear()
                    break
                if idx > 0:
                    del buf[:idx]
                if len(buf) < PACKET_SIZE:
                    break
                pkt = decode_packet(bytes(buf[:PACKET_SIZE]))
                del buf[:PACKET_SIZE]
                if pkt:
                    self._on_spike(pkt)
            time.sleep(0.001)

        spi.close()

    # ----------------------------------------------------------
    #  Callback: nowy spike
    # ----------------------------------------------------------
    def _on_spike(self, pkt: dict):
        with self.window_lock:
            self.spike_window.append(pkt)
            self.last_spike_time = pkt["received_at"]
            self.stats["total_spikes"] += 1

        if self.cfg.verbose:
            ch = pkt["channel"]
            e  = pkt["energy"]
            print(f"[SPIKE] ch={ch} energy={e:.2f} ts={pkt['timestamp_ms']}")

    # ----------------------------------------------------------
    #  Czyść stare spiki z okna czasowego
    # ----------------------------------------------------------
    def _prune_window(self):
        now   = time.time()
        cutoff = now - self.cfg.window_sec
        with self.window_lock:
            while self.spike_window and self.spike_window[0]["received_at"] < cutoff:
                self.spike_window.popleft()

    # ----------------------------------------------------------
    #  Zbierz statystyki okna
    # ----------------------------------------------------------
    def _window_stats(self) -> tuple[list[int], list[float]]:
        with self.window_lock:
            counts   = [0] * N_CHANNELS
            energies = [0.0] * N_CHANNELS
            for pkt in self.spike_window:
                ch = min(pkt["channel"], N_CHANNELS - 1)
                counts[ch]   += 1
                energies[ch] += pkt["energy"]
            for ch in range(N_CHANNELS):
                if counts[ch] > 0:
                    energies[ch] /= counts[ch]
        return counts, energies

    # ----------------------------------------------------------
    #  Pętla Wake-Up
    # ----------------------------------------------------------
    def _process_loop(self):
        """
        IDLE:    sprawdza okno co 50ms (bardzo mało CPU)
        ACTIVE:  co 20ms — przetwarza i wywołuje agenta
        """
        print("[DECODER] Pętla przetwarzania uruchomiona")
        print(f"[DECODER] Próg: {self.cfg.spike_threshold} spików w {self.cfg.window_sec}s\n")

        while self._running:
            self._prune_window()
            counts, energies = self._window_stats()
            total = sum(counts)
            now   = time.time()

            # === IDLE → ACTIVE ===
            if self.state == SystemState.IDLE:
                if total >= self.cfg.spike_threshold:
                    self.confirm_counter += 1
                    if self.confirm_counter >= self.cfg.confirm_count:
                        self._wake_up(counts, energies)
                else:
                    self.confirm_counter = 0
                time.sleep(0.05)  # 50ms polling — minimalne CPU

            # === ACTIVE ===
            else:
                cls = self.classifier.classify(counts, energies)

                if cls == 1:
                    event = {
                        "timestamp":       now,
                        "spike_count":     total,
                        "channel_counts":  counts,
                        "mean_energy":     sum(energies) / max(1, sum(1 for e in energies if e > 0)),
                        "active_channels": [i for i, c in enumerate(counts) if c > 0],
                    }
                    result = self.agent.analyze(event)
                    self.agent.execute_action(result["action"], result)
                    self.stats["events"] += 1
                    self.gpio.set(True)
                    time.sleep(1.0)  # nie spamuj agenta
                    self.gpio.set(False)

                # Timeout → powrót do IDLE
                if now - self.last_spike_time > self.cfg.idle_timeout_sec:
                    self._go_idle()

                time.sleep(0.02)  # 20ms w trybie aktywnym

    def _wake_up(self, counts, energies):
        self.state = SystemState.ACTIVE
        self.confirm_counter = 0
        print(f"\n[WAKE-UP] Wykryto aktywność: counts={counts} energies={[f'{e:.2f}' for e in energies]}")

    def _go_idle(self):
        self.state = SystemState.IDLE
        print(f"[IDLE] Brak spików przez {self.cfg.idle_timeout_sec}s → powrót do uśpienia")
        print(f"[STATS] Łącznie: spikes={self.stats['total_spikes']} events={self.stats['events']}")

    # ----------------------------------------------------------
    #  Start / Stop
    # ----------------------------------------------------------
    def start(self):
        self._running = True

        listener = (self._spi_listener if self.cfg.use_spi else self._udp_listener)
        self._listener_thread = threading.Thread(target=listener, daemon=True)
        self._process_thread  = threading.Thread(target=self._process_loop, daemon=True)

        self._listener_thread.start()
        self._process_thread.start()

        print(f"[DECODER] Stan: {self.state.name}")
        print(f"[DECODER] Tryb: {'SPI' if self.cfg.use_spi else 'UDP'}\n")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[DECODER] Zatrzymuję...")
        finally:
            self._running = False
            self.gpio.cleanup()

# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SNN Decoder (RPi)")
    parser.add_argument("--port",    type=int,  default=9000)
    parser.add_argument("--spi",     action="store_true", help="Użyj SPI zamiast UDP")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--threshold", type=int, default=5,   help="Spiki w oknie → wake-up")
    parser.add_argument("--window",    type=float, default=0.5, help="Okno czasowe [s]")
    parser.add_argument("--timeout",   type=float, default=10.0, help="Idle timeout [s]")
    parser.add_argument("--led",       type=int, default=17, help="GPIO pin diody BCM (-1=wyłącz)")
    args = parser.parse_args()

    cfg = DecoderConfig(
        udp_port=args.port,
        use_spi=args.spi,
        verbose=args.verbose,
        spike_threshold=args.threshold,
        window_sec=args.window,
        idle_timeout_sec=args.timeout,
        led_gpio_pin=args.led,
    )

    Decoder(cfg).start()


if __name__ == "__main__":
    main()
