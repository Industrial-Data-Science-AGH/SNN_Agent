"""
encoder.py — Symulacja/test enkodera bez fizycznego Arduino.

Dwa tryby:
  1. SIMULATE  — generuje syntetyczne spiki (dobre do testowania decodera)
  2. SERIAL    — czyta spiki z Arduino przez USB Serial (tryb bridge)

Użycie:
  python encoder.py --mode simulate --host 192.168.1.X --port 9000
  python encoder.py --mode serial   --serial /dev/ttyUSB0
"""

import argparse
import socket
import struct
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

# ============================================================
#  Konfiguracja
# ============================================================
SPIKE_HEADER     = 0xAE
PACKET_SIZE      = 6
N_CHANNELS       = 3
SAMPLE_RATE_HZ   = 8000
FRAME_SIZE       = 64   # = 8ms

@dataclass
class EncoderConfig:
    mode:             str   = "simulate"   # "simulate" | "serial"
    host:             str   = "127.0.0.1"  # decoder IP (RPi)
    port:             int   = 9000
    serial_port:      str   = "/dev/ttyUSB0"
    serial_baud:      int   = 115200
    energy_threshold: float = 180.0
    noise_floor:      float = 40.0
    baseline_alpha:   float = 0.005        # szybkość adaptacji

# ============================================================
#  Kodowanie / dekodowanie pakietu
# ============================================================
def encode_packet(channel: int, energy_u8: int, timestamp_ms: int) -> bytes:
    ts_h = (timestamp_ms >> 8) & 0xFF
    ts_l = timestamp_ms & 0xFF
    chk  = SPIKE_HEADER ^ ts_h ^ ts_l ^ energy_u8 ^ channel
    return bytes([SPIKE_HEADER, ts_h, ts_l, energy_u8, channel, chk])


def decode_packet(raw: bytes) -> Optional[dict]:
    if len(raw) < PACKET_SIZE or raw[0] != SPIKE_HEADER:
        return None
    header, ts_h, ts_l, energy, channel, checksum = raw[:6]
    expected_chk = header ^ ts_h ^ ts_l ^ energy ^ channel
    if checksum != expected_chk:
        return None
    return {
        "channel":      channel,
        "energy":       energy,
        "timestamp_ms": (ts_h << 8) | ts_l,
    }

# ============================================================
#  Generator sygnału audio (syntetyczny)
# ============================================================
class AudioSimulator:
    """Generuje próbki audio symulujące mikrofon."""

    def __init__(self, rng_seed: int = 42):
        self.rng  = np.random.default_rng(rng_seed)
        self._t   = 0
        self._anomaly_countdown = 0

    def next_frame(self, n_samples: int = FRAME_SIZE) -> np.ndarray:
        """Zwraca ramkę próbek (ADC 0–1023, DC ~512)."""
        self._t += 1
        noise = self.rng.normal(512, 15, n_samples).astype(np.float32)

        # Co ~5 sekund wstrzyknij burstową anomalię
        if self._anomaly_countdown <= 0 and self._t % 625 == 0:
            self._anomaly_countdown = 4  # burst trwa ~4 ramki = 32ms
            print("[SIM] *** ANOMALIA — burst wstrzyknięty ***")

        if self._anomaly_countdown > 0:
            burst = self.rng.normal(0, 200, n_samples)
            noise += burst
            self._anomaly_countdown -= 1

        return np.clip(noise, 0, 1023).astype(np.int16)

# ============================================================
#  Enkoder (logika analogiczna → spiki)
# ============================================================
class Encoder:
    def __init__(self, cfg: EncoderConfig):
        self.cfg      = cfg
        self.baseline = np.full(N_CHANNELS, 50.0, dtype=np.float32)

    def compute_energy(self, frame: np.ndarray, downsample: int) -> float:
        samples = frame[::downsample].astype(np.float32)
        if len(samples) == 0:
            return 0.0
        mean = samples.mean()
        return float(np.sqrt(((samples - mean) ** 2).mean()))

    def process_frame(self, frame: np.ndarray, ts_ms: int) -> list[bytes]:
        packets = []
        for ch in range(N_CHANNELS):
            ds     = 1 << ch  # kanał 0: każda próbka, 1: co 2, 2: co 4
            energy = self.compute_energy(frame, ds)

            # Adaptacja baseline
            if energy < self.baseline[ch] * 2.0:
                a = self.cfg.baseline_alpha
                self.baseline[ch] = self.baseline[ch] * (1 - a) + energy * a

            delta = energy - self.baseline[ch]

            if delta > self.cfg.energy_threshold and energy > self.cfg.noise_floor:
                energy_u8 = int(min(255.0, (delta / 512.0) * 255.0))
                pkt = encode_packet(ch, energy_u8, ts_ms & 0xFFFF)
                packets.append(pkt)

        return packets

# ============================================================
#  Tryb SIMULATE — wysyła spiki przez UDP do decodera
# ============================================================
class SimulateMode:
    def __init__(self, cfg: EncoderConfig):
        self.cfg     = cfg
        self.encoder = Encoder(cfg)
        self.sim     = AudioSimulator()
        self.sock    = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.frame_interval = FRAME_SIZE / SAMPLE_RATE_HZ  # ~8ms

    def run(self):
        print(f"[SIMULATE] Enkoder → {self.cfg.host}:{self.cfg.port} (UDP)")
        print(f"[SIMULATE] Ctrl+C żeby zatrzymać\n")
        start = time.time()
        spike_count = 0
        last_status = time.time()

        while True:
            t0    = time.time()
            ts_ms = int((t0 - start) * 1000)
            frame = self.sim.next_frame()

            for pkt in self.encoder.process_frame(frame, ts_ms):
                self.sock.sendto(pkt, (self.cfg.host, self.cfg.port))
                spike_count += 1

            # Status co 2s
            if time.time() - last_status > 2.0:
                bl = " | ".join(f"ch{i}={self.encoder.baseline[i]:.1f}" for i in range(N_CHANNELS))
                print(f"[STATUS] {bl}  |  spikes/2s: {spike_count}")
                spike_count = 0
                last_status = time.time()

            # Taktowanie
            elapsed = time.time() - t0
            sleep   = self.frame_interval - elapsed
            if sleep > 0:
                time.sleep(sleep)

# ============================================================
#  Tryb SERIAL — bridge: Arduino USB Serial → UDP
# ============================================================
class SerialMode:
    def __init__(self, cfg: EncoderConfig):
        self.cfg = cfg

    def run(self):
        try:
            import serial
        except ImportError:
            print("[ERROR] pip install pyserial")
            return

        print(f"[SERIAL] Bridge {self.cfg.serial_port} → {self.cfg.host}:{self.cfg.port}")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        with serial.Serial(self.cfg.serial_port, self.cfg.serial_baud, timeout=1) as ser:
            buf = b""
            while True:
                data = ser.read(PACKET_SIZE)
                if not data:
                    continue
                buf += data
                while len(buf) >= PACKET_SIZE:
                    # Szukaj nagłówka
                    idx = buf.find(SPIKE_HEADER)
                    if idx < 0:
                        buf = b""
                        break
                    if idx > 0:
                        buf = buf[idx:]
                    if len(buf) < PACKET_SIZE:
                        break
                    pkt_raw = buf[:PACKET_SIZE]
                    buf = buf[PACKET_SIZE:]
                    parsed = decode_packet(pkt_raw)
                    if parsed:
                        sock.sendto(pkt_raw, (self.cfg.host, self.cfg.port))

# ============================================================
#  CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="SNN Encoder")
    parser.add_argument("--mode",   default="simulate", choices=["simulate", "serial"])
    parser.add_argument("--host",   default="127.0.0.1")
    parser.add_argument("--port",   default=9000, type=int)
    parser.add_argument("--serial", default="/dev/ttyUSB0")
    args = parser.parse_args()

    cfg = EncoderConfig(
        mode=args.mode,
        host=args.host,
        port=args.port,
        serial_port=args.serial,
    )

    if args.mode == "simulate":
        SimulateMode(cfg).run()
    else:
        SerialMode(cfg).run()


if __name__ == "__main__":
    main()
