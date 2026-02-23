# experiments/encoder_sim.py
"""
Usage:
    python -m experiments.encoder_sim
    python -m experiments.encoder_sim --port 9999 --rate 5 --duration 10
"""

import argparse
import logging
import socket
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

HEADER = 0xAE
ENERGY_THRESHOLD = 150
WINDOW_SIZE = 256


def build_packet(timestamp: int, energy: int, spike_id: int = 0) -> bytes:
    """[HEADER][TS_HI][TS_LO][ENERGY][ID][CHECKSUM], checksum = XOR of all prior bytes."""
    ts = timestamp & 0xFFFF
    hi = (ts >> 8) & 0xFF
    lo = ts & 0xFF
    energy = energy & 0xFF
    spike_id = spike_id & 0xFF
    checksum = (HEADER ^ hi ^ lo ^ energy ^ spike_id) & 0xFF
    return bytes([HEADER, hi, lo, energy, spike_id, checksum])


class EncoderSim:

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9999,
        rate_hz: float = 5.0,
        threshold: int = ENERGY_THRESHOLD,
    ):
        self.host = host
        self.port = port
        self.rate_hz = rate_hz
        self.threshold = threshold
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.sock: Optional[socket.socket] = None
        self.packets_sent = 0
        self.sample_counter = 0

    def connect(self, timeout: float = 5.0) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(timeout)
        self.sock.connect((self.host, self.port))
        logger.info(f"EncoderSim connected to {self.host}:{self.port}")

    def start(self) -> None:
        self.connect()
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.sock:
            try:
                self.sock.close()
            except Exception:
                pass
        logger.info(f"EncoderSim stopped – {self.packets_sent} packets sent")

    def _sample_energy(self) -> int:
        return int(
            np.random.choice(
                [30, 40, 60, 90, 120, 180, 220],
                p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05],
            )
        )

    def _loop(self) -> None:
        spike_id = 0
        while self.running:
            energy = self._sample_energy()
            self.sample_counter += WINDOW_SIZE

            if energy > self.threshold:
                pkt = build_packet(
                    timestamp=self.sample_counter & 0xFFFF,
                    energy=energy,
                    spike_id=spike_id & 0xFF,
                )
                spike_id += 1
                try:
                    self.sock.sendall(pkt)  # type: ignore[union-attr]
                    self.packets_sent += 1
                    logger.info(
                        f"[ENC TX] energy={energy} ts={self.sample_counter & 0xFFFF:#06x} "
                        f"id={spike_id - 1}"
                    )
                except (BrokenPipeError, OSError) as e:
                    logger.error(f"EncoderSim send failed: {e}")
                    self.running = False
                    break
            else:
                logger.debug(f"[ENC] energy={energy} (below threshold, skipped)")

            time.sleep(max(0.001, 1.0 / self.rate_hz))

    def send_one(self, energy: int, spike_id: int = 0) -> None:
        self.sample_counter += WINDOW_SIZE
        pkt = build_packet(
            timestamp=self.sample_counter & 0xFFFF,
            energy=energy,
            spike_id=spike_id,
        )
        self.sock.sendall(pkt)  # type: ignore[union-attr]
        self.packets_sent += 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Encoder simulator (TCP)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--rate", type=float, default=5.0, help="Packets/sec")
    parser.add_argument("--duration", type=float, default=0, help="0 = run forever")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(message)s",
    )

    sim = EncoderSim(host=args.host, port=args.port, rate_hz=args.rate)
    sim.start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        sim.stop()


if __name__ == "__main__":
    main()
