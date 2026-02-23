# experiments/decoder.py
import time
import threading
import queue
import socket
import numpy as np
import sys
import os
from dataclasses import dataclass
from collections import deque
from datetime import datetime
import logging
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("/tmp/decoder.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

SPI_BUS = 0
SPI_DEVICE = 0
SPI_SPEED = 4_000_000

AUDIO_BUFFER_SIZE = 16000
circular_buffer = deque(maxlen=AUDIO_BUFFER_SIZE)

SPIKE_QUEUE = queue.Queue(maxsize=100)

ANOMALY_THRESHOLD = 150
SPIKE_COOLDOWN_MS = 100

LLM_MODEL = "tinyllama"
OLLAMA_HOST = "http://localhost:11434"

GPIO_PIN = 17


class GPIOController:
    def __init__(self, pin: int):
        self.pin = pin
        self._gpio = None
        self.enabled = False
        try:
            import RPi.GPIO as GPIO  # type: ignore

            self._gpio = GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.pin, GPIO.OUT)
            self.enabled = True
            logger.info("GPIO initialized (RPi.GPIO)")
        except Exception:
            self.enabled = False
            logger.warning("GPIO not available - LED disabled")

    def pulse(self, seconds: float = 0.05) -> None:
        if not self.enabled or self._gpio is None:
            return
        try:
            self._gpio.output(self.pin, 1)
            time.sleep(seconds)
            self._gpio.output(self.pin, 0)
        except Exception as e:
            logger.warning(f"GPIO pulse failed: {e}")

    def cleanup(self) -> None:
        if not self.enabled or self._gpio is None:
            return
        try:
            self._gpio.cleanup()
        except Exception:
            pass


gpio = GPIOController(GPIO_PIN)


class SpiPacket:
    HEADER = 0xAE
    SIZE = 6

    def __init__(self, raw_bytes: list[int] | bytes | bytearray):
        if len(raw_bytes) < self.SIZE:
            raise ValueError(f"Packet too short: {len(raw_bytes)} < {self.SIZE}")

        b = list(raw_bytes)
        self.header = b[0] & 0xFF
        self.timestamp = ((b[1] & 0xFF) << 8) | (b[2] & 0xFF)
        self.energy_level = b[3] & 0xFF
        self.spike_id = b[4] & 0xFF
        self.checksum = b[5] & 0xFF
        self.received_time = time.time()

    def validate(self) -> bool:
        if self.header != self.HEADER:
            return False
        hi = (self.timestamp >> 8) & 0xFF
        lo = self.timestamp & 0xFF
        expected = (self.header ^ hi ^ lo ^ self.energy_level ^ self.spike_id) & 0xFF
        if self.checksum != expected:
            logger.warning(
                f"Checksum mismatch: got {hex(self.checksum)}, expected {hex(expected)}"
            )
            return False
        return True

    def __repr__(self) -> str:
        return f"Spike(T={self.timestamp:05d}, E={self.energy_level:3d}, ID={self.spike_id})"


@dataclass
class MockPacket:
    timestamp: int
    energy_level: int
    spike_id: int

    def validate(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"MockSpike(T={self.timestamp:05d}, E={self.energy_level:3d}, ID={self.spike_id})"


class SpiReaderBase:
    def start(self) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError


class SpiReader(SpiReaderBase):
    def __init__(self, bus=0, device=0, speed=4_000_000):
        try:
            import spidev  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "spidev is not available on this system. "
                "Install it only on Linux (e.g., Raspberry Pi) or run with mock SPI."
            ) from e

        self.spi = spidev.SpiDev()
        self.spi.open(bus, device)
        self.spi.max_speed_hz = speed
        self.spi.mode = 0
        self.spi.lsb_first = False

        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.packet_count = 0
        self.error_count = 0

        logger.info(
            f"SPI Reader initialized: Bus={bus}, Device={device}, Speed={speed / 1e6:.1f}MHz"
        )

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
        logger.info("SPI Reader thread started")

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        try:
            self.spi.close()
        except Exception:
            pass
        logger.info(
            f"SPI Reader stopped - {self.packet_count} packets received, {self.error_count} errors"
        )

    def _read_loop(self) -> None:
        last_spike_time_ms = 0.0
        while self.running:
            try:
                rx_data = self.spi.readbytes(SpiPacket.SIZE)
                pkt = SpiPacket(rx_data)
                if not pkt.validate():
                    self.error_count += 1
                    continue

                self.packet_count += 1
                now_ms = time.time() * 1000.0
                if now_ms - last_spike_time_ms < SPIKE_COOLDOWN_MS:
                    continue
                last_spike_time_ms = now_ms

                try:
                    SPIKE_QUEUE.put_nowait((pkt, datetime.now()))
                    logger.info(f"[SPIKE] {pkt} | Energy rise detected!")
                    gpio.pulse(0.05)
                except queue.Full:
                    logger.warning("Spike queue full - dropping packet")

            except KeyboardInterrupt:
                break
            except Exception as e:
                self.error_count += 1
                logger.error(f"SPI read error: {e}")
                time.sleep(0.01)


class MockSpiReader(SpiReaderBase):
    def __init__(self, rate_hz: float = 2.0):
        self.rate_hz = rate_hz
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.packet_count = 0
        logger.warning("Using MockSpiReader (no hardware SPI).")

    def start(self) -> None:
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(f"Mock SPI stopped - {self.packet_count} spikes generated")

    def _loop(self) -> None:
        spike_id = 0
        while self.running:
            energy = int(
                np.random.choice(
                    [30, 40, 60, 90, 120, 180, 220],
                    p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05],
                )
            )
            pkt = MockPacket(
                timestamp=int(time.time() * 1000) & 0xFFFF,
                energy_level=energy,
                spike_id=spike_id & 0xFF,
            )
            spike_id += 1

            try:
                SPIKE_QUEUE.put_nowait((pkt, datetime.now()))
                self.packet_count += 1
                logger.info(f"[SPIKE] {pkt} | (mock)")
            except queue.Full:
                pass

            time.sleep(max(0.001, 1.0 / self.rate_hz))


TCP_HOST = "127.0.0.1"
TCP_PORT = 9999


class TcpReader(SpiReaderBase):
    def __init__(self, host: str = TCP_HOST, port: int = TCP_PORT):
        self.host = host
        self.port = port
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.server_socket: Optional[socket.socket] = None
        self.packet_count = 0
        self.error_count = 0
        self._ready = threading.Event()
        logger.info(f"TcpReader initialized: {host}:{port}")

    @property
    def bound_port(self) -> int:
        if self.server_socket is not None:
            return self.server_socket.getsockname()[1]
        return self.port

    def start(self) -> None:
        self.running = True
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.settimeout(1.0)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.thread = threading.Thread(target=self._accept_loop, daemon=True)
        self.thread.start()
        self._ready.set()
        logger.info(f"TcpReader listening on {self.host}:{self.bound_port}")

    def wait_ready(self, timeout: float = 5.0) -> bool:
        return self._ready.wait(timeout)

    def stop(self) -> None:
        self.running = False
        if self.server_socket:
            try:
                self.server_socket.close()
            except Exception:
                pass
        if self.thread:
            self.thread.join(timeout=2)
        logger.info(
            f"TcpReader stopped – {self.packet_count} packets, {self.error_count} errors"
        )

    def _accept_loop(self) -> None:
        while self.running:
            try:
                conn, addr = self.server_socket.accept()  # type: ignore[union-attr]
                logger.info(f"TcpReader: client connected from {addr}")
                self._handle_client(conn)
            except socket.timeout:
                continue
            except OSError:
                break

    def _handle_client(self, conn: socket.socket) -> None:
        conn.settimeout(1.0)
        buf = b""
        last_spike_time_ms = 0.0
        try:
            while self.running:
                try:
                    chunk = conn.recv(4096)
                    if not chunk:
                        break
                    buf += chunk
                except socket.timeout:
                    continue

                while len(buf) >= SpiPacket.SIZE:
                    raw = buf[: SpiPacket.SIZE]
                    buf = buf[SpiPacket.SIZE :]

                    try:
                        pkt = SpiPacket(raw)
                    except ValueError:
                        self.error_count += 1
                        continue

                    if not pkt.validate():
                        self.error_count += 1
                        continue

                    self.packet_count += 1
                    now_ms = time.time() * 1000.0
                    if now_ms - last_spike_time_ms < SPIKE_COOLDOWN_MS:
                        continue
                    last_spike_time_ms = now_ms

                    try:
                        SPIKE_QUEUE.put_nowait((pkt, datetime.now()))
                        logger.info(f"[SPIKE] {pkt} | via TCP")
                        gpio.pulse(0.05)
                    except queue.Full:
                        logger.warning("Spike queue full – dropping TCP packet")
        except Exception as e:
            logger.error(f"TcpReader client error: {e}")
        finally:
            conn.close()


class AnomalyDetector:
    def __init__(self, spike_queue: queue.Queue, buffer: deque):
        self.spike_queue = spike_queue
        self.buffer = buffer
        self.spike_history = deque(maxlen=10)
        self.event_threshold = 3
        self.window_s = 0.5

    def process(self) -> bool:
        try:
            pkt, ts = self.spike_queue.get(timeout=0.1)

            energy = int(getattr(pkt, "energy_level", 0))
            spike_id = int(getattr(pkt, "spike_id", 0))
            self.buffer.append(energy)

            if energy < ANOMALY_THRESHOLD:
                return False

            self.spike_history.append({"energy": energy, "time": ts, "id": spike_id})

            if len(self.spike_history) >= self.event_threshold:
                dt = (
                    self.spike_history[-1]["time"] - self.spike_history[0]["time"]
                ).total_seconds()
                if dt <= self.window_s:
                    avg_energy = sum(s["energy"] for s in self.spike_history) / len(
                        self.spike_history
                    )
                    logger.info(
                        f"[EVENT] Spike cluster: {len(self.spike_history)} spikes in {dt:.2f}s | avg energy: {avg_energy:.0f}"
                    )
                    self.spike_history.clear()
                    return True

            return False

        except queue.Empty:
            return False
        except Exception as e:
            logger.error(f"Error in AnomalyDetector: {e}")
            return False


class LLMAgent:
    def __init__(self, model=LLM_MODEL, host=OLLAMA_HOST):
        self.model = model
        self.host = host
        self.is_running = False
        try:
            import requests  # type: ignore

            self.requests = requests
            self.enabled = True
            logger.info(f"LLM Agent initialized: model={model}")
        except Exception:
            self.requests = None
            self.enabled = False
            logger.warning("requests not installed - LLM calls disabled")

    def wake_up_async(self, spike_data: float, context_buffer: list[int]) -> None:
        if not self.enabled:
            return
        t = threading.Thread(
            target=self.wake_up, args=(spike_data, context_buffer), daemon=True
        )
        t.start()

    def wake_up(self, spike_data: float, context_buffer: list[int]) -> None:
        if not self.enabled or self.requests is None:
            return
        if self.is_running:
            logger.warning("LLM already running - skipping")
            return

        self.is_running = True
        logger.info("[LLM] WAKING UP AGENT")

        context_tail = context_buffer[-100:] if context_buffer else []
        context_str = f"Energy(avg_recent)={spike_data:.1f} | Last {len(context_tail)} samples: {context_tail}"

        prompt = (
            "You are an anomaly detector. The home sensor just detected an acoustic event:\n\n"
            f"{context_str}\n\n"
            "Respond in ONE LINE with action:\n"
            "- ALARM: If danger detected (glass breaking, siren, etc)\n"
            "- WARNING: If suspicious\n"
            "- NORMAL: If false alarm\n\n"
            "Action:"
        )

        try:
            resp = self.requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,
                },
                timeout=30,
            )

            if resp.status_code == 200:
                result = resp.json()
                action = (result.get("response", "") or "").strip().split("\n")[0]
                logger.info(f"[LLM RESPONSE] {action}")

                up = action.upper()
                if "ALARM" in up:
                    logger.critical("ALARM TRIGGERED")
                    self._trigger_alarm()
                elif "WARNING" in up:
                    logger.warning("WARNING")
                else:
                    logger.info("NORMAL")
            else:
                logger.error(
                    f"Ollama error: HTTP {resp.status_code} | {resp.text[:200]}"
                )
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
        finally:
            self.is_running = False

    def _trigger_alarm(self) -> None:
        logger.critical("ALARM ACTION: implement siren/email/push here")


class NeuromorphicDecoder:
    def __init__(self):
        self.spi_reader: SpiReaderBase = self._make_spi_reader()
        self.anomaly_detector = AnomalyDetector(SPIKE_QUEUE, circular_buffer)
        self.llm_agent = LLMAgent()
        self.running = False

    def _make_spi_reader(self) -> SpiReaderBase:
        mode = str(os.environ.get("SNN_READER", "")).strip().lower()

        if mode == "tcp":
            tcp_host = os.environ.get("SNN_TCP_HOST", TCP_HOST)
            tcp_port = int(os.environ.get("SNN_TCP_PORT", str(TCP_PORT)))
            return TcpReader(host=tcp_host, port=tcp_port)

        if str(os.environ.get("SNN_USE_MOCK_SPI", "")).strip() == "1":
            return MockSpiReader()

        if sys.platform.startswith("linux"):
            try:
                return SpiReader(SPI_BUS, SPI_DEVICE, SPI_SPEED)
            except Exception as e:
                logger.warning(f"Real SPI unavailable, falling back to mock: {e}")
                return MockSpiReader()
        return MockSpiReader()

    def start(self) -> None:
        self.spi_reader.start()
        self.running = True
        try:
            self._main_loop()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            self.stop()

    def _main_loop(self) -> None:
        while self.running:
            event_detected = self.anomaly_detector.process()
            if event_detected:
                recent = list(circular_buffer)[-256:] if circular_buffer else [0]
                avg_energy = float(np.mean(recent)) if recent else 0.0
                self.llm_agent.wake_up_async(
                    spike_data=avg_energy, context_buffer=recent
                )
            time.sleep(0.01)

    def stop(self) -> None:
        self.running = False
        try:
            self.spi_reader.stop()
        finally:
            gpio.cleanup()
        logger.info("System stopped")


if __name__ == "__main__":
    try:
        decoder = NeuromorphicDecoder()
        decoder.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
