import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import threading
import time
from datetime import datetime

import pytest

from experiments.decoder import SpiPacket, AnomalyDetector, SPIKE_QUEUE, circular_buffer

def encode_packet(timestamp: int, energy: int, spike_id: int) -> bytes:
    header = 0xAE
    ts = timestamp & 0xFFFF
    hi = (ts >> 8) & 0xFF
    lo = ts & 0xFF
    energy &= 0xFF
    spike_id &= 0xFF
    checksum = (header ^ hi ^ lo ^ energy ^ spike_id) & 0xFF
    return bytes([header, hi, lo, energy, spike_id, checksum])

class FakeLLM:
    def __init__(self):
        self.calls = []

    def wake_up_async(self, spike_data, context_buffer):
        self.calls.append((spike_data, list(context_buffer)))

class TcpSpiReader:
    def __init__(self, host, port, packet_size, on_packet):
        import socket
        self.socket = socket
        self.host = host
        self.port = port
        self.packet_size = packet_size
        self.on_packet = on_packet
        self.running = False
        self.thread = None
        self.server = None
        self.conn = None

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        try:
            if self.conn:
                self.conn.close()
        except Exception:
            pass
        try:
            if self.server:
                self.server.close()
        except Exception:
            pass
        if self.thread:
            self.thread.join(timeout=2)

    def _recv_exact(self, n):
        buf = bytearray()
        while len(buf) < n:
            chunk = self.conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("client disconnected")
            buf.extend(chunk)
        return bytes(buf)

    def _loop(self):
        s = self.socket.socket(self.socket.AF_INET, self.socket.SOCK_STREAM)
        s.setsockopt(self.socket.SOL_SOCKET, self.socket.SO_REUSEADDR, 1)
        s.bind((self.host, self.port))
        s.listen(1)
        self.server = s
        self.conn, _ = s.accept()
        self.conn.settimeout(1.0)

        while self.running:
            try:
                raw = self._recv_exact(self.packet_size)
                self.on_packet(raw, datetime.now())
            except self.socket.timeout:
                continue
            except Exception:
                time.sleep(0.01)

def test_remote_encoder_decoder_triggers_llm(monkeypatch):
    while not SPIKE_QUEUE.empty():
        SPIKE_QUEUE.get_nowait()
    circular_buffer.clear()

    fake_llm = FakeLLM()
    detector = AnomalyDetector(SPIKE_QUEUE, circular_buffer)

    def on_packet(raw, ts):
        pkt = SpiPacket(raw)
        assert pkt.validate()
        SPIKE_QUEUE.put_nowait((pkt, ts))

    host, port = "127.0.0.1", 50555
    reader = TcpSpiReader(host, port, SpiPacket.SIZE, on_packet)
    reader.start()

    def encoder_send():
        import socket
        with socket.create_connection((host, port), timeout=5) as c:
            base_ts = int(time.time() * 1000) & 0xFFFF
            c.sendall(encode_packet(base_ts + 1, 200, 1))
            time.sleep(0.05)
            c.sendall(encode_packet(base_ts + 2, 210, 2))
            time.sleep(0.05)
            c.sendall(encode_packet(base_ts + 3, 220, 3))

    t = threading.Thread(target=encoder_send, daemon=True)
    t.start()

    deadline = time.time() + 2.0
    triggered = False
    while time.time() < deadline:
        if detector.process():
            recent = list(circular_buffer)[-256:] if circular_buffer else [0]
            avg_energy = sum(recent) / len(recent)
            fake_llm.wake_up_async(avg_energy, recent)
            triggered = True
            break
        time.sleep(0.01)

    reader.stop()

    assert triggered, "event should be detected"
    assert len(fake_llm.calls) == 1, "LLM should be called exactly once"