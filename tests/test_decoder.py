# tests/test_decoder.py

import queue
import sys
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, PropertyMock
import pytest
import numpy as np


# Mockujemy RPi.GPIO i spidev zanim moduł zostanie zaimportowany
sys.modules.setdefault("RPi", MagicMock())
sys.modules.setdefault("RPi.GPIO", MagicMock())
sys.modules.setdefault("spidev", MagicMock())

sys.path.insert(0, ".")

from experiments.decoder import (  # noqa: E402  (import po mockach)
    SpiPacket,
    MockPacket,
    MockSpiReader,
    TcpReader,
    AnomalyDetector,
    LLMAgent,
    GPIOController,
    NeuromorphicDecoder,
    SPIKE_QUEUE,
    ANOMALY_THRESHOLD,
)

from experiments.encoder_sim import EncoderSim, build_packet  # noqa: E402


def make_raw_packet(header=0xAE, timestamp=0x0064, energy=200, spike_id=1):
    """Buduje poprawny 6-bajtowy surowy pakiet z wyliczonym checksumem."""
    hi = (timestamp >> 8) & 0xFF
    lo = timestamp & 0xFF
    checksum = (header ^ hi ^ lo ^ energy ^ spike_id) & 0xFF
    return bytes([header, hi, lo, energy, spike_id, checksum])


def make_spike_packet(**kwargs):
    raw = make_raw_packet(**kwargs)
    return SpiPacket(raw)


class TestSpiPacket:

    def test_parse_valid_packet(self):
        raw = make_raw_packet(header=0xAE, timestamp=100, energy=200, spike_id=1)
        pkt = SpiPacket(raw)
        assert pkt.header == 0xAE
        assert pkt.timestamp == 100
        assert pkt.energy_level == 200
        assert pkt.spike_id == 1

    def test_validate_correct_checksum(self):
        raw = make_raw_packet(header=0xAE, timestamp=100, energy=200, spike_id=1)
        pkt = SpiPacket(raw)
        assert pkt.validate() is True

    def test_validate_wrong_header(self):
        raw = make_raw_packet(header=0xFF, timestamp=100, energy=200, spike_id=1)
        pkt = SpiPacket(raw)
        assert pkt.validate() is False

    def test_validate_corrupted_checksum(self):
        raw = bytearray(make_raw_packet(energy=200))
        raw[5] ^= 0xFF
        pkt = SpiPacket(bytes(raw))
        assert pkt.validate() is False

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="Packet too short"):
            SpiPacket(bytes([0xAE, 0x00, 0x64]))

    def test_repr_contains_key_fields(self):
        pkt = make_spike_packet(timestamp=9999, energy=42, spike_id=7)
        r = repr(pkt)
        assert "9999" in r
        assert "42" in r
        assert "7" in r

    def test_timestamp_high_bytes(self):
        ts = 0x1234
        raw = make_raw_packet(timestamp=ts, energy=50, spike_id=0)
        pkt = SpiPacket(raw)
        assert pkt.timestamp == ts

    def test_zero_energy_packet(self):
        raw = make_raw_packet(energy=0, spike_id=0)
        pkt = SpiPacket(raw)
        assert pkt.validate() is True


class TestMockPacket:

    def test_always_valid(self):
        pkt = MockPacket(timestamp=1000, energy_level=255, spike_id=5)
        assert pkt.validate() is True

    def test_repr(self):
        pkt = MockPacket(timestamp=42, energy_level=111, spike_id=3)
        r = repr(pkt)
        assert "42" in r
        assert "111" in r

class TestAnomalyDetector:

    def setup_method(self):
        while not SPIKE_QUEUE.empty():
            try:
                SPIKE_QUEUE.get_nowait()
            except queue.Empty:
                break
        self.buf = deque(maxlen=16000)
        self.detector = AnomalyDetector(SPIKE_QUEUE, self.buf)

    def _push_spike(self, energy, dt_offset_ms=0):
        pkt = MockPacket(timestamp=int(time.time() * 1000) & 0xFFFF,
                         energy_level=energy, spike_id=0)
        ts = datetime.now() + timedelta(milliseconds=dt_offset_ms)
        SPIKE_QUEUE.put((pkt, ts))

    def test_low_energy_no_event(self):
        self._push_spike(energy=ANOMALY_THRESHOLD - 1)
        result = self.detector.process()
        assert result is False

    def test_exactly_threshold_no_event(self):
        self._push_spike(energy=ANOMALY_THRESHOLD)
        result = self.detector.process()
        assert result is False

    def test_cluster_triggers_event(self):
        for _ in range(3):
            self._push_spike(energy=ANOMALY_THRESHOLD + 50)
        results = [self.detector.process() for _ in range(3)]
        assert True in results

    def test_no_event_on_single_spike(self):
        self._push_spike(energy=ANOMALY_THRESHOLD + 100)
        result = self.detector.process()
        assert result is False

    def test_buffer_receives_energy(self):
        self._push_spike(energy=77)
        self.detector.process()
        assert 77 in self.buf

    def test_empty_queue_returns_false(self):
        result = self.detector.process()
        assert result is False

    def test_history_clears_after_event(self):
        for _ in range(3):
            self._push_spike(energy=ANOMALY_THRESHOLD + 80)
        for _ in range(10):
            if self.detector.process():
                break
        assert len(self.detector.spike_history) == 0


class TestLLMAgent:

    def setup_method(self):
        self.agent = LLMAgent()
        self.agent.enabled = True

    def _mock_response(self, text: str, status=200):
        mock_resp = MagicMock()
        mock_resp.status_code = status
        mock_resp.json.return_value = {"response": text}
        mock_resp.text = text
        return mock_resp

    def test_wake_up_alarm(self):
        with patch.object(self.agent, "requests") as mock_req:
            mock_req.post.return_value = self._mock_response("ALARM: glass breaking detected")
            with patch.object(self.agent, "_trigger_alarm") as mock_alarm:
                self.agent.wake_up(spike_data=220.0, context_buffer=[220]*10)
                mock_alarm.assert_called_once()

    def test_wake_up_normal(self):
        with patch.object(self.agent, "requests") as mock_req:
            mock_req.post.return_value = self._mock_response("NORMAL: background noise")
            with patch.object(self.agent, "_trigger_alarm") as mock_alarm:
                self.agent.wake_up(spike_data=30.0, context_buffer=[30]*10)
                mock_alarm.assert_not_called()

    def test_wake_up_warning(self):
        with patch.object(self.agent, "requests") as mock_req:
            mock_req.post.return_value = self._mock_response("WARNING: suspicious sound")
            with patch.object(self.agent, "_trigger_alarm") as mock_alarm:
                self.agent.wake_up(spike_data=160.0, context_buffer=[160]*5)
                mock_alarm.assert_not_called()

    def test_http_error_handled(self):
        with patch.object(self.agent, "requests") as mock_req:
            mock_req.post.return_value = self._mock_response("", status=503)
            self.agent.wake_up(spike_data=200.0, context_buffer=[])

    def test_network_exception_handled(self):
        with patch.object(self.agent, "requests") as mock_req:
            mock_req.post.side_effect = ConnectionError("timeout")
            self.agent.wake_up(spike_data=200.0, context_buffer=[])

    def test_skip_when_already_running(self):
        self.agent.is_running = True
        with patch.object(self.agent, "requests") as mock_req:
            self.agent.wake_up(spike_data=200.0, context_buffer=[])
            mock_req.post.assert_not_called()
        self.agent.is_running = False

    def test_disabled_agent_does_nothing(self):
        self.agent.enabled = False
        self.agent.wake_up(spike_data=200.0, context_buffer=[])

    def test_prompt_contains_energy_value(self):
        captured_prompt = {}
        def fake_post(url, json=None, timeout=None):
            captured_prompt["json"] = json
            return self._mock_response("NORMAL")
        with patch.object(self.agent, "requests") as mock_req:
            mock_req.post.side_effect = fake_post
            self.agent.wake_up(spike_data=185.5, context_buffer=[185]*5)
        assert "185.5" in captured_prompt["json"]["prompt"]


class TestGPIOController:

    def test_pulse_when_disabled_does_not_raise(self):
        ctrl = GPIOController(pin=17)
        ctrl.enabled = False
        ctrl.pulse(0.01)

    def test_pulse_calls_gpio_output(self):
        mock_gpio = MagicMock()
        ctrl = GPIOController(pin=17)
        ctrl._gpio = mock_gpio
        ctrl.enabled = True
        ctrl.pulse(0.001)
        calls = [c[0][1] for c in mock_gpio.output.call_args_list]
        assert 1 in calls
        assert 0 in calls

    def test_cleanup_when_disabled_does_not_raise(self):
        ctrl = GPIOController(pin=17)
        ctrl.enabled = False
        ctrl.cleanup()


class TestMockSpiReader:

    def test_reader_generates_packets(self):
        test_queue = queue.Queue(maxsize=50)
        reader = MockSpiReader(rate_hz=100.0)
        import experiments.decoder as dec
        original_queue = dec.SPIKE_QUEUE
        dec.SPIKE_QUEUE = test_queue
        reader.start()
        time.sleep(0.15)
        reader.stop()
        dec.SPIKE_QUEUE = original_queue
        assert not test_queue.empty()

    def test_reader_stops_cleanly(self):
        reader = MockSpiReader(rate_hz=10.0)
        reader.start()
        time.sleep(0.05)
        reader.stop()
        assert not reader.running

    def test_mock_energy_distribution(self):
        samples = [
            int(np.random.choice([30, 40, 60, 90, 120, 180, 220],
                                 p=[0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.05]))
            for _ in range(200)
        ]
        unique = set(samples)
        assert len(unique) > 2


class TestNeuromorphicDecoder:

    def test_decoder_uses_mock_spi_outside_linux(self):
        import experiments.decoder as dec
        with patch.object(dec.sys, "platform", "win32"):
            d = NeuromorphicDecoder()
        assert isinstance(d.spi_reader, MockSpiReader)

    def test_decoder_uses_mock_spi_via_env(self):
        import experiments.decoder as dec
        with patch.dict(dec.os.environ, {"SNN_USE_MOCK_SPI": "1"}):
            d = NeuromorphicDecoder()
        assert isinstance(d.spi_reader, MockSpiReader)

    def test_main_loop_stops_on_flag(self):
        d = NeuromorphicDecoder()
        d.running = False
        t = threading.Thread(target=d._main_loop, daemon=True)
        t.start()
        t.join(timeout=0.5)
        assert not t.is_alive()

    def test_stop_does_not_raise(self):
        d = NeuromorphicDecoder()
        d.stop()

    def test_llm_called_on_event_detection(self):
        d = NeuromorphicDecoder()
        d.running = True
        called = threading.Event()

        def fake_wake_up_async(**kwargs):
            called.set()
            d.running = False

        d.llm_agent.wake_up_async = fake_wake_up_async

        with patch.object(d.anomaly_detector, "process", side_effect=[True, False]):
            t = threading.Thread(target=d._main_loop, daemon=True)
            t.start()
            called.wait(timeout=2.0)
            d.running = False
            t.join(timeout=1.0)

        assert called.is_set()


class TestEndToEnd:

    def test_full_pipeline_spike_to_llm(self):
        buf = deque(maxlen=16000)
        test_q = queue.Queue(maxsize=100)
        detector = AnomalyDetector(test_q, buf)
        agent = LLMAgent()
        agent.enabled = True

        llm_called = threading.Event()

        with patch.object(agent, "requests") as mock_req:
            mock_req.post.return_value = MagicMock(
                status_code=200,
                json=lambda: {"response": "ALARM: test"}
            )
            with patch.object(agent, "_trigger_alarm", side_effect=lambda: llm_called.set()):
                now = datetime.now()
                for i in range(3):
                    pkt = MockPacket(timestamp=i, energy_level=200, spike_id=i)
                    test_q.put((pkt, now))

                triggered = False
                for _ in range(10):
                    if detector.process():
                        avg = float(np.mean(list(buf)[-256:]) if buf else 200.0)
                        agent.wake_up(spike_data=avg, context_buffer=list(buf))
                        triggered = True
                        break

                assert triggered
                assert llm_called.is_set()

    def test_checksum_roundtrip(self):
        header = 0xAE
        timestamp = 0x0190  # 400
        energy = 180
        spike_id = 0x00
        hi = (timestamp >> 8) & 0xFF
        lo = timestamp & 0xFF
        checksum = (header ^ hi ^ lo ^ energy ^ spike_id) & 0xFF

        raw = bytes([header, hi, lo, energy, spike_id, checksum])
        pkt = SpiPacket(raw)
        assert pkt.validate() is True
        assert pkt.energy_level == energy
        assert pkt.timestamp == timestamp

class TestTcpTransport:
    """Real-time simulation tests over TCP (localhost)."""

    def _drain_queue(self):
        while not SPIKE_QUEUE.empty():
            try:
                SPIKE_QUEUE.get_nowait()
            except queue.Empty:
                break

    def test_build_packet_valid(self):
        raw = build_packet(timestamp=500, energy=200, spike_id=3)
        assert len(raw) == 6
        pkt = SpiPacket(raw)
        assert pkt.validate() is True
        assert pkt.timestamp == 500
        assert pkt.energy_level == 200
        assert pkt.spike_id == 3

    def test_build_packet_checksum_matches_arduino_logic(self):
        ts = 0x1234
        energy = 180
        sid = 7
        raw = build_packet(timestamp=ts, energy=energy, spike_id=sid)
        hi = (ts >> 8) & 0xFF
        lo = ts & 0xFF
        expected_cs = (0xAE ^ hi ^ lo ^ energy ^ sid) & 0xFF
        assert raw[5] == expected_cs

    def test_tcp_reader_starts_and_stops(self):
        reader = TcpReader(host="127.0.0.1", port=0)
        reader.start()
        assert reader.wait_ready(timeout=2)
        assert reader.bound_port > 0
        reader.stop()
        assert not reader.running

    def test_single_packet_arrives_via_tcp(self):
        self._drain_queue()
        import experiments.decoder as dec
        original_queue = dec.SPIKE_QUEUE
        test_q = queue.Queue(maxsize=50)
        dec.SPIKE_QUEUE = test_q

        reader = TcpReader(host="127.0.0.1", port=0)
        reader.start()
        reader.wait_ready()

        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", reader.bound_port))
            raw = build_packet(timestamp=100, energy=200, spike_id=1)
            sock.sendall(raw)
            time.sleep(0.3)
            sock.close()

            assert not test_q.empty()
            pkt, ts = test_q.get_nowait()
            assert pkt.energy_level == 200
            assert pkt.timestamp == 100
            assert pkt.spike_id == 1
        finally:
            reader.stop()
            dec.SPIKE_QUEUE = original_queue

    def test_corrupted_packet_rejected(self):
        self._drain_queue()
        import experiments.decoder as dec
        original_queue = dec.SPIKE_QUEUE
        test_q = queue.Queue(maxsize=50)
        dec.SPIKE_QUEUE = test_q

        reader = TcpReader(host="127.0.0.1", port=0)
        reader.start()
        reader.wait_ready()

        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect(("127.0.0.1", reader.bound_port))

            bad = bytearray(build_packet(timestamp=100, energy=200, spike_id=1))
            bad[5] ^= 0xFF
            sock.sendall(bytes(bad))
            time.sleep(0.3)
            sock.close()

            assert test_q.empty()
            assert reader.error_count >= 1
        finally:
            reader.stop()
            dec.SPIKE_QUEUE = original_queue

    def test_encoder_sim_sends_packets(self):
        self._drain_queue()
        import experiments.decoder as dec
        original_queue = dec.SPIKE_QUEUE
        test_q = queue.Queue(maxsize=50)
        dec.SPIKE_QUEUE = test_q

        reader = TcpReader(host="127.0.0.1", port=0)
        reader.start()
        reader.wait_ready()

        try:
            sim = EncoderSim(host="127.0.0.1", port=reader.bound_port)
            sim.connect()
            sim.send_one(energy=220, spike_id=0)
            sim.send_one(energy=180, spike_id=1)
            time.sleep(0.3)
            sim.stop()

            assert not test_q.empty()
            pkt, _ = test_q.get_nowait()
            assert pkt.energy_level in (220, 180)
        finally:
            reader.stop()
            dec.SPIKE_QUEUE = original_queue

    def test_tcp_end_to_end_pipeline(self):
        self._drain_queue()
        import experiments.decoder as dec
        original_queue = dec.SPIKE_QUEUE
        test_q = queue.Queue(maxsize=100)
        dec.SPIKE_QUEUE = test_q

        reader = TcpReader(host="127.0.0.1", port=0)
        reader.start()
        reader.wait_ready()

        try:
            sim = EncoderSim(host="127.0.0.1", port=reader.bound_port)
            sim.connect()
            for i in range(3):
                sim.send_one(energy=220, spike_id=i)
            time.sleep(0.5)
            sim.stop()

            buf = deque(maxlen=16000)
            detector = AnomalyDetector(test_q, buf)

            triggered = False
            for _ in range(20):
                if detector.process():
                    triggered = True
                    break

            assert reader.packet_count >= 1
            assert len(buf) >= 1
            if triggered:
                agent = LLMAgent()
                agent.enabled = True
                llm_called = threading.Event()

                with patch.object(agent, "requests") as mock_req:
                    mock_req.post.return_value = MagicMock(
                        status_code=200,
                        json=lambda: {"response": "ALARM: tcp test"},
                    )
                    with patch.object(agent, "_trigger_alarm", side_effect=lambda: llm_called.set()):
                        avg = float(np.mean(list(buf)[-256:])) if buf else 220.0
                        agent.wake_up(spike_data=avg, context_buffer=list(buf))
                        assert llm_called.is_set()
        finally:
            reader.stop()
            dec.SPIKE_QUEUE = original_queue

    def test_decoder_picks_tcp_reader_via_env(self):
        import experiments.decoder as dec
        with patch.dict(dec.os.environ, {"SNN_READER": "tcp", "SNN_TCP_PORT": "0"}):
            d = NeuromorphicDecoder()
        assert isinstance(d.spi_reader, TcpReader)
        d.stop()