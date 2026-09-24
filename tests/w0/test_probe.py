import os
import subprocess
import sys

from rpi_agents.agent.probe import probe
from rpi_agents.agent.sources import ReplaySource, SerialPortSource
from rpi_agents.agent.synthetic import scenario


def test_probe_summarises_a_clean_glass_stream():
    result = probe(ReplaySource(scenario("glass", 200)), 5.0)
    assert (result["boots"], result["frames"], result["priming"]) == (1, 200, 50)
    assert result["spike_frames"] == 8 and result["gaps"] == [] and result["rejected"] == {}
    assert result["boot"]["n_ch"] == 7 and result["batches"] == 8  # 200 hops = exactly 8 closed batches of 25
    assert result["spikes"] == 24  # 8 burst hops x 3 channels
    assert 9.9 < result["device_hop_ms"] < 10.1


def test_probe_reports_gaps_rejections_and_resets():
    gap = probe(ReplaySource(scenario("gap", 200)), 5.0)
    assert gap["gaps"] == [{"first_seq": 80, "missing": 10, "cause": "tx_drop"}]
    corrupt = probe(ReplaySource(scenario("corrupt", 200)), 5.0)
    assert corrupt["rejected"] == {"BAD_CRC": 4}
    assert probe(ReplaySource(scenario("reset", 100)), 5.0)["boots"] == 2
    late = probe(ReplaySource(scenario("late", 200)), 5.0)
    assert late["anomalies"]["MERGED"] == 1 and late["gaps"] == []


class Silent:
    """Source that never delivers bytes and records commands; a fake clock advances on every read."""

    def __init__(self):
        self.now, self.written = 0.0, []

    def clock(self):
        return self.now

    def read(self, max_bytes, timeout_s):
        self.now += 0.5
        return b""

    def write(self, data):
        self.written.append((self.now, data))

    def close(self):
        pass


def test_scheduled_commands_are_sent_once_each_in_time_order():
    src = Silent()
    result = probe(src, 5.0, sends=[(2.0, b"T"), (1.0, b"G")], clock=src.clock)
    assert [d for _, d in src.written] == [b"G", b"T"]
    assert all(t >= due for (t, _), due in zip(src.written, (1.0, 2.0)))
    assert result["frames"] == 0 and result["stalls"] >= 1  # a silent port is reported, not hidden


def test_serial_source_can_send_commands_to_the_device():
    master, slave = os.openpty()
    try:
        src = SerialPortSource(os.ttyname(slave))
        src.write(b"I")
        assert os.read(master, 8) == b"I"
        src.close()
    finally:
        os.close(master)
        os.close(slave)


def test_probe_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.probe"], check=True)
