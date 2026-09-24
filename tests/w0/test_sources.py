import os
import subprocess
import sys
from itertools import count

import pytest

from rpi_agents.agent.batching import BatchAssembler
from rpi_agents.agent.serial_protocol import (
    BootEvent,
    FrameEvent,
    GapEvent,
    RejectedEvent,
    StreamTracker,
)
from rpi_agents.agent.sources import (
    ReplaySource,
    SerialPortSource,
    SourceDisconnected,
    SourceEnded,
    StallEvent,
    TickEvent,
    event_stream,
)
from rpi_agents.agent.synthetic import scenario

CHANNELS = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]


def tracker():
    ids = count(1)
    return StreamTracker(lambda: f"boot-{next(ids)}")


def events(data: bytes, chunk: int = 64):
    return list(event_stream(ReplaySource(data, chunk), tracker(), poll_s=0.01, stall_s=1e9))


def of(kind, evs):
    return [e for e in evs if isinstance(e, kind)]


def test_replay_source_serves_everything_then_ends():
    src = ReplaySource(b"abcdefg", chunk=3)
    assert [src.read(100, 0) for _ in range(3)] == [b"abc", b"def", b"g"]
    with pytest.raises(SourceEnded):
        src.read(100, 0)
    with pytest.raises(ValueError):
        ReplaySource(b"x", chunk=0)


def test_silence_has_priming_then_no_spikes_and_no_gaps():
    evs = events(scenario("silence", 200))
    frames = of(FrameEvent, evs)
    assert len(of(BootEvent, evs)) == 1 and len(frames) == 200
    assert [f.priming for f in frames].count(True) == 50 and not any(f.mask for f in frames)
    assert not of(GapEvent, evs) and not of(RejectedEvent, evs)


def test_glass_bursts_appear_only_after_priming():
    frames = of(FrameEvent, events(scenario("glass", 200)))
    assert [f.seq for f in frames if f.mask] == list(range(100, 108))
    assert all(f.mask == 0b1110000 and not f.priming for f in frames if f.mask)


def test_gap_scenarios_distinguish_uno_tx_drop_from_silent_loss():
    (gap,) = of(GapEvent, events(scenario("gap", 200)))
    assert (gap.first_seq, gap.missing_hops, gap.txdrop_delta, gap.cause) == (80, 10, 10, "tx_drop")
    (gap,) = of(GapEvent, events(scenario("lost", 200)))
    assert (gap.first_seq, gap.missing_hops, gap.txdrop_delta, gap.cause) == (80, 10, 0, "lost_or_corrupt")


def test_late_frame_is_merged_not_a_gap():
    evs = events(scenario("late", 200))
    (merged,) = [f for f in of(FrameEvent, evs) if f.covered_hops > 1]
    assert merged.seq == 100 and "MERGED" in merged.anomalies and not of(GapEvent, evs)


def test_corrupt_lines_are_rejected_and_show_as_gaps():
    evs = events(scenario("corrupt", 200))
    assert [r.code for r in of(RejectedEvent, evs)] == ["BAD_CRC"] * 4
    assert all("b'$F," in r.detail for r in of(RejectedEvent, evs))  # the log says which line was bad
    assert [(g.first_seq, g.missing_hops) for g in of(GapEvent, evs)] == [(49, 1), (99, 1), (149, 1)]


def test_reset_scenario_starts_a_new_boot_without_seq_errors():
    evs = events(scenario("reset", 100))
    assert [b.boot_id for b in of(BootEvent, evs)] == ["boot-1", "boot-2"]
    assert not of(RejectedEvent, evs) and len(of(FrameEvent, evs)) == 200


@pytest.mark.parametrize("name", ["silence", "glass", "gap", "late", "corrupt", "reset"])
def test_chunking_and_crlf_do_not_change_the_events(name):
    def same(evs):  # a rejected line carries a free-text preview of the raw bytes, which differs by line ending
        return [RejectedEvent(e.code, "") if isinstance(e, RejectedEvent) else e for e in evs]

    data = scenario(name, 120)
    baseline = same(events(data, chunk=4096))
    assert same(events(data, chunk=1)) == baseline and same(events(data, chunk=7)) == baseline
    assert same(events(scenario(name, 120, eol=b"\r\n"), chunk=5)) == baseline


def test_unknown_scenario_is_refused():
    with pytest.raises(ValueError):
        scenario("nope")


def test_overlong_line_is_reported_and_the_stream_recovers():
    data = b"#" + b"x" * 200 + b"\n" + scenario("silence", 5)
    evs = events(data, chunk=16)
    assert [r.code for r in of(RejectedEvent, evs)] == ["TOO_LONG"] and len(of(FrameEvent, evs)) == 5


def test_synthetic_glass_becomes_spikes_in_the_expected_batch():
    tr, asm, drafts = tracker(), None, []
    for ev in event_stream(ReplaySource(scenario("glass", 200)), tr, poll_s=0.01, stall_s=1e9):
        if isinstance(ev, BootEvent):
            asm = BatchAssembler(boot_id=ev.boot_id, channels=CHANNELS, fs_hz=19231, hop=192)
        elif asm is not None:
            drafts += asm.feed(ev)
    drafts += asm.flush()
    with_spikes = [d for d in drafts if d.spikes]
    assert len(with_spikes) == 1 and len(with_spikes[0].spikes) == 8 * 3
    assert {c for _, c in with_spikes[0].spikes} == {"flux", "hf_lo", "hf_hi"}


class Scripted:
    """Source that returns scripted chunks and advances a fake clock on every read."""

    def __init__(self, chunks, clock):
        self.chunks, self.clock = list(chunks), clock

    def read(self, max_bytes, timeout_s):
        self.clock.now += 0.5
        if not self.chunks:
            raise SourceEnded
        return self.chunks.pop(0)

    def close(self):
        pass


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


def test_silent_port_is_a_visible_stall_reported_once_per_silence():
    clock = Clock()
    boot, first = scenario("silence", 1).split(b"\n", 1)[0] + b"\n", scenario("silence", 3)
    src = Scripted([boot, b"", b"", b"", first, b"", b"", b"", b""], clock)
    stream = list(event_stream(src, tracker(), poll_s=0.5, stall_s=1.0, clock=clock))
    kinds = [type(e).__name__ for e in stream]
    assert kinds.count("StallEvent") == 2 and isinstance(stream[-1], StallEvent)
    assert kinds.index("StallEvent") < kinds.index("FrameEvent")  # first silence before any frame


def test_idle_polls_yield_ticks_only_when_asked_for():
    clock = Clock()
    src = Scripted([b"", b"", b""], clock)
    kinds = [type(e) for e in event_stream(src, tracker(), poll_s=0.5, stall_s=1e9, ticks=True, clock=clock)]
    assert kinds == [TickEvent] * 3
    src = Scripted([b"", b""], Clock())
    assert list(event_stream(src, tracker(), poll_s=0.5, stall_s=1e9, clock=src.clock)) == []


def test_a_boot_line_restarts_the_stall_timer():
    clock = Clock()
    boot = scenario("silence", 1).split(b"\n", 1)[0] + b"\n"
    src = Scripted([b"", b"", b"", boot, b"", b"", b""], clock)
    stream = list(event_stream(src, tracker(), poll_s=0.5, stall_s=1.0, clock=clock))
    kinds = [type(e).__name__ for e in stream]
    assert kinds == ["StallEvent", "BootEvent", "StallEvent"]  # silence before AND after the boot


def test_serial_port_source_reads_a_pseudo_terminal_and_reports_disconnect():
    master, slave = os.openpty()
    try:
        src = SerialPortSource(os.ttyname(slave), 115200)
        assert src.read(64, 0.05) == b""
        os.write(master, b"$F,1*00\n")
        assert src.read(64, 1.0) == b"$F,1*00\n"
        os.close(master)
        master = None
        with pytest.raises(SourceDisconnected):
            for _ in range(5):
                src.read(64, 0.2)
        src.close()
    finally:
        if master is not None:
            os.close(master)
        os.close(slave)


def test_serial_port_source_refuses_missing_device_and_bad_baud():
    with pytest.raises(SourceDisconnected, match="cannot open"):
        SerialPortSource("/nonexistent/tty")
    with pytest.raises(ValueError, match="baud"):
        SerialPortSource("/nonexistent/tty", 12345)


def test_modules_import_without_site_packages():
    code = "import rpi_agents.agent.sources, rpi_agents.agent.synthetic"
    subprocess.run([sys.executable, "-S", "-c", code], check=True)
