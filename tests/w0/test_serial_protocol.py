import subprocess
import sys
from itertools import count

import pytest

from rpi_agents.agent.serial_protocol import (
    BootEvent,
    BootLine,
    FrameEvent,
    FrameLine,
    GapEvent,
    LineAssembler,
    LineOverflow,
    ProtocolError,
    RejectedEvent,
    StreamTracker,
    crc8,
    parse_line,
)

# Vectors sent to the firmware owner; they must match bit for bit.
BOOT = b"$B,1,3F2A91C0,19231,192,7,6000,swap*BA"
VECTORS = [
    BOOT,
    b"$F,0,1234567,192,00,1,0*AF",
    b"$F,600,6001234,192,0A,0,0*FA",
    b"$F,602,6021000,384,00,0,0*9C",
    b"$F,4294967295,4294967295,192,7F,0,65535*43",
]
HOP_US = 192 * 1_000_000 // 19231  # 9984


def line(body: str) -> bytes:
    data = body.encode()
    return b"$" + data + b"*" + b"%02X" % crc8(data)


def frame(seq, t_us=None, n=192, mask=0, flags=0, txdrop=0) -> bytes:
    t_us = seq * HOP_US if t_us is None else t_us
    return line(f"F,{seq},{t_us},{n},{mask:02X},{flags:X},{txdrop}")


def tracker() -> StreamTracker:
    ids = count(1)
    return StreamTracker(lambda: f"boot-{next(ids)}")


def feed(t: StreamTracker, *lines: bytes):
    return [event for raw in lines for event in t.feed_line(raw)]


def test_crc8_check_value_and_vectors():
    assert crc8(b"123456789") == 0xF4
    for vector in VECTORS:
        assert parse_line(vector) is not None


def test_parse_boot_and_frame_fields():
    assert parse_line(BOOT) == BootLine(1, "3F2A91C0", 19231, 192, 7, 6000, "swap")
    assert parse_line(b"$F,600,6001234,192,0A,0,0*FA") == FrameLine(600, 6001234, 192, 0x0A, 0, 0)
    assert parse_line(b"$F,0,1234567,192,00,1,0*AF").priming
    assert parse_line(b"$F,600,6001234,192,0A,0,0*FA\r") is not None  # println sends CRLF


@pytest.mark.parametrize("raw", [b"", b"# encoder_v2 dt=10ms", b"\r"])
def test_comments_and_blank_lines_are_ignored(raw):
    assert parse_line(raw) is None


@pytest.mark.parametrize(
    "raw,code",
    [
        (b"$F,600,6001234,192,0A,0,0*FB", "BAD_CRC"),
        (b"$F,601,6001234,192,0A,0,0*FA", "BAD_CRC"),
        (b"$F,600,6001234,192,0A,0*FA", "BAD_FORMAT"),
        (line("F,600,6001234,192,0A,0,0,9"), "BAD_FORMAT"),
        (line("F,600,6001234,192,0a,0,0"), "BAD_FORMAT"),
        (line("F,0600,6001234,192,0A,0,0"), "BAD_FORMAT"),
        (line("F,-1,6001234,192,0A,0,0"), "BAD_FORMAT"),
        (line("F,٩,6001234,192,0A,0,0"), "BAD_FORMAT"),
        (line("F,4294967296,0,192,00,0,0"), "BAD_RANGE"),
        (line("F,1,4294967296,192,00,0,0"), "BAD_RANGE"),
        (line("F,1,1,0,00,0,0"), "BAD_RANGE"),
        (line("F,1,1,192,00,0,65536"), "BAD_RANGE"),
        (line("B,2,3F2A91C0,19231,192,7,6000,swap"), "BAD_RANGE"),
        (line("B,1,3F2A91C0,19231,192,9,6000,swap"), "BAD_RANGE"),
        (line("B,1,3F2A91C0,19231,192,7,6000,swap") + b"x" * 20, "TOO_LONG"),
        (b"frame,s0,s1", "NOT_PROTOCOL"),
        (b"$X,1*00", "BAD_FORMAT"),
    ],
)
def test_reject_untrusted_lines(raw, code):
    with pytest.raises(ProtocolError) as error:
        parse_line(raw)
    assert error.value.code == code


def test_assembler_splits_chunks_and_bounds_memory():
    a = LineAssembler(limit=16)
    out = list(a.feed(b"$F,1,2,3"))
    out += a.feed(b",4,5,6*00\r\n$B,")
    assert out == [LineOverflow()]  # over the 16-byte bound: dropped, buffer never grew
    assert list(a.feed(b"x\nshort\r\n")) == [b"$B,x", b"short\r"]


def test_assembler_recovers_after_overflow():
    a = LineAssembler(limit=8)
    events = list(a.feed(b"0123456789ABCDEF\nok\n"))
    assert events == [LineOverflow(), b"ok"]


def test_continuous_stream_has_no_gap_and_unwraps_time():
    t = tracker()
    events = feed(t, BOOT, frame(0), frame(1), frame(2))
    assert isinstance(events[0], BootEvent)
    frames = events[1:]
    assert all(isinstance(e, FrameEvent) and not e.anomalies for e in frames)
    assert [e.source_us for e in frames] == [0, HOP_US, 2 * HOP_US]
    assert {e.boot_id for e in frames} == {"boot-1"}


def test_lost_lines_become_an_explicit_gap():
    events = feed(tracker(), BOOT, frame(0), frame(1), frame(5))
    gap = next(e for e in events if isinstance(e, GapEvent))
    assert (gap.first_seq, gap.missing_hops, gap.cause) == (2, 3, "lost_or_corrupt")
    assert events.index(gap) == len(events) - 2  # reported before the frame that reveals it


def test_uno_reported_tx_drop_explains_the_gap():
    events = feed(tracker(), BOOT, frame(0), frame(3, txdrop=2))
    gap = next(e for e in events if isinstance(e, GapEvent))
    assert (gap.missing_hops, gap.txdrop_delta, gap.cause) == (2, 2, "tx_drop")
    assert "TXDROP" in events[-1].anomalies


def test_corrupt_line_is_rejected_then_shows_up_as_gap():
    bad = frame(1)[:-2] + b"00"
    events = feed(tracker(), BOOT, frame(0), bad, frame(2))
    assert [type(e).__name__ for e in events[1:]] == ["FrameEvent", "RejectedEvent", "GapEvent", "FrameEvent"]
    assert events[-2].missing_hops == 1


def test_late_merged_frame_is_flagged_not_hidden():
    events = feed(tracker(), BOOT, frame(0), frame(2, n=384))
    assert not any(isinstance(e, GapEvent) for e in events)
    assert events[-1].covered_hops == 2 and "MERGED" in events[-1].anomalies


def test_merged_frame_with_extra_missing_hops_still_gaps():
    events = feed(tracker(), BOOT, frame(0), frame(5, n=384))
    gap = next(e for e in events if isinstance(e, GapEvent))
    assert (gap.first_seq, gap.missing_hops) == (1, 3)


def test_boot_line_starts_a_new_boot_id_and_seq_restarts():
    t = tracker()
    events = feed(t, BOOT, frame(0), frame(1), BOOT, frame(0), frame(1))
    assert [e.boot_id for e in events if isinstance(e, FrameEvent)] == ["boot-1"] * 2 + ["boot-2"] * 2
    assert not any(isinstance(e, (GapEvent, RejectedEvent)) for e in events)


def test_frames_before_boot_are_rejected():
    events = feed(tracker(), frame(7))
    assert [e.code for e in events] == ["NO_BOOT"]


def test_seq_regression_without_boot_forces_resync():
    t = tracker()
    events = feed(t, BOOT, frame(10), frame(11), frame(0), frame(1))
    codes = [e.code for e in events if isinstance(e, RejectedEvent)]
    assert codes == ["SEQ_REGRESSION", "NO_BOOT"] and not t.synced
    assert isinstance(feed(t, BOOT)[0], BootEvent) and t.synced


def test_duplicate_seq_is_ignored():
    events = feed(tracker(), BOOT, frame(0), frame(0), frame(1))
    assert [e.code for e in events if isinstance(e, RejectedEvent)] == ["SEQ_DUPLICATE"]
    assert sum(isinstance(e, FrameEvent) for e in events) == 2


def test_mask_bit_above_channel_count_is_rejected():
    events = feed(tracker(), BOOT, frame(0, mask=0x80))
    assert isinstance(events[-1], RejectedEvent) and events[-1].code == "BAD_RANGE"


def test_priming_frames_are_carried_and_marked():
    events = feed(tracker(), BOOT, frame(0, flags=1), frame(1, flags=1), frame(2))
    assert [e.priming for e in events[1:]] == [True, True, False]


def test_micros_wraparound_keeps_source_time_monotonic():
    wrap = 2**32
    t = tracker()
    events = feed(t, BOOT, frame(0, t_us=wrap - HOP_US), frame(1, t_us=0), frame(2, t_us=HOP_US))
    times = [e.source_us for e in events if isinstance(e, FrameEvent)]
    assert times == [wrap - HOP_US, wrap, wrap + HOP_US]


def test_clock_skew_between_seq_and_time_is_reported():
    events = feed(tracker(), BOOT, frame(0), frame(1, t_us=HOP_US + 5000))
    assert "TIME_SKEW" in events[-1].anomalies


def test_module_imports_without_site_packages():
    subprocess.run([sys.executable, "-S", "-c", "import rpi_agents.agent.serial_protocol"], check=True)
