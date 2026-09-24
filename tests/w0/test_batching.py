import re

import pytest
from fastapi.testclient import TestClient

from contracts.validation import fixture, validate
from rpi_agents.agent.batching import BatchAssembler, request_id, to_spike_batch
from rpi_agents.agent.serial_protocol import BootEvent, FrameEvent, RejectedEvent, StreamTracker, crc8
from rpi_agents.cloud.app.mock_api import create_app

FS, HOP = 19231, 192
MANIFEST = fixture("model-manifest")
CHANNELS = [c["channel"] for c in MANIFEST["encoder_profile"]["channel_map"]]
ENCODER_HASH = MANIFEST["encoder_hash"]
BOOT_LINE = b"$B,1,3F2A91C0,19231,192,7,6000,swap*BA"
HOP_US = 9984  # nominal, close enough for synthetic streams; the assembler uses the hop grid


def g(hop_index: int) -> int:
    """Hop grid position in us relative to the origin (same integer rule as the assembler)."""
    return (hop_index * HOP * 1_000_000 + FS // 2) // FS


def frame(seq, *, mask=0, n=192, flags=0, t_us=None, txdrop=0) -> bytes:
    t_us = seq * HOP_US if t_us is None else t_us
    body = f"F,{seq},{t_us},{n},{mask:02X},{flags:X},{txdrop}".encode()
    return b"$" + body + b"*" + b"%02X" % crc8(body)


def assembler(boot_id="demo-boot", **kw) -> BatchAssembler:
    return BatchAssembler(boot_id=boot_id, channels=CHANNELS, fs_hz=FS, hop=HOP, **kw)


def run(*lines: bytes, boot_id="demo-boot", flush=True, **kw):
    tracker = StreamTracker(lambda: boot_id)
    asm = assembler(boot_id, **kw)
    drafts = []
    for raw in lines:
        for event in tracker.feed_line(raw):
            if not isinstance(event, BootEvent):
                drafts += asm.feed(event)
    return (drafts + asm.flush() if flush else drafts), asm


def payloads(drafts):
    return [
        to_spike_batch(d, device_id="demo-pi", session_id="s1", epoch=1, encoder_hash=ENCODER_HASH)
        for d in drafts
    ]


def test_silence_tiles_the_timeline_and_matches_the_contract():
    drafts, asm = run(BOOT_LINE, *[frame(i) for i in range(60)])
    assert [(d.batch_seq, d.source_start_us, d.source_end_us) for d in drafts] == [
        (0, g(0), g(25)),
        (1, g(25), g(50)),
        (2, g(50), g(60)),  # flush closes the last partial batch
    ]
    assert asm.stream_start_us == drafts[0].source_start_us == 0
    assert all(not d.spikes and d.dropped_events == 0 for d in drafts)
    for payload in payloads(drafts):
        validate("SpikeBatch", payload, manifest=MANIFEST)
    assert asm.stats.frames == 60 and asm.stats.batches == 3 and asm.stats.gap_hops == 0


def test_spikes_sit_at_the_hop_start_with_batch_relative_offsets():
    drafts, _ = run(BOOT_LINE, *[frame(i, mask={7: 0x08, 30: 0x21}.get(i, 0)) for i in range(40)])
    assert drafts[0].spikes == ((g(7) - g(0), "zcr"),)
    assert drafts[1].spikes == ((g(30) - g(25), "peak"), (g(30) - g(25), "hf_lo"))
    for payload in payloads(drafts):
        validate("SpikeBatch", payload, manifest=MANIFEST)


def test_serial_gap_becomes_a_time_hole_with_consecutive_batch_seq():
    lines = [BOOT_LINE, *[frame(i) for i in range(10)], *[frame(i) for i in range(20, 30)]]
    drafts, asm = run(*lines)
    assert [d.batch_seq for d in drafts] == [0, 1]
    assert (drafts[0].source_start_us, drafts[0].source_end_us) == (g(0), g(10))  # closed at the hole
    assert drafts[1].source_start_us == g(20) > drafts[0].source_end_us
    assert asm.stats.gap_hops == 10


def test_merged_frame_is_reported_as_degraded_not_hidden():
    drafts, asm = run(BOOT_LINE, *[frame(i) for i in range(10)], frame(11, n=384, mask=0x02))
    assert [(d.source_start_us, d.source_end_us) for d in drafts] == [(g(0), g(12))]  # hop 10 is not a hole
    assert drafts[0].dropped_events == 1
    assert drafts[0].spikes == ((g(11), "peak_cnt"),)
    assert asm.stats.merged_frames == 1 and asm.stats.dropped_hops == 1


def test_priming_hops_are_observed_silence():
    drafts, asm = run(BOOT_LINE, *[frame(i, flags=1) for i in range(30)])
    assert len(drafts) == 2 and not any(d.spikes for d in drafts)
    assert asm.stats.priming_frames == 30


def test_timeline_is_anchored_to_uno_micros_across_rollover():
    wrap = 2**32
    start = wrap - 300_000
    lines = [BOOT_LINE, *[frame(i, t_us=(start + i * HOP_US) & (wrap - 1)) for i in range(60)]]
    drafts, asm = run(*lines)
    assert asm.stream_start_us == pytest.approx(start - HOP_US, abs=HOP_US)
    assert drafts[-1].source_end_us > wrap  # monotonic through the 32-bit rollover
    assert all(a.source_end_us == b.source_start_us for a, b in zip(drafts, drafts[1:]))


def test_wrong_inputs_are_refused_loudly():
    tracker = StreamTracker(lambda: "demo-boot")
    boot = tracker.feed_line(BOOT_LINE)[0]
    good = tracker.feed_line(frame(0))[0]
    asm = assembler()
    with pytest.raises(ValueError, match="new session"):
        asm.feed(boot)
    with pytest.raises(ValueError, match="assembler is for"):
        asm.feed(FrameEvent("other", 0, 0, 192, 0, False, 1, 0))
    with pytest.raises(ValueError, match="beyond 7 channels"):
        asm.feed(FrameEvent("demo-boot", 0, 0, 192, 0x80, False, 1, 0))
    assert asm.feed(RejectedEvent("BAD_CRC", "x")) == []
    asm.feed(good)
    assert asm.feed(good) == []  # an already covered frame adds nothing
    assert asm.stats.frames == 1


@pytest.mark.parametrize("kwargs", [{"batch_us": 2_000_000}, {"boot_id": "bad id"}])
def test_constructor_validation(kwargs):
    with pytest.raises(ValueError):
        assembler(**kwargs)


def test_request_id_is_stable_unique_and_within_the_contract_pattern():
    ids = {request_id("s1", 0), request_id("s1", 1), request_id("s2", 0)}
    assert len(ids) == 3 and request_id("s1", 0) == request_id("s1", 0)
    long_id = request_id("x" * 64, 12345)
    assert len(long_id) <= 64 and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", long_id)


@pytest.fixture
def client():
    with TestClient(create_app(), base_url="http://127.0.0.1") as c:
        yield c


def post(client, path, body):
    return client.post(path, json=body, headers={"Idempotency-Key": body["request_id"]})


def open_session(client, asm):
    body = fixture("session-create") | {"source_start_us": asm.stream_start_us}
    r = post(client, "/v1/sessions?scenario=silence", body)
    assert r.status_code == 201, r.text
    return r.json()


def send_all(client, lines):
    drafts, asm = run(*lines)
    s = open_session(client, asm)
    acks = []
    for draft in drafts:
        body = to_spike_batch(
            draft, device_id=s["device_id"], session_id=s["session_id"], epoch=s["epoch"],
            encoder_hash=ENCODER_HASH,
        )  # fmt: skip
        r = post(client, f"/v1/sessions/{s['session_id']}/batches", body)
        assert r.status_code == 200, r.text
        validate("BatchAck", r.json())
        acks.append(r.json())
    return drafts, acks


def test_backend_accepts_a_clean_stream_without_gaps(client):
    _, acks = send_all(client, [BOOT_LINE, *[frame(i) for i in range(60)]])
    assert [a["status"] for a in acks] == ["running"] * 3
    assert all(a["gaps"] == [] for a in acks)


def test_backend_reports_a_serial_hole_as_missing_batch(client):
    lines = [BOOT_LINE, *[frame(i) for i in range(10)], *[frame(i) for i in range(20, 30)]]
    drafts, acks = send_all(client, lines)
    assert acks[0]["gaps"] == []
    (gap,) = acks[1]["gaps"]
    assert (gap["reason"], gap["source_start_us"], gap["source_end_us"]) == (
        "missing_batch", drafts[0].source_end_us, drafts[1].source_start_us,
    )  # fmt: skip
    assert acks[1]["status"] == "gap"


def test_backend_reports_a_merged_frame_as_dropped_events(client):
    _, acks = send_all(client, [BOOT_LINE, *[frame(i) for i in range(10)], frame(11, n=384)])
    assert [g_["reason"] for g_ in acks[0]["gaps"]] == ["dropped_events"]
