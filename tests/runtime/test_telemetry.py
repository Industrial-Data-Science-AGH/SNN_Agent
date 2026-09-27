"""Telemetry: what a viewer is allowed to believe about a frame (task P3).

The acceptance criterion for P3 is that the dashboard sees a faithful raster and
potential on a known test vector, and that pausing the view does not stop the
backend. Both are properties of the frames, so they are asserted here: a frame
validates against the contract *against the loaded manifest*, reading frames
never moves the simulation, and a hole in the input is never rendered as quiet.
"""

from __future__ import annotations

import pytest

from contracts.validation import ContractError, fixture, validate
from snn_runtime import LuiRuntime, RuntimeStateError
from snn_runtime.telemetry import TelemetryFeed


def started(manifest, **kwargs):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(manifest)
    runtime.reset(epoch=1, source_time_us=0, device_id="dev1", session_id="sess1", **kwargs)
    return runtime


# ------------------------------------------------------------------- the frame


def test_a_frame_is_a_contract_frame_for_the_loaded_model(lui8, stream, make_batch):
    """Not merely schema-valid: valid *against this manifest*.

    ``validate`` with a manifest is what checks the three things a viewer would
    otherwise have to trust: same model hash, same topology version, every
    neuron present. That is the whole point of sending them in the frame.
    """
    runtime = started(lui8)
    runtime.step(make_batch(stream, 0, 20))
    frame = runtime.snapshot()

    validate("NeuronFrame", frame, manifest=lui8)
    assert [n["neuron_id"] for n in frame["neurons"]] == [
        n["neuron_id"] for n in lui8["topology"]["neurons"]
    ]
    assert frame["potential_unit"] == lui8["runtime"]["potential_unit"]
    assert frame["source_time_us"] == runtime.source_time_us


def test_the_potential_is_the_peak_the_board_reached(lui8, stream, make_batch):
    """A neuron that fired must not report the value it was reset to.

    Reading the membrane after the reset would tell a viewer the LED is dark at
    the exact instant the board flashed, which is the one moment that matters.
    """
    runtime = started(lui8)
    seen = False
    for first in range(0, 300, 10):
        runtime.step(make_batch(stream, first, 10, seq=first // 10))
        frame = runtime.snapshot()
        for neuron in frame["neurons"]:
            if neuron["spiked"]:
                seen = True
                assert neuron["v_mem"] >= neuron["v_threshold"]
    assert seen, "the test stream never made any board fire"


def test_reading_telemetry_does_not_move_the_simulation(lui8, stream, make_batch):
    runtime = started(lui8)
    runtime.step(make_batch(stream, 0, 20))

    first, second = runtime.snapshot(), runtime.snapshot()
    assert first["source_time_us"] == second["source_time_us"]
    assert [n["v_mem"] for n in first["neurons"]] == [n["v_mem"] for n in second["neurons"]]
    # Only the counter moves, so a client can tell two readings apart.
    assert second["frame_seq"] == first["frame_seq"] + 1


def test_pausing_the_view_leaves_the_stream_running(lui8, stream, make_batch, grid):
    """Nobody has to call snapshot() for the decisions to keep coming."""
    runtime = started(lui8)
    for first in range(0, 100, 10):
        runtime.step(make_batch(stream, first, 10, seq=first // 10))
    assert runtime.source_time_us == grid(100)
    assert runtime.snapshot()["frame_seq"] == 0  # first frame anyone asked for


# ------------------------------------------------------------------ the status


def test_before_any_input_the_session_says_warmup(lui8):
    """Rest is an assumption until a batch confirms it, and warmup says so."""
    runtime = started(lui8)
    assert runtime.snapshot()["status"] == "warmup"


def test_an_uninterrupted_stream_reaches_running(lui8, stream, make_batch):
    runtime = started(lui8)
    for first in range(0, 60, 10):
        runtime.step(make_batch(stream, first, 10, seq=first // 10))
    statuses = {runtime.snapshot()["status"] for _ in range(3)}
    assert statuses == {"running"}


def test_a_hole_in_the_input_is_reported_once_and_not_as_silence(lui8, stream, make_batch):
    runtime = started(lui8)
    for first in range(0, 60, 10):
        runtime.step(make_batch(stream, first, 10, seq=first // 10))
    assert runtime.snapshot()["status"] == "running"

    # Frames 60..90 are produced by the device and never arrive; the next batch
    # starts at 90, which is a real hole in the audio and not quiet audio.
    runtime.step(make_batch(stream, 90, 10, seq=9))

    # Two separate snapshots on purpose: the first one to look after the hole
    # carries the announcement, and taking it consumes the flag. Reading these
    # as one contradictory assertion is an easy mistake, hence the names.
    first_after_the_hole = runtime.snapshot()
    the_one_after_that = runtime.snapshot()
    assert first_after_the_hole["status"] == "gap"
    assert the_one_after_that["status"] != "gap"


def test_a_stopped_session_says_stopped(lui8, stream, make_batch):
    runtime = started(lui8)
    runtime.step(make_batch(stream, 0, 10))
    runtime.stop()
    assert runtime.snapshot()["status"] == "stopped"


# -------------------------------------------------------------- what it claims


def test_an_uncalibrated_model_never_claims_a_measurement(lui8, stream, make_batch):
    runtime = started(lui8)
    runtime.step(make_batch(stream, 0, 10))
    frame = runtime.snapshot()
    assert frame["provenance"] == "simulated"
    assert frame["potential_unit"] == "a.u."


def test_a_scripted_package_reports_no_potential_at_all(stream, make_batch):
    """``v_mem: null`` is the contract's way of saying "no reading"."""
    runtime = started(fixture("model-manifest"))
    frame = runtime.snapshot()
    assert frame["provenance"] == "demo"
    assert all(n["v_mem"] is None for n in frame["neurons"])
    validate("NeuronFrame", frame)


# ------------------------------------------------------------------- identity


def test_a_frame_without_a_stream_identity_is_refused(lui8):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(lui8)
    runtime.reset(epoch=1, source_time_us=0)
    with pytest.raises(RuntimeStateError) as err:
        runtime.snapshot()
    assert err.value.code == "NO_STREAM_IDENTITY"


def test_the_first_batch_names_the_stream(lui8, stream, make_batch):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(lui8)
    runtime.reset(epoch=1, source_time_us=0)
    runtime.step(make_batch(stream, 0, 10))
    frame = runtime.snapshot()
    assert (frame["device_id"], frame["session_id"]) == ("dev1", "sess1")


def test_a_batch_from_another_session_is_refused(lui8, stream, make_batch):
    runtime = started(lui8)
    with pytest.raises(RuntimeStateError) as err:
        runtime.step({**make_batch(stream, 0, 10), "session_id": "somebody-elses"})
    assert err.value.code == "SESSION_MISMATCH"


# ----------------------------------------------------------------- the feed


def test_the_feed_thins_on_source_time_not_on_frame_count(lui8, stream, make_batch):
    """100 frames a second is for the integrator; a chart wants a handful."""
    dt = lui8["runtime"]["dt_us"]
    runtime = started(lui8)
    feed = TelemetryFeed(min_interval_us=10 * dt)

    sent = []
    for first in range(0, 100, 5):
        runtime.step(make_batch(stream, first, 5, seq=first // 5))
        frame = feed.offer(runtime.snapshot())
        if frame is not None:
            sent.append(frame)

    assert 1 < len(sent) < 20
    gaps = [b["source_time_us"] - a["source_time_us"] for a, b in zip(sent, sent[1:])]
    assert all(gap >= 10 * dt for gap in gaps)
    assert feed.skipped > 0


def test_the_feed_never_thins_away_a_change_of_status(lui8):
    """A gap that fell between two sampling points would be a lie."""
    feed = TelemetryFeed(min_interval_us=1_000_000)
    base = {"status": "running", "source_time_us": 0}
    assert feed.offer(base) is not None
    assert feed.offer({"status": "running", "source_time_us": 10_000}) is None
    passed = feed.offer({"status": "gap", "source_time_us": 20_000})
    assert passed is not None and passed["status"] == "gap"


def test_the_feed_can_be_told_to_send_everything(lui8):
    feed = TelemetryFeed(min_interval_us=0)
    for index in range(5):
        assert feed.offer({"status": "running", "source_time_us": index * 10}) is not None
    assert feed.skipped == 0


def test_a_negative_interval_is_refused():
    with pytest.raises(ValueError):
        TelemetryFeed(min_interval_us=-1)


def test_an_unknown_status_never_reaches_a_viewer(lui8, stream, make_batch):
    from snn_runtime.telemetry import build_frame

    with pytest.raises(ValueError):
        build_frame(
            device_id="d", session_id="s", epoch=1, source_time_us=0, model_hash="sha256:" + "0" * 64,
            frame_seq=0, topology_version="v1", status="paused", provenance="simulated",
            potential_unit="a.u.", neurons=[], membrane=[], spiked=[],
        )


def test_the_fixture_a_dashboard_developed_against_is_still_the_shape_we_send():
    """Karolina's demo frames come from this fixture; it has to stay in step."""
    demo = fixture("neuron-frame")
    validate("NeuronFrame", demo)
    with pytest.raises(ContractError):
        validate("NeuronFrame", {**demo, "neurons": demo["neurons"] * 2})


# ------------------------------------------------------------ the demo replay


def test_the_demo_replay_tool_emits_frames_the_contract_accepts(lui8):
    """Karolina's demo data comes out of the real runtime, not out of a guess."""
    from snn_runtime.tools.make_demo_frames import build

    frames = build(lui8, frames=100, every_us=50_000, density=0.25, seed=1)
    assert frames and all(f["provenance"] == "simulated" for f in frames)
    assert [f["frame_seq"] for f in frames] == sorted(f["frame_seq"] for f in frames)
    for frame in frames:
        validate("NeuronFrame", frame, manifest=lui8)


def test_inhibition_can_drive_a_potential_below_reset(lui8):
    """The LED formula has to clip: v_mem is not bounded by v_reset."""
    from snn_runtime.tools.make_demo_frames import build

    values = [n["v_mem"] for f in build(lui8, frames=200, every_us=50_000, density=0.25, seed=1) for n in f["neurons"]]
    assert min(values) < 0


def test_frame_seq_counts_frames_sent_not_frames_taken(lui8, stream, make_batch):
    """TELEMETRY.md promises a continuous sequence, so thinning cannot hole it.

    The snapshot counter cannot serve that promise: the feed drops frames, so
    reusing it would hand the viewer gaps in `frame_seq` that mean nothing.
    A hole there is how a viewer detects it lost frames, so it has to be real.
    """
    from snn_runtime.telemetry import TelemetryFeed

    runtime = started(lui8)
    feed = TelemetryFeed(min_interval_us=250_000)

    sent = []
    for first in range(0, 200, 5):
        runtime.step(make_batch(stream, first, 5, seq=first // 5))
        frame = feed.offer(runtime.snapshot())
        if frame is not None:
            sent.append(frame)

    assert feed.skipped > 0, "nothing was thinned, so the test proves nothing"
    assert [f["frame_seq"] for f in sent] == list(range(len(sent)))
    assert feed.sent == len(sent)
