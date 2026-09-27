"""P2 acceptance: one stream, many batch sizes, one answer.

"Ten sam strumien podzielony na rozne batch sizes daje zgodne spike'y i decyzje
w ustalonej tolerancji; luka generuje gap/warmup." The tolerance here is zero:
the runtime carries exact state across batch boundaries, so cutting the same
stream differently is not an approximation, it is the same computation. A test
that allowed a tolerance would be hiding a bug.
"""

from __future__ import annotations

import pytest

from contracts.validation import fixture
from snn_runtime import LuiRuntime, RuntimeStateError

BATCH_SIZES = (1, 2, 3, 7, 25, 200, 400)


def session(manifest, *, epoch: int = 1, origin_us: int = 0) -> LuiRuntime:
    # These tests are about the integrator, not the weights on disk, so they
    # take the explicit opt-out rather than staging artifact files.
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(manifest)
    runtime.reset(epoch=epoch, source_time_us=origin_us)
    return runtime


def drive(manifest, stream, make_batch, size: int, *, frames: int = 400) -> list[dict]:
    runtime = session(manifest)
    return [
        runtime.step(make_batch(stream, first, min(size, frames - first), seq=i))
        for i, first in enumerate(range(0, frames, size))
    ]


def alarm_frames(decisions, size: int, frames: int = 400) -> list[range]:
    """The frame window of every batch that reported a trigger."""
    return [
        range(i * size, min((i + 1) * size, frames))
        for i, decision in enumerate(decisions)
        if decision["trigger"]
    ]


# --------------------------------------------------------------- the criterion


def test_batch_size_does_not_change_the_answer(lui8, stream, make_batch):
    reference = drive(lui8, stream, make_batch, 1)
    spikes = sum(d["score"] for d in reference)
    fired = [w.start for w in alarm_frames(reference, 1)]
    assert spikes > 0 and fired, "the fixture stream must exercise both, or nothing is tested"

    for size in BATCH_SIZES:
        decisions = drive(lui8, stream, make_batch, size)
        assert sum(d["score"] for d in decisions) == spikes, f"batch size {size} changed the spike count"
        windows = alarm_frames(decisions, size)
        assert len(windows) == len(fired), f"batch size {size} changed the number of alarms"
        for window, frame in zip(windows, fired):
            assert frame in window, f"batch size {size} moved an alarm out of its frame"


def test_state_really_crosses_the_boundary(lui8, stream, make_batch):
    """A runtime restarted between batches must give a different answer.

    Without this, the test above would also pass on a stateless runtime, and
    P2 would be untested.
    """
    continuous = sum(d["score"] for d in drive(lui8, stream, make_batch, 25))
    restarted = 0
    for i, first in enumerate(range(0, 400, 25)):
        runtime = session(lui8, origin_us=first * lui8["runtime"]["dt_us"])
        restarted += runtime.step(make_batch(stream, first, 25, seq=i))["score"]
    assert restarted != continuous, "resetting between batches changed nothing, so nothing is being carried"


# ---------------------------------------------------------------------- gaps


def test_a_gap_forces_warmup_and_suppresses_the_trigger(lui8, stream, make_batch):
    runtime = session(lui8)
    runtime.step(make_batch(stream, 0, 100, seq=0))
    assert not runtime.warming_up

    after = runtime.step(make_batch(stream, 150, 20, seq=1))  # frames 100..149 never arrived
    assert after["status"] == "warmup"
    assert after["trigger"] is False
    assert runtime.warming_up


def test_the_runtime_comes_back_out_of_warmup(lui8, stream, make_batch):
    runtime = session(lui8)
    runtime.step(make_batch(stream, 0, 50, seq=0))
    runtime.step(make_batch(stream, 100, 10, seq=1))  # gap
    assert runtime.warming_up
    statuses = [runtime.step(make_batch(stream, first, 20, seq=9))["status"] for first in range(110, 390, 20)]
    assert "warmup" in statuses and statuses[-1] == "valid"


def test_a_long_gap_restarts_from_rest(lui8, stream, make_batch):
    """Past a few membrane time constants there is nothing left to carry.

    A 200 frame hole is longer than the settle time, so the state that survives
    it must be indistinguishable from a session that just started; a 5 frame
    hole must not be, or the runtime is throwing away real state on every
    hiccup. Both halves are asserted, because only the pair pins the boundary.
    """
    dt = lui8["runtime"]["dt_us"]
    at_rest = session(lui8, origin_us=300 * dt)
    expected = at_rest.step(make_batch(stream, 300, 100, seq=0))["score"]

    long_gap = session(lui8)
    long_gap.step(make_batch(stream, 0, 100, seq=0))
    assert long_gap.step(make_batch(stream, 300, 100, seq=1))["score"] == expected

    short_gap = session(lui8)
    short_gap.step(make_batch(stream, 0, 295, seq=0))
    assert short_gap.step(make_batch(stream, 300, 100, seq=1))["score"] != expected


def test_time_advances_only_by_what_was_consumed(lui8, stream, make_batch):
    dt = lui8["runtime"]["dt_us"]
    runtime = session(lui8)
    runtime.step(make_batch(stream, 0, 30, seq=0))
    assert runtime.source_time_us == 30 * dt
    runtime.step(make_batch(stream, 60, 10, seq=1))
    assert runtime.source_time_us == 70 * dt


# ------------------------------------------------------------------- refusals


def test_a_batch_from_another_epoch_is_refused(lui8, stream, make_batch):
    runtime = session(lui8, epoch=4)
    with pytest.raises(RuntimeStateError) as excinfo:
        runtime.step(make_batch(stream, 0, 10, epoch=5))
    assert excinfo.value.code == "EPOCH_MISMATCH"


def test_a_batch_from_another_encoder_is_refused(lui8, stream, make_batch):
    runtime = session(lui8)
    batch = make_batch(stream, 0, 10) | {"encoder_hash": "sha256:" + "0" * 64}
    with pytest.raises(RuntimeStateError) as excinfo:
        runtime.step(batch)
    assert excinfo.value.code == "ENCODER_MISMATCH"


def test_replayed_time_is_invalid_rather_than_integrated_twice(lui8, stream, make_batch):
    runtime = session(lui8)
    runtime.step(make_batch(stream, 0, 30, seq=0))
    before = runtime.source_time_us
    decision = runtime.step(make_batch(stream, 10, 30, seq=1))
    assert decision["status"] == "invalid" and decision["trigger"] is False
    assert runtime.source_time_us == before, "an overlapping batch must not move the clock"


@pytest.mark.parametrize("shift", [1, 4999, -3])
def test_a_window_off_the_frame_grid_is_invalid(lui8, stream, make_batch, shift):
    runtime = session(lui8)
    batch = make_batch(stream, 0, 10)
    batch["source_end_us"] += shift
    assert runtime.step(batch)["status"] == "invalid"


def test_calls_before_a_session_are_refused(lui8):
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    with pytest.raises(RuntimeStateError) as excinfo:
        runtime.step({})
    assert excinfo.value.code == "NOT_LOADED"
    runtime.load(lui8)
    with pytest.raises(RuntimeStateError) as excinfo:
        runtime.step({})
    assert excinfo.value.code == "NOT_STARTED"


def test_a_scripted_package_refuses_to_pretend(lui8, stream, make_batch):
    """The W0 demo fixture declares integrator "none": it has no physics."""
    runtime = LuiRuntime(allow_unverified_artifacts=True)
    runtime.load(fixture("model-manifest"))
    runtime.reset(epoch=1, source_time_us=0)
    decision = runtime.step(make_batch(stream, 0, 10))
    assert decision == {
        "trigger": False,
        "status": "invalid",
        "score": None,
        "score_kind": "unavailable",
        "provenance": "demo",
    }


# ------------------------------------------------------ the backend's contract


def test_the_decision_is_shaped_the_way_ingest_reads_it(lui8, stream, make_batch):
    """Assert against the backend's own constants, not against a copy of them."""
    from rpi_agents.cloud.app import ingest

    for decision in drive(lui8, stream, make_batch, 50):
        assert set(decision) == {"trigger", "status", "score", "score_kind", "provenance"}
        assert isinstance(decision["trigger"], bool)
        assert decision["status"] in ingest._RUNTIME_STATUSES
        assert decision["score_kind"] in ingest._SCORE_KINDS
        assert decision["provenance"] in ingest._PROVENANCE
        assert decision["score"] is None or isinstance(decision["score"], float)


def test_an_uncalibrated_model_never_claims_a_measurement(lui8, stream, make_batch):
    runtime = session(lui8)
    decision = runtime.step(make_batch(stream, 0, 10))
    assert decision["provenance"] == "simulated"
    assert decision["score_kind"] == "uncalibrated"
