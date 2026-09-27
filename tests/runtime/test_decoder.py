"""The streaming decoder, and its equivalence with the project's FA/h rule.

``snn_pipeline/stream_eval.py:count_alarms`` is the single source of the alarm
rule: every FA/h number the project selects models by comes out of it. The live
decoder has to be the same function or the operating point chosen offline is not
the one that runs on the Pi. ``test_equals_count_alarms`` asserts exactly that,
on random trains across a grid of k, w and cooldown.

It is skipped when ``snn_pipeline`` is not checked out, because the ``pr-gate``
only fetches ``snn_runtime`` and ``tests/runtime`` and numpy is not in
``requirements-w0.lock``. Skipping is honest: the equivalence is checked on a
full checkout, and the decoder's own behaviour is pinned by the tests below,
which need nothing.
"""

from __future__ import annotations

import random

import pytest

from snn_runtime import KOfWDecoder


def run(decoder: KOfWDecoder, train) -> int:
    return sum(decoder.step(bool(x)) for x in train)


def test_k_of_one_fires_on_every_spike_outside_the_cooldown():
    decoder = KOfWDecoder(k=1, window_frames=1, cooldown_frames=3)
    fired = [decoder.step(bool(x)) for x in [1, 1, 1, 1, 1, 0, 1]]
    #        frame:                            0  1  2  3  4  5  6
    # frame 0 fires and blocks 1..2; frame 3 fires and blocks 4..5; frame 6 fires
    assert fired == [True, False, False, True, False, False, True]


def test_a_window_that_is_too_wide_does_not_fire():
    decoder = KOfWDecoder(k=3, window_frames=4, cooldown_frames=10)
    assert run(decoder, [1, 0, 0, 0, 1, 0, 0, 0, 1]) == 0


def test_three_spikes_inside_the_window_fire_once():
    decoder = KOfWDecoder(k=3, window_frames=4, cooldown_frames=10)
    assert run(decoder, [1, 1, 0, 1, 0, 0]) == 1


def test_the_cooldown_swallows_the_spikes_it_covers():
    """A spike inside the cooldown is dropped, not merely unreported.

    It must not count towards the next window either, which is what
    ``searchsorted`` does in the offline rule.
    """
    decoder = KOfWDecoder(k=2, window_frames=2, cooldown_frames=5)
    fired = [decoder.step(bool(x)) for x in [1, 1, 1, 1, 1, 1, 1, 1]]
    assert fired == [False, True, False, False, False, False, False, True]


def test_flush_forgets_the_spikes_but_keeps_the_cooldown():
    decoder = KOfWDecoder(k=2, window_frames=3, cooldown_frames=4)
    assert decoder.step(True) is False
    decoder.flush()
    assert decoder.step(True) is False, "the pre-gap spike must not complete the window"
    assert decoder.step(True) is True


def test_skip_advances_the_clock_so_a_window_cannot_span_a_gap():
    decoder = KOfWDecoder(k=2, window_frames=3, cooldown_frames=4)
    assert decoder.step(True) is False
    decoder.skip(100)
    assert decoder.step(True) is False, "100 missing frames put the two spikes far apart"


def test_reset_clears_everything():
    decoder = KOfWDecoder(k=1, window_frames=1, cooldown_frames=50)
    assert decoder.step(True) is True
    decoder.reset()
    assert decoder.step(True) is True
    assert decoder.frame == 1


@pytest.mark.parametrize("bad", [{"k": 0}, {"window_frames": 0}, {"cooldown_frames": -1}])
def test_a_rule_that_cannot_be_evaluated_is_refused(bad):
    kwargs = {"k": 1, "window_frames": 1, "cooldown_frames": 0} | bad
    with pytest.raises(ValueError):
        KOfWDecoder(**kwargs)


def test_from_manifest_converts_microseconds_to_frames(lui8):
    decoder = KOfWDecoder.from_manifest(lui8["decoder"], lui8["runtime"]["dt_us"])
    assert (decoder.k, decoder.window_frames, decoder.cooldown_frames) == (1, 1, 500)


@pytest.mark.parametrize("k, window, cooldown", [(1, 1, 1), (1, 1, 20), (2, 3, 5), (3, 10, 40), (4, 25, 25), (5, 50, 300)])
def test_equals_count_alarms(k, window, cooldown):
    """Same alarm count as the offline rule, on every train we throw at it."""
    stream_eval = pytest.importorskip(
        "snn_pipeline.stream_eval",
        reason="snn_pipeline is outside the pr-gate checkout and needs numpy",
    )
    numpy = pytest.importorskip("numpy")

    rng = random.Random(f"{k}-{window}-{cooldown}")
    for density in (0.02, 0.1, 0.4, 0.9):
        train = [1 if rng.random() < density else 0 for _ in range(2000)]
        offline = stream_eval.count_alarms(numpy.array(train, dtype=bool), k, window, cooldown)
        assert run(KOfWDecoder(k=k, window_frames=window, cooldown_frames=cooldown), train) == offline, (
            f"density {density}: the live decoder and the FA/h rule disagree"
        )
