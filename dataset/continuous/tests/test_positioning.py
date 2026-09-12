"""
test_positioning.py — testy logiki losowania pozycji zdarzeń, bez żadnego I/O
na plikach audio (czysta logika matematyczna, szybkie, uruchamialne w CI).
"""
import random
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from eval.stream_builder import PlacementError, sample_event_positions


def _no_overlap(starts, durations, min_gap_s):
    intervals = sorted(zip(starts, durations), key=lambda x: x[0])
    for (s0, d0), (s1, d1) in zip(intervals, intervals[1:]):
        e0 = s0 + d0
        assert s1 >= e0 + min_gap_s - 1e-9, (
            f"nakładanie/zbyt mały odstęp: [{s0},{e0}] vs start {s1} (min_gap={min_gap_s})"
        )


def test_five_events_no_overlap_typical_case():
    rng = random.Random(123)
    durations = [0.8, 1.2, 0.5, 2.0, 0.9]
    starts = sample_event_positions(rng, stream_duration_s=600.0, event_durations_s=durations,
                                     min_gap_s=2.0, edge_margin_s=1.0)
    assert len(starts) == 5
    _no_overlap(starts, durations, min_gap_s=2.0)
    for s, d in zip(starts, durations):
        assert s >= 1.0
        assert s + d <= 600.0 - 1.0 + 1e-9


def test_deterministic_with_same_seed():
    durations = [0.8, 1.2, 0.5, 2.0, 0.9]
    starts_a = sample_event_positions(random.Random(42), 600.0, durations, 2.0, 1.0)
    starts_b = sample_event_positions(random.Random(42), 600.0, durations, 2.0, 1.0)
    assert starts_a == starts_b


def test_different_seeds_give_different_positions():
    durations = [0.8, 1.2, 0.5, 2.0, 0.9]
    starts_a = sample_event_positions(random.Random(1), 600.0, durations, 2.0, 1.0)
    starts_b = sample_event_positions(random.Random(2), 600.0, durations, 2.0, 1.0)
    assert starts_a != starts_b


def test_too_short_stream_raises():
    durations = [10.0] * 5
    with pytest.raises(PlacementError):
        sample_event_positions(random.Random(0), stream_duration_s=20.0,
                                event_durations_s=durations, min_gap_s=2.0, edge_margin_s=1.0)


def test_tight_but_feasible_stream():
    # 5 zdarzeń po 1s, min_gap 0.5s, edge_margin 0 -> minimalna potrzebna
    # długość = 5*1 + 6*0.5 = 8s. Dajemy trochę zapasu.
    durations = [1.0] * 5
    starts = sample_event_positions(random.Random(7), stream_duration_s=12.0,
                                     event_durations_s=durations, min_gap_s=0.5,
                                     edge_margin_s=0.0)
    _no_overlap(starts, durations, min_gap_s=0.5)


def test_varying_durations_many_seeds_never_overlap():
    durations = [0.3, 2.5, 0.6, 1.1, 0.4]
    for seed in range(50):
        starts = sample_event_positions(random.Random(seed), 300.0, durations, 1.5, 0.5)
        _no_overlap(starts, durations, min_gap_s=1.5)
        for s, d in zip(starts, durations):
            assert s >= 0.5 - 1e-9
            assert s + d <= 300.0 - 0.5 + 1e-9
