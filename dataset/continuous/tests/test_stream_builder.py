"""
test_stream_builder.py — integracyjne testy build_stream na syntetycznym audio.
Nie wymagają plików z dysku: audio generowane jako numpy arrays in-memory.
"""
import os, sys, random, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.stream_builder import (
    BackgroundPool, PlacedEvent, GeneratedStream, build_stream,
    collect_background_pool, N_EVENTS,
)
from eval.annotations import GlassClip
from eval.audio_io import AudioStandard, write_audio

SR = 44100
STD = AudioStandard(sample_rate=SR)


def _make_wav(path: str, duration_s: float = 5.0, freq: float = 440.0):
    """Tworzy krótki WAV z tonem sinusoidalnym — tylko do testów."""
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
    write_audio(path, 0.3 * np.sin(2 * np.pi * freq * t), STD)


def _fake_clips(n: int, source_stem: str, duration_s: float = 1.0) -> list[GlassClip]:
    return [
        GlassClip(
            source_stem=source_stem,
            start_s=float(i),
            end_s=float(i) + duration_s,
            is_contaminated=False,
            overlapping_labels=(),
        )
        for i in range(n)
    ]


@pytest.fixture()
def audio_root(tmp_path):
    """Jeden plik WAV źródłowy na 60 s, żeby wycinanie klipów miało z czego brać."""
    _make_wav(str(tmp_path / "synthetic_001.wav"), duration_s=60.0)
    return str(tmp_path)


@pytest.fixture()
def background_pool(tmp_path):
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    for i in range(3):
        _make_wav(str(bg_dir / f"bg{i:02d}.wav"), duration_s=30.0, freq=200 + i * 50)
    pool = collect_background_pool([str(bg_dir)])
    return pool


def test_build_stream_exactly_5_events(audio_root, background_pool):
    clips = _fake_clips(10, "synthetic_001", duration_s=1.0)
    result = build_stream(
        duration_s=60.0, glass_clips=clips,
        audio_root_for_glass=audio_root,
        background_pool=background_pool,
        seed=42, min_gap_s=1.0, edge_margin_s=0.5, standard=STD,
    )
    assert len(result.events) == N_EVENTS == 5


def test_build_stream_no_event_overlap(audio_root, background_pool):
    clips = _fake_clips(10, "synthetic_001", duration_s=1.0)
    result = build_stream(
        duration_s=60.0, glass_clips=clips,
        audio_root_for_glass=audio_root,
        background_pool=background_pool,
        seed=7, min_gap_s=1.0, edge_margin_s=0.5, standard=STD,
    )
    evs = sorted(result.events, key=lambda e: e.start_s)
    for a, b in zip(evs, evs[1:]):
        assert a.end_s <= b.start_s + 1e-9


def test_build_stream_deterministic(audio_root, background_pool):
    clips = _fake_clips(10, "synthetic_001", duration_s=1.0)
    kw = dict(duration_s=60.0, glass_clips=clips,
               audio_root_for_glass=audio_root, background_pool=background_pool,
               seed=99, min_gap_s=1.0, edge_margin_s=0.5, standard=STD)
    r1 = build_stream(**kw)
    r2 = build_stream(**kw)
    assert np.array_equal(r1.audio, r2.audio)
    assert [(e.start_s, e.end_s) for e in r1.events] == \
           [(e.start_s, e.end_s) for e in r2.events]


def test_build_stream_different_seeds_differ(audio_root, background_pool):
    clips = _fake_clips(10, "synthetic_001", duration_s=1.0)
    kw = dict(duration_s=60.0, glass_clips=clips,
               audio_root_for_glass=audio_root, background_pool=background_pool,
               min_gap_s=1.0, edge_margin_s=0.5, standard=STD)
    r1 = build_stream(**kw, seed=1)
    r2 = build_stream(**kw, seed=2)
    starts1 = [e.start_s for e in r1.events]
    starts2 = [e.start_s for e in r2.events]
    assert starts1 != starts2


def test_build_stream_audio_length(audio_root, background_pool):
    clips = _fake_clips(10, "synthetic_001", duration_s=1.0)
    duration_s = 60.0
    result = build_stream(
        duration_s=duration_s, glass_clips=clips,
        audio_root_for_glass=audio_root,
        background_pool=background_pool,
        seed=0, min_gap_s=1.0, edge_margin_s=0.5, standard=STD,
    )
    assert abs(result.audio.size / SR - duration_s) < 0.1


def test_build_stream_no_clipping(audio_root, background_pool):
    clips = _fake_clips(10, "synthetic_001", duration_s=1.0)
    result = build_stream(
        duration_s=60.0, glass_clips=clips,
        audio_root_for_glass=audio_root,
        background_pool=background_pool,
        seed=5, standard=STD, min_gap_s=1.0, edge_margin_s=0.5,
    )
    assert float(np.max(np.abs(result.audio))) <= 1.0 + 1e-6


def test_collect_background_pool_missing_dir_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        collect_background_pool([str(tmp_path / "nonexistent")])


def test_collect_background_pool_empty_dir_fails(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError):
        collect_background_pool([str(empty)])
