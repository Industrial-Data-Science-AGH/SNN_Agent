"""
test_val_test_pair.py — testy rozdziału continuous-val / continuous-test z jednego seeda
(K3 pkt 2/4). W konwencji test_stream_builder.py: audio in-memory, bez plików produkcyjnych.
"""
import os, sys, random
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.stream_builder import (
    BackgroundPool, build_val_test_pair, validate_val_test_disjoint, derive_seed, N_EVENTS,
)
from eval.annotations import GlassClip
from eval.audio_io import AudioStandard, write_audio

SR = 44100
STD = AudioStandard(sample_rate=SR)
N_SOURCES = 3 * N_EVENTS  # z zapasem: val i test biorą po N_EVENTS z rozłącznych puli source_stem


def _make_wav(path: str, duration_s: float = 5.0, freq: float = 440.0):
    t = np.linspace(0, duration_s, int(duration_s * SR), endpoint=False, dtype=np.float32)
    write_audio(path, 0.3 * np.sin(2 * np.pi * freq * t), STD)


@pytest.fixture()
def audio_root(tmp_path):
    """N_SOURCES osobnych plików źródłowych szkła — każdy inny source_stem."""
    for i in range(N_SOURCES):
        _make_wav(str(tmp_path / f"synthetic_{i:03d}.wav"), duration_s=5.0)
    return str(tmp_path)


def _clips(n=N_SOURCES, duration_s=1.0):
    return [
        GlassClip(source_stem=f"synthetic_{i:03d}", start_s=0.0, end_s=duration_s,
                  is_contaminated=False, overlapping_labels=())
        for i in range(n)
    ]


@pytest.fixture()
def background_dirs(tmp_path, monkeypatch):
    """50 plików tła (dużo więcej niż jeden build_stream typowo zużywa dla 60s streamu),
    żeby test sprawdzał logikę wykluczania val/test, a nie przypadkowo trafiał na
    wyczerpanie puli przez pechowe offsety w _fill_background (patrz K3: przy 15
    plikach i offsetach bez zawijania jeden val potrafił zużyć wszystkie 15)."""
    bg_dir = tmp_path / "bg"
    bg_dir.mkdir()
    for i in range(50):
        _make_wav(str(bg_dir / f"1-{i:06d}-A-0.wav"), duration_s=10.0, freq=200 + i)
    kind_map = {f"1-{i:06d}-A-0.wav": "stationary" for i in range(50)}
    monkeypatch.setattr("eval.stream_builder._load_esc50_kind_map", lambda d: kind_map)
    return [str(bg_dir)]


def _pair_kwargs(audio_root, background_dirs, **overrides):
    kw = dict(
        duration_s=60.0, glass_clips=_clips(), audio_root_for_glass=audio_root,
        background_dirs=background_dirs, train_manifest_csv=None,
        min_gap_s=1.0, warmup_s=10.0, end_margin_s=0.5, standard=STD,
    )
    kw.update(overrides)
    return kw


def test_derive_seed_differs_by_role_and_seed():
    assert derive_seed(42, "val") != derive_seed(42, "test")
    assert derive_seed(42, "val") == derive_seed(42, "val")
    assert derive_seed(42, "val") != derive_seed(43, "val")


def test_val_and_test_never_share_background_group_id(audio_root, background_dirs):
    pair = build_val_test_pair(seed=42, **_pair_kwargs(audio_root, background_dirs))
    val_ids = {s["group_id"] for s in pair.val.background_segments}
    test_ids = {s["group_id"] for s in pair.test.background_segments}
    assert val_ids.isdisjoint(test_ids)
    assert val_ids == pair.val_background_group_ids


def test_val_and_test_never_share_glass_source_stem(audio_root, background_dirs):
    pair = build_val_test_pair(seed=42, **_pair_kwargs(audio_root, background_dirs))
    val_stems = {e.source_stem for e in pair.val.events}
    test_stems = {e.source_stem for e in pair.test.events}
    assert val_stems.isdisjoint(test_stems)
    assert val_stems == pair.val_glass_source_stems


def test_validate_val_test_disjoint_passes_on_a_real_pair(audio_root, background_dirs):
    pair = build_val_test_pair(seed=42, **_pair_kwargs(audio_root, background_dirs))
    validate_val_test_disjoint(pair)  # nie powinno rzucić


def test_validate_val_test_disjoint_catches_injected_collision(audio_root, background_dirs):
    pair = build_val_test_pair(seed=42, **_pair_kwargs(audio_root, background_dirs))
    pair.val_background_group_ids |= {s["group_id"] for s in pair.test.background_segments}
    with pytest.raises(ValueError, match="tła ESC-50"):
        validate_val_test_disjoint(pair)


def test_pair_is_deterministic_for_the_same_seed(audio_root, background_dirs):
    kw = _pair_kwargs(audio_root, background_dirs)
    p1 = build_val_test_pair(seed=7, **kw)
    p2 = build_val_test_pair(seed=7, **kw)
    assert np.array_equal(p1.val.audio, p2.val.audio)
    assert np.array_equal(p1.test.audio, p2.test.audio)
    assert [e.source_stem for e in p1.val.events] == [e.source_stem for e in p2.val.events]
    assert [e.source_stem for e in p1.test.events] == [e.source_stem for e in p2.test.events]


def test_different_seeds_give_different_pairs(audio_root, background_dirs):
    kw = _pair_kwargs(audio_root, background_dirs)
    p1 = build_val_test_pair(seed=1, **kw)
    p2 = build_val_test_pair(seed=2, **kw)
    assert [e.source_stem for e in p1.val.events] != [e.source_stem for e in p2.val.events]


def test_val_and_test_are_each_internally_valid_streams(audio_root, background_dirs):
    """val i test to osobno poprawne strumienie: dokładnie N_EVENTS, bez nakładania."""
    pair = build_val_test_pair(seed=3, **_pair_kwargs(audio_root, background_dirs))
    for stream in (pair.val, pair.test):
        assert len(stream.events) == N_EVENTS
        evs = sorted(stream.events, key=lambda e: e.start_s)
        for a, b in zip(evs, evs[1:]):
            assert a.end_s <= b.start_s + 1e-9


def test_too_few_unique_sources_raises_not_silently_reuses(audio_root, background_dirs):
    """Gdy source_stem starcza dokładnie na val, test MUSI rzucić - nie po cichu
    wybrać z powrotem z puli val."""
    few = _clips(n=N_EVENTS)  # dokładnie tyle co potrzeba na SAM val
    with pytest.raises(RuntimeError, match="za mało"):
        build_val_test_pair(seed=42, **_pair_kwargs(audio_root, background_dirs, glass_clips=few))