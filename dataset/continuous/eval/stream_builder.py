"""
stream_builder.py — budowa pojedynczego ciągłego strumienia ewaluacyjnego
z dokładnie N_EVENTS zdarzeniami glassbreak w losowych, nienachodzących
pozycjach.

Ten moduł nie robi żadnego I/O poza tym co dostaje przez audio_io — dzięki
temu logika losowania pozycji (najtrudniejsza część do przetestowania) jest
testowalna bez plików na dysku (patrz tests/test_stream_builder.py).
"""
from __future__ import annotations

import glob
import hashlib
import os
import random
import subprocess
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .annotations import GlassClip
from .audio_io import AudioStandard, load_audio_mono, peak_normalize, write_audio

N_EVENTS = 5
GENERATOR_VERSION = "1.0.0"


# ============================================================ pozycjonowanie

@dataclass(frozen=True)
class PlacedEvent:
    start_s: float
    end_s: float
    source_stem: str
    source_start_s: float
    source_end_s: float
    is_contaminated: bool
    overlapping_labels: tuple[str, ...]
    gain_db: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_manifest_dict(self) -> dict:
        return {
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
            "source_stem": self.source_stem,
            "source_start_s": round(self.source_start_s, 6),
            "source_end_s": round(self.source_end_s, 6),
            "is_contaminated": self.is_contaminated,
            "overlapping_labels": list(self.overlapping_labels),
            "gain_db": round(self.gain_db, 3),
        }


class PlacementError(RuntimeError):
    """Nie udało się znaleźć N_EVENTS nienachodzących pozycji w strumieniu."""


def sample_event_positions(
    rng: random.Random,
    stream_duration_s: float,
    event_durations_s: Sequence[float],
    min_gap_s: float,
    edge_margin_s: float,
    max_attempts: int = 20_000,
) -> list[float]:
    """Losuje start każdego zdarzenia tak, by:
      - żadne zdarzenie nie wychodziło poza [edge_margin_s, stream_duration_s - edge_margin_s]
      - między końcem jednego a początkiem następnego był co najmniej min_gap_s
      - zdarzenia się nie nakładały (gwarantowane przez min_gap_s > 0)

    Zwraca listę startów w tej samej kolejności co event_durations_s (czyli NIE
    posortowaną chronologicznie — kolejność wstrzykiwania nie ma znaczenia,
    liczy się tylko finalne rozmieszczenie w czasie).

    Deterministyczne przy tym samym rng (przekaż random.Random(seed)).
    """
    n = len(event_durations_s)
    usable = stream_duration_s - 2 * edge_margin_s
    total_needed = sum(event_durations_s) + min_gap_s * (n + 1)
    if usable <= 0 or total_needed > usable:
        raise PlacementError(
            f"strumień {stream_duration_s:.1f}s za krótki na {n} zdarzeń "
            f"(potrzeba >= {total_needed + 2*edge_margin_s:.1f}s z marginesami)"
        )

    for _attempt in range(max_attempts):
        # losuj kolejność wstawiania, potem losuj start w dostępnej przestrzeni
        order = list(range(n))
        rng.shuffle(order)
        placed: list[tuple[float, float]] = []  # (start, end) posortowane rosnąco
        ok = True
        for idx in order:
            dur = event_durations_s[idx]
            lo, hi = edge_margin_s, stream_duration_s - edge_margin_s - dur
            if hi < lo:
                ok = False
                break
            start = rng.uniform(lo, hi)
            end = start + dur
            conflict = any(
                start < pe + min_gap_s and ps - min_gap_s < end
                for ps, pe in placed
            )
            if conflict:
                ok = False
                break
            placed.append((start, end))
        if ok:
            # zmapuj z powrotem na oryginalną kolejność event_durations_s
            starts_by_idx = {}
            placed_sorted_by_order = list(zip(order, placed))
            for idx, (s, _e) in placed_sorted_by_order:
                starts_by_idx[idx] = s
            return [starts_by_idx[i] for i in range(n)]

    raise PlacementError(
        f"nie znaleziono nienachodzących pozycji dla {n} zdarzeń po "
        f"{max_attempts} próbach (strumień {stream_duration_s:.1f}s, "
        f"min_gap {min_gap_s:.1f}s) — wydłuż strumień albo zmniejsz min_gap"
    )


# ============================================================ budowa strumienia

@dataclass
class BackgroundPool:
    """Pula plików tła. Każdy wpis to (ścieżka, źródło_nazwa) dla provenance."""
    files: list[tuple[str, str]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.files) == 0


def collect_background_pool(background_dirs: Sequence[str]) -> BackgroundPool:
    """Zbiera pliki .wav z podanych katalogów (rekurencyjnie). Twardy fail,
    jeśli którykolwiek katalog nie istnieje lub pula końcowa jest pusta —
    świadomie odwrotnie niż znany bug build-manifest, które po cichu pomija
    braki."""
    pool = BackgroundPool()
    for d in background_dirs:
        if not os.path.isdir(d):
            raise FileNotFoundError(f"katalog tła nie istnieje: {d}")
        found = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
        if not found:
            raise FileNotFoundError(f"0 plików .wav w katalogu tła: {d}")
        source_name = os.path.basename(os.path.normpath(d))
        pool.files.extend((f, source_name) for f in found)

    if pool.is_empty():
        raise FileNotFoundError(
            "pula tła jest pusta po przejrzeniu wszystkich --background-dir; "
            "generator wymaga niepustego tła (twardy fail zamiast cichego pominięcia)"
        )
    return pool


def _rms_dbfs(samples: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if samples.size else 0.0
    if rms < 1e-12:
        return -120.0
    return 20.0 * np.log10(rms)


def _fill_background(
    rng: random.Random,
    pool: BackgroundPool,
    total_samples: int,
    sample_rate: int,
    standard: AudioStandard,
) -> tuple[np.ndarray, list[dict]]:
    """Konkatenuje losowe pliki tła (z powtórzeniami, w losowej kolejności,
    losowym offsetem startu) aż do wypełnienia total_samples. Zwraca bufor
    oraz listę użytych segmentów tła (do provenance w manifeście)."""
    buf = np.zeros(total_samples, dtype=np.float32)
    used: list[dict] = []
    pos = 0
    files = list(pool.files)
    rng.shuffle(files)
    fi = 0
    guard = 0
    while pos < total_samples:
        guard += 1
        if guard > 100_000:
            raise RuntimeError("nie udało się wypełnić tła — pula plików prawdopodobnie pusta/zbyt krótka")
        path, source_name = files[fi % len(files)]
        fi += 1
        samples = load_audio_mono(path, standard)
        if samples.size == 0:
            continue
        # losowy offset startu wewnątrz pliku, żeby te same pliki brzmiały
        # różnie przy różnych seedach nawet gdy kolejność się powtórzy
        offset = rng.randrange(0, samples.size) if samples.size > 1 else 0
        rotated = np.concatenate([samples[offset:], samples[:offset]])
        take = min(rotated.size, total_samples - pos)
        buf[pos:pos + take] = rotated[:take]
        used.append({
            "path": os.path.relpath(path),
            "source": source_name,
            "stream_start_s": round(pos / sample_rate, 6),
            "stream_end_s": round((pos + take) / sample_rate, 6),
        })
        pos += take
    return buf, used


@dataclass
class GeneratedStream:
    audio: np.ndarray
    sample_rate: int
    events: list[PlacedEvent]
    background_segments: list[dict]
    background_gain_db: float


def build_stream(
    *,
    duration_s: float,
    glass_clips: Sequence[GlassClip],
    audio_root_for_glass: str,
    background_pool: BackgroundPool,
    seed: int,
    min_gap_s: float = 2.0,
    edge_margin_s: float = 1.0,
    event_gain_db_range: tuple[float, float] = (-3.0, 3.0),
    standard: AudioStandard = AudioStandard(),
    clip_guard_peak: float = 0.97,
) -> GeneratedStream:
    """Buduje jeden deterministyczny strumień. Ten sam seed + te same wejścia
    (glass_clips, background_pool, duration_s, min_gap_s, edge_margin_s) ->
    identyczny wynik (audio i manifest), bo cała losowość idzie przez jeden
    random.Random(seed) w ustalonej kolejności operacji.

    audio_root_for_glass: katalog z plikami .wav odpowiadającymi
    GlassClip.source_stem (np. dataset/clean/clean/audio/).
    """
    if len(glass_clips) < N_EVENTS:
        raise RuntimeError(
            f"za mało kandydujących klipów glassbreak: {len(glass_clips)} < {N_EVENTS}"
        )

    rng = random.Random(seed)

    # 1) wybór DOKŁADNIE N_EVENTS klipów szkła z puli (bez powtórzeń klipu)
    chosen_clips = rng.sample(list(glass_clips), N_EVENTS)

    # 2) wczytanie audio wybranych klipów + wycięcie natywnego interwału
    event_audios: list[np.ndarray] = []
    for clip in chosen_clips:
        src_path = os.path.join(audio_root_for_glass, f"{clip.source_stem}.wav")
        full = load_audio_mono(src_path, standard)
        i0 = int(round(clip.start_s * standard.sample_rate))
        i1 = int(round(clip.end_s * standard.sample_rate))
        i0 = max(0, min(i0, full.size))
        i1 = max(i0, min(i1, full.size))
        segment = full[i0:i1]
        if segment.size == 0:
            raise RuntimeError(
                f"pusty segment glassbreak z {src_path} [{clip.start_s}, {clip.end_s}]"
            )
        event_audios.append(segment)

    event_durations_s = [a.size / standard.sample_rate for a in event_audios]

    # 3) losowanie pozycji (nienachodzące, w granicach strumienia)
    starts = sample_event_positions(
        rng, duration_s, event_durations_s, min_gap_s=min_gap_s, edge_margin_s=edge_margin_s
    )

    # 4) losowanie skal głośności per zdarzenie
    gains_db = [rng.uniform(*event_gain_db_range) for _ in range(N_EVENTS)]

    # 5) budowa tła
    total_samples = int(round(duration_s * standard.sample_rate))
    bg_buf, bg_segments = _fill_background(rng, background_pool, total_samples,
                                            standard.sample_rate, standard)
    bg_gain_db = rng.uniform(-6.0, 0.0)
    bg_buf = bg_buf * (10.0 ** (bg_gain_db / 20.0))

    # 6) miksowanie zdarzeń na tło
    mix = bg_buf.copy()
    placed_events: list[PlacedEvent] = []
    for clip, seg, start_s, gain_db in zip(chosen_clips, event_audios, starts, gains_db):
        scaled = seg * (10.0 ** (gain_db / 20.0))
        i0 = int(round(start_s * standard.sample_rate))
        i1 = i0 + scaled.size
        i1c = min(i1, mix.size)
        mix[i0:i1c] += scaled[: i1c - i0]
        placed_events.append(PlacedEvent(
            start_s=start_s,
            end_s=start_s + seg.size / standard.sample_rate,
            source_stem=clip.source_stem,
            source_start_s=clip.start_s,
            source_end_s=clip.end_s,
            is_contaminated=clip.is_contaminated,
            overlapping_labels=clip.overlapping_labels,
            gain_db=gain_db,
        ))

    # 7) zabezpieczenie przed clippingiem: jeśli szczyt przekracza guard, skaluj
    #    CAŁY miks w dół (zachowuje względne proporcje zdarzenie/tło, w
    #    przeciwieństwie do per-sample clip, który zniekształciłby kształt fali
    #    dokładnie w oknach zdarzeń — najgorsze możliwe miejsce na artefakt)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > clip_guard_peak:
        mix = mix * (clip_guard_peak / peak)

    placed_events.sort(key=lambda e: e.start_s)

    return GeneratedStream(
        audio=mix.astype(np.float32),
        sample_rate=standard.sample_rate,
        events=placed_events,
        background_segments=bg_segments,
        background_gain_db=bg_gain_db,
    )


# ============================================================ provenance / hash

def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_short() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None
