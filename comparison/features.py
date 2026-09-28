"""Frame features for the Fourier side of the SNN vs Fourier comparison.

Both variants produce one feature row per encoder frame, on exactly the grid the
Lu.i encoder uses: 192 samples at about 19231 Hz, so a frame every 9984 us. That
is not cosmetic. The comparison is decided by a k-of-w rule over frames and by
false alarms per hour, so if the two sides counted frames differently the
numbers would not be comparable at all.

Two variants, because "Fourier is worse" is only a result if it is not simply
an undertuned baseline:

``full``
    What a Fourier front end achieves with no hardware budget at all: a 512
    point rFFT per frame, 24 log spaced bands, and the usual shape descriptors.
    This is the ceiling. If the SNN lands near it, that is the interesting
    finding; if it lands far below, the encoder is not what is holding us back.

``mcu``
    What an ATmega328P at 16 MHz could actually run, which is the claim the
    project is really making. See ``mcu_budget.py``: Kacper measured the current
    encoder's ISR at 590.5 cycles, 70.97 % of the CPU, on a real board. A 512
    point FFT does not fit in what is left, a handful of single bin evaluations
    does. So this variant is six band magnitudes plus total energy, computed on
    the 192 sample hop with no extra window.

Both variants then get the same context treatment and the same classifier, so
the only thing that differs between them is how much spectrum the hardware
could afford to look at.

The band magnitudes in ``mcu`` are computed with a direct DFT matrix rather than
the Goertzel recurrence. The two are the same number; Goertzel is how you would
compute it on the MCU cheaply, and the cost of doing so is accounted for in
``mcu_budget.py``, not here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# The encoder's own grid. 191/438 is 44100 Hz scaled to 19230.8 Hz, which is the
# rate Kacper measured on the board (measurements.json: fs_hz 19230.5/19230.8);
# an exact 19231 would need a polyphase filter with 19231 phases and buys
# nothing at one part in 10^5.
RESAMPLE_UP, RESAMPLE_DOWN = 191, 438
FS_HZ = 44100 * RESAMPLE_UP / RESAMPLE_DOWN
HOP = 192
WINDOW = 512

# Six bands spanning what the microphone and an 19 kHz ADC can actually carry.
# Glass breaking is broadband with most of its signature well above speech, so
# the spacing is denser at the top than a purely log placement would be.
MCU_BANDS_HZ = (400.0, 1000.0, 2200.0, 4000.0, 6000.0, 8200.0)
N_FULL_BANDS = 24
BAND_LO_HZ, BAND_HI_HZ = 100.0, 9000.0

EPS = 1e-10


@dataclass(frozen=True)
class FeatureSet:
    name: str
    columns: tuple[str, ...]

    @property
    def width(self) -> int:
        return len(self.columns)


def _with_deltas(names: tuple[str, ...]) -> tuple[str, ...]:
    return names + tuple(f"d_{n}" for n in names)


FULL = FeatureSet(
    "full",
    _with_deltas(
        tuple(f"band{i:02d}" for i in range(N_FULL_BANDS))
        + ("centroid", "flatness", "rolloff85", "energy")
    ),
)
MCU = FeatureSet(
    "mcu",
    _with_deltas(tuple(f"g{int(f)}" for f in MCU_BANDS_HZ) + ("energy",)),
)
SETS = {FULL.name: FULL, MCU.name: MCU}


def load_frames(path: str) -> np.ndarray:
    """Read a clip, put it on the encoder's sample rate, cut it into hops."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 44100:
        # The v2.0.0 manifest is all 44.1 kHz; anything else goes through the
        # slow path rather than being silently mis-rated.
        from fractions import Fraction

        ratio = Fraction(int(round(FS_HZ)), sample_rate).limit_denominator(500)
        audio = resample_poly(audio, ratio.numerator, ratio.denominator)
    else:
        audio = resample_poly(audio, RESAMPLE_UP, RESAMPLE_DOWN)
    n_frames = len(audio) // HOP
    if n_frames < 2:
        return np.zeros((0, HOP), dtype=np.float32)
    return audio[: n_frames * HOP].reshape(n_frames, HOP).astype(np.float32)


def _band_edges() -> np.ndarray:
    return np.geomspace(BAND_LO_HZ, BAND_HI_HZ, N_FULL_BANDS + 1)


def _full_features(hops: np.ndarray) -> np.ndarray:
    """512 point rFFT per frame, overlapping the previous two hops."""
    n_frames = len(hops)
    signal = hops.reshape(-1)
    window = np.hanning(WINDOW).astype(np.float32)
    pad = WINDOW - HOP
    padded = np.concatenate([np.zeros(pad, dtype=np.float32), signal])
    # A strided view rather than a Python loop: at 7.5 million frames across
    # the corpus the loop dominates the FFT it feeds.
    frames = np.lib.stride_tricks.sliding_window_view(padded, WINDOW)[::HOP][:n_frames] * window

    spectrum = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32)
    power = spectrum**2
    freqs = np.fft.rfftfreq(WINDOW, 1.0 / FS_HZ)

    edges = _band_edges()
    bands = np.empty((n_frames, N_FULL_BANDS), dtype=np.float32)
    for i in range(N_FULL_BANDS):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        bands[:, i] = power[:, mask].sum(axis=1) if mask.any() else 0.0

    total = power.sum(axis=1) + EPS
    centroid = (power * freqs).sum(axis=1) / total
    flatness = np.exp(np.log(power + EPS).mean(axis=1)) / (total / power.shape[1])
    cumulative = np.cumsum(power, axis=1)
    rolloff = freqs[np.argmax(cumulative >= 0.85 * cumulative[:, -1:], axis=1)]

    return np.column_stack(
        [
            np.log10(bands + EPS),
            np.log10(centroid + EPS),
            np.log10(flatness + EPS),
            np.log10(rolloff + EPS),
            np.log10(total),
        ]
    ).astype(np.float32)


def _mcu_features(hops: np.ndarray) -> np.ndarray:
    """Six band magnitudes on the bare 192 sample hop, plus its energy."""
    n = HOP
    sample_index = np.arange(n)
    basis = np.exp(-2j * np.pi * np.outer(np.asarray(MCU_BANDS_HZ) / FS_HZ, sample_index))
    magnitudes = np.abs(hops.astype(np.float64) @ basis.T.conj())
    energy = (hops.astype(np.float64) ** 2).sum(axis=1)
    return np.column_stack([np.log10(magnitudes + EPS), np.log10(energy + EPS)]).astype(np.float32)


def frame_features(path: str, which: str) -> np.ndarray:
    """One row per encoder frame: the variant's features and their deltas.

    The delta is the change since the previous frame. Glass breaking is a
    transient, so the rate of change carries much of the evidence, and giving
    it to the Fourier side explicitly is part of not handicapping it.
    """
    hops = load_frames(path)
    if len(hops) == 0:
        return np.zeros((0, SETS[which].width), dtype=np.float32)
    base = _full_features(hops) if which == "full" else _mcu_features(hops)
    delta = np.diff(base, axis=0, prepend=base[:1])
    return np.hstack([base, delta])
