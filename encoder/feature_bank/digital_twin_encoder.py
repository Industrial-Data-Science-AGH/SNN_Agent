"""
digital_twin_encoder.py
========================
Cyfrowy bliźniak enkodera `encoder_v2.ino` w Pythonie.
"""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping

import numpy as np
from scipy.signal import lfilter

try:
    from feature_metrics import FEATURE_FUNCTIONS
except ImportError:
    from .feature_metrics import FEATURE_FUNCTIONS


@dataclass
class McuComplexity:
    add: int = 0
    mul: int = 0
    cmp: int = 0
    div: int = 0
    sqrt: int = 0

    def cycles_cortex_m4f(self) -> int:
        """Szacowana liczba cykli dla ARM Cortex-M4F (FPU SP: ADD/MUL/CMP = 1, DIV/SQRT = 14)."""
        return (
            self.add * 1 + self.mul * 1 + self.cmp * 1 + self.div * 14 + self.sqrt * 14
        )


class FeatureExtractor:
    def __init__(
        self,
        fn: Callable[[Mapping[str, object]], float],
        is_spectral: bool = False,
        complexity_fn: Callable[[int, int], McuComplexity] | None = None,
    ):
        self.fn = fn
        self.is_spectral = is_spectral
        self._complexity_fn = complexity_fn

    def __call__(self, stats: Mapping[str, object]) -> float:
        return self.fn(stats)

    def complexity(self, n: int, n_fft: int) -> McuComplexity:
        if self._complexity_fn:
            return self._complexity_fn(n, n_fft)
        return McuComplexity(add=1)


def _build_extractors() -> dict[str, FeatureExtractor]:
    def comp_peak(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(cmp=n)

    def comp_div1(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(div=1)

    def comp_cv(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(div=1, sqrt=1)

    def comp_tkeo(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(add=n, mul=2 * n, div=1)

    def comp_curve(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(add=n, cmp=n)

    def comp_kurtosis(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(mul=2, div=1)

    def comp_spectral_centroid(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(add=n_fft, mul=n_fft, div=1)

    def comp_dominant_freq(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(cmp=n_fft)

    def comp_band(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(add=n_fft, div=1)

    def comp_flatness(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(add=n_fft, div=1)

    def comp_flux(n: int, n_fft: int) -> McuComplexity:
        return McuComplexity(add=n_fft, cmp=n_fft)

    spectral_names = {
        "spectral_centroid",
        "dominant_freq",
        "band_energy_low",
        "band_energy_mid",
        "band_energy_high",
        "spectral_flatness",
        "spectral_flux",
    }

    complexity_map = {
        "peak": comp_peak,
        "peak_cnt": comp_peak,
        "crest": comp_div1,
        "cv": comp_cv,
        "zcr": comp_div1,
        "flux": comp_div1,
        "hjorth_mobility": comp_cv,
        "tkeo_mean": comp_tkeo,
        "curve_length": comp_curve,
        "autocorr_lag1": comp_div1,
        "kurtosis": comp_kurtosis,
        "spectral_centroid": comp_spectral_centroid,
        "dominant_freq": comp_dominant_freq,
        "band_energy_low": comp_band,
        "band_energy_mid": comp_band,
        "band_energy_high": comp_band,
        "spectral_flatness": comp_flatness,
        "spectral_flux": comp_flux,
    }

    extractors = {}
    for name, fn in FEATURE_FUNCTIONS.items():
        is_spec = name in spectral_names
        c_fn = complexity_map.get(name)
        extractors[name] = FeatureExtractor(fn, is_spectral=is_spec, complexity_fn=c_fn)

    return extractors


CHANNEL_EXTRACTORS = _build_extractors()

FS_HZ_HW = 19231.0
HOP_MS = 10.0
DC_ALPHA = 1.0 / 512.0
PRIME_FRAMES = 50
SPIKE_THR_INIT = 40.0

HW_CHANNELS = [
    "peak",
    "peak_cnt",
    "crest",
    "cv",
    "zcr",
    "flux",
]

NEW_TIME_CHANNELS = [
    "hjorth_mobility",
    "tkeo_mean",
    "curve_length",
    "autocorr_lag1",
    "kurtosis",
]

FFT_CHANNELS = [
    c for c in CHANNEL_EXTRACTORS if c not in HW_CHANNELS and c not in NEW_TIME_CHANNELS
]


def simulate_adc_raw(
    normalized_samples: np.ndarray,
    gain: float = 1.0,
) -> np.ndarray:
    x = normalized_samples.astype(np.float64)
    return np.clip(512.0 + x * 512.0 * gain, 0.0, 1023.0)


def remove_dc(raw: np.ndarray) -> np.ndarray:
    b = [DC_ALPHA]
    a = [1.0, -(1.0 - DC_ALPHA)]
    zi = [raw[0] * (1.0 - DC_ALPHA)]
    dc_est, _ = lfilter(b, a, raw, zi=zi)
    return raw - dc_est


class EncoderState:
    __slots__ = (
        "rms_prev",
        "floor_peak",
        "mad_peak",
        "spike_thr",
        "floors_primed",
        "frame_idx",
        "prev_spectrum",
    )

    def __init__(self) -> None:
        self.rms_prev = 0.0
        self.floor_peak = 0.0
        self.mad_peak = 0.0
        self.spike_thr = SPIKE_THR_INIT
        self.floors_primed = False
        self.frame_idx = 0
        self.prev_spectrum: np.ndarray | None = None


def _update_peak_floor(
    state: EncoderState,
    peak_val: float,
) -> None:
    a = 0.0015 if peak_val > state.floor_peak else 0.0300

    state.floor_peak += a * (peak_val - state.floor_peak)

    d = abs(peak_val - state.floor_peak)
    state.mad_peak += 0.0100 * (d - state.mad_peak)

    z = (peak_val - state.floor_peak) / (state.mad_peak + 1e-6)

    if z > 4.0:
        state.floor_peak -= 0.0015 * (peak_val - state.floor_peak)


def _frame_spectrum(
    x_frame: np.ndarray,
    fs: float,
) -> tuple[np.ndarray, dict]:
    n = len(x_frame)
    window = np.hanning(n)

    spec = np.abs(np.fft.rfft(x_frame * window))
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    mag_sum = spec.sum() + 1e-6

    centroid = float((freqs * spec).sum() / mag_sum)

    dominant_freq = float(freqs[int(np.argmax(spec))])

    nyquist = fs / 2.0

    low_mask = freqs < nyquist * 0.15
    mid_mask = (freqs >= nyquist * 0.15) & (freqs < nyquist * 0.5)
    high_mask = freqs >= nyquist * 0.5

    band_low = float(spec[low_mask].sum() / mag_sum)
    band_mid = float(spec[mid_mask].sum() / mag_sum)
    band_high = float(spec[high_mask].sum() / mag_sum)

    log_spec = np.log(spec + 1e-6)

    flatness = float(np.exp(log_spec.mean()) / (spec.mean() + 1e-6))

    feats = {
        "spectral_centroid": centroid,
        "dominant_freq": dominant_freq,
        "band_energy_low": band_low,
        "band_energy_mid": band_mid,
        "band_energy_high": band_high,
        "spectral_flatness": flatness,
    }

    return spec, feats


def encode_signal(
    normalized_samples: np.ndarray,
    fs: float,
    gain: float = 1.0,
    channels: Iterable[str] | None = None,
) -> list[dict]:
    hop_samples = max(
        1,
        round(fs * HOP_MS / 1000.0),
    )

    selected = (
        list(channels) if channels is not None else list(CHANNEL_EXTRACTORS.keys())
    )

    unknown = set(selected) - set(CHANNEL_EXTRACTORS)

    if unknown:
        raise ValueError(f"Nieznane kanały: {sorted(unknown)}")

    raw = simulate_adc_raw(
        normalized_samples,
        gain=gain,
    )

    x = remove_dc(raw)
    ax = np.abs(x)

    signs = np.sign(x)
    signs[signs == 0.0] = 1.0

    crossings = signs[1:] != signs[:-1]

    n_frames = len(x) // hop_samples

    state = EncoderState()
    rows: list[dict] = []

    for f in range(n_frames):
        i0 = f * hop_samples
        i1 = (f + 1) * hop_samples

        x_f = x[i0:i1]
        ax_f = ax[i0:i1]

        peak = float(ax_f.max())
        mean_abs = float(ax_f.mean())

        mean_sq = float((x_f**2).mean())

        var_abs = max(
            0.0,
            mean_sq - mean_abs**2,
        )

        rms = math.sqrt(mean_sq)

        zc = int(np.count_nonzero(crossings[max(i0 - 1, 0) : max(i1 - 1, 0)]))

        peak_cnt = int(np.count_nonzero(ax_f > state.spike_thr))

        if len(x_f) > 1:
            dx = np.diff(x_f)
            var_dx = float(np.var(dx))
            num_ac = float(np.sum(x_f[:-1] * x_f[1:]))
            den_ac = float(np.sum(x_f**2)) + 1e-6
        else:
            var_dx = 0.0
            num_ac = 0.0
            den_ac = 1e-6

        mean_4 = float(np.mean(x_f**4))

        spec, spec_feats = _frame_spectrum(
            x_f,
            fs,
        )

        if state.prev_spectrum is not None and len(state.prev_spectrum) == len(spec):
            spectral_flux = float(
                np.sum(
                    np.maximum(
                        spec - state.prev_spectrum,
                        0.0,
                    )
                )
            )
        else:
            spectral_flux = 0.0

        state.prev_spectrum = spec

        stats = {
            "peak": peak,
            "mean_abs": mean_abs,
            "var_abs": var_abs,
            "rms": rms,
            "rms_prev": state.rms_prev,
            "zc": zc,
            "n": hop_samples,
            "peak_cnt": peak_cnt,
            "spectral_flux": spectral_flux,
            "var_dx": var_dx,
            "num_ac": num_ac,
            "den_ac": den_ac,
            "mean_4": mean_4,
            "frame": x_f,
            **spec_feats,
        }

        row = {
            "timestamp_ms": round(
                f * HOP_MS,
                2,
            )
        }

        for name in selected:
            row[name] = round(
                CHANNEL_EXTRACTORS[name](stats),
                6,
            )

        if not state.floors_primed:
            row["_priming"] = True

        rows.append(row)

        if not state.floors_primed:
            state.floor_peak = peak
            state.mad_peak = 0.1 * abs(peak) + 1e-6

            state.frame_idx += 1

            if state.frame_idx > PRIME_FRAMES:
                state.floors_primed = True

        else:
            _update_peak_floor(
                state,
                peak,
            )

            state.frame_idx += 1

        state.spike_thr = float(
            np.clip(
                3.0 * (state.floor_peak + 1e-6),
                8.0,
                1023.0,
            )
        )

        state.rms_prev = rms

    return rows


def _read_wav_normalized(
    file_path: str,
) -> tuple[np.ndarray, int]:
    with wave.open(file_path, "rb") as w:
        fs = w.getframerate()
        n_channels = w.getnchannels()
        n_samples = w.getnframes()
        sampwidth = w.getsampwidth()
        raw_bytes = w.readframes(n_samples)

    if sampwidth == 2:
        data = (
            np.frombuffer(
                raw_bytes,
                dtype=np.int16,
            ).astype(np.float32)
            / 32768.0
        )

    elif sampwidth == 4:
        data = (
            np.frombuffer(
                raw_bytes,
                dtype=np.int32,
            ).astype(np.float32)
            / 2147483648.0
        )

    else:
        data = (
            np.frombuffer(
                raw_bytes,
                dtype=np.uint8,
            ).astype(np.float32)
            - 128.0
        ) / 128.0

    if n_channels > 1:
        data = data[::n_channels]

    return data, fs


def encode_wav_file(
    file_path: str,
    channels: Iterable[str] | None = None,
) -> list[dict]:
    data, fs = _read_wav_normalized(file_path)

    return encode_signal(
        data,
        fs,
        channels=channels,
    )
