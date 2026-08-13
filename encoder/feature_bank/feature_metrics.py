"""Implementacje cech używane przez cyfrowego bliźniaka enkodera.

Każda funkcja jest osobno importowalna. Funkcje otrzymują słownik statystyk
jednej ramki; dla cech wymagających próbek ramki używane jest pole ``frame``.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

EPS = 1e-6


def peak(s: Mapping[str, object]) -> float:
    return float(s["peak"])


def peak_cnt(s: Mapping[str, object]) -> float:
    return float(s["peak_cnt"])


def crest(s: Mapping[str, object]) -> float:
    return float(s["peak"]) / (float(s["rms"]) + EPS)


def cv(s: Mapping[str, object]) -> float:
    return math.sqrt(float(s["var_abs"])) / (float(s["mean_abs"]) + EPS)


def zcr(s: Mapping[str, object]) -> float:
    return float(s["zc"]) / float(s["n"])


def flux(s: Mapping[str, object]) -> float:
    return max(
        0.0,
        math.log(float(s["rms"]) + 1.0) - math.log(float(s["rms_prev"]) + 1.0),
    )


def hjorth_mobility(s: Mapping[str, object]) -> float:
    return math.sqrt(float(s["var_dx"]) / (float(s["var_abs"]) + EPS))


def tkeo_mean(s: Mapping[str, object]) -> float:
    x = np.asarray(s["frame"], dtype=np.float64)
    if len(x) <= 2:
        return 0.0
    return float(np.mean((x[1:-1] ** 2) - (x[:-2] * x[2:])))


def curve_length(s: Mapping[str, object]) -> float:
    x = np.asarray(s["frame"], dtype=np.float64)
    if len(x) <= 1:
        return 0.0
    return float(np.sum(np.abs(np.diff(x))))


def autocorr_lag1(s: Mapping[str, object]) -> float:
    # Dokładnie jak w poprzedniej wersji enkodera: EPS jest tylko w mianowniku.
    return float(s["num_ac"]) / float(s["den_ac"])


def kurtosis(s: Mapping[str, object]) -> float:
    return float(s["mean_4"]) / (float(s["var_abs"]) ** 2 + EPS)


def spectral_centroid(s: Mapping[str, object]) -> float:
    return float(s["spectral_centroid"])


def dominant_freq(s: Mapping[str, object]) -> float:
    return float(s["dominant_freq"])


def band_energy_low(s: Mapping[str, object]) -> float:
    return float(s["band_energy_low"])


def band_energy_mid(s: Mapping[str, object]) -> float:
    return float(s["band_energy_mid"])


def band_energy_high(s: Mapping[str, object]) -> float:
    return float(s["band_energy_high"])


def spectral_flatness(s: Mapping[str, object]) -> float:
    return float(s["spectral_flatness"])


def spectral_flux(s: Mapping[str, object]) -> float:
    return float(s["spectral_flux"])


FEATURE_FUNCTIONS = {
    "peak": peak,
    "peak_cnt": peak_cnt,
    "crest": crest,
    "cv": cv,
    "zcr": zcr,
    "flux": flux,
    "hjorth_mobility": hjorth_mobility,
    "tkeo_mean": tkeo_mean,
    "curve_length": curve_length,
    "autocorr_lag1": autocorr_lag1,
    "kurtosis": kurtosis,
    "spectral_centroid": spectral_centroid,
    "dominant_freq": dominant_freq,
    "band_energy_low": band_energy_low,
    "band_energy_mid": band_energy_mid,
    "band_energy_high": band_energy_high,
    "spectral_flatness": spectral_flatness,
    "spectral_flux": spectral_flux,
}
