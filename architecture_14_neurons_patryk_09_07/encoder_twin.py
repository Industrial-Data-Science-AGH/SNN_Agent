#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
encoder_twin.py — cyfrowy bliźniak firmware'u encoder_v2.ino (Delta Spike / Wake-Up AI).

Odtwarza tę samą logikę DSP co Arduino (akumulacja cech w ISR + adaptacyjny próg
z-score + refrakcja), na plikach .wav zamiast na żywym mikrofonie, i zapisuje wynik
w formacie CSV (frame,s0..s5,label) oczekiwanym przez SpikeClips w snn_hw_pipeline.py.

Wierność wobec encoder_v2.ino — i gdzie świadomie odstępujemy:
  - Audio jest resamplowane do FS_HZ = 19231 Hz, więc HOP_SAMPLES = 192 nadal
    odpowiada dokładnie 10 ms. To jest wybrane rozwiązanie problemu niezgodności
    częstotliwości próbkowania (nagrania są 44.1 kHz, Arduino ADC ~19.231 kHz):
    resampling zamiast przeliczania długości ramki w próbkach, bo dzięki temu
    WSZYSTKIE stałe firmware'u (THR_Z, A_UP/A_DN/A_MAD, spike_thr, HOP_SAMPLES)
    przenoszą się 1:1 bez przeliczeń na inne jednostki.
  - Sygnał jest rekwantowany do syntetycznych 10-bitowych kodów ADC (0..1023,
    DC bias ~511.5 — mikrofon spolaryzowany na Vcc/2), żeby stałe `spike_thr`
    (clamp 8..1023) i wartość startowa DC (512<<4) miały sens w tych samych
    jednostkach co firmware.
  - Arytmetyka jest w floatach, nie w stałoprzecinkowej Q4/int16 z ATmega328P —
    to różnica w implementacji (ograniczenia MCU), nie w modelu DSP, więc
    świadomie jej nie odtwarzamy.
  - Kolumna `frame` w CSV to lokalny indeks ramki w obrębie jednego klipu
    (liczony od 0 po fazie priming), a nie bezwzględny licznik uptime'u Arduino —
    SpikeClips czyta tylko kolejność ramek, nie ich bezwzględny numer.

WAŻNE — dlaczego stan enkodera NIE jest resetowany per plik:
  Pierwsza wersja tego skryptu resetowała EncoderState przy każdym 3-sekundowym
  klipie (jak power-on-reset Arduino). Zmierzone na realnych danych
  (notebooks/dataset/glass vs negative): dawało to niemal identyczny odsetek
  ramek ze spikiem dla glass (~25%) i negative (~27%) — enkoder nie
  rozróżniał klas. Przyczyna: 0.5 s primingu (PRIME_FRAMES=50) to za mało,
  żeby floor/MAD się ustabilizowały — stała czasowa wzrostu floora (A_UP)
  odpowiada ~6.7 s, więc w 3-sekundowym klipie floor prawie nigdy nie dogania
  prawdziwego poziomu tła. Do tego mechanizm "zamrożenia floora w trakcie
  zdarzenia" utrwala błędny (zbyt niski) floor, bo każda ramka powyżej niego
  wygląda jak trwające zdarzenie.
  W realnym wdrożeniu enkoder działa CIĄGLE — floor kalibruje się raz, przez
  minuty pracy, nie resetuje się między nagraniami. Dlatego `build_dataset`
  rozgrzewa JEDEN wspólny EncoderState na realnym tle (`--warmup-seconds`,
  domyślnie 30 s) i utrzymuje go bez resetu przez cały zbiór (negatywy +
  pozytywy), tak jak ciągła praca urządzenia. Koszt: pierwsze 1-2 ramki na
  granicy między plikami mogą dziedziczyć odrobinę stanu z poprzedniego klipu
  (np. rms_prev) — to nieuniknione przy sklejaniu niezależnie nagranych
  klipów w jeden strumień, i jest dużo mniejszym zniekształceniem niż reset
  co 3 sekundy.

Użycie:
    python encoder_twin.py build --glass ../notebooks/dataset/glass \
        --negative ../notebooks/dataset/negative --out ./spikes_csv
    python encoder_twin.py preview sciezka/do/pliku.wav
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple
import csv as _csv
import copy, random

import librosa
import numpy as np
from scipy.signal import lfilter

# =============================================================================
# STAŁE — skopiowane 1:1 z encoder_v2.ino
# =============================================================================
FS_HZ = 19231            # rzeczywiste fs przy prescalerze ADC = 32
HOP_SAMPLES = 192        # 192 / 19231 Hz ~= 10.0 ms  <- dt sieci
N_CH = 7
# v3: `crest` (martwa w treningu 14-neur.) zastąpiona dwiema cechami WIDMOWYMI.
# hf_ratio = udział energii pasma górnego (>~2.2 kHz) w energii ramki — rozróżnia
# szkło (energia 4-10 kHz) od głośnych nie-szkieł (łomot/strzał, niskopasmowe).
#
# WAŻNE: hf_ratio jest kodowana progiem BEZWZGLĘDNYM, nie adaptacyjnym z-score.
# Adaptacyjny floor reaguje na ZMIANĘ względem tła kanału, a hf_ratio to POZIOM
# (opis kształtu widma), nie transient — floor adaptowałby się do trwale wysokiego
# HF szkła i kanał przestawał strzelać (zmierzone: hf_ratio-z-score strzelał
# RZADZIEJ dla szkła niż dla negatywów, sygnał odwrócony). Próg bezwzględny na
# poziomie ~0.28/0.35 rozdziela wprost (szkło 58/41% ramek, esc50/quiet/voice ~0-8%).
# Dwa progi = kod termometrowy: hf_hi to mocny dowód na szkło.
CHANNELS = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]
CH_PEAK, CH_PEAKCNT, CH_CV, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI = range(N_CH)

# progi z-score (adaptacyjny floor) — używane tylko dla kanałów bez progu bezwzgl.
THR_Z = np.array([4.0, 3.5, 3.0, 2.5, 3.5, np.nan, np.nan])
# progi bezwzględne (nan = kanał używa z-score). hf_lo czuły, hf_hi specyficzny.
ABS_THR = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.28, 0.35])

A_UP, A_DN, A_MAD = 0.0015, 0.0300, 0.0100
EPS = 1e-6   # ogólny epsilon (dzielenia poza z-score: hf_ratio, cv, spike_thr, bramka HF)

# EPS PER KANAŁ dla mianownika z-score w _update_floor — każdy kanał ma inną
# jednostkę, więc jeden wspólny EPS=1e-6 kolapsuje przy cyfrowej ciszy (mad->0,
# mianownik->EPS, dowolne drgnięcie = spike). Wartości to przybliżona
# rozdzielczość kwantyzacji KAŻDEGO kanału, żeby EPS odpowiadał "1 LSB ADC":
#   peak      : 1 LSB kodu ADC wprost                    -> 1.0
#   peak_cnt  : rozdzielczość to 1 próbka/ramkę (int)     -> 1.0
#   cv, zcr   : ułamek jednej próbki na HOP_SAMPLES próbek statystyki ramki -> 1/HOP_SAMPLES
#   flux      : rozdzielczość log1p(rms) przy najmniejszej mierzalnej zmianie
#               rms (rząd 1 LSB / sqrt(HOP_SAMPLES), tłumione przez log1p)
EPS_FLOOR = np.array([
    1.0,                                   # peak
    1.0,                                   # peak_cnt
    1.0 / HOP_SAMPLES,                     # cv
    1.0 / HOP_SAMPLES,                     # zcr
    1.0 / (1.0 + HOP_SAMPLES ** 0.5),       # flux
    1.0, 1.0,                        # hf_lo/hf_hi: próg bezwzględny, nieużywane tu bo kanał na progu bezwzględnym
])

REFRAC_FRAMES = 1
PRIME_FRAMES = 50        # ~0.5 s przy 100 Hz ramek

# stały offset per split zamiast hash(split) — hash() na stringach jest solony
# per-proces (PYTHONHASHSEED), więc dawało inny RNG przy każdym uruchomieniu
# mimo tego samego seed. Determinizm wymaga stałej, jawnej mapy.
SPLIT_OFFSET = {"train": 0, "val": 1, "test": 2}

# 1-pole highpass pasma górnego: lp += (x-lp)>>HF_HP_SHIFT, hf = x - lp.
# SHIFT=1 => a=0.5 => cutoff ~2.2 kHz @19231 Hz. Kaskada nie poprawiała separacji
# (AUC 0.730 vs 0.739), więc zostaje pojedynczy stopień — najtańszy w firmware.
HF_HP_SHIFT = 1
# hf_ratio na ciszy = szum (dzielenie przez ~0), więc kanały widmowe strzelają
# tylko gdy ramka jest ZDARZENIEM: peak powyżej HF_GATE_MULT * floor kanału peak.
HF_GATE_MULT = 1.5

# --- emulacja ADC 10-bit (patrz uwagi o wierności powyżej) ---
ADC_FULL_SCALE = 1023.0
ADC_BIAS = 511.5
DC_EMA_SHIFT = 9          # dc_est += (raw<<4 - dc_est) >> 9  =>  k = 1/512
SPIKE_THR_INIT = 40.0
SPIKE_THR_MIN, SPIKE_THR_MAX = 8.0, ADC_FULL_SCALE

GAIN_PERCENTILE = 99.9   # percentyl amplitudy trafiający w pełną skalę ADC
GAIN_METHOD = "all-files"   # "all-files": percentyl z połączonego rozkładu próbek
                            # (poprawny odpowiednik docstringu). "per-file": percentyl
                            # z listy maksimów/percentyli per plik — używać z niższym
                            # percentylem (mediana albo ~95), bo przy percentylu 99.9
                            # i setkach plików to praktycznie maksimum z peaks, czyli
                            # jeden najgłośniejszy plik dyktuje wzmocnienie całego zbioru.


def compute_global_gain(paths, percentile: float = GAIN_PERCENTILE,
                        method: str = GAIN_METHOD,
                        fs_hz: int = FS_HZ) -> float:
    """Liczy JEDNO globalne wzmocnienie z listy plików (percentyl amplitudy
    -> pełna skala ADC). WYWOŁYWAĆ WYŁĄCZNIE na zbiorze TRENINGOWYM i zamrozić
    wynik (np. zapisać do JSON obok datasetu) — patrz build_manifest().
    Zastępuje dawną normalizację `y = y / peak` per plik, która chowała
    poziom bezwzględny (przesłankę niedostępną na płytce: sprzęt nie ma AGC).

    `method="all-files"` (domyślny): zbiera |y| ze WSZYSTKICH plików do jednego
    połączonego rozkładu próbek i liczy z niego jeden globalny percentyl —
    to jest dosłowne odwzorowanie „percentyl amplitudy trafia w pełną skalę
    ADC" z docstringu, bo percentyl liczony jest na rozkładzie próbek, nie
    maksimów. Kosztowniejsze pamięciowo (trzyma wszystkie próbki naraz).

    `method="per-file"`: liczy percentyl `percentile` osobno dla każdego pliku,
    potem bierze `percentile` z tej listy per-plikowych wartości — czyli
    percentyl z percentyli. Tańsze (jedna liczba na plik), ale przy wysokim
    percentyle (99.9) i wielu plikach efektywnie wybiera najgłośniejszy plik
    ze zbioru. Jeśli używasz tego wariantu, ustaw niższy `percentile`
    (mediana = 50, albo ~95), NIE 99.9."""
    if method not in ("all-files", "per-file"):
        raise ValueError(f"nieznana metoda gain: {method!r} (all-files/per-file)")
GAIN_PERCENTILE = 99.9   # percentyl amplitudy trafiający w pełną skalę ADC
GAIN_METHOD = "all-files"   # "all-files": percentyl z połączonego rozkładu próbek
                            # (poprawny odpowiednik docstringu). "per-file": percentyl
                            # z listy maksimów/percentyli per plik — używać z niższym
                            # percentylem (mediana albo ~95), bo przy percentylu 99.9
                            # i setkach plików to praktycznie maksimum z peaks, czyli
                            # jeden najgłośniejszy plik dyktuje wzmocnienie całego zbioru.


def compute_global_gain(paths, percentile: float = GAIN_PERCENTILE,
                        method: str = GAIN_METHOD,
                        fs_hz: int = FS_HZ) -> float:
    """Liczy JEDNO globalne wzmocnienie z listy plików (percentyl amplitudy
    -> pełna skala ADC). WYWOŁYWAĆ WYŁĄCZNIE na zbiorze TRENINGOWYM i zamrozić
    wynik (np. zapisać do JSON obok datasetu) — patrz build_manifest().
    Zastępuje dawną normalizację `y = y / peak` per plik, która chowała
    poziom bezwzględny (przesłankę niedostępną na płytce: sprzęt nie ma AGC).

    `method="all-files"` (domyślny): zbiera |y| ze WSZYSTKICH plików do jednego
    połączonego rozkładu próbek i liczy z niego jeden globalny percentyl —
    to jest dosłowne odwzorowanie „percentyl amplitudy trafia w pełną skalę
    ADC" z docstringu, bo percentyl liczony jest na rozkładzie próbek, nie
    maksimów. Kosztowniejsze pamięciowo (trzyma wszystkie próbki naraz).

    `method="per-file"`: liczy percentyl `percentile` osobno dla każdego pliku,
    potem bierze `percentile` z tej listy per-plikowych wartości — czyli
    percentyl z percentyli. Tańsze (jedna liczba na plik), ale przy wysokim
    percentyle (99.9) i wielu plikach efektywnie wybiera najgłośniejszy plik
    ze zbioru. Jeśli używasz tego wariantu, ustaw niższy `percentile`
    (mediana = 50, albo ~95), NIE 99.9."""
    if method not in ("all-files", "per-file"):
        raise ValueError(f"nieznana metoda gain: {method!r} (all-files/per-file)")

    if method == "all-files":
        chunks = []
        for p in paths:
            y, _ = librosa.load(p, sr=fs_hz, mono=True)
            if len(y):
                chunks.append(np.abs(y))
        if not chunks:
            return 1.0
        all_abs = np.concatenate(chunks)
        ref = float(np.percentile(all_abs, percentile))
    else:
        peaks = []
        for p in paths:
            y, _ = librosa.load(p, sr=fs_hz, mono=True)
            if len(y):
                peaks.append(np.percentile(np.abs(y), percentile))
        ref = float(np.percentile(peaks, percentile)) if peaks else 1.0
    return 1.0 / max(ref, 1e-9)


def wav_to_adc_codes(path: str, gain: float, fs_hz: int = FS_HZ,
                     aug_gain_db: float = 0.0, rng=None) -> np.ndarray:
    """Wczytuje audio i odtwarza to, co widziałby ADC Arduino: resample do FS_HZ
    + rekwantyzacja do 10-bit kodów wyśrodkowanych na ADC_BIAS.

    `gain`: globalne, ZAMROŻONE wzmocnienie z compute_global_gain() (liczone
    raz na train) — zastępuje normalizację per-plik, więc poziom bezwzględny
    (głośność) zostaje zachowany jako przesłanka, tak jak widzi go sprzęt.
    `aug_gain_db`: augmentacja treningowa — losowe dodatkowe wzmocnienie w
    zakresie ±aug_gain_db dB, stosowane PRZED rekwantyzacją do kodów ADC (żeby
    modelować rozrzut poziomu wejściowego, a nie chować go jak stara normalizacja).
    Zostaw 0.0 dla val/test/produkcji."""
    y, _ = librosa.load(path, sr=fs_hz, mono=True)
    y = y * gain
    if aug_gain_db:
        r = rng if rng is not None else np.random
        y = y * (10.0 ** (r.uniform(-aug_gain_db, aug_gain_db) / 20.0))
    codes = ADC_BIAS + y * (ADC_FULL_SCALE / 2.0)
    return np.clip(codes, 0.0, ADC_FULL_SCALE)


def _remove_dc(codes: np.ndarray) -> np.ndarray:
    """Wektorowy odpowiednik EMA usuwania DC z ISR:
        dc_est[n] = dc_est[n-1] + (16*raw[n] - dc_est[n-1]) / 512
    Stała czasowa filtru (~512 próbek @ 19231 Hz ≈ 27 ms) jest krótsza niż
    okno primingu (~500 ms), więc zerowy stan początkowy w lfilter w pełni
    się ustala zanim floory zaczną cokolwiek liczyć — nie trzeba specjalnego
    stanu startowego."""
    k = 1.0 / (1 << DC_EMA_SHIFT)
    dc = lfilter([k], [1.0, -(1.0 - k)], codes)
    return codes - dc


def _high_band(x: np.ndarray) -> np.ndarray:
    """Pasmo górne przez ten sam 1-pole EMA co usuwanie DC, tyle że ostrzejszy:
        lp[n] = lp[n-1] + (x[n] - lp[n-1]) >> HF_HP_SHIFT ;  hf = x - lp
    Wierny odpowiednik jednej dodatkowej linii w ISR firmware. Stan zerowy w
    lfilter ustala się w kilka próbek (a=0.5), więc nie wpływa na cechy po
    primingu."""
    a = 1.0 / (1 << HF_HP_SHIFT)
    lp = lfilter([a], [1.0, -(1.0 - a)], x)
    return x - lp


@dataclass
class EncoderState:
    """Stan enkodera przenoszony między ramkami — odpowiednik zmiennych
    `static`/`volatile` w encoder_v2.ino."""

    floor_v: np.ndarray = field(default_factory=lambda: np.zeros(N_CH))
    mad_v: np.ndarray = field(default_factory=lambda: np.zeros(N_CH))
    refrac: np.ndarray = field(default_factory=lambda: np.zeros(N_CH, dtype=int))
    rms_prev: float = 0.0
    hf_rms_prev: float = 0.0
    spike_thr: float = SPIKE_THR_INIT
    floors_primed: bool = False
    n_seen: int = 0


def _update_floor(state: EncoderState, c: int, v: float) -> float:
    """Odpowiednik updateFloor(): asymetryczna adaptacja (rośnie wolno A_UP,
    spada szybko A_DN), potem MAD, zwraca z-score."""
    a = A_UP if v > state.floor_v[c] else A_DN
    state.floor_v[c] += a * (v - state.floor_v[c])
    d = abs(v - state.floor_v[c])
    state.mad_v[c] += A_MAD * (d - state.mad_v[c])
    return (v - state.floor_v[c]) / (state.mad_v[c] + EPS_FLOOR[c])


def encode_file(path: str, gain: float, state: Optional[EncoderState] = None,
                aug_gain_db: float = 0.0, rng=None) -> np.ndarray:
    """Zwraca macierz [n_frames_po_primingu, 7] spike'ów (0/1) dla jednego pliku
    audio — dokładnie to, co encoder_v2.ino wypisałby na Serial jako s0..s6."""
    codes = wav_to_adc_codes(path, gain=gain, aug_gain_db=aug_gain_db, rng=rng)
    x = _remove_dc(codes)
    hf = _high_band(x)                       # pasmo górne (~>2.2 kHz) — cechy widmowe
    ax = np.abs(x)
    sign = np.where(x >= 0, 1, -1)
    sign_ext = np.concatenate(([0], sign))  # prev_sign startuje od 0, jak w .ino
    crossings = (sign_ext[1:] != sign_ext[:-1]).astype(np.int32)

    n_frames = len(x) // HOP_SAMPLES
    if n_frames == 0:
        return np.zeros((0, N_CH), dtype=np.uint8)

    x_f = x[: n_frames * HOP_SAMPLES].reshape(n_frames, HOP_SAMPLES)
    ax_f = ax[: n_frames * HOP_SAMPLES].reshape(n_frames, HOP_SAMPLES)
    zc_f = crossings[: n_frames * HOP_SAMPLES].reshape(n_frames, HOP_SAMPLES)
    hf_f = hf[: n_frames * HOP_SAMPLES].reshape(n_frames, HOP_SAMPLES)

    # Cechy niezależne od progu (spike_thr wpływa tylko na peak_cnt) — w pełni
    # zwektoryzowane, licz raz dla całego pliku.
    acc_abs = ax_f.sum(axis=1)
    acc_sq = (x_f ** 2).sum(axis=1)
    acc_max = ax_f.max(axis=1)
    acc_zc = zc_f.sum(axis=1)
    acc_hf_sq = (hf_f ** 2).sum(axis=1)      # energia pasma górnego w ramce

    n = float(HOP_SAMPLES)
    mean_abs = acc_abs / n
    rms = np.sqrt(acc_sq / n)
    peak = acc_max
    var_abs = np.maximum(0.0, acc_sq / n - mean_abs ** 2)
    cv = np.sqrt(var_abs) / (mean_abs + EPS)
    zcr = acc_zc / n
    hf_ratio = acc_hf_sq / (acc_sq + EPS)    # udział energii HF (szkło >> łomot/strzał)

    if state is None:
        state = EncoderState()

    out_rows = []
    for k in range(n_frames):
        # peak_cnt zależy od spike_thr ustawionego na końcu POPRZEDNIEJ ramki —
        # to jedyna cecha, której nie dało się policzyć z góry.
        peak_cnt = float(np.count_nonzero(ax_f[k] > state.spike_thr))

        lr = np.log1p(rms[k])
        lrp = np.log1p(state.rms_prev)
        flux = max(0.0, lr - lrp)
        state.rms_prev = rms[k]

        # próg mikro-szpilki na NASTĘPNĄ ramkę — 3x poziom tła kanału `peak`
        state.spike_thr = float(np.clip(3.0 * (state.floor_v[CH_PEAK] + EPS),
                                         SPIKE_THR_MIN, SPIKE_THR_MAX))

        # kanały hf_lo/hf_hi dostają tę samą wartość hf_ratio (różnią się progiem)
        feat = np.array([peak[k], peak_cnt, cv[k], zcr[k], flux,
                         hf_ratio[k], hf_ratio[k]])

        if not state.floors_primed:
            state.floor_v[:] = feat
            state.mad_v[:] = 0.1 * np.abs(feat) + EPS
            if state.n_seen > PRIME_FRAMES:
                state.floors_primed = True
            state.n_seen += 1
            continue
        state.n_seen += 1

        # bramka zdarzenia dla kanałów widmowych: ramka musi mieć transient
        # (peak nad floorem), inaczej hf_ratio na ciszy jest szumem z dzielenia
        hf_gated = feat[CH_PEAK] > HF_GATE_MULT * (state.floor_v[CH_PEAK] + EPS)

        bits = np.zeros(N_CH, dtype=np.uint8)
        for c in range(N_CH):
            if not np.isnan(ABS_THR[c]):
                above = hf_gated and (feat[c] > ABS_THR[c])   # próg bezwzględny
            else:
                above = _update_floor(state, c, feat[c]) > THR_Z[c]
                if above:
                    # zamrożenie adaptacji floora w trakcie zdarzenia (encoder nie
                    # ma się "przyzwyczajać" do szkła jako nowego tła)
                    state.floor_v[c] -= A_UP * (feat[c] - state.floor_v[c])
            if above and state.refrac[c] == 0:
                bits[c] = 1
                state.refrac[c] = REFRAC_FRAMES
            elif state.refrac[c] > 0:
                state.refrac[c] -= 1

        out_rows.append(bits)

    return np.array(out_rows, dtype=np.uint8) if out_rows else np.zeros((0, N_CH), dtype=np.uint8)


# =============================================================================
# BUDOWA DATASETU CSV — wejście dla snn_hw_pipeline.py train --data ...
# =============================================================================

def _warmup_state(negative_files: list, warmup_seconds: float,
                  gain: float) -> Tuple[EncoderState, int]:
    """Rozgrzewa jeden EncoderState na kolejnych plikach tła, aż zbierze co
    najmniej `warmup_seconds` sekund audio PO fazie primingu (floors_primed).
    Wyjście tych plików jest odrzucane — służą wyłącznie do ustabilizowania
    floor/MAD, tak jak pierwsze minuty pracy realnego urządzenia.

    Zwraca (state, liczba_zużytych_plików) — te pliki NIE są potem powtórnie
    używane jako przykłady treningowe, żeby nie wyciekły do dwóch ról naraz.
    """
    state = EncoderState()
    frame_dt = HOP_SAMPLES / FS_HZ
    warmed_frames = 0
    used = 0
    for f in negative_files:
        spikes = encode_file(f, gain=gain, state=state)
        used += 1
        warmed_frames += spikes.shape[0]
        if state.floors_primed and warmed_frames * frame_dt >= warmup_seconds:
            break
    else:
        print(f"[!] warmup: zabrakło plików tła, rozgrzano tylko "
              f"{warmed_frames * frame_dt:.1f}s zamiast {warmup_seconds:.1f}s")
    print(f"[warmup] rozgrzano stan enkodera na {used} plikach tła "
          f"({warmed_frames * frame_dt:.1f}s realnego audio, odrzucone)")
    return state, used


def _glob_dirs(dirs) -> list:
    dirs = [dirs] if isinstance(dirs, str) else list(dirs)
    files = []
    for d in dirs:
        found = sorted(glob.glob(os.path.join(d, "**", "*.wav"), recursive=True))
        if not found:
            print(f"[!] brak plików .wav w {d}")
        files.extend(found)
    return files


def build_dataset(glass_dirs, negative_dirs, out_dir: str,
                   warmup_seconds: float = 30.0, warmup_dir: Optional[str] = None) -> None:
    """Buduje CSV-y dla SpikeClips, utrzymując JEDEN ciągły EncoderState przez
    cały zbiór (patrz uwaga na górze pliku — brak resetu per plik, tak jak
    ciągła praca realnego urządzenia).

    `glass_dirs`/`negative_dirs` mogą być pojedynczą ścieżką albo listą ścieżek
    (np. żeby połączyć oryginalne nagrania z klipami wyciętymi przez
    voice_extract.py). `warmup_dir` (domyślnie: pierwszy z negative_dirs)
    MUSI wskazywać na prawdziwie stacjonarne tło — krótkie wycinki "trudnych
    negatywów" (gunshot/babycry) nie nadają się do rozgrzewania floora, bo same
    są zdarzeniami, a nie ciszą/tłem."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    glass_files = _glob_dirs(glass_dirs)
    negative_files = _glob_dirs(negative_dirs)
    if not glass_files or not negative_files:
        return

    gain = compute_global_gain(glass_files + negative_files)
    print(f"[gain] globalne wzmocnienie {gain:.4f}", flush=True)

    warmup_dir = warmup_dir or (negative_dirs[0] if isinstance(negative_dirs, list) else negative_dirs)
    warmup_files = sorted(glob.glob(os.path.join(warmup_dir, "**", "*.wav"), recursive=True))

    state, n_warmup_used = _warmup_state(warmup_files, warmup_seconds, gain=gain)
    used_for_warmup = set(warmup_files[:n_warmup_used])
    remaining_negative = [f for f in negative_files if f not in used_for_warmup]

    # Przeplatamy klasy w kolejności zbliżonej do realnego wdrożenia: głównie
    # tło, z rzadka przerywane zdarzeniem szkła — enkoder cały czas widzi ten
    # sam, kontynuowany stan.
    tasks: list = [(0, f) for f in remaining_negative] + [(1, f) for f in glass_files]

    counts = {0: 0, 1: 0}
    for i, (label, f) in enumerate(tasks):
        spikes = encode_file(f, gain=gain, state=state)
        if spikes.shape[0] == 0:
            print(f"[!] {f}: plik za krótki na choćby jedną ramkę — pomijam")
            continue

        stem = Path(f).stem[:40]
        tag = "glass" if label else "negative"
        out_path = out / f"{tag}_{counts[label]:04d}_{stem}.csv"
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("frame," + ",".join(f"s{c}" for c in range(N_CH)) + ",label\n")
            for frame_idx, row in enumerate(spikes):
                fh.write(f"{frame_idx}," + ",".join(str(int(v)) for v in row) + f",{label}\n")
        counts[label] += 1

    print(f"[ok] glass: {counts[1]}/{len(glass_files)} plików -> CSV w {out_dir}")
    print(f"[ok] negative: {counts[0]}/{len(remaining_negative)} plików "
          f"(+ {n_warmup_used} zużytych na warmup) -> CSV w {out_dir}")


def _load_or_compute_gain(train_paths, out_dir: str, percentile: float,
                          method: str, gain_file: Optional[str] = None) -> float:
    """Czyta zamrożone globalne wzmocnienie z JSON, jeśli plik istnieje i
    zgadza się percentile+n_files (i metoda) — inaczej liczy je od nowa
    (jeden librosa.load na plik train) i zapisuje. Unika podwójnego liczenia
    przy kolejnych uruchomieniach na tym samym zbiorze, i sprawia, że dobudowa
    kolejnej wersji datasetu nie dostaje po cichu innego wzmocnienia."""
    gain_path = Path(gain_file) if gain_file else Path(out_dir) / "global_gain.json"
    if gain_path.exists():
        try:
            cached = json.load(open(gain_path))
            if (cached.get("percentile") == percentile
                    and cached.get("method") == method
                    and cached.get("n_files") == len(train_paths)):
                print(f"[gain] wczytano zamrożone wzmocnienie {cached['gain']:.4f} "
                      f"z {gain_path} (percentyl {percentile}, metoda {method})",
                      flush=True)
                return float(cached["gain"])
            print(f"[gain] {gain_path} istnieje, ale percentile/method/n_files się "
                  f"nie zgadzają — liczę od nowa", flush=True)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[gain] nie udało się odczytać {gain_path} ({e}) — liczę od nowa",
                  flush=True)

    gain = compute_global_gain(train_paths, percentile=percentile, method=method)
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"gain": gain, "percentile": percentile, "method": method,
              "computed_on": "split=train", "n_files": len(train_paths)},
              open(gain_path, "w"), indent=2)
    print(f"[gain] globalne wzmocnienie {gain:.4f} (z {len(train_paths)} "
          f"plików train, percentyl {percentile}, metoda {method}) -> {gain_path}",
          flush=True)
    return gain


def _sha256_file(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _assert_groups_disjoint(rows: list) -> None:
    """Przerywa budowe, jesli jakas grupa zrodlowa jest w wiecej niz jednym splicie.

    To jest ten test, ktorego brak kosztowal trzy kampanie: w spikes_manifest7
    194 z 194 miksow VOICe obecnych w tescie byly rowniez w treningu, wiec kazda
    raportowana metryka byla liczona na przecieku. Lepiej nie zbudowac artefaktu
    niz zbudowac taki, ktorego wyniku nie da sie uzyc.
    """
    if not rows or "group_id" not in rows[0]:
        raise SystemExit(
            "[BLAD] manifest nie ma kolumny group_id, nie da sie sprawdzic "
            "rozlacznosci splitow. Uzyj dataset/versions/<wersja>/manifest.csv")
    splits_of: dict = {}
    for r in rows:
        splits_of.setdefault(r["group_id"], set()).add(r["split"])
    bad = {g: s for g, s in splits_of.items() if len(s) > 1}
    if bad:
        przyklad = ", ".join(f"{g}: {sorted(bad[g])}" for g in sorted(bad)[:3])
        raise SystemExit(
            f"[BLAD] przeciek miedzy splitami: {len(bad)} grup zrodlowych wystepuje "
            f"w wiecej niz jednym splicie (np. {przyklad}). Artefakt nie powstal.")
    print(f"[ok] rozlacznosc grup: {len(splits_of)} grup, kazda w jednym splicie")


def _interleave_by_class(split_rows: list) -> list:
    """Przeplata pozytywy z negatywami zamiast sortowac po etykiecie.

    Enkoder utrzymuje JEDEN ciagly stan (floor/MAD) przez caly zbior. Gdy
    wszystkie negatywy ida przed wszystkimi pozytywami, stan enkodera jest
    skorelowany z etykieta i siec moze uczyc sie pozycji w strumieniu zamiast
    dzwieku. Rozkladamy obie klasy rownomiernie: element o randze i w klasie
    o licznosci n dostaje pozycje (i + 0.5) / n.
    """
    pos = sorted((r for r in split_rows if r["label"] == "positive"),
                 key=lambda r: r["abspath"])
    neg = sorted((r for r in split_rows if r["label"] != "positive"),
                 key=lambda r: r["abspath"])
    if not pos or not neg:
        return neg + pos
    keyed = [((i + 0.5) / len(pos), 1, r) for i, r in enumerate(pos)] + \
            [((i + 0.5) / len(neg), 0, r) for i, r in enumerate(neg)]
    keyed.sort(key=lambda kr: (kr[0], kr[1]))
    return [r for _, _, r in keyed]


def _stream_balance(ordered: list) -> str:
    """Srednia pozycja w strumieniu per klasa, obie powinny wyjsc ~0.500."""
    n = len(ordered)
    if n < 2:
        return "n/a"
    poz, neg = [], []
    for i, r in enumerate(ordered):
        (poz if r["label"] == "positive" else neg).append(i / (n - 1))
    fmt = lambda v: f"{sum(v) / len(v):.3f}" if v else "n/a"
    return f"pozytywy {fmt(poz)}, negatywy {fmt(neg)}"


def _infer_version(manifest_path: str) -> Optional[str]:
    """v2.0.0 z dataset/versions/v2.0.0/manifest.csv."""
    parent = Path(manifest_path).resolve().parent
    return parent.name if parent.name.startswith("v") else None


def _load_or_compute_gain(train_paths, out_dir: str, percentile: float,
                          method: str, gain_file: Optional[str] = None) -> float:
    """Czyta zamrożone globalne wzmocnienie z JSON, jeśli plik istnieje i
    zgadza się percentile+n_files (i metoda) — inaczej liczy je od nowa
    (jeden librosa.load na plik train) i zapisuje. Unika podwójnego liczenia
    przy kolejnych uruchomieniach na tym samym zbiorze, i sprawia, że dobudowa
    kolejnej wersji datasetu nie dostaje po cichu innego wzmocnienia."""
    gain_path = Path(gain_file) if gain_file else Path(out_dir) / "global_gain.json"
    if gain_path.exists():
        try:
            cached = json.load(open(gain_path))
            if (cached.get("percentile") == percentile
                    and cached.get("method") == method
                    and cached.get("n_files") == len(train_paths)):
                print(f"[gain] wczytano zamrożone wzmocnienie {cached['gain']:.4f} "
                      f"z {gain_path} (percentyl {percentile}, metoda {method})",
                      flush=True)
                return float(cached["gain"])
            print(f"[gain] {gain_path} istnieje, ale percentile/method/n_files się "
                  f"nie zgadzają — liczę od nowa", flush=True)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"[gain] nie udało się odczytać {gain_path} ({e}) — liczę od nowa",
                  flush=True)

    gain = compute_global_gain(train_paths, percentile=percentile, method=method)
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"gain": gain, "percentile": percentile, "method": method,
              "computed_on": "split=train", "n_files": len(train_paths)},
              open(gain_path, "w"), indent=2)
    print(f"[gain] globalne wzmocnienie {gain:.4f} (z {len(train_paths)} "
          f"plików train, percentyl {percentile}, metoda {method}) -> {gain_path}",
          flush=True)
    return gain


def build_manifest(manifest_path: str, out_dir: str, root: str = ".",
                   warmup_seconds: float = 30.0, seed: int = 0,
                   aug_gain_db: float = 12.0,
                   gain_percentile: float = GAIN_PERCENTILE,
                   gain_method: str = GAIN_METHOD,
                   gain_file: Optional[str] = None) -> None:

    """Buduje CSV-y wg wersjonowanego manifestu (dataset/versions/<ver>/manifest.csv).

    Stan enkodera per split: WSPÓLNY rozgrzany stan bazowy (na tle treningowym)
    jest KOPIOWANY niezależnie dla train/val/test — więc kodowanie pliku w val
    nie zależy od tego, co wcześniej przeszło przez train, ani przez jaki
    kolejny plik w val. W obrębie jednego splitu pliki są przeplatane
    (seedowany shuffle) zamiast sortowane po etykiecie, żeby historia stanu
    nie korelowała z klasą (patrz pomiar w zadaniu: negatywy zawsze przed
    pozytywami psuło floor). To NIE jest pełna niezależność od kolejności
    wewnątrz splitu (stan wciąż akumuluje się między kolejnymi plikami tego
    samego splitu) — tylko usunięcie korelacji stanu z etykietą i determinizm
    przez seed.
    Globalne wzmocnienie (gain) liczone RAZ na negatywach+pozytywach splitu
    train i zamrożone: używane identycznie dla train/val/test. Augmentacja
    losowym wzmocnieniem (domyślnie ±12 dB) stosowana TYLKO na splicie train.

    Zachowuje podział train/val/test z manifestu jako podkatalogi wyjścia, więc
    trener dostaje --data out/train --val-data out/val, czyli split zrobiony raz,
    wspólny dla wszystkich eksperymentów.

    Trzy zabezpieczenia, których wcześniej nie było:
      1. rozłączność `group_id` między splitami sprawdzana TWARDO przed kodowaniem
         (bez tego powstał spikes_manifest7 z przeciekiem 194/194 miksów VOICe),
      2. pozytywy przeplatane z negatywami, żeby ciągły stan enkodera nie był
         skorelowany z etykietą,
      3. warmup floora wyłącznie na tle stacjonarnym (`kind == stationary`),
         bo krótkie zdarzenia (gunshot, dzwony) rozgrzewają floor do złego poziomu.

    Do każdego splitu trafia `channels.json` z pochodzeniem i `files.csv` z listą
    plików źródłowych i ich etykietami, żeby `validate_dataset.py` (K9) mógł
    sprawdzić, czy artefakt nie rozjechał się z manifestem, na który się powołuje.
    """
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            r["abspath"] = os.path.join(root, r["filepath"])
            rows.append(r)
    print(f"[manifest] {len(rows)} plików", flush=True)

    missing = [r for r in rows if not os.path.exists(r["abspath"])]
    if missing:
        print(f"[!] {len(missing)} plików z manifestu nie istnieje, np. "
              f"{missing[0]['abspath']} — pomijam je")
        rows = [r for r in rows if os.path.exists(r["abspath"])]

    # --- globalne wzmocnienie: liczone WYŁĄCZNIE na train, potem zamrożone ---
    train_paths = [r["abspath"] for r in rows if r["split"] == "train"]
    gain = _load_or_compute_gain(train_paths, out_dir, gain_percentile,
                                 gain_method, gain_file=gain_file)

    _assert_groups_disjoint(rows)
    # --- globalne wzmocnienie: liczone WYŁĄCZNIE na train, potem zamrożone ---
    train_paths = [r["abspath"] for r in rows if r["split"] == "train"]
    gain = compute_global_gain(train_paths)
    gain_path = Path(out_dir) / "global_gain.json"
    gain_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(
        {
            "gain": gain,
            "percentile": GAIN_PERCENTILE,
            "computed_on": "split=train",
            "n_files": len(train_paths),
        },
        open(gain_path, "w"),
        indent=2,
    )
    print(
        f"[gain] globalne wzmocnienie {gain:.4f} (z {len(train_paths)} "
        f"plików train, percentyl {GAIN_PERCENTILE}) -> {gain_path}",
        flush=True,
    )
    gain = _load_or_compute_gain(train_paths, out_dir, gain_percentile,
                                 gain_method, gain_file=gain_file)

    version = dataset_version or _infer_version(manifest_path)
    prov = {
        "dataset_version": version,
        "manifest_path": os.path.relpath(os.path.abspath(manifest_path),
                                         os.path.abspath(root)).replace(os.sep, "/"),
        "manifest_sha256": _sha256_file(manifest_path),
        "encoder_file": "encoder_twin.py",
        "encoder_sha256": _sha256_file(os.path.abspath(__file__)),
        "warmup_seconds": warmup_seconds,
        "stream_order": "pozytywy przeplatane z negatywami, stan enkodera ciągły",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[pochodzenie] audio={version}  manifest={prov['manifest_sha256'][:12]}  "
          f"enkoder={(prov['encoder_sha256'] or '')[:12]}", flush=True)

    # Warmup floora tylko na tle stacjonarnym: krótkie zdarzenia nie są ciszą.
    warmup_files = sorted(r["abspath"] for r in rows
            if r["label"] == "negative" and r["split"] == "train"
                          and r.get("kind") == "stationary")
    if not warmup_files:
        print("[!] brak negatywów kind=stationary w train — warmup na dowolnym tle")
        warmup_files = sorted(r["abspath"] for r in rows
                              if r["label"] == "negative" and r["split"] == "train")
    base_state, n_used = _warmup_state(warmup_files, warmup_seconds, gain=gain)
    used_for_warmup = set(warmup_files[:n_used])

    out = Path(out_dir)
    counts: dict = {}
    rng = random.Random(seed)
    written: dict = {"train": [], "val": [], "test": []}
    for split in ("train", "val", "test"):
        (out / split).mkdir(parents=True, exist_ok=True)
        split_rows = [
            r
            for r in rows
            if r["split"] == split and r["abspath"] not in used_for_warmup
        ]
        rng.shuffle(split_rows)  # przeplot klas, deterministyczny (seed) —
        # zamiast sortowania po etykiecie
        split_state = copy.deepcopy(base_state)  # niezależny stan per split
        split_aug = aug_gain_db if split == "train" else 0.0
        split_rng = np.random.default_rng(seed * 10 + SPLIT_OFFSET[split])
        split_rows = _interleave_by_class([r for r in rows if r["split"] == split])
        print(f"[{split}] średnia pozycja w strumieniu: {_stream_balance(split_rows)}",
              flush=True)
        for r in split_rows:
            label = 1 if r["label"] == "positive" else 0
            spikes = encode_file(
                r["abspath"],
                gain=gain,
                state=split_state,
                aug_gain_db=split_aug,
                rng=split_rng,
            )
            if spikes.shape[0] == 0:
                print(f"[!] {r['abspath']}: za krótki na choćby jedną ramkę — pomijam")
                continue
            tag = "glass" if label else "negative"
            key = (split, label)
            counts[key] = counts.get(key, 0)
            stem = Path(r["abspath"]).stem[:40]
            out_path = out / split / f"{tag}_{counts[key]:05d}_{stem}.csv"
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("frame," + ",".join(f"s{c}" for c in range(N_CH)) + ",label\n")
                for frame_idx, row_bits in enumerate(spikes):
                    fh.write(
                        f"{frame_idx},"
                        + ",".join(str(int(v)) for v in row_bits)
                        + f",{label}\n"
                    )
            counts[key] += 1
            written[split].append((r, name))
            done = sum(counts.values())
            if done % 500 == 0:
                print(f"[postęp] {done} plików zakodowanych", flush=True)

        with open(out / split / "channels.json", "w", encoding="utf-8") as fh:
            json.dump({"channels": CHANNELS, "n_hw": N_CH, "provenance": prov},
                      fh, ensure_ascii=False, indent=2)
        with open(out / split / "files.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["filepath", "label", "kind", "source", "group_id", "csv"])
            for r, name in written[split]:
                w.writerow([r["filepath"], r["label"], r.get("kind", ""),
                            r.get("source", ""), r.get("group_id", ""), name])

    for split in ("train", "val", "test"):
        print(
            f"[ok] {split}: glass {counts.get((split, 1), 0)}, "
            f"negative {counts.get((split, 0), 0)} -> {out / split}",
            flush=True,
        )


# =============================================================================
# CLI
# =============================================================================


def _cmd_build(args) -> None:
    build_dataset(
        args.glass,
        args.negative,
        args.out,
        warmup_seconds=args.warmup_seconds,
        warmup_dir=args.warmup_dir,
    )


def _cmd_preview(args) -> None:
    gain = 1.0
    gain_path = Path(args.gain_file) if args.gain_file else None
    if gain_path and gain_path.exists():
        cached = json.load(open(gain_path))
        gain = float(cached["gain"])
        print(f"[gain] wczytano zamrożone wzmocnienie {gain:.4f} z {gain_path}")
    else:
        print(f"[!] brak pliku wzmocnienia ({gain_path or '--gain-file nie podano'}) "
              f"— podgląd BEZ wzmocnienia (gain=1.0). To NIE jest to, co enkoder "
              f"zrobi z tym plikiem w zbiorze (tam gain jest zamrożone z train) — "
              f"podaj --gain-file, żeby zobaczyć realny wynik.")
    gain = compute_global_gain(
        [args.wav]
    )  # podgląd pojedynczego pliku — brak train do kalibracji
    gain = 1.0
    gain_path = Path(args.gain_file) if args.gain_file else None
    if gain_path and gain_path.exists():
        cached = json.load(open(gain_path))
        gain = float(cached["gain"])
        print(f"[gain] wczytano zamrożone wzmocnienie {gain:.4f} z {gain_path}")
    else:
        print(f"[!] brak pliku wzmocnienia ({gain_path or '--gain-file nie podano'}) "
              f"— podgląd BEZ wzmocnienia (gain=1.0). To NIE jest to, co enkoder "
              f"zrobi z tym plikiem w zbiorze (tam gain jest zamrożone z train) — "
              f"podaj --gain-file, żeby zobaczyć realny wynik.")
    spikes = encode_file(args.wav, gain=gain)
    print(f"plik: {args.wav}")
    print(f"ramek po primingu: {spikes.shape[0]}")
    if spikes.shape[0] == 0:
        return
    rates = spikes.mean(axis=0) * 100.0
    for name, rate in zip(CHANNELS, rates):
        print(f"  {name:10s}: {rate:5.1f}% ramek ze spikiem")
    print(f"  ktokolwiek strzelił w co najmniej 1 ramce: {(spikes.any(axis=1)).mean()*100:.1f}% ramek")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Zbuduj CSV-y ze zbioru glass/negative")
    b.add_argument("--glass", required=True, nargs="+",
                   help="jeden lub więcej folderów z .wav tłuczonego szkła")
    b.add_argument("--negative", required=True, nargs="+",
                   help="jeden lub więcej folderów z .wav tła/negatywów")
    b.add_argument("--out", default="./spikes_csv")
    b.add_argument("--warmup-seconds", type=float, default=30.0,
                   help="ile sekund realnego tła zużyć na rozgrzanie floor/MAD przed "
                        "zapisem jakichkolwiek CSV (patrz uwaga w docstringu modułu)")
    b.add_argument("--warmup-dir", default=None,
                   help="folder z PRAWDZIWIE stacjonarnym tłem do rozgrzewki "
                        "(domyślnie: pierwszy z --negative). Nie podawaj tu folderu "
                        "z krótkimi wycinkami zdarzeń (np. hard_negative z VOICe).")
    b.set_defaults(func=_cmd_build)

    m = sub.add_parser("build-manifest",
                       help="Zbuduj CSV-y wg manifestu (filepath,label,...,split)")
    m.add_argument("--manifest", required=True)
    m.add_argument("--root", default=".",
                   help="katalog, względem którego rozwiązywane są ścieżki z manifestu")
    m.add_argument("--out", default="./spikes_manifest")
    m.add_argument("--warmup-seconds", type=float, default=30.0)
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--aug-gain-db", type=float, default=12.0)
    m.add_argument("--gain-percentile", type=float, default=GAIN_PERCENTILE,
                   help="percentyl amplitudy trafiający w pełną skalę ADC")
    m.add_argument("--gain-method", default=GAIN_METHOD, choices=["all-files", "per-file"],
                   help="all-files: percentyl z połączonego rozkładu próbek (domyślne, "
                        "poprawne dla percentile=99.9). per-file: percentyl z listy "
                        "peaków per plik — użyj z niższym --gain-percentile (np. 50/95)")
    m.add_argument("--gain-file", default=None,
                   help="ścieżka do global_gain.json (domyślnie <out>/global_gain.json); "
                        "jeśli istnieje i pasuje percentile/method/n_files, wczytywany "
                        "zamiast liczony od nowa")
    m.add_argument("--dataset-version", default=None,
                   help="etykieta wersji do channels.json; domyślnie z nazwy "
                        "katalogu manifestu (dataset/versions/vX.Y.Z/manifest.csv)")
    m.add_argument("--seed", type=int, default=0)
    m.add_argument("--aug-gain-db", type=float, default=12.0)
    m.add_argument("--gain-percentile", type=float, default=GAIN_PERCENTILE,
                   help="percentyl amplitudy trafiający w pełną skalę ADC")
    m.add_argument("--gain-method", default=GAIN_METHOD, choices=["all-files", "per-file"],
                   help="all-files: percentyl z połączonego rozkładu próbek (domyślne, "
                        "poprawne dla percentile=99.9). per-file: percentyl z listy "
                        "peaków per plik — użyj z niższym --gain-percentile (np. 50/95)")
    m.add_argument("--gain-file", default=None,
                   help="ścieżka do global_gain.json (domyślnie <out>/global_gain.json); "
                        "jeśli istnieje i pasuje percentile/method/n_files, wczytywany "
                        "zamiast liczony od nowa")
    m.set_defaults(func=lambda a: build_manifest(a.manifest, a.out, root=a.root,
                                                 warmup_seconds=a.warmup_seconds,
                                                 seed=a.seed, aug_gain_db=a.aug_gain_db,
                                                 gain_percentile=a.gain_percentile,
                                                 gain_method=a.gain_method,
                                                 gain_file=a.gain_file))

    p = sub.add_parser("preview", help="Podgląd spike-rate jednego pliku (debug)")
    p.add_argument("wav")
    p.add_argument("--gain-file", default=None,
                   help="global_gain.json z zamrożonym wzmocnieniem (z build-manifest); "
                        "bez niego podgląd używa gain=1.0 i wypisuje ostrzeżenie")
    p.set_defaults(func=_cmd_preview)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()