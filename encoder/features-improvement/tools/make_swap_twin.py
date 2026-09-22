#!/usr/bin/env python3
"""Generuje encoder_twin_swap.py z oryginalnego encoder_twin.py. Oryginał nietknięty; każda kotwica
musi wystąpić dokładnie raz. Domyślnie ENCODER_CHANNEL_SET=baseline => wyniki bit-w-bit jak oryginał."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

def sub(old, new):
    global s
    assert s.count(old) == 1, f"kotwica występuje {s.count(old)}x: {old[:70]!r}"
    s = s.replace(old, new)

# ---------------------------------------------------------------- definicja zestawów kanałów
sub('''CHANNELS = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]
CH_PEAK, CH_PEAKCNT, CH_CV, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI = range(N_CH)

# progi z-score (adaptacyjny floor) — używane tylko dla kanałów bez progu bezwzgl.
THR_Z = np.array([4.0, 3.5, 3.0, 2.5, 3.5, np.nan, np.nan])
# progi bezwzględne (nan = kanał używa z-score). hf_lo czuły, hf_hi specyficzny.
ABS_THR = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.28, 0.35])
''', '''# ZESTAW KANAŁÓW: "baseline" (oryginał) albo "swap" (peak_cnt -> hjorth_mobility, cv -> autocorr_lag1,
# wymiana POZYCYJNA: kanały 1 i 2, więc maski sieci i piny zostają bez zmian). Wybór: zmienna środowiskowa
# ENCODER_CHANNEL_SET; domyślnie baseline. Oba ramiona A/B budowane z TEGO SAMEGO pliku.
CHANNEL_SET = os.environ.get("ENCODER_CHANNEL_SET", "baseline")
assert CHANNEL_SET in ("baseline", "swap"), f"ENCODER_CHANNEL_SET={CHANNEL_SET!r}"

# Nowe kanały to POZIOMY (kształt widma), nie zdarzenia -> próg BEZWZGLĘDNY + bramka jak hf_lo/hf_hi
# (z-score odwracałby sygnał — patrz komentarz wyżej). Kierunek: mobility d>0 (szkło wyższa) -> odpala
# powyżej progu; autocorr_lag1 d<0 (szkło niższa) -> odpala poniżej progu.
# <<< WSTAW progi z phase0_analysis.py (recommended_thresholds). Domyślnie kanały MILCZĄ (próg +-1e9).
MOB_FIRE_BELOW = False
AC_FIRE_BELOW = True
MOB_THR = float(os.environ.get("ENCODER_MOB_THR", -1e9 if MOB_FIRE_BELOW else 1e9))
AC_THR = float(os.environ.get("ENCODER_AC_THR", -1e9 if AC_FIRE_BELOW else 1e9))

if CHANNEL_SET == "swap":
    CHANNELS = ["peak", "hjorth_mobility", "autocorr_lag1", "zcr", "flux", "hf_lo", "hf_hi"]
    THR_Z = np.array([4.0, np.nan, np.nan, 2.5, 3.5, np.nan, np.nan])
    ABS_THR = np.array([np.nan, MOB_THR, AC_THR, np.nan, np.nan, 0.28, 0.35])
    ABS_DIR = np.array([1.0, -1.0 if MOB_FIRE_BELOW else 1.0, -1.0 if AC_FIRE_BELOW else 1.0, 1.0, 1.0, 1.0, 1.0])
else:
    CHANNELS = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]
    # progi z-score (adaptacyjny floor) — używane tylko dla kanałów bez progu bezwzgl.
    THR_Z = np.array([4.0, 3.5, 3.0, 2.5, 3.5, np.nan, np.nan])
    # progi bezwzględne (nan = kanał używa z-score). hf_lo czuły, hf_hi specyficzny.
    ABS_THR = np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.28, 0.35])
    ABS_DIR = np.ones(N_CH)
CH_PEAK, CH_PEAKCNT, CH_CV, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI = range(N_CH)   # POZYCJE (w swap: 1=mobility, 2=autocorr)
''')

# ---------------------------------------------------------------- cechy per ramka: jedna funkcja (źródło prawdy)
sub('''def encode_file(path: str, gain: float, state: Optional[EncoderState] = None,''',
'''def _frame_features(x: np.ndarray) -> dict:
    """Cechy per ramka dla sygnału po usunięciu DC (wektorowo). Jedno źródło prawdy dla encode_file,
    phase0_analysis.py i parity_test.py. Cechy zależne od STANU (peak_cnt, flux) liczy encode_file.
    Nowe cechy są CIĄGŁE po granicach ramek (x_prev[0]=0), tak jak akumulatory w firmware:
      mobility = sqrt(var(dx)/(var|x| + EPS)),  var(dx) = E[dx^2] - (sum dx / n)^2
      autocorr = sum x[n]x[n-1] / (sum x^2 + EPS)"""
    hf = _high_band(x)                       # pasmo górne (~>2.2 kHz) — cechy widmowe
    ax = np.abs(x)
    sign = np.where(x >= 0, 1, -1)
    sign_ext = np.concatenate(([0], sign))  # prev_sign startuje od 0, jak w .ino
    crossings = (sign_ext[1:] != sign_ext[:-1]).astype(np.int32)

    n_frames = len(x) // HOP_SAMPLES
    if n_frames == 0:
        return {"n_frames": 0}
    L = n_frames * HOP_SAMPLES
    x_f = x[:L].reshape(n_frames, HOP_SAMPLES)
    ax_f = ax[:L].reshape(n_frames, HOP_SAMPLES)
    zc_f = crossings[:L].reshape(n_frames, HOP_SAMPLES)
    hf_f = hf[:L].reshape(n_frames, HOP_SAMPLES)

    acc_abs = ax_f.sum(axis=1)
    acc_sq = (x_f ** 2).sum(axis=1)
    acc_max = ax_f.max(axis=1)
    acc_zc = zc_f.sum(axis=1)
    acc_hf_sq = (hf_f ** 2).sum(axis=1)      # energia pasma górnego w ramce

    n = float(HOP_SAMPLES)
    mean_abs = acc_abs / n
    rms = np.sqrt(acc_sq / n)
    var_abs = np.maximum(0.0, acc_sq / n - mean_abs ** 2)

    x_prev = np.concatenate(([0.0], x[:L - 1]))
    dx_f = (x[:L] - x_prev).reshape(n_frames, HOP_SAMPLES)
    xp_f = x_prev.reshape(n_frames, HOP_SAMPLES)
    acc_dx2 = (dx_f ** 2).sum(axis=1)
    acc_xx1 = (x_f * xp_f).sum(axis=1)
    var_dx = np.maximum(0.0, acc_dx2 / n - (dx_f.sum(axis=1) / n) ** 2)

    return {
        "n_frames": n_frames, "ax_f": ax_f, "rms": rms, "peak": acc_max,
        "cv": np.sqrt(var_abs) / (mean_abs + EPS),
        "zcr": acc_zc / n,
        "hf_ratio": acc_hf_sq / (acc_sq + EPS),   # udział energii HF (szkło >> łomot/strzał)
        "mobility": np.sqrt(var_dx / (var_abs + EPS)),
        "autocorr": acc_xx1 / (acc_sq + EPS),
    }


def encode_file(path: str, gain: float, state: Optional[EncoderState] = None,''')

# encode_file: użyj _frame_features zamiast kodu inline (te same wzory, ta sama kolejność operacji)
i0 = s.index("    x = _remove_dc(codes)\n    hf = _high_band(x)")
i1 = s.index("    if state is None:\n        state = EncoderState()\n\n    out_rows = []")
s = s[:i0] + '''    x = _remove_dc(codes)
    ff = _frame_features(x)
    n_frames = ff["n_frames"]
    if n_frames == 0:
        return np.zeros((0, N_CH), dtype=np.uint8)
    ax_f, rms, peak, cv, zcr, hf_ratio = (ff[k] for k in ("ax_f", "rms", "peak", "cv", "zcr", "hf_ratio"))
    mobility, autocorr = ff["mobility"], ff["autocorr"]

''' + s[i1:]

# wektor cech: pozycje 1 i 2 zależą od zestawu
sub('''        feat = np.array([peak[k], peak_cnt, cv[k], zcr[k], flux,
                         hf_ratio[k], hf_ratio[k]])
''', '''        if CHANNEL_SET == "swap":
            feat = np.array([peak[k], mobility[k], autocorr[k], zcr[k], flux,
                             hf_ratio[k], hf_ratio[k]])
        else:
            feat = np.array([peak[k], peak_cnt, cv[k], zcr[k], flux,
                             hf_ratio[k], hf_ratio[k]])
''')

# próg bezwzględny z kierunkiem (baseline: ABS_DIR=1 => identycznie jak wcześniej)
sub('''                above = hf_gated and (feat[c] > ABS_THR[c])   # próg bezwzględny''',
    '''                above = hf_gated and (feat[c] * ABS_DIR[c] > ABS_THR[c] * ABS_DIR[c])   # próg bezwzględny (kierunek: ABS_DIR)''')


# ---------------------------------------------------------------- opcjonalny zwrot cech (dla phase0_analysis.py / parity)
sub("def encode_file(path: str, gain: float, state: Optional[EncoderState] = None,\n                aug_gain_db: float = 0.0, rng=None) -> np.ndarray:\n",
    "def encode_file(path: str, gain: float, state: Optional[EncoderState] = None,\n                aug_gain_db: float = 0.0, rng=None, return_features: bool = False):\n")
sub("    out_rows = []\n", "    out_rows = []\n    feat_rows, gate_rows = [], []      # tylko dla return_features=True\n")
sub("        out_rows.append(bits)\n",
    "        out_rows.append(bits)\n        if return_features:\n            feat_rows.append(feat.copy()); gate_rows.append(bool(hf_gated))\n")
sub("    return np.array(out_rows, dtype=np.uint8) if out_rows else np.zeros((0, N_CH), dtype=np.uint8)\n",
    "    bits_arr = np.array(out_rows, dtype=np.uint8) if out_rows else np.zeros((0, N_CH), dtype=np.uint8)\n"
    "    if return_features:   # (bity, wartości cech [n,7], bramka hf_gated [n]) — dla ramek po primingu\n"
    "        return bits_arr, (np.array(feat_rows) if feat_rows else np.zeros((0, N_CH))), np.array(gate_rows, dtype=bool)\n"
    "    return bits_arr\n")

# import os potrzebny do ENCODER_CHANNEL_SET (jest już 'import os' w pliku — asercja)
assert "\nimport os\n" in s
open(dst, "w", encoding="utf-8").write(s)
print("OK ->", dst)
