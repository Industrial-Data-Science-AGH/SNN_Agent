#!/usr/bin/env python3
"""Generuje encoder_v2_swap.ino z oryginalnego encoder_v2.ino (wstawki #if, oryginał nietknięty).
Każda podmiana ma assert 'kotwica występuje dokładnie raz' — jeśli oryginał się zmieni, skrypt krzyczy."""
import sys
src, dst = sys.argv[1], sys.argv[2]
s = open(src, encoding="utf-8").read()

def sub(old, new, count=1):
    global s
    assert s.count(old) == count, f"kotwica występuje {s.count(old)}x (oczekiwano {count}): {old[:60]!r}"
    s = s.replace(old, new)

# ---------------------------------------------------------------- flagi
sub("#include <util/atomic.h>\n", '''#include <util/atomic.h>

// ================================================================ FLAGI KOMPILACJI
// Wszystkie domyślnie 0 => kod IDENTYCZNY z encoder_v2.ino (sprawdzane w tools/run_predictions.sh).
#ifndef ENC_SET_SWAP
#define ENC_SET_SWAP 0    // 1: peak_cnt -> hjorth_mobility, cv -> autocorr_lag1 (pozycyjnie: kanały 1 i 2)
#endif
#ifndef ENC_PARITY
#define ENC_PARITY 0      // 1: włącza trzy poprawki zgodności z encoder_twin.py (poniżej), każdą można też osobno
#endif
#ifndef ENC_DC_FIX
#define ENC_DC_FIX ENC_PARITY   // usuwanie DC bez martwej strefy (Q4>>9 miało ~32 LSB) — patrz ISR
#endif
#ifndef ENC_HF_ROUND
#define ENC_HF_ROUND ENC_PARITY // zaokrąglanie w 1-biegunowym LP pasma górnego (>>1 ścinało w dół)
#endif
#ifndef ENC_EPS_FLOOR
#define ENC_EPS_FLOOR ENC_PARITY // EPS per kanał w mianowniku z-score jak EPS_FLOOR w twinie (zamiast 1e-6)
#endif
#ifndef ENC_ACC32
#define ENC_ACC32 0       // 1: acc_sq/acc_hf_sq jako uint32 (192*1023^2 = 2.0e8 < 2^32) — bez __adddi3
#endif
#ifndef ENC_DEBUG_FEAT
#define ENC_DEBUG_FEAT 0  // 1: w linii debug wypisz też wartości cech (do testu parytetu z twinem)
#endif
#ifndef ENC_BAUD
#define ENC_BAUD 115200
#endif
#ifndef ENC_BENCH
#define ENC_BENCH 0       // 1: tryb pomiarowy (polecenie 'B' przez Serial) — patrz sekcja BENCH
#endif

// Progi bezwzględne nowych kanałów (poziom kształtu widma, jak hf_lo/hf_hi — NIE z-score).
// WSTAW wartości z phase0_analysis.py (recommended_thresholds). Domyślnie kanał MILCZY.
#ifndef MOB_FIRE_BELOW
#define MOB_FIRE_BELOW 0  // mobility: szkło ma WYŻSZĄ mobility (d>0) -> odpala gdy mob > MOB_THR
#endif
#ifndef AC_FIRE_BELOW
#define AC_FIRE_BELOW 1   // autocorr_lag1: szkło ma NIŻSZĄ autokorelację (d<0) -> odpala gdy ac < AC_THR
#endif
#ifndef MOB_THR
#define MOB_THR (MOB_FIRE_BELOW ? -1.0e9f : 1.0e9f)   // <<< WSTAW z fazy 0 (tu: nigdy nie strzela)
#endif
#ifndef AC_THR
#define AC_THR  (AC_FIRE_BELOW  ? -1.0e9f : 1.0e9f)   // <<< WSTAW z fazy 0 (tu: nigdy nie strzela)
#endif
''')

# ---------------------------------------------------------------- enum + progi
sub("enum Ch { CH_PEAK = 0, CH_PEAKCNT, CH_CV, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI };\n",
    "#if ENC_SET_SWAP\nenum Ch { CH_PEAK = 0, CH_MOB, CH_AC, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI };\n"
    "#else\nenum Ch { CH_PEAK = 0, CH_PEAKCNT, CH_CV, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI };\n#endif\n")
sub("static float THR_Z[N_CH] = { 4.0, 3.5, 3.0, 2.5, 3.5, 0.0, 0.0 };\n",
    "#if ENC_SET_SWAP\nstatic float THR_Z[N_CH] = { 4.0, 0.0, 0.0, 2.5, 3.5, 0.0, 0.0 };   // kanały 1,2 na progach bezwzgl.\n"
    "#else\nstatic float THR_Z[N_CH] = { 4.0, 3.5, 3.0, 2.5, 3.5, 0.0, 0.0 };\n#endif\n")

# ---------------------------------------------------------------- stan ISR
sub("volatile uint64_t acc_sq;       // suma x^2\nvolatile uint64_t acc_hf_sq;    // suma hf^2 (energia pasma górnego)\n",
    "#if ENC_ACC32\nvolatile uint32_t acc_sq;       // suma x^2 (32 bity wystarczą: 192*1023^2 = 2.0e8)\n"
    "volatile uint32_t acc_hf_sq;    // suma hf^2\n#else\n"
    "volatile uint64_t acc_sq;       // suma x^2\nvolatile uint64_t acc_hf_sq;    // suma hf^2 (energia pasma górnego)\n#endif\n")
sub("volatile uint16_t acc_pk;       // próbki powyżej progu mikro-szpilki\n",
    "#if !ENC_SET_SWAP\nvolatile uint16_t acc_pk;       // próbki powyżej progu mikro-szpilki\n#endif\n")
sub("volatile int16_t  dc_est   = 512 << 4;  // Q4, średnia bieżąca ADC\n",
    "#if ENC_DC_FIX\nstatic   int32_t  dc_q9    = (int32_t)512 << 9;   // Q9: dc = dc_q9/512 (tylko ISR; Q4>>9 miało martwą strefę ~32 LSB)\n"
    "#else\nvolatile int16_t  dc_est   = 512 << 4;  // Q4, średnia bieżąca ADC\n#endif\n")
sub("volatile int16_t  spike_thr = 40;       // próg mikro-szpilki, aktualizowany co ramkę\n",
    "#if !ENC_SET_SWAP\nvolatile int16_t  spike_thr = 40;       // próg mikro-szpilki, aktualizowany co ramkę\n#endif\n")
sub("static   int16_t  prev_sign = 0;\n",
    "static   int16_t  prev_sign = 0;\n#if ENC_SET_SWAP\n"
    "volatile int16_t  x_prev   = 0;         // poprzednia próbka (po usunięciu DC)\n"
    "volatile uint32_t acc_dx2;              // suma (x[n]-x[n-1])^2   (max 192*2046^2 = 8.0e8 < 2^32)\n"
    "volatile int32_t  acc_xx1;              // suma x[n]*x[n-1]       (|.| <= 192*1023^2 = 2.0e8 < 2^31)\n#endif\n")
sub("static float rms_prev = 0.0f;\n", "static float rms_prev = 0.0f;\n#if ENC_SET_SWAP\nstatic int16_t xlast_prev = 0;   // ostatnia próbka poprzedniej ramki (do sum(dx) = x_last - xlast_prev)\n#endif\n")

# ---------------------------------------------------------------- ISR
sub("""  // usunięcie DC: EMA w Q4
  dc_est += (int16_t)(((int32_t)(raw << 4) - dc_est) >> 9);
  int16_t x = raw - (dc_est >> 4);
""", """  // usunięcie DC: EMA k=1/512
#if ENC_DC_FIX
  // EMA k=1/512 bez martwej strefy. Y = 512*dc (Q9). Krok: Y += raw - round(Y/512).
  // round(Y/512) = (d1+1)>>1 gdzie d1 = Y>>8 (przesunięcie o bajt — tanie na AVR, w odróżnieniu od
  // pętli 9x asr). Martwa strefa: +-0.5 LSB zamiast ~32 LSB (Q4>>9). x liczone PO aktualizacji, jak w twinie.
  int16_t d1 = (int16_t)(dc_q9 >> 8);
  dc_q9 += raw - ((d1 + 1) >> 1);
  d1 = (int16_t)(dc_q9 >> 8);
  int16_t x = raw - ((d1 + 1) >> 1);
#else
  // EMA w Q4
  dc_est += (int16_t)(((int32_t)(raw << 4) - dc_est) >> 9);
  int16_t x = raw - (dc_est >> 4);
#endif
""")
sub("  if (ax > spike_thr) acc_pk++;\n", "#if !ENC_SET_SWAP\n  if (ax > spike_thr) acc_pk++;\n#endif\n")
sub("  int16_t s = (x >= 0) ? 1 : -1;\n", """#if ENC_SET_SWAP
  // hjorth_mobility i autocorr_lag1: dwa dodatkowe akumulatory (ciągłe po granicach ramek)
  int16_t dx = x - x_prev;
  acc_dx2 += (uint32_t)((int32_t)dx * dx);
  acc_xx1 += (int32_t)x * x_prev;
  x_prev = x;
#endif

  int16_t s = (x >= 0) ? 1 : -1;
""")

# ---------------------------------------------------------------- loop: snapshot
sub("  int32_t  s_abs; uint64_t s_sq, s_hf_sq; int16_t s_max; uint16_t s_zc, s_pk, s_n;\n",
    "#if ENC_ACC32\n  int32_t  s_abs; uint32_t s_sq, s_hf_sq; int16_t s_max; uint16_t s_zc, s_pk, s_n;\n#else\n"
    "  int32_t  s_abs; uint64_t s_sq, s_hf_sq; int16_t s_max; uint16_t s_zc, s_pk, s_n;\n#endif\n"
    "#if ENC_SET_SWAP\n  uint32_t s_dx2; int32_t s_xx1; int16_t s_xlast;\n#endif\n")
sub("    s_zc = acc_zc;  s_pk = acc_pk;  s_n = n_samp;\n    acc_abs = 0; acc_sq = 0; acc_hf_sq = 0; acc_max = 0; acc_zc = 0; acc_pk = 0; n_samp = 0;\n",
    "#if ENC_SET_SWAP\n    s_zc = acc_zc;  s_pk = 0;       s_n = n_samp;\n"
    "    s_dx2 = acc_dx2; s_xx1 = acc_xx1; s_xlast = x_prev;\n"
    "    acc_abs = 0; acc_sq = 0; acc_hf_sq = 0; acc_max = 0; acc_zc = 0; acc_dx2 = 0; acc_xx1 = 0; n_samp = 0;\n"
    "#else\n    s_zc = acc_zc;  s_pk = acc_pk;  s_n = n_samp;\n"
    "    acc_abs = 0; acc_sq = 0; acc_hf_sq = 0; acc_max = 0; acc_zc = 0; acc_pk = 0; n_samp = 0;\n#endif\n")

# ---------------------------------------------------------------- loop: cechy
sub("  float cv       = sqrtf(var_abs) / (mean_abs + EPS);\n\n  float zcr      = (float)s_zc * inv_n;\n  float peak_cnt = (float)s_pk;\n",
    """#if ENC_SET_SWAP
  // hjorth_mobility = sqrt(var(dx) / var(|x|));  var(dx) = E[dx^2] - (sum dx / n)^2, sum dx = x_last - xlast_prev
  float mean_dx  = (float)(s_xlast - xlast_prev) * inv_n;
  float var_dx   = fmaxf(0.0f, (float)s_dx2 * inv_n - mean_dx * mean_dx);
  float mob      = sqrtf(var_dx / (var_abs + EPS));
  // autocorr_lag1 = sum x[n]x[n-1] / (sum x^2 + eps)
  float ac       = (float)s_xx1 / ((float)s_sq + EPS);
  xlast_prev     = s_xlast;
  float zcr      = (float)s_zc * inv_n;
#else
  float cv       = sqrtf(var_abs) / (mean_abs + EPS);

  float zcr      = (float)s_zc * inv_n;
  float peak_cnt = (float)s_pk;
#endif
""")
sub("  spike_thr = (int16_t)fminf(1023.0f, fmaxf(8.0f, 3.0f * (floor_v[CH_PEAK] + EPS)));\n",
    "#if !ENC_SET_SWAP\n  spike_thr = (int16_t)fminf(1023.0f, fmaxf(8.0f, 3.0f * (floor_v[CH_PEAK] + EPS)));\n#endif\n")
sub("  float feat[N_CH] = { peak, peak_cnt, cv, zcr, flux, hf_ratio, hf_ratio };\n",
    "#if ENC_SET_SWAP\n  float feat[N_CH] = { peak, mob, ac, zcr, flux, hf_ratio, hf_ratio };\n#else\n"
    "  float feat[N_CH] = { peak, peak_cnt, cv, zcr, flux, hf_ratio, hf_ratio };\n#endif\n")

# ---------------------------------------------------------------- loop: kodowanie
sub("""    } else if (c == CH_HFHI) {
      above = hf_gated && (hf_ratio > HF_HI_THR);
    } else {""", """    } else if (c == CH_HFHI) {
      above = hf_gated && (hf_ratio > HF_HI_THR);
#if ENC_SET_SWAP
    } else if (c == CH_MOB) {                 // poziom, próg bezwzględny + bramka jak hf
      above = hf_gated && (MOB_FIRE_BELOW ? (mob < MOB_THR) : (mob > MOB_THR));
    } else if (c == CH_AC) {
      above = hf_gated && (AC_FIRE_BELOW ? (ac < AC_THR) : (ac > AC_THR));
#endif
    } else {""")


# ---------------------------------------------------------------- HF rounding + EPS_FLOOR
sub("  hf_lp += (int16_t)((x - hf_lp) >> HF_HP_SHIFT);\n",
    "#if ENC_HF_ROUND\n  hf_lp += (int16_t)((x - hf_lp + (1 << (HF_HP_SHIFT - 1))) >> HF_HP_SHIFT);   // z zaokrągleniem\n"
    "#else\n  hf_lp += (int16_t)((x - hf_lp) >> HF_HP_SHIFT);\n#endif\n")
sub("#define EPS     1e-6f\n", """#define EPS     1e-6f
#if ENC_EPS_FLOOR
// EPS PER KANAŁ w mianowniku z-score (1:1 z EPS_FLOOR w encoder_twin.py). Wspólne 1e-6 powoduje eksplozję
// z-score przy cyfrowej ciszy (MAD->0). Kolejność = pozycje kanałów; kanały na progu bezwzgl. nieużywane.
static const float EPS_FLOOR_V[7] = { 1.0f, 1.0f, 1.0f / 192.0f, 1.0f / 192.0f, 1.0f / (1.0f + 13.856406f), 1.0f, 1.0f };
#endif
""")
sub("  return (v - floor_v[c]) / (mad_v[c] + EPS);\n",
    "#if ENC_EPS_FLOOR\n  return (v - floor_v[c]) / (mad_v[c] + EPS_FLOOR_V[c]);\n#else\n  return (v - floor_v[c]) / (mad_v[c] + EPS);\n#endif\n")

# ---------------------------------------------------------------- debug + baud
sub("    Serial.print(frame_idx);\n    for (uint8_t c = 0; c < N_CH; c++) {\n      Serial.print(',');\n      Serial.print((fired >> c) & 1);\n",
    "    Serial.print(frame_idx);\n#if ENC_DEBUG_FEAT\n    for (uint8_t c = 0; c < N_CH; c++) { Serial.print(','); Serial.print(feat[c], 5); }\n#endif\n"
    "    for (uint8_t c = 0; c < N_CH; c++) {\n      Serial.print(',');\n      Serial.print((fired >> c) & 1);\n")
sub("  Serial.begin(115200);\n", "  Serial.begin(ENC_BAUD);\n")

# ---------------------------------------------------------------- prescaler ADC (flaga) + pin ISR (flaga) + BENCH
import os as _os
sub("#ifndef ENC_BENCH\n", """#ifndef ENC_ADC_PRESCALER
#define ENC_ADC_PRESCALER 32   // 32 = oryginał (ADC 500 kHz => ~38.5 kHz); 64 => ADC 250 kHz => 19231 Hz (wartość FS_HZ)
#endif
#ifndef ENC_ISR_PIN
#define ENC_ISR_PIN 0     // 1: D9 wysoko na czas ISR (oscyloskop/analizator). Pulsu NIE obejmuje prologu/epilogu ISR
#endif
#ifndef ENC_BENCH
""")
sub("         | _BV(ADPS2) | _BV(ADPS0);             // prescaler 32\n",
    "#if ENC_ADC_PRESCALER == 64\n         | _BV(ADPS2) | _BV(ADPS1);             // prescaler 64\n"
    "#elif ENC_ADC_PRESCALER == 128\n         | _BV(ADPS2) | _BV(ADPS1) | _BV(ADPS0); // prescaler 128\n"
    "#else\n         | _BV(ADPS2) | _BV(ADPS0);             // prescaler 32\n#endif\n")
sub("  int16_t raw = ADC;\n", "#if ENC_ISR_PIN\n  PORTB |= _BV(1);          // D9 wysoko\n#endif\n  int16_t raw = ADC;\n")
sub("  if (++n_samp >= HOP_SAMPLES) frame_ready = true;\n}\n",
    "  if (++n_samp >= HOP_SAMPLES) frame_ready = true;\n#if ENC_ISR_PIN\n  PORTB &= ~_BV(1);         // D9 nisko\n#endif\n}\n")
sub("  for (uint8_t p = 2; p <= 8; p++) pinMode(p, OUTPUT);   // D2..D8 = 7 kanałów\n",
    "  for (uint8_t p = 2; p <= 8; p++) pinMode(p, OUTPUT);   // D2..D8 = 7 kanałów\n#if ENC_ISR_PIN\n  pinMode(9, OUTPUT);\n#endif\n")
bench = open(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "bench_block.inc"), encoding="utf-8").read()
sub("// ---------------------------------------------------------------- CALIB\n", bench + "// ---------------------------------------------------------------- CALIB\n")
sub("  } else if (cmd == 'D') {\n", "#if ENC_BENCH\n  } else if (cmd == 'B') {\n    bench_start();\n#endif\n  } else if (cmd == 'D') {\n")
sub("void loop() {\n  handleSerial();\n", "void loop() {\n  handleSerial();\n  BENCH_TICK();\n")
sub("  // --- cechy ---------------------------------------------------------------\n", "  BENCH_FRAME_START(s_n);\n\n  // --- cechy ---------------------------------------------------------------\n")
sub("    frame_idx++;\n    return;\n", "    frame_idx++;\n    BENCH_FRAME_END();\n    return;\n")
sub("    Serial.println();\n  }\n  frame_idx++;\n}\n", "    Serial.println();\n  }\n  frame_idx++;\n  BENCH_FRAME_END();\n}\n")

open(dst, "w", encoding="utf-8").write(s)
print("OK ->", dst)
