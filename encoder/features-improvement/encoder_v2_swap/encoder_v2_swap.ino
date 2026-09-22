/*
 * encoder_v2.ino — Delta Spike / Wake-Up AI
 * Enkoder 6-kanałowy dla sieci Lu.i o topologii 6->4->3->1.
 *
 * Zmiany względem fixed_encoder_08_06_26.ino:
 *  1. Ramkowo-synchroniczne spike'i: max 1 impuls na kanał na ramkę (hop = 10 ms).
 *     => dt symulacji == dt sprzętu == 10 ms. Trening liczy to samo, co robi płytka.
 *  2. Stała szerokość impulsu 6 ms. Ładunek w synapsie ~ szerokość impulsu,
 *     więc stała szerokość = stała, znana waga bazowa (bez zgadywania "wąski impuls x rate").
 *  3. Adaptacyjny floor + MAD per kanał. Progi w jednostkach z-score, nie w absolutnych.
 *     Adaptacja zamrożona w trakcie zdarzenia (inaczej encoder "przyzwyczaja się" do szkła).
 *  4. Kanał `mean` usunięty (nie różnicował klas), dodany `flux` (dodatni przyrost log-RMS).
 *  5. Tryb CALIB: generator impulsów testowych do kalibracji wag płytek Lu.i.
 *
 * Piny: D2..D7 = spike out (PORTD bity 2..7 -> wszystkie kanały zapalane jedną instrukcją,
 *       zero skewu między kanałami). D0/D1 zostawione dla UART.
 *
 * Mikrofon: wejście analogowe A0, ADC free-running, prescaler 32 -> fs ~= 19231 Hz.
 * Target: ATmega328P (Uno/Nano). Na innym MCU popraw setupADC() i FS.
 */

#include <avr/io.h>
#include <avr/interrupt.h>
#include <util/atomic.h>

// ================================================================ FLAGI KOMPILACJI
// Wszystkie domyślnie 0 => kod IDENTYCZNY z encoder_v2.ino (sprawdzane w tools/run_predictions.sh).
#ifndef ENC_SET_SWAP
#define ENC_SET_SWAP 1    // 1: peak_cnt -> hjorth_mobility, cv -> autocorr_lag1 (pozycyjnie: kanały 1 i 2)
#endif
#ifndef ENC_PARITY
#define ENC_PARITY 1      // 1: włącza trzy poprawki zgodności z encoder_twin.py (poniżej), każdą można też osobno
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
#define ENC_ACC32 1       // 1: acc_sq/acc_hf_sq jako uint32 (192*1023^2 = 2.0e8 < 2^32) — bez __adddi3
#endif
#ifndef ENC_DEBUG_FEAT
#define ENC_DEBUG_FEAT 0  // 1: w linii debug wypisz też wartości cech (do testu parytetu z twinem)
#endif
#ifndef ENC_BAUD
#define ENC_BAUD 115200
#endif
#ifndef ENC_ADC_PRESCALER
#define ENC_ADC_PRESCALER 64   // 32 = oryginał (ADC 500 kHz => ~38.5 kHz); 64 => ADC 250 kHz => 19231 Hz (wartość FS_HZ)
#endif
#ifndef ENC_ISR_PIN
#define ENC_ISR_PIN 0     // 1: D9 wysoko na czas ISR (oscyloskop/analizator). Pulsu NIE obejmuje prologu/epilogu ISR
#endif
#ifndef ENC_BENCH
#define ENC_BENCH 1       // 1: tryb pomiarowy (polecenie 'B' przez Serial) — patrz sekcja BENCH
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

// ---------------------------------------------------------------- konfiguracja

#define FS_HZ        19231UL   // realna fs przy prescalerze 32
#define HOP_SAMPLES  192       // 192 / 19231 Hz ~= 10.0 ms  <-- to jest dt sieci
#define PULSE_MS     6         // < HOP_MS, inaczej impulsy się skleją
#define N_CH         7

// Piny wyjściowe: kanały 0..5 -> PORTD bity 2..7 (D2..D7), kanał 6 -> PORTB bit 0 (D8).
// Dwie instrukcje wystawienia (PORTD, PORTB) dają ~62 ns skewu — pomijalny vs impuls 6 ms.
#define PIND_BASE    2         // kanały 0..5 na D2..D7
#define PIND_MASK    0b11111100
#define PINB_MASK    0b00000001 // kanał 6 na D8 (PORTB bit 0)

// v3: kolejność kanałów == kolumny s0..s6. `crest` (martwa) zastąpiona hf_lo/hf_hi.
#if ENC_SET_SWAP
enum Ch { CH_PEAK = 0, CH_MOB, CH_AC, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI };
#else
enum Ch { CH_PEAK = 0, CH_PEAKCNT, CH_CV, CH_ZCR, CH_FLUX, CH_HFLO, CH_HFHI };
#endif

// progi z-score = (feature - floor) / (MAD + eps) — TYLKO kanały czasowe 0..4.
// hf_lo/hf_hi (5,6) NIE używają z-score (patrz niżej), ich pola tu są nieużywane.
#if ENC_SET_SWAP
static float THR_Z[N_CH] = { 4.0, 0.0, 0.0, 2.5, 3.5, 0.0, 0.0 };   // kanały 1,2 na progach bezwzgl.
#else
static float THR_Z[N_CH] = { 4.0, 3.5, 3.0, 2.5, 3.5, 0.0, 0.0 };
#endif

// --- cechy widmowe: hf_ratio = energia pasma górnego / energia ramki ---
// hf_ratio to POZIOM (kształt widma), nie transient, więc NIE adaptujemy floora —
// próg bezwzględny. Adaptacyjny floor odjąłby trwale wysokie HF szkła (zmierzone
// w symulacji: z-score odwracał sygnał). hf_hi jest mocnym dowodem na szkło.
#define HF_HP_SHIFT  1         // lp += (x-lp)>>1 => cutoff ~2.2 kHz; hf = x - lp
#define HF_LO_THR    0.28f     // czuły: szkło ~58% ramek, negatywy ~2-4%
#define HF_HI_THR    0.35f     // specyficzny: szkło ~41%, esc50/cisza/voice ~0-7%
#define HF_GATE_MULT 1.5f      // hf strzela tylko gdy peak > 1.5*floor (nie na ciszy)

// asymetryczna adaptacja floora: rośnie wolno, spada szybko -> śledzi poziom ciszy
#define A_UP    0.0015f
#define A_DN    0.0300f
#define A_MAD   0.0100f
#define EPS     1e-6f
#if ENC_EPS_FLOOR
// EPS PER KANAŁ w mianowniku z-score (1:1 z EPS_FLOOR w encoder_twin.py). Wspólne 1e-6 powoduje eksplozję
// z-score przy cyfrowej ciszy (MAD->0). Kolejność = pozycje kanałów; kanały na progu bezwzgl. nieużywane.
static const float EPS_FLOOR_V[7] = { 1.0f, 1.0f, 1.0f / 192.0f, 1.0f / 192.0f, 1.0f / (1.0f + 13.856406f), 1.0f, 1.0f };
#endif

// refrakcja w ramkach (1 = kanał może strzelać co ramkę, tj. 100 Hz)
#define REFRAC_FRAMES 1

// ---------------------------------------------------------------- stan ISR

volatile int32_t  acc_abs;      // suma |x|
#if ENC_ACC32
volatile uint32_t acc_sq;       // suma x^2 (32 bity wystarczą: 192*1023^2 = 2.0e8)
volatile uint32_t acc_hf_sq;    // suma hf^2
#else
volatile uint64_t acc_sq;       // suma x^2
volatile uint64_t acc_hf_sq;    // suma hf^2 (energia pasma górnego)
#endif
volatile int16_t  acc_max;      // max |x|
volatile uint16_t acc_zc;       // przejścia przez zero
#if !ENC_SET_SWAP
volatile uint16_t acc_pk;       // próbki powyżej progu mikro-szpilki
#endif
volatile uint16_t n_samp;
volatile bool     frame_ready;

#if ENC_DC_FIX
static   int32_t  dc_q9    = (int32_t)512 << 9;   // Q9: dc = dc_q9/512 (tylko ISR; Q4>>9 miało martwą strefę ~32 LSB)
#else
volatile int16_t  dc_est   = 512 << 4;  // Q4, średnia bieżąca ADC
#endif
volatile int16_t  hf_lp    = 0;         // 1-pole lowpass pasma górnego
#if !ENC_SET_SWAP
volatile int16_t  spike_thr = 40;       // próg mikro-szpilki, aktualizowany co ramkę
#endif
static   int16_t  prev_sign = 0;
#if ENC_SET_SWAP
volatile int16_t  x_prev   = 0;         // poprzednia próbka (po usunięciu DC)
volatile uint32_t acc_dx2;              // suma (x[n]-x[n-1])^2   (max 192*2046^2 = 8.0e8 < 2^32)
volatile int32_t  acc_xx1;              // suma x[n]*x[n-1]       (|.| <= 192*1023^2 = 2.0e8 < 2^31)
#endif

// ---------------------------------------------------------------- stan ramki

static float floor_v[N_CH] = {0}, mad_v[N_CH] = {0};
static uint8_t refrac[N_CH] = {0};
static float rms_prev = 0.0f;
#if ENC_SET_SWAP
static int16_t xlast_prev = 0;   // ostatnia próbka poprzedniej ramki (do sum(dx) = x_last - xlast_prev)
#endif
static bool  floors_primed = false;
static uint32_t frame_idx = 0;

// pulse-off bez blokowania
static uint32_t pulse_off_us = 0;
static bool     pulse_active = false;

// tryby
static bool debug_csv = false;
static bool calib_mode = false;

// ---------------------------------------------------------------- ADC

void setupADC() {
  ADMUX  = _BV(REFS0);                          // AVcc, kanał A0
  ADCSRB = 0;                                   // free running
  ADCSRA = _BV(ADEN) | _BV(ADSC) | _BV(ADATE) | _BV(ADIE)
#if ENC_ADC_PRESCALER == 64
         | _BV(ADPS2) | _BV(ADPS1);             // prescaler 64
#elif ENC_ADC_PRESCALER == 128
         | _BV(ADPS2) | _BV(ADPS1) | _BV(ADPS0); // prescaler 128
#else
         | _BV(ADPS2) | _BV(ADPS0);             // prescaler 32
#endif
}

ISR(ADC_vect) {
#if ENC_ISR_PIN
  PORTB |= _BV(1);          // D9 wysoko
#endif
  int16_t raw = ADC;

  // usunięcie DC: EMA k=1/512
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

  int16_t ax = x < 0 ? -x : x;

  acc_abs += ax;
  acc_sq  += (uint32_t)((int32_t)x * x);
  if (ax > acc_max) acc_max = ax;
#if !ENC_SET_SWAP
  if (ax > spike_thr) acc_pk++;
#endif

  // pasmo górne: 1-pole lowpass i odjęcie (hf = x - lp). Cutoff ~2.2 kHz.
#if ENC_HF_ROUND
  hf_lp += (int16_t)((x - hf_lp + (1 << (HF_HP_SHIFT - 1))) >> HF_HP_SHIFT);   // z zaokrągleniem
#else
  hf_lp += (int16_t)((x - hf_lp) >> HF_HP_SHIFT);
#endif
  int16_t hf = x - hf_lp;
  acc_hf_sq += (uint32_t)((int32_t)hf * hf);

#if ENC_SET_SWAP
  // hjorth_mobility i autocorr_lag1: dwa dodatkowe akumulatory (ciągłe po granicach ramek)
  int16_t dx = x - x_prev;
  acc_dx2 += (uint32_t)((int32_t)dx * dx);
  acc_xx1 += (int32_t)x * x_prev;
  x_prev = x;
#endif

  int16_t s = (x >= 0) ? 1 : -1;
  if (s != prev_sign) { acc_zc++; prev_sign = s; }

  if (++n_samp >= HOP_SAMPLES) frame_ready = true;
#if ENC_ISR_PIN
  PORTB &= ~_BV(1);         // D9 nisko
#endif
}

// ---------------------------------------------------------------- pomocnicze

static inline float updateFloor(uint8_t c, float v, bool freeze) {
  if (!freeze) {
    float a = (v > floor_v[c]) ? A_UP : A_DN;
    floor_v[c] += a * (v - floor_v[c]);
    float d = fabsf(v - floor_v[c]);
    mad_v[c]  += A_MAD * (d - mad_v[c]);
  }
#if ENC_EPS_FLOOR
  return (v - floor_v[c]) / (mad_v[c] + EPS_FLOOR_V[c]);
#else
  return (v - floor_v[c]) / (mad_v[c] + EPS);
#endif
}

// ---------------------------------------------------------------- BENCH (ENC_BENCH=1)
// Polecenie 'B' przez Serial (115200, Newline) uruchamia pomiar (~13 s) i wypisuje linie "#BENCH klucz=wartość".
//  Faza A (1 s, ADC-ISR wyłączony)  : puste iteracje pętli -> tempo referencyjne
//  Faza B (1 s, ADC-ISR włączony)   : te same iteracje    -> ułamek CPU zabrany przez ISR (metoda "wolnych cykli",
//                                     obejmuje prolog/epilog ISR i wszystko, co kradnie cykle), oraz fs = n_samp / czas
//  Faza p1 (5 s, normalna praca z wypisywaniem linii) i p2 (5 s, bez wypisywania): czas przetwarzania ramki w loop()
//                                     (zegarowy, wraz z wywłaszczeniem przez ISR), okres ramek, ramki spóźnione (s_n > HOP)
// Wynik zapisz: tools/capture_bench.py (czyta te linie z portu i wpisuje do measurements.json).
#if ENC_BENCH
struct BenchStats { uint32_t frames, late, proc_sum, proc_max, per_min, per_max, per_n, per_sum; uint16_t sn_max; };
static BenchStats bench_s[2];
static bool     bench_active = false, bench_rec = false, bench_have_last = false;
static uint8_t  bench_phase  = 0;
static uint32_t bench_t0_ms = 0, bench_t_start_us = 0, bench_last_us = 0;

static uint32_t bench_spin(uint32_t us) {
  uint32_t t0 = micros(), n = 0;
  while ((uint32_t)(micros() - t0) < us) n++;
  return n;
}
static void bench_kv(const __FlashStringHelper *k, float v, uint8_t dec) {
  Serial.print(F("#BENCH ")); Serial.print(k); Serial.print('='); Serial.println(v, dec);
}
static void bench_reset_acc() {
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    acc_abs = 0; acc_sq = 0; acc_hf_sq = 0; acc_max = 0; acc_zc = 0; n_samp = 0; frame_ready = false;
#if ENC_SET_SWAP
    acc_dx2 = 0; acc_xx1 = 0;
#else
    acc_pk = 0;
#endif
  }
}
static void bench_start() {
  Serial.println(F("#BENCH begin"));
  bench_kv(F("cfg_swap"), ENC_SET_SWAP, 0);   bench_kv(F("cfg_dc_fix"), ENC_DC_FIX, 0);
  bench_kv(F("cfg_hf_round"), ENC_HF_ROUND, 0); bench_kv(F("cfg_eps_floor"), ENC_EPS_FLOOR, 0);
  bench_kv(F("cfg_acc32"), ENC_ACC32, 0);     bench_kv(F("adcsra_ps_bits"), ADCSRA & 7, 0);
  debug_csv = false;
  ADCSRA &= ~_BV(ADIE);                        // A: bez ISR
  Serial.flush();
  uint32_t a0 = micros();
  uint32_t n0 = bench_spin(1000000UL);
  uint32_t a1 = micros();
  bench_reset_acc();
  ADCSRA |= _BV(ADIE);                         // B: z ISR (enkoder liczy akumulatory, loop nie przetwarza ramek)
  uint32_t b0 = micros();
  uint32_t n1 = bench_spin(1000000UL);
  uint32_t b1 = micros();
  ADCSRA &= ~_BV(ADIE);
  uint16_t ns; ATOMIC_BLOCK(ATOMIC_RESTORESTATE) { ns = n_samp; }
  float r0 = (float)n0 / (float)(a1 - a0), r1 = (float)n1 / (float)(b1 - b0);   // iteracje na µs
  float util = 1.0f - r1 / r0;
  float fs   = (float)ns * 1e6f / (float)(b1 - b0);
  bench_kv(F("spin_ref_per_us"), r0, 4);
  bench_kv(F("isr_cpu_pct"), 100.0f * util, 2);
  bench_kv(F("fs_hz"), fs, 1);
  bench_kv(F("isr_us_mean"), 1e6f * util / fs, 2);
  bench_kv(F("isr_cycles_mean"), 16.0f * 1e6f * util / fs, 1);
  // faza p1/p2: normalna praca od zera (priming), statystyki tylko dla ramek po primingu
  bench_reset_acc();
  floors_primed = false; frame_idx = 0; rms_prev = 0.0f;
  for (uint8_t c = 0; c < N_CH; c++) { floor_v[c] = 0; mad_v[c] = 0; }
  memset(bench_s, 0, sizeof(bench_s));
  debug_csv = false; bench_phase = 1; bench_t0_ms = millis(); bench_have_last = false; bench_rec = false; bench_active = true;
  ADCSRA |= _BV(ADIE);
}
static void bench_report() {
  for (uint8_t i = 0; i < 2; i++) {
    BenchStats &b = bench_s[i];
    Serial.print(F("#BENCH p")); Serial.print(i + 1);
    Serial.print(F(" frames="));       Serial.print(b.frames);
    Serial.print(F(" late="));         Serial.print(b.late);
    Serial.print(F(" sn_max="));       Serial.print(b.sn_max);
    Serial.print(F(" proc_us_mean=")); Serial.print(b.frames ? b.proc_sum / b.frames : 0);
    Serial.print(F(" proc_us_max="));  Serial.print(b.proc_max);
    Serial.print(F(" per_us_min="));   Serial.print(b.per_min);
    Serial.print(F(" per_us_mean="));  Serial.print(b.per_n ? b.per_sum / b.per_n : 0);
    Serial.print(F(" per_us_max="));   Serial.println(b.per_max);
  }
  Serial.println(F("#BENCH end"));
}
static void bench_tick() {
  if (!bench_active) return;
  uint32_t el = millis() - bench_t0_ms;
  if (bench_phase == 1 && el >= 5000UL) { bench_phase = 2; debug_csv = false; bench_t0_ms = millis(); bench_have_last = false; }
  else if (bench_phase == 2 && el >= 5000UL) { bench_active = false; bench_phase = 0; debug_csv = true; bench_report(); }
}
static inline void bench_frame_start(uint16_t s_n) {
  if (!bench_active || !floors_primed) { bench_rec = false; return; }
  uint32_t now = micros();
  BenchStats &b = bench_s[bench_phase - 1];
  bench_rec = true; b.frames++;
  if (s_n > HOP_SAMPLES) b.late++;
  if (s_n > b.sn_max) b.sn_max = s_n;
  if (bench_have_last) {
    uint32_t p = now - bench_last_us;
    if (b.per_n == 0 || p < b.per_min) b.per_min = p;
    if (p > b.per_max) b.per_max = p;
    b.per_sum += p; b.per_n++;
  }
  bench_last_us = now; bench_have_last = true; bench_t_start_us = now;
}
static inline void bench_frame_end() {
  if (!bench_rec) return;
  bench_rec = false;
  uint32_t d = micros() - bench_t_start_us;
  BenchStats &b = bench_s[bench_phase - 1];
  b.proc_sum += d; if (d > b.proc_max) b.proc_max = d;
}
#define BENCH_TICK()          bench_tick()
#define BENCH_FRAME_START(n)  bench_frame_start(n)
#define BENCH_FRAME_END()     bench_frame_end()
#else
#define BENCH_TICK()
#define BENCH_FRAME_START(n)
#define BENCH_FRAME_END()
#endif

// ---------------------------------------------------------------- CALIB

/*  Format: "C <pin> <n> <rate_hz>\n"
 *  np. "C 2 7 100" -> 7 impulsów po 6 ms na D2 z częstotliwością 100 Hz.
 *  Używane w Fazie A/C kalibracji: szukasz najmniejszego n, przy którym płytka odpala.
 */
void runCalibPulses(uint8_t pin, uint16_t n, uint16_t rate_hz) {
  if (pin < 2 || pin > 8) return;   // D2..D8 = 7 kanałów
  uint32_t period_us = 1000000UL / (rate_hz ? rate_hz : 100);
  if (period_us < (uint32_t)PULSE_MS * 1000UL + 500UL) period_us = (uint32_t)PULSE_MS * 1000UL + 500UL;

  for (uint16_t i = 0; i < n; i++) {
    uint32_t t0 = micros();
    digitalWrite(pin, HIGH);
    delay(PULSE_MS);
    digitalWrite(pin, LOW);
    while (micros() - t0 < period_us) { /* spin */ }
  }
  Serial.print(F("# calib done pin=")); Serial.print(pin);
  Serial.print(F(" n=")); Serial.print(n);
  Serial.print(F(" rate=")); Serial.println(rate_hz);
}

void handleSerial() {
  if (!Serial.available()) return;
  char cmd = Serial.read();
  if (cmd == 'C') {
    uint8_t  pin  = Serial.parseInt();
    uint16_t n    = Serial.parseInt();
    uint16_t rate = Serial.parseInt();
    calib_mode = true;
    ADCSRA &= ~_BV(ADIE);          // wyłącz enkoder na czas testu
    runCalibPulses(pin, n, rate);
    ADCSRA |= _BV(ADIE);
    calib_mode = false;
#if ENC_BENCH
  } else if (cmd == 'B') {
    bench_start();
#endif
  } else if (cmd == 'D') {
    debug_csv = !debug_csv;
  } else if (cmd == 'R') {         // reset adaptacji floorów
    floors_primed = false;
    for (uint8_t c = 0; c < N_CH; c++) { floor_v[c] = 0; mad_v[c] = 0; }
  }
}

// ---------------------------------------------------------------- setup / loop

void setup() {
  Serial.begin(ENC_BAUD);
  for (uint8_t p = 2; p <= 8; p++) pinMode(p, OUTPUT);   // D2..D8 = 7 kanałów
#if ENC_ISR_PIN
  pinMode(9, OUTPUT);
#endif
  PORTD &= ~PIND_MASK;
  PORTB &= ~PINB_MASK;
  setupADC();
  sei();
  Serial.println(F("# encoder_v2 dt=10ms pulse=6ms ch=peak,peak_cnt,cv,zcr,flux,hf_lo,hf_hi"));
  Serial.println(F("frame,s0,s1,s2,s3,s4,s5,s6"));
}

void loop() {
  handleSerial();
  BENCH_TICK();

  // wyłączenie impulsu — nieblokująco, wszystkie kanały razem
  if (pulse_active && (int32_t)(micros() - pulse_off_us) >= 0) {
    PORTD &= ~PIND_MASK;
    PORTB &= ~PINB_MASK;
    pulse_active = false;
  }

  if (!frame_ready || calib_mode) return;

#if ENC_ACC32
  int32_t  s_abs; uint32_t s_sq, s_hf_sq; int16_t s_max; uint16_t s_zc, s_pk, s_n;
#else
  int32_t  s_abs; uint64_t s_sq, s_hf_sq; int16_t s_max; uint16_t s_zc, s_pk, s_n;
#endif
#if ENC_SET_SWAP
  uint32_t s_dx2; int32_t s_xx1; int16_t s_xlast;
#endif
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    s_abs = acc_abs; s_sq = acc_sq; s_hf_sq = acc_hf_sq; s_max = acc_max;
#if ENC_SET_SWAP
    s_zc = acc_zc;  s_pk = 0;       s_n = n_samp;
    s_dx2 = acc_dx2; s_xx1 = acc_xx1; s_xlast = x_prev;
    acc_abs = 0; acc_sq = 0; acc_hf_sq = 0; acc_max = 0; acc_zc = 0; acc_dx2 = 0; acc_xx1 = 0; n_samp = 0;
#else
    s_zc = acc_zc;  s_pk = acc_pk;  s_n = n_samp;
    acc_abs = 0; acc_sq = 0; acc_hf_sq = 0; acc_max = 0; acc_zc = 0; acc_pk = 0; n_samp = 0;
#endif
    frame_ready = false;
  }

  BENCH_FRAME_START(s_n);

  // --- cechy ---------------------------------------------------------------
  float inv_n    = 1.0f / (float)s_n;
  float mean_abs = (float)s_abs * inv_n;
  float rms      = sqrtf((float)s_sq * inv_n);
  float peak     = (float)s_max;

  // wariancja |x| -> CV. E[x^2] - E[|x|]^2 jest nieobciążone dla |x| tylko przy
  // symetrycznym rozkładzie; dla audio po usunięciu DC to bardzo dobre przybliżenie.
  float var_abs  = fmaxf(0.0f, (float)s_sq * inv_n - mean_abs * mean_abs);
#if ENC_SET_SWAP
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

  // hf_ratio = udział energii pasma górnego w energii ramki (cecha widmowa)
  float hf_ratio = (float)s_hf_sq / ((float)s_sq + EPS);

  // flux = tylko dodatni przyrost log-RMS (detektor ataku, nie wygaszania)
  float lr   = logf(rms + 1.0f);
  float lrp  = logf(rms_prev + 1.0f);
  float flux = fmaxf(0.0f, lr - lrp);
  rms_prev   = rms;

  // próg mikro-szpilki na następną ramkę: 3x poziom tła obwiedni
#if !ENC_SET_SWAP
  spike_thr = (int16_t)fminf(1023.0f, fmaxf(8.0f, 3.0f * (floor_v[CH_PEAK] + EPS)));
#endif

  // hf_lo/hf_hi dostają tę samą wartość hf_ratio (różni je próg bezwzględny niżej)
#if ENC_SET_SWAP
  float feat[N_CH] = { peak, mob, ac, zcr, flux, hf_ratio, hf_ratio };
#else
  float feat[N_CH] = { peak, peak_cnt, cv, zcr, flux, hf_ratio, hf_ratio };
#endif

  // pierwsze ~0.5 s tylko primuje floory, nie strzela
  if (!floors_primed) {
    for (uint8_t c = 0; c < N_CH; c++) { floor_v[c] = feat[c]; mad_v[c] = 0.1f * fabsf(feat[c]) + EPS; }
    if (frame_idx > 50) floors_primed = true;
    frame_idx++;
    BENCH_FRAME_END();
    return;
  }

  // bramka zdarzenia dla kanałów widmowych: hf_ratio na ciszy to szum z dzielenia,
  // więc hf_lo/hf_hi strzelają tylko gdy ramka ma transient (peak nad floorem)
  bool hf_gated = peak > HF_GATE_MULT * (floor_v[CH_PEAK] + EPS);

  // --- progowanie + refrakcja ---------------------------------------------
  // fired: bitmapa logiczna (bit c = kanał c strzelił), niezależna od mapowania pinów
  uint8_t fired = 0;
  for (uint8_t c = 0; c < N_CH; c++) {
    bool above;
    if (c == CH_HFLO) {
      above = hf_gated && (hf_ratio > HF_LO_THR);    // próg bezwzględny, bez floora
    } else if (c == CH_HFHI) {
      above = hf_gated && (hf_ratio > HF_HI_THR);
#if ENC_SET_SWAP
    } else if (c == CH_MOB) {                 // poziom, próg bezwzględny + bramka jak hf
      above = hf_gated && (MOB_FIRE_BELOW ? (mob < MOB_THR) : (mob > MOB_THR));
    } else if (c == CH_AC) {
      above = hf_gated && (AC_FIRE_BELOW ? (ac < AC_THR) : (ac > AC_THR));
#endif
    } else {
      above = (updateFloor(c, feat[c], /*freeze=*/false) > THR_Z[c]);
      // zamrożenie adaptacji, gdy kanał jest w zdarzeniu: floor nie ma się uczyć szkła
      if (above) { floor_v[c] -= A_UP * (feat[c] - floor_v[c]); }
    }

    if (above && refrac[c] == 0) {
      fired |= (1 << c);
      refrac[c] = REFRAC_FRAMES;
    } else if (refrac[c]) {
      refrac[c]--;
    }
  }

  // --- wystawienie impulsów: kanały 0..5 na PORTD, kanał 6 na PORTB ---

  if (fired) {
    uint8_t bitsD = (fired & 0x3F) << PIND_BASE;    // kanały 0..5 -> bity 2..7
    uint8_t bitsB = (fired >> 6) & PINB_MASK;        // kanał 6     -> PORTB bit 0
    PORTD |= bitsD;
    PORTB |= bitsB;
    pulse_off_us = micros() + (uint32_t)PULSE_MS * 1000UL;
    pulse_active = true;
  }

  if (debug_csv) {
    Serial.print(frame_idx);
#if ENC_DEBUG_FEAT
    for (uint8_t c = 0; c < N_CH; c++) { Serial.print(','); Serial.print(feat[c], 5); }
#endif
    for (uint8_t c = 0; c < N_CH; c++) {
      Serial.print(',');
      Serial.print((fired >> c) & 1);
    }
    Serial.println();
  }
  frame_idx++;
  BENCH_FRAME_END();
}
