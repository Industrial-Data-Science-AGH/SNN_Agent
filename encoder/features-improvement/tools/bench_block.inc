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
  debug_csv = true; bench_phase = 1; bench_t0_ms = millis(); bench_have_last = false; bench_rec = false; bench_active = true;
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

