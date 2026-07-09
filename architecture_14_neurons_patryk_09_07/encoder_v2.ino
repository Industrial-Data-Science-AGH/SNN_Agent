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

// ---------------------------------------------------------------- konfiguracja

#define FS_HZ        19231UL   // realna fs przy prescalerze 32
#define HOP_SAMPLES  192       // 192 / 19231 Hz ~= 10.0 ms  <-- to jest dt sieci
#define PULSE_MS     6         // < HOP_MS, inaczej impulsy się skleją
#define N_CH         6

#define PIN_BASE     2         // D2..D7
#define PIN_MASK     0b11111100

// kolejność kanałów == kolejność kolumn s0..s5 w CSV i w pipeline
enum Ch { CH_PEAK = 0, CH_PEAKCNT, CH_CREST, CH_CV, CH_ZCR, CH_FLUX };

// progi w jednostkach z = (feature - floor) / (MAD + eps)
// dostrój na swoim zbiorze: podnieś, jeśli tło strzela; obniż, jeśli szkło ucieka
static float THR_Z[N_CH] = { 4.0, 3.5, 3.0, 3.0, 2.5, 3.5 };

// asymetryczna adaptacja floora: rośnie wolno, spada szybko -> śledzi poziom ciszy
#define A_UP    0.0015f
#define A_DN    0.0300f
#define A_MAD   0.0100f
#define EPS     1e-6f

// refrakcja w ramkach (1 = kanał może strzelać co ramkę, tj. 100 Hz)
#define REFRAC_FRAMES 1

// ---------------------------------------------------------------- stan ISR

volatile int32_t  acc_abs;      // suma |x|
volatile uint64_t acc_sq;       // suma x^2
volatile int16_t  acc_max;      // max |x|
volatile uint16_t acc_zc;       // przejścia przez zero
volatile uint16_t acc_pk;       // próbki powyżej progu mikro-szpilki
volatile uint16_t n_samp;
volatile bool     frame_ready;

volatile int16_t  dc_est   = 512 << 4;  // Q4, średnia bieżąca ADC
volatile int16_t  spike_thr = 40;       // próg mikro-szpilki, aktualizowany co ramkę
static   int16_t  prev_sign = 0;

// ---------------------------------------------------------------- stan ramki

static float floor_v[N_CH] = {0}, mad_v[N_CH] = {0};
static uint8_t refrac[N_CH] = {0};
static float rms_prev = 0.0f;
static bool  floors_primed = false;
static uint32_t frame_idx = 0;

// pulse-off bez blokowania
static uint32_t pulse_off_us = 0;
static bool     pulse_active = false;

// tryby
static bool debug_csv = true;
static bool calib_mode = false;

// ---------------------------------------------------------------- ADC

void setupADC() {
  ADMUX  = _BV(REFS0);                          // AVcc, kanał A0
  ADCSRB = 0;                                   // free running
  ADCSRA = _BV(ADEN) | _BV(ADSC) | _BV(ADATE) | _BV(ADIE)
         | _BV(ADPS2) | _BV(ADPS0);             // prescaler 32
}

ISR(ADC_vect) {
  int16_t raw = ADC;

  // usunięcie DC: EMA w Q4
  dc_est += (int16_t)(((int32_t)(raw << 4) - dc_est) >> 9);
  int16_t x = raw - (dc_est >> 4);

  int16_t ax = x < 0 ? -x : x;

  acc_abs += ax;
  acc_sq  += (uint32_t)((int32_t)x * x);
  if (ax > acc_max) acc_max = ax;
  if (ax > spike_thr) acc_pk++;

  int16_t s = (x >= 0) ? 1 : -1;
  if (s != prev_sign) { acc_zc++; prev_sign = s; }

  if (++n_samp >= HOP_SAMPLES) frame_ready = true;
}

// ---------------------------------------------------------------- pomocnicze

static inline float updateFloor(uint8_t c, float v, bool freeze) {
  if (!freeze) {
    float a = (v > floor_v[c]) ? A_UP : A_DN;
    floor_v[c] += a * (v - floor_v[c]);
    float d = fabsf(v - floor_v[c]);
    mad_v[c]  += A_MAD * (d - mad_v[c]);
  }
  return (v - floor_v[c]) / (mad_v[c] + EPS);
}

// ---------------------------------------------------------------- CALIB

/*  Format: "C <pin> <n> <rate_hz>\n"
 *  np. "C 2 7 100" -> 7 impulsów po 6 ms na D2 z częstotliwością 100 Hz.
 *  Używane w Fazie A/C kalibracji: szukasz najmniejszego n, przy którym płytka odpala.
 */
void runCalibPulses(uint8_t pin, uint16_t n, uint16_t rate_hz) {
  if (pin < PIN_BASE || pin > PIN_BASE + N_CH - 1) return;
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
  } else if (cmd == 'D') {
    debug_csv = !debug_csv;
  } else if (cmd == 'R') {         // reset adaptacji floorów
    floors_primed = false;
    for (uint8_t c = 0; c < N_CH; c++) { floor_v[c] = 0; mad_v[c] = 0; }
  }
}

// ---------------------------------------------------------------- setup / loop

void setup() {
  Serial.begin(115200);
  for (uint8_t p = PIN_BASE; p < PIN_BASE + N_CH; p++) pinMode(p, OUTPUT);
  PORTD &= ~PIN_MASK;
  setupADC();
  sei();
  Serial.println(F("# encoder_v2 dt=10ms pulse=6ms ch=peak,peak_cnt,crest,cv,zcr,flux"));
  Serial.println(F("frame,s0,s1,s2,s3,s4,s5"));
}

void loop() {
  handleSerial();

  // wyłączenie impulsu — nieblokująco, wszystkie kanały razem
  if (pulse_active && (int32_t)(micros() - pulse_off_us) >= 0) {
    PORTD &= ~PIN_MASK;
    pulse_active = false;
  }

  if (!frame_ready || calib_mode) return;

  int32_t  s_abs; uint64_t s_sq; int16_t s_max; uint16_t s_zc, s_pk, s_n;
  ATOMIC_BLOCK(ATOMIC_RESTORESTATE) {
    s_abs = acc_abs; s_sq = acc_sq; s_max = acc_max;
    s_zc = acc_zc;  s_pk = acc_pk;  s_n = n_samp;
    acc_abs = 0; acc_sq = 0; acc_max = 0; acc_zc = 0; acc_pk = 0; n_samp = 0;
    frame_ready = false;
  }

  // --- cechy ---------------------------------------------------------------
  float inv_n    = 1.0f / (float)s_n;
  float mean_abs = (float)s_abs * inv_n;
  float rms      = sqrtf((float)s_sq * inv_n);
  float peak     = (float)s_max;

  // wariancja |x| -> CV. E[x^2] - E[|x|]^2 jest nieobciążone dla |x| tylko przy
  // symetrycznym rozkładzie; dla audio po usunięciu DC to bardzo dobre przybliżenie.
  float var_abs  = fmaxf(0.0f, (float)s_sq * inv_n - mean_abs * mean_abs);
  float cv       = sqrtf(var_abs) / (mean_abs + EPS);

  float crest    = peak / (rms + EPS);
  float zcr      = (float)s_zc * inv_n;
  float peak_cnt = (float)s_pk;

  // flux = tylko dodatni przyrost log-RMS (detektor ataku, nie wygaszania)
  float lr   = logf(rms + 1.0f);
  float lrp  = logf(rms_prev + 1.0f);
  float flux = fmaxf(0.0f, lr - lrp);
  rms_prev   = rms;

  // próg mikro-szpilki na następną ramkę: 3x poziom tła obwiedni
  spike_thr = (int16_t)fminf(1023.0f, fmaxf(8.0f, 3.0f * (floor_v[CH_PEAK] + EPS)));

  float feat[N_CH] = { peak, peak_cnt, crest, cv, zcr, flux };

  // pierwsze ~0.5 s tylko primuje floory, nie strzela
  if (!floors_primed) {
    for (uint8_t c = 0; c < N_CH; c++) { floor_v[c] = feat[c]; mad_v[c] = 0.1f * fabsf(feat[c]) + EPS; }
    if (frame_idx > 50) floors_primed = true;
    frame_idx++;
    return;
  }

  // --- progowanie + refrakcja ---------------------------------------------
  uint8_t bits = 0;
  for (uint8_t c = 0; c < N_CH; c++) {
    bool above = (updateFloor(c, feat[c], /*freeze=*/false) > THR_Z[c]);

    // zamrożenie adaptacji, gdy kanał jest w zdarzeniu: floor nie ma się uczyć szkła
    if (above) { floor_v[c] -= A_UP * (feat[c] - floor_v[c]); }

    if (above && refrac[c] == 0) {
      bits |= (1 << (PIN_BASE + c));
      refrac[c] = REFRAC_FRAMES;
    } else if (refrac[c]) {
      refrac[c]--;
    }
  }

  // --- wystawienie impulsów: wszystkie kanały w tej samej mikrosekundzie ---
  if (bits) {
    PORTD |= bits;
    pulse_off_us = micros() + (uint32_t)PULSE_MS * 1000UL;
    pulse_active = true;
  }

  if (debug_csv) {
    Serial.print(frame_idx);
    for (uint8_t c = 0; c < N_CH; c++) {
      Serial.print(',');
      Serial.print((bits >> (PIN_BASE + c)) & 1);
    }
    Serial.println();
  }
  frame_idx++;
}
