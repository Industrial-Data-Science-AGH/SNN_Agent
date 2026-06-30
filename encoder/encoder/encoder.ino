// ============================================================
//  SNN Spike Encoder — Arduino  (Wersja 3-Kanałowa)
//  Projekt: detekcja rozbijanego szkła
//
//  Wejście:  mikrofon (WM61-A) na pinie A0
//  Wyjścia:  3 piny cyfrowe (Kanał 0: Peak, Kanał 1: Mean, Kanał 2: Std)
//  Modyfikacja: Dostosowano pod układ LUI (Szerokość impulsu ~15ms)
// ============================================================

#define ENCODER_MODE RATE_CODING

#define RATE_CODING 1
#define TTFS 2

// czy kalibrujemy i obliczamy max vals
#define AUTO_CALIBRATE 0

// PINOUT
#define PIN_MIC A0
#define PIN_SPIKE_PEAK 6 // peak energy
#define PIN_SPIKE_MEAN 7 // mean energy
#define PIN_SPIKE_CV 8   // coeef of var
#define PIN_DEBUG_LED 13

// okno analizy audio
#define FRAME_WINDOW_MS 20

const uint8_t SPIKE_PINS[3] = {PIN_SPIKE_PEAK, PIN_SPIKE_MEAN, PIN_SPIKE_CV};

#define RC_MIN_RATE_HZ 5 // min czestotilowsc spike

// [ZMIANA DLA LUI] Obniżono z 200 Hz do 50 Hz. 
// Przy 50 Hz maksymalny okres (ISI) to 20ms. Pozwala to na pełny impuls 15ms + 5ms przerwy.
#define RC_MAX_RATE_HZ 50 

#define RC_NOISE_FLOOR 0.05f // ponizej 5% normalized cisza
#define TTFS_THRESHOLD 0.1f // 10%

// 1.0 = brak filtra wygladzenia
const float LP_ALPHA[3] = {1.0f, 0.4f, 0.4f};


// [ZMIANA DLA LUI] Zmniejszono z 20000us na 5000us (5ms).
// Maksymalne opóźnienie szpilki (5ms) + czas trwania szpilki (15ms) = 20ms.
// Dzięki temu szpilka TTFS zawsze zamknie się w obrębie bieżącej ramki i nie zablokuje kolejnej.
#define TTFS_WINDOW_US 5000UL

#if AUTO_CALIBRATE == 0
  // domyślne wartości dla mikrofonu ze wzmocnieniem i bez
  #define STRENGTHENED 0
    #if STRENGTHENED
      #define MAX_PEAK_VAL   950.0f 
      #define MAX_MEAN_VAL   520.0f
      #define MAX_STD_VAL    0.4f
    #else
      #define MAX_PEAK_VAL   556.6f 
      #define MAX_MEAN_VAL   519.2f
      #define MAX_STD_VAL    0.03f
    #endif
  
  const float MAX_VALS[3] = {MAX_PEAK_VAL, MAX_MEAN_VAL, MAX_STD_VAL};
#endif

// [ZMIANA DLA LUI] Zwiększono z 500us do 15000us (15ms), ponieważ LUI 
// ignoruje amplitudę (>1.8V), a siłę synapsy skaluje liniowo szerokością pulsu.
#define SPIKE_WIDTH_US 15000UL

#define HPF_ENABLED 0
#define ALPHA 0.99f

// wellford online algo dla std
static float wf_mean = 0.0f;
static float wf_M2 = 0.0f;
static uint32_t wf_n = 0;

static float frameMax = 0.0f;
static float hp_filtered = 0.0f;
static float prev_raw = 450.0f;

static float channelValues[3] = {0.0f, 0.0f, 0.0f};
static float smoothedVals[3] = {0.0f, 0.0f, 0.0f};

static uint32_t frameStartMs    = 0;
static bool     newFrameReady   = false;

// Spike generation state
static uint32_t currISI_us[3]    = {0, 0, 0};
static uint32_t lastSpikeTime_us[3] = {0, 0, 0};
static bool     spikeActive[3]   = {false, false, false};
static uint32_t spikeStartUs[3]  = {0, 0, 0};

// TTFS state
static bool     ttfsSpiked[3]       = {false, false, false};
static uint32_t ttfsSpikeAt_us[3]   = {0, 0, 0};  
static uint32_t ttfsFrameStart_us   = 0;

#if AUTO_CALIBRATE
static float cal_maxPeak = 0.0f;
static float cal_maxMean = 0.0f;
static float cal_maxCV   = 0.0f;
static uint32_t calStartMs = 0;
static bool calDone = false;
#define CAL_DURATION_MS 10000
#endif

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 3; i++) {
    pinMode(SPIKE_PINS[i], OUTPUT);
    digitalWrite(SPIKE_PINS[i], LOW);
  }
  pinMode(PIN_DEBUG_LED, OUTPUT);
  digitalWrite(PIN_DEBUG_LED, LOW);

  // adc preskaler na 16
  ADCSRA = (ADCSRA & ~0x07) | 0x04;

  frameStartMs      = millis();
  ttfsFrameStart_us = micros();

  Serial.println("\n\n\n");

#if AUTO_CALIBRATE
  calStartMs = millis();
  Serial.println(F("=== TRYB KALIBRACJI — 5 sekund ==="));
#else
  Serial.println(F("=== SNN 3-Channel Encoder START ==="));
  #if ENCODER_MODE == RATE_CODING
    Serial.println(F("Tryb: RATE CODING"));
  #else
    Serial.println(F("Tryb: TTFS"));
  #endif
#endif
}

void processAudio() {
  float raw = (float)analogRead(PIN_MIC);

#if HPF_ENABLED
  hp_filtered = ALPHA * (hp_filtered + raw - prev_raw);
  prev_raw    = raw;
  float val   = fabs(hp_filtered);
#else
  float val = raw;
#endif

  wf_n++;
  float delta  = val - wf_mean;
  wf_mean     += delta / (float)wf_n;
  float delta2 = val - wf_mean;
  wf_M2       += delta * delta2;

  if (val > frameMax) frameMax = val;

  uint32_t now = millis();
  if (now - frameStartMs >= FRAME_WINDOW_MS) {
    if (wf_n < 2) { wf_n = 2; wf_M2 = 0.001f; } 

    float mean_val = wf_mean;
    float std_val  = sqrt(wf_M2 / (float)(wf_n - 1));  
    float cv_val   = (mean_val > 1.0f) ? (std_val / mean_val) : 0.0f;

    channelValues[0] = frameMax;   
    channelValues[1] = mean_val;   
    channelValues[2] = cv_val;     

    wf_mean  = 0.0f;
    wf_M2    = 0.0f;
    wf_n     = 0;
    frameMax = 0.0f;
    frameStartMs = now;
    newFrameReady = true;
  }
}

void fireSpike(uint8_t ch) {
  if (spikeActive[ch]) return;  
  digitalWrite(SPIKE_PINS[ch], HIGH);
  digitalWrite(PIN_DEBUG_LED, HIGH);
  spikeActive[ch]  = true;
  spikeStartUs[ch] = micros();
}

void updateSpikes() {
  uint32_t now_us     = micros();
  bool     anyActive  = false;
  for (int i = 0; i < 3; i++) {
    if (spikeActive[i]) {
      if (now_us - spikeStartUs[i] >= SPIKE_WIDTH_US) {
        digitalWrite(SPIKE_PINS[i], LOW);
        spikeActive[i] = false;
      } else {
        anyActive = true;
      }
    }
  }
  if (!anyActive) digitalWrite(PIN_DEBUG_LED, LOW);
}

#if AUTO_CALIBRATE == 0
void loopRateCoding() {
  processAudio();

  if (newFrameReady) {
    newFrameReady = false;

    for (int i = 0; i < 3; i++) {
      float a    = LP_ALPHA[i];
      smoothedVals[i] = a * channelValues[i] + (1.0f - a) * smoothedVals[i];

      float norm = min(smoothedVals[i] / MAX_VALS[i], 1.0f);

      if (norm < RC_NOISE_FLOOR) {
        currISI_us[i] = 0;  
      } else {
        float rate_hz  = RC_MIN_RATE_HZ + norm * (RC_MAX_RATE_HZ - RC_MIN_RATE_HZ);
        currISI_us[i]  = (uint32_t)(1e6f / rate_hz);
      }
    }

    Serial.print(F("P=")); Serial.print(smoothedVals[0], 1);
    Serial.print(F(" M=")); Serial.print(smoothedVals[1], 1);
    Serial.print(F(" CV=")); Serial.print(smoothedVals[2], 3);
    Serial.print(F(" | ISI(us): "));
    for (int i = 0; i < 3; i++) { Serial.print(currISI_us[i]); Serial.print(' '); }
    Serial.println();
  }

  uint32_t now_us = micros();
  for (int i = 0; i < 3; i++) {
    if (currISI_us[i] > 0 && (now_us - lastSpikeTime_us[i]) >= currISI_us[i]) {
      lastSpikeTime_us[i] = now_us;
      fireSpike(i);
    }
  }
}
#endif

#if AUTO_CALIBRATE == 0
void loopTTFS() {
  processAudio();

  if (newFrameReady) {
    newFrameReady = false;
    ttfsFrameStart_us = micros();

    for (int i = 0; i < 3; i++) {
      ttfsSpiked[i] = false;
      float norm = min(smoothedVals[i] / MAX_VALS[i], 1.0f);
      
      // opóźniam spike'a proporcjonalnie do siły sygnału z mikrofonu
      if (norm < TTFS_THRESHOLD) {
        ttfsSpikeAt_us[i] = 0;  
      } else {
        float delay_frac   = 1.0f - norm; 
        uint32_t delay_us  = (uint32_t)(delay_frac * TTFS_WINDOW_US);
        ttfsSpikeAt_us[i]  = ttfsFrameStart_us + delay_us;
      }
    }
  }

  uint32_t now_us = micros();
  for (int i = 0; i < 3; i++) {
    if (!ttfsSpiked[i] && ttfsSpikeAt_us[i] > 0) {
      if (now_us >= ttfsSpikeAt_us[i]) {
        fireSpike(i);
        ttfsSpiked[i] = true;
      }
    }
  }
}
#endif

#if AUTO_CALIBRATE
void loopCalibrate() {
  processAudio();
  if (newFrameReady) {
    newFrameReady = false;
    if (channelValues[0] > cal_maxPeak) cal_maxPeak = channelValues[0];
    if (channelValues[1] > cal_maxMean) cal_maxMean = channelValues[1];
    if (channelValues[2] > cal_maxCV)   cal_maxCV   = channelValues[2];
  }

  if (!calDone && (millis() - calStartMs >= CAL_DURATION_MS)) {
    calDone = true;
    Serial.println(F("\n=== KALIBRACJA ZAKONCZONA ==="));
    Serial.print(F("#define MAX_PEAK_VAL  ")); Serial.println(cal_maxPeak * 1.1f, 1);
    Serial.print(F("#define MAX_MEAN_VAL  ")); Serial.println(cal_maxMean * 1.1f, 1);
    Serial.print(F("#define MAX_CV_VAL    ")); Serial.println(cal_maxCV   * 1.1f, 2);
  }
}
#endif

void loop() {
  updateSpikes();

#if AUTO_CALIBRATE
  loopCalibrate();
  return;
#else
  #if ENCODER_MODE == RATE_CODING
    loopRateCoding();
  #else
    loopTTFS();
  #endif
#endif
}