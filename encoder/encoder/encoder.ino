// ============================================================
//  SNN Spike Encoder — Arduino  (Wersja 3-Kanałowa)
//  Projekt: detekcja rozbijanego szkła
//
//  Wejście:  mikrofon MAX 4466 na pinie A0
//  Wyjścia:  3 piny cyfrowe (Kanał 0: Peak, Kanał 1: Mean, Kanał 2: Std)
//  Modyfikacja: Dostosowano pod układ LUI (Szerokość impulsu ~15ms)
// ============================================================

#include "params.h"
#include "wellford.h"


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
  Serial.println(F("=== TRYB KALIBRACJI — 10 sekund ==="));
  Serial.println(F("Puszczaj dzwieki stluczonego szkla"));
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

#if !AUTO_CALIBRATE
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

#if !AUTO_CALIBRATE
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