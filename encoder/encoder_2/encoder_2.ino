#include "params.h"
#include "features.h"

void setup() {
  Serial.begin(115200);

  for (int i = 0; i < 4; i++) {
    pinMode(SPIKE_PINS[i], OUTPUT);
    digitalWrite(SPIKE_PINS[i], LOW);
  }
  pinMode(PIN_DEBUG_LED, OUTPUT);
  digitalWrite(PIN_DEBUG_LED, LOW);

  // ADC preskaler na 16 (szybsze próbkowanie audio)
  ADCSRA = (ADCSRA & ~0x07) | 0x04;

  frameStartMs      = millis();
  ttfsFrameStart_us = micros();

  Serial.println(F("\n=== SNN 4-Channel Feature Encoder START ==="));
}

void processAudio() {
  float raw = (float)analogRead(PIN_MIC); // Odpowiednik symulacji "raw" z Pytona
  globalSampleIdx++;

  float val_centered = 0.0f;
  float val_abs = 0.0f;

#if HPF_ENABLED
  hp_filtered = ALPHA * (hp_filtered + raw - prev_raw);
  prev_raw    = raw;
  val_centered = hp_filtered;
  val_abs      = fabs(hp_filtered);
#else
  val_centered = raw - 512.0f; 
  val_abs      = raw; 
#endif

  // --- ALGORYTM 1: ZERO CROSSING RATE ---
  if (globalSampleIdx > 1) {
    if ((val_centered > 0.0f && prev_val_centered <= 0.0f) || 
        (val_centered < 0.0f && prev_val_centered >= 0.0f)) {
      zcr_count++;
    }
  }
  prev_val_centered = val_centered;

  // --- ALGORYTM 2 & 3: AKUMULACJA ENERGII I SZCZYTU ---
  de_sum_val += val_abs;
  de_sample_cnt++;

  cf_sum_val += val_abs;
  cf_sample_cnt++;
  if (val_abs > cf_frame_max) {
    cf_frame_max = val_abs;
  }

  // --- ALGORYTM 4: PEAK COUNTING (Lokalne maksima) ---
  if (globalSampleIdx > 2) {
    if (p_val > pp_val && p_val > val_abs) {
      local_peaks_count++;
    }
  }
  pp_val = p_val;
  p_val = val_abs;

  // --- ZAMKNIĘCIE RAMKI CZASOWEJ ---
  uint32_t now = millis();
  if (now - frameStartMs >= FRAME_WINDOW_MS) {
    
    // Obliczenia końcowe dla Ramki:
    
    // Kanał 0: ZCR
    channelValues[0] = (float)zcr_count;
    
    // Kanał 1: Delta Energii
    float de_current_mean = de_sum_val / (de_sample_cnt > 0 ? de_sample_cnt : 1);
    channelValues[1] = fabs(de_current_mean - prev_frame_mean); // Wartość bezwzględna zmiany
    prev_frame_mean = de_current_mean;

    // Kanał 2: Crest Factor
    float cf_mean_val = cf_sum_val / (cf_sample_cnt > 0 ? cf_sample_cnt : 1);
    channelValues[2] = (cf_mean_val > 1.0f) ? (cf_frame_max / cf_mean_val) : 1.0f;

    // Kanał 3: Peak Counting
    channelValues[3] = (float)local_peaks_count;

    // Reset zmiennych ramkowych
    zcr_count = 0;
    de_sum_val = 0.0f; de_sample_cnt = 0;
    cf_sum_val = 0.0f; cf_frame_max = 0.0f; cf_sample_cnt = 0;
    local_peaks_count = 0;
    if(globalSampleIdx > 10000) globalSampleIdx = 3; // Zabezpieczenie przed przepełnieniem licznika

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
  uint32_t now_us = micros();
  bool anyActive  = false;
  for (int i = 0; i < 4; i++) {
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

void loopRateCoding() {
  processAudio();

  if (newFrameReady) {
    newFrameReady = false;

    for (int i = 0; i < 4; i++) {
      float a = LP_ALPHA[i];
      smoothedVals[i] = a * channelValues[i] + (1.0f - a) * smoothedVals[i];
      float norm = min(smoothedVals[i] / MAX_VALS[i], 1.0f);

      if (norm < RC_NOISE_FLOOR) {
        currISI_us[i] = 0;  
      } else {
        float rate_hz = RC_MIN_RATE_HZ + norm * (RC_MAX_RATE_HZ - RC_MIN_RATE_HZ);
        currISI_us[i] = (uint32_t)(1e6f / rate_hz);
      }
    }

    // Debug cech w konsoli
    Serial.print(F("ZCR="));  Serial.print(smoothedVals[0], 2);
    Serial.print(F(" DLT="));  Serial.print(smoothedVals[1], 2);
    Serial.print(F(" CRST=")); Serial.print(smoothedVals[2], 2);
    Serial.print(F(" PKS="));  Serial.println(smoothedVals[3], 2);
  }

  uint32_t now_us = micros();
  for (int i = 0; i < 4; i++) {
    if (currISI_us[i] > 0 && (now_us - lastSpikeTime_us[i]) >= currISI_us[i]) {
      lastSpikeTime_us[i] = now_us;
      fireSpike(i);
    }
  }
}

void loopTTFS() {
  processAudio();

  if (newFrameReady) {
    newFrameReady = false;
    ttfsFrameStart_us = micros();

    for (int i = 0; i < 4; i++) {
      ttfsSpiked[i] = false;
      float norm = min(smoothedVals[i] / MAX_VALS[i], 1.0f);
      
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
  for (int i = 0; i < 4; i++) {
    if (!ttfsSpiked[i] && ttfsSpikeAt_us[i] > 0) {
      if (now_us >= ttfsSpikeAt_us[i]) {
        fireSpike(i);
        ttfsSpiked[i] = true;
      }
    }
  }
}

void loop() {
  updateSpikes();

#if ENCODER_MODE == RATE_CODING
  loopRateCoding();
#else
  loopTTFS();
#endif
}