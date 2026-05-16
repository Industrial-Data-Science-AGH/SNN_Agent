// ============================================================
//  SNN Spike Encoder — Arduino Mega (Wersja 3-Kanałowa)
//  Projekt: detekcja rozbijanego szkła
//
//  Wejście:  mikrofon (WM61-A) na pinie A0
//  Wyjścia:  3 piny cyfrowe (Kanał 0: Peak, Kanał 1: Mean, Kanał 2: Std)
// ============================================================

// ---- TRYB ENKODOWANIA ----
#define ENCODER_MODE RATE_CODING   // Rate coding lub TTFS

#define RATE_CODING 1
#define TTFS        2

// ---- PINY ----
#define PIN_MIC        A0   // Wejście mikrofonu
#define PIN_SPIKE_PEAK 6    // Kanał 0: Peak energy
#define PIN_SPIKE_MEAN 7    // Kanał 1: Mean energy
#define PIN_SPIKE_STD  8    // Kanał 2: Std energy
#define PIN_DEBUG_LED  13   // LED

// Tablica pinów [Peak, Mean, Std]
const uint8_t SPIKE_PINS[3] = {PIN_SPIKE_PEAK, PIN_SPIKE_MEAN, PIN_SPIKE_STD};

// ---- PARAMETRY WSPÓLNE ----
#define FRAME_WINDOW_MS 20  // ms - rozmiar ramki czasowej do analizy energii

// Znormalizowane maksima dla poszczególnych cech (do kalibracji)
// Pozwalają mapować wartości z ADC na zakres [0.0 - 1.0]
#define MAX_PEAK_VAL   200.0f 
#define MAX_MEAN_VAL   40.0f
#define MAX_STD_VAL    30.0f
const float MAX_VALS[3] = {MAX_PEAK_VAL, MAX_MEAN_VAL, MAX_STD_VAL};

// ---- PARAMETRY RATE CODING ----
#define RC_MIN_RATE_HZ 5      // Hz
#define RC_MAX_RATE_HZ 200    // Hz
#define RC_NOISE_FLOOR 0.05f  // Odcięcie (5%) - poniżej traktujemy jako ciszę

// ---- PARAMETRY TTFS ----
#define TTFS_THRESHOLD 0.1f   // Odcięcie (10% znormalizowanego sygnału)

// ---- PARAMETRY IMPULSU ----
#define SPIKE_WIDTH_US 500    // µs (0.5ms)
#define SPIKE_VOLTAGE  HIGH

// ---- STAN GLOBALNY ----
static uint32_t lastWindowStart  = 0;
static uint32_t lastSpikeTime[3] = {0, 0, 0};
static uint32_t currISI_us[3] = {0, 0, 0}; // Inter-spike interval lub timestamp TTFS
static bool     ttfsSpiked[3]    = {false, false, false};
static uint32_t spikeCount[3]    = {0, 0, 0};

float channelValues[3] = {0.0f, 0.0f, 0.0f}; // [Peak, Mean, Std]
float smoothedVals[3]  = {0.0f, 0.0f, 0.0f}; // Filtracja Low-pass dla RC

// ---- ZMIENNE FILTRA HPF ----
float hp_filtered = 0.0f;
float prev_raw = 610.0f; // spodziewana wartość spoczynkowa: 610 dla 3.3V, 940 dla 5V
#define ALPHA 0.99f // współczynnik odcięcia DC (ok 0.95-0.99)


// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);

  for(int i=0; i<3; i++) {
    pinMode(SPIKE_PINS[i], OUTPUT);
    digitalWrite(SPIKE_PINS[i], LOW);
  }
  pinMode(PIN_DEBUG_LED, OUTPUT);
  digitalWrite(PIN_DEBUG_LED, LOW);

  // Przyspieszenie ADC dla gęstszego próbkowania w oknie (prescaler = 16)
  ADCSRA = (ADCSRA & ~0x07) | 0x04;

  Serial.println(F("=== SNN 3-Channel Encoder START ==="));
  #if ENCODER_MODE == RATE_CODING
    Serial.println(F("Tryb: RATE CODING"));
  #else
    Serial.println(F("Tryb: TTFS"));
  #endif

  lastWindowStart = millis();
}

// ============================================================
//  ODCZYT STATYSTYK MIKROFONU (Energy Extraction)
//  Wylicza parametry na liczbach całkowitych dla oszczędności CPU.
// zastosowano filtr HPF
// ============================================================

void extractFrameFeatures(uint16_t windowMs) {
  uint32_t start = millis();
  float maxAc = 0;
  float sumAc = 0;
  float sumSq = 0;
  uint16_t count = 0;

  while (millis() - start < windowMs) {
    float raw = (float) analogRead(PIN_MIC);

    hp_filtered = ALPHA * (hp_filtered + raw - prev_raw);
    prev_raw = raw;

    float val = abs(hp_filtered);
    
    maxAc = max(maxAc, val);
    sumAc += val;
    sumSq += val * val;
    count++;
  }

  // Zabezpieczenie przed dzieleniem przez zero (w razie zacięcia zegara)
  if(count == 0) count = 1;

  // Obliczenia końcowe (float używany tylko tu, raz na okno)
  channelValues[0] = maxAc;         // PEAK
  channelValues[1] = sumAc / (float) count; // MEAN
  float variance = sumSq / (float) count;
  channelValues[2] = sqrt(variance);        // STD
}

// ============================================================
//  GENERUJ SPIKE NA KONKRETNYM KANALE
// ============================================================
void fireSpike(uint8_t channel) {
  Serial.print("SPIKE "); Serial.println(channel);
  digitalWrite(SPIKE_PINS[channel], SPIKE_VOLTAGE);
  digitalWrite(PIN_DEBUG_LED, HIGH);
  delayMicroseconds(SPIKE_WIDTH_US);
  digitalWrite(SPIKE_PINS[channel], LOW);
  digitalWrite(PIN_DEBUG_LED, LOW);
  spikeCount[channel] ++;
}

// ============================================================
//  ENKODER 1: RATE CODING (Dla 3 Kanałów)
// ============================================================
void loopRateCoding() {
  uint32_t now = millis();

  // ---- Odśwież cechy co ramkę czasową ----
  if (now - lastWindowStart >= FRAME_WINDOW_MS) {
    lastWindowStart = now;
    extractFrameFeatures(FRAME_WINDOW_MS);

    for(int i=0; i<3; i++) {
      // Filtracja LP wygładzająca dla rate-coding (alpha=0.3)
      smoothedVals[i] = 0.3f * channelValues[i] + 0.7f * smoothedVals[i];

      // Normalizacja do zakresu [0.0 - 1.0]
      float normalised = min(smoothedVals[i] / MAX_VALS[i], 1.0f);
      if(normalised < RC_NOISE_FLOOR) {
        currISI_us[i] = 0; // cisza
      } else {
        float rate_hz = RC_MIN_RATE_HZ + normalised * (RC_MAX_RATE_HZ - RC_MIN_RATE_HZ);
        currISI_us[i] = (uint32_t)(1e6f / rate_hz);
      }
    }
  }

  // ---- Generuj spike'i asynchronicznie dla wszystkich 3 kanałów ----
  uint32_t now_us = micros();
  for(int i=0; i<3; i++) {
    if(currISI_us[i] > 0) {
      if(now_us - lastSpikeTime[i] >= currISI_us[i]) {
        lastSpikeTime[i] = now_us;
        fireSpike(i);
      }
    }
  }

  //  debug
  if(smoothedVals[2] > 1.0) {
    Serial.print("A0 = "); Serial.print(analogRead(0));
  Serial.print(" Peak = "); Serial.print(smoothedVals[0]);
  Serial.print(" Mean = "); Serial.print(smoothedVals[1]);
  Serial.print(" Stdev = "); Serial.print(smoothedVals[2]);
  Serial.println();
  }
  


}

// ============================================================
//  ENKODER 2: TIME-TO-FIRST-SPIKE (TTFS) (Dla 3 Kanałów)
// ============================================================
void loopTTFS() {
  uint32_t now = millis();
  uint32_t elapsed = now - lastWindowStart;

  // ---- Nowa Ramka ----
  if(elapsed >= (uint32_t) FRAME_WINDOW_MS) {
    lastWindowStart = now;

    /* W trybie TTFS okno ekstrakcji musi być krótsze, żeby zostawić
      miejsce w czasie rzeczywistym na wysłanie impulsów.
      Analizujemy tylko pierwsze 4ms z ramki 20ms */
    extractFrameFeatures(4);

    for(int i=0; i<3; i++) {
      ttfsSpiked[i] = false;
      float normalised = min(channelValues[i] / MAX_VALS[i], 1.0f);

      if(normalised < TTFS_THRESHOLD) {
        currISI_us[i] = 0;
      } else {
        uint32_t delay_ms = (uint32_t)(FRAME_WINDOW_MS * (1.0f - normalised));
        currISI_us[i] = (uint32_t)(lastWindowStart + delay_ms);
      }
    }
  }

  // ---- Sprawdź czy nadszedł czas wysłania spike'ów ----
  for(int i=0; i<3; i++) {
    if(!ttfsSpiked[i] && currISI_us[i] > 0) {
      if(millis() >= currISI_us[i]) {
        fireSpike(i);
        ttfsSpiked[i] = true;
      }
    }
  }
}

// ============================================================
//  GŁÓWNA PĘTLA
// ============================================================
void loop() {
  #if ENCODER_MODE == RATE_CODING
    loopRateCoding();
  #else
    loopTTFS();
  #endif
}
