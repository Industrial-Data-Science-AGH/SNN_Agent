// ============================================================
//  SNN Spike Encoder — Arduino Mega
//  Projekt: detekcja rozbijanego szkła
//  Branch:  sim_train → Faza 1 hardware (SYMULACJA PRZEZ SERIAL)
//
//  Wejście:  Dane (amplituda) wysyłane z Pythona przez port szeregowy
//  Wyjście:  impulsy napięciowe na PIN_SPIKE → neuron PCB
//
//  Tryb 1: RATE CODING  — częstotliwość spike'ów ∝ amplituda
//  Tryb 2: TTFS         — czas do 1. spike'a ∝ 1/amplituda
// ============================================================

// ---- WYBIERZ TRYB ENKODOWANIA ----
#define RATE_CODING 1
#define TTFS        2

#define ENCODER_MODE RATE_CODING   // lub: TTFS

// ---- PINY ----
#define PIN_SPIKE      6    // wyjście → neuron PCB (przez rezystor 100Ω)
#define PIN_DEBUG_LED  13   // wbudowana LED — miga przy każdym spike'u

// ---- PARAMETRY RATE CODING ----
#define RC_WINDOW_MS      10      // ms — okno zbierania próbek
#define RC_MIN_RATE_HZ    5       // Hz — minimalna częstotliwość spike'ów
#define RC_MAX_RATE_HZ    200     // Hz — maksymalna (ograniczona przez PCB neuron)
#define RC_NOISE_FLOOR    40      // ADC counts poniżej których = cisza

// ---- PARAMETRY TTFS ----
#define TTFS_WINDOW_MS    20      // ms — czas jednej "ramki" TTFS
#define TTFS_THRESHOLD    50      // ADC counts — poniżej: brak spike'a (cisza)
#define TTFS_MAX_AMP      400     // ADC counts — amplituda = maksymalny sygnał

// ---- PARAMETRY IMPULSU ----
#define SPIKE_WIDTH_US    500     // µs — szerokość impulsu (0.5ms)
#define SPIKE_VOLTAGE     HIGH    // HIGH = 5V lub 3.3V

// ---- STAN GLOBALNY SNN ----
static uint32_t lastWindowStart  = 0;
static uint32_t lastSpikeTime    = 0;
static uint32_t ttfsWindowStart  = 0;
static bool     ttfsSpiked       = false;
static uint32_t currentISI_us    = 0;    
static uint32_t spikeCount       = 0;    
static float    smoothedAmp      = 0.0f; 

// ---- STAN GLOBALNY AFE (CYFROWY FILTR) ----
float hp_alpha = 0.85;  // Filtr HP: 0.85 (mocno wycina basy, zostawia trzask)
float filtered_amp = 0;
float last_raw_amp = 0;

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(5); // KLUCZOWE: skraca czas czekania na liczbę z Seriala

  pinMode(PIN_SPIKE, OUTPUT);
  pinMode(PIN_DEBUG_LED, OUTPUT);
  digitalWrite(PIN_SPIKE, LOW);
  digitalWrite(PIN_DEBUG_LED, LOW);

  // Uwaga: Rezygnujemy ze zmiany prescalera ADC, ponieważ nie używamy
  // analogRead() w tym trybie. Płytka czyta z bufora Serial.

  Serial.println(F("=== SNN Encoder START (TRYB SYMULACJI SERIAL) ==="));
#if ENCODER_MODE == RATE_CODING
  Serial.println(F("Tryb: RATE CODING"));
  Serial.print(F("Rate: ")); Serial.print(RC_MIN_RATE_HZ);
  Serial.print(F("-")); Serial.print(RC_MAX_RATE_HZ); Serial.println(F(" Hz"));
#else
  Serial.println(F("Tryb: TTFS"));
  Serial.print(F("Okno: ")); Serial.print(TTFS_WINDOW_MS); Serial.println(F(" ms"));
#endif
  Serial.println(F("Format: [ms] AMP ISI_us SPIKES_total"));

  lastWindowStart = millis();
  ttfsWindowStart = millis();
}

// ============================================================
//  ODCZYT SYMULOWANY Z PYTHONA Z FILTREM AFE
// ============================================================
uint16_t readAmplitude(uint16_t windowMs) {
  uint32_t start = millis();
  uint16_t maxAmp = 0;

  // Czekamy przez zadane okno czasowe (zachowanie zgodne z oryginalnym kodem)
  while (millis() - start < windowMs) {
    if (Serial.available() > 0) {
      int raw = Serial.parseInt(); // Odczyt 0-1023 przesłany z Pythona
      
      // --- CYFROWE AFE (FILTR GÓRNOPRZEPUSTOWY) ---
      filtered_amp = hp_alpha * (filtered_amp + raw - last_raw_amp);
      last_raw_amp = raw;
      
      uint16_t currentAmp = (uint16_t)abs(filtered_amp);
      
      if (currentAmp > maxAmp) {
        maxAmp = currentAmp;
      }
    }
  }
  
  return maxAmp;
}

// ============================================================
//  GENERUJ SPIKE
// ============================================================
void fireSPike() {
  digitalWrite(PIN_SPIKE, SPIKE_VOLTAGE);
  digitalWrite(PIN_DEBUG_LED, HIGH);
  delayMicroseconds(SPIKE_WIDTH_US);
  digitalWrite(PIN_SPIKE, LOW);
  digitalWrite(PIN_DEBUG_LED, LOW);
  spikeCount++;
}

// ============================================================
//  ENKODER 1: RATE CODING
// ============================================================
void loopRateCoding() {
  uint32_t now = millis();

  // ---- Odśwież rate co RC_WINDOW_MS ----
  if (now - lastWindowStart >= RC_WINDOW_MS) {
    lastWindowStart = now;

    uint16_t rawAmp = readAmplitude(RC_WINDOW_MS);

    // Odejmij noise floor, clamp do 0
    int16_t netAmp = (int16_t)rawAmp - RC_NOISE_FLOOR;
    if (netAmp < 0) netAmp = 0;

    // Low-pass filter (α=0.3) — wygładza skoki
    smoothedAmp = 0.3f * netAmp + 0.7f * smoothedAmp;

    // Mapuj amplitudę → rate [Hz]
    float maxNet = (float)(1023 - RC_NOISE_FLOOR);
    float normalized = smoothedAmp / maxNet;           // [0..1]
    if (normalized > 1.0f) normalized = 1.0f;

    float rate_hz = RC_MIN_RATE_HZ + normalized * (RC_MAX_RATE_HZ - RC_MIN_RATE_HZ);

    // Rate → ISI [µs]
    if (netAmp < 2) {
      currentISI_us = 0;   // cisza — nie generuj spike'ów
    } else {
      currentISI_us = (uint32_t)(1e6f / rate_hz);
    }

    // Debug output
    Serial.print(millis()); Serial.print(F("\t"));
    Serial.print(rawAmp);   Serial.print(F("\t"));
    Serial.print(currentISI_us); Serial.print(F("\t"));
    Serial.println(spikeCount);
  }

  // ---- Generuj spike'i z bieżącym ISI ----
  if (currentISI_us > 0) {
    uint32_t nowUs = micros();
    if (nowUs - lastSpikeTime >= currentISI_us) {
      lastSpikeTime = nowUs;
      fireSPike();
    }
  }
}

// ============================================================
//  ENKODER 2: TIME-TO-FIRST-SPIKE (TTFS)
// ============================================================
void loopTTFS() {
  uint32_t now = millis();
  uint32_t elapsed = now - ttfsWindowStart;

  // ---- Nowe okno ----
  if (elapsed >= (uint32_t)TTFS_WINDOW_MS) {
    ttfsWindowStart = now;
    ttfsSpiked = false;

    // Zmierz amplitudę (4ms próbkowania)
    uint16_t rawAmp = readAmplitude(4);  

    if (rawAmp < TTFS_THRESHOLD) {
      // Poniżej progu — cisza, brak spike'a w tym oknie
      currentISI_us = 0;

      Serial.print(millis()); Serial.print(F("\t"));
      Serial.print(rawAmp);   Serial.print(F("\tNO_SPIKE\t"));
      Serial.println(spikeCount);
      return;
    }

    // Clamp amplitudy do zakresu
    uint16_t clampedAmp = rawAmp;
    if (clampedAmp > TTFS_MAX_AMP) clampedAmp = TTFS_MAX_AMP;

    // Normalizacja [0..1]
    float normalized = (float)(clampedAmp - TTFS_THRESHOLD)
                     / (float)(TTFS_MAX_AMP - TTFS_THRESHOLD);

    // Opóźnienie spike'a
    uint32_t delayMs = (uint32_t)(TTFS_WINDOW_MS * (1.0f - normalized));

    // Zapisz jako "czas kiedy spike ma wystąpić"
    currentISI_us = (uint32_t)(ttfsWindowStart + delayMs);  

    Serial.print(millis()); Serial.print(F("\t"));
    Serial.print(rawAmp);   Serial.print(F("\tdelay_ms="));
    Serial.print(delayMs);  Serial.print(F("\t"));
    Serial.println(spikeCount);
  }

  // ---- Wyślij spike w odpowiednim momencie ----
  if (!ttfsSpiked && currentISI_us > 0) {
    if (millis() >= currentISI_us) {
      fireSPike();
      ttfsSpiked = true;
    }
  }
}

// ============================================================
//  LOOP
// ============================================================
void loop() {
#if ENCODER_MODE == RATE_CODING
  loopRateCoding();
#else
  loopTTFS();
#endif
}
