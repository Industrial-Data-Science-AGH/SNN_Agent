// ============================================================
//  SNN Spike Encoder — Arduino Mega
//  Projekt: detekcja rozbijanego szkła
//  Branch:  sim_train → Faza 1 hardware
//
//  Wejście:  mikrofon (MAX4466 lub podobny) na pinie A0
//  Wyjście:  impulsy napięciowe na PIN_SPIKE → neuron PCB
//
//  Tryb 1: RATE CODING  — częstotliwość spike'ów ∝ amplituda
//  Tryb 2: TTFS         — czas do 1. spike'a ∝ 1/amplituda
//
//  Wybierz tryb poniżej, wgraj, obserwuj Serial Monitor (115200).
// ============================================================

// ---- WYBIERZ TRYB ENKODOWANIA ----
#define ENCODER_MODE RATE_CODING   // lub: TTFS

#define RATE_CODING 1
#define TTFS        2

// ---- PINY ----
#define PIN_MIC        A0   // wejście mikrofonu (przez dzielnik jeśli 5V mic → 3.3V ADC)
#define PIN_SPIKE      6    // wyjście → neuron PCB (przez rezystor 100Ω)
#define PIN_DEBUG_LED  13   // wbudowana LED — miga przy każdym spike'u

// ---- PARAMETRY RATE CODING ----
//   Okno analizy: co RC_WINDOW_MS ms liczymy amplitudę i ustawiamy rate
//   Min/max rate: poniżej MIN neuron nie dostaje nic, powyżej MAX saturacja
#define RC_WINDOW_MS      10      // ms — okno zbierania próbek (10ms = 100Hz refresh)
#define RC_MIN_RATE_HZ    5       // Hz — minimalna częstotliwość spike'ów
#define RC_MAX_RATE_HZ    200     // Hz — maksymalna (ograniczona przez PCB neuron)
#define RC_NOISE_FLOOR    40      // ADC counts (0–1023) poniżej których = cisza

// ---- PARAMETRY TTFS ----
//   Okno kodowania: TTFS_WINDOW_MS ms → w tym czasie jeden spike lub zero
//   Próg: sygnał musi przekroczyć TTFS_THRESHOLD żeby w ogóle wygenerować spike
#define TTFS_WINDOW_MS    20      // ms — czas jednego "ramki" TTFS
#define TTFS_THRESHOLD    50      // ADC counts — poniżej: brak spike'a (cisza)
#define TTFS_MAX_AMP      400     // ADC counts — amplituda = maksymalny sygnał (satura.)

// ---- PARAMETRY IMPULSU ----
//   Czas trwania pojedynczego spike'a wyjściowego.
//   Dopasuj do tego co PCB neuron "widzi" jako poprawny impuls.
#define SPIKE_WIDTH_US    500     // µs — szerokość impulsu (0.5ms)
#define SPIKE_VOLTAGE     HIGH    // HIGH = 5V (Arduino Mega) lub 3.3V (Pro Mini)

// ---- STAN GLOBALNY ----
static uint32_t lastWindowStart  = 0;
static uint32_t lastSpikeTime    = 0;
static uint32_t ttfsWindowStart  = 0;
static bool     ttfsSpiked       = false;
static uint32_t currentISI_us    = 0;   // inter-spike interval dla Rate Coding
static uint32_t spikeCount       = 0;   // do Serial debuggingu
static float    smoothedAmp      = 0.0f; // low-pass na amplitudę

// ============================================================
//  SETUP
// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(PIN_SPIKE, OUTPUT);
  pinMode(PIN_DEBUG_LED, OUTPUT);
  digitalWrite(PIN_SPIKE, LOW);
  digitalWrite(PIN_DEBUG_LED, LOW);

  // Przyspieszenie ADC: domyślny prescaler 128 → zmieniamy na 16
  // Daje ~77kHz sampling zamiast ~9.6kHz (potrzebne do okien 10ms)
  ADCSRA = (ADCSRA & ~0x07) | 0x04;  // prescaler = 16

  // Kalibracja DC offset (mikrofon ma DC bias ~Vcc/2)
  Serial.println(F("=== SNN Encoder START ==="));
#if ENCODER_MODE == RATE_CODING
  Serial.println(F("Tryb: RATE CODING"));
  Serial.print(F("Rate: ")); Serial.print(RC_MIN_RATE_HZ);
  Serial.print(F("–")); Serial.print(RC_MAX_RATE_HZ); Serial.println(F(" Hz"));
#else
  Serial.println(F("Tryb: TTFS"));
  Serial.print(F("Okno: ")); Serial.print(TTFS_WINDOW_MS); Serial.println(F(" ms"));
#endif
  Serial.println(F("Format: [ms] AMP ISI_us SPIKES_total"));

  lastWindowStart = millis();
  ttfsWindowStart = millis();
}

// ============================================================
//  ODCZYT MIKROFONU — peak-to-peak w oknie czasowym
//
//  Mikrofon MAX4466: sygnał AC wycentrowany na ~512 (10-bit ADC)
//  Mierzymy peak-to-peak w oknie = przybliżenie amplitudy RMS * 2
// ============================================================
uint16_t readAmplitude(uint16_t windowMs) {
  uint32_t start = millis();
  uint16_t sampleMax = 0;
  uint16_t sampleMin = 1023;

  while (millis() - start < windowMs) {
    uint16_t sample = analogRead(PIN_MIC);
    if (sample > sampleMax) sampleMax = sample;
    if (sample < sampleMin) sampleMin = sample;
  }
  return sampleMax - sampleMin;  // peak-to-peak [0..1023]
}

// ============================================================
//  GENERUJ SPIKE — impuls napięciowy na PIN_SPIKE
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
//
//  Zasada: amplituda → częstotliwość spike'ów
//
//  Co RC_WINDOW_MS ms:
//    1. Zmierz amplitudę (peak-to-peak)
//    2. Odejmij noise floor
//    3. Mapuj liniowo na zakres [MIN_RATE .. MAX_RATE] Hz
//    4. Przelicz rate → ISI (inter-spike interval)
//
//  Między oknami: generuj spike'i z bieżącym ISI
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
//
//  Zasada: czas od początku okna do 1. spike'a ∝ 1/amplituda
//
//  Co TTFS_WINDOW_MS ms (nowe okno):
//    1. Zmierz amplitudę
//    2. Oblicz opóźnienie: delay = window * (1 - normalized_amp)
//       → silny sygnał: spike prawie natychmiast (t≈0)
//       → słaby sygnał: spike późno lub wcale
//    3. Czekaj delay, potem jeden spike
//
//  Zalety TTFS: bardziej biologicznie wierny, lepszy dla silnych
//  zdarzeń impulsowych (jak rozbicie szkła = krótki energetyczny burst)
// ============================================================
void loopTTFS() {
  uint32_t now = millis();
  uint32_t elapsed = now - ttfsWindowStart;

  // ---- Nowe okno ----
  if (elapsed >= (uint32_t)TTFS_WINDOW_MS) {
    ttfsWindowStart = now;
    ttfsSpiked = false;

    // Zmierz amplitudę — krótkie okno żeby nie blokować
    uint16_t rawAmp = readAmplitude(4);  // 4ms próbkowania na start okna

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

    // Opóźnienie spike'a: silny sygnał → małe delay, słaby → duże
    // delay_ms = window * (1 - normalized)
    uint32_t delayMs = (uint32_t)(TTFS_WINDOW_MS * (1.0f - normalized));

    // Zapisz jako "czas kiedy spike ma wystąpić"
    // Używamy currentISI_us do przechowania absolutnego czasu
    currentISI_us = (uint32_t)(ttfsWindowStart + delayMs);  // target time [ms]

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
