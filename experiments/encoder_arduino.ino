// ============================================================
//  SNN ENCODER — Arduino Uno/Nano
//  Mikrofon analogowy → detekcja anomalii → spiki SPI do RPi
//
//  Pinout SPI:
//    D10 (CS)   → RPi GPIO 8  (CE0)
//    D11 (MOSI) → RPi GPIO 10 (MOSI)
//    D12 (MISO) → RPi GPIO 9  (MISO)  [nieużywane]
//    D13 (SCLK) → RPi GPIO 11 (SCLK)
//    GND        → RPi GND  (WSPÓLNY!)
// ============================================================

#include <SPI.h>

// === KONFIGURACJA — zmień pod swój układ ===
#define MIC_PIN          A0
#define SPI_CS_PIN       10
#define LED_DEBUG        LED_BUILTIN

#define SAMPLE_RATE_HZ   8000    // Hz — próbkowanie mikrofonu
#define FRAME_SIZE       64      // próbek na ramkę (8ms @ 8kHz)
#define N_CHANNELS       3       // liczba pasm częstotliwości (= N_INPUTS w SNN)

// Progi detekcji — KALIBRUJ po uruchomieniu (patrz Serial Monitor)
#define ENERGY_THRESHOLD   180   // próg anomalii (domyślnie ~3× baseline)
#define NOISE_FLOOR         40   // minimalna energia tła

// Format pakietu SPI (6 bajtów)
#define SPIKE_HEADER    0xAE
// ============================================================

uint32_t sample_interval_us;
uint32_t last_sample_us = 0;
uint32_t spike_counter  = 0;

// Bufor próbek dla każdego kanału
int16_t  frame_buf[FRAME_SIZE];
uint16_t frame_idx = 0;

// Baseline energii (adaptacyjny)
float baseline_energy[N_CHANNELS] = {50.0f, 50.0f, 50.0f};
#define BASELINE_ALPHA 0.005f  // szybkość adaptacji (wolna)

// Prosta stała czasowa — podział na N_CHANNELS pasm przez podpróbkowanie
// Kanał 0: każda próbka (HF), kanał 1: co 2, kanał 2: co 4 (LF)

// ============================================================
void setup() {
  Serial.begin(115200);
  pinMode(MIC_PIN,    INPUT);
  pinMode(SPI_CS_PIN, OUTPUT);
  pinMode(LED_DEBUG,  OUTPUT);
  digitalWrite(SPI_CS_PIN, HIGH);

  SPI.begin();
  SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));

  sample_interval_us = 1000000UL / SAMPLE_RATE_HZ;

  Serial.println(F("[ENCODER] SNN Audio Encoder ready"));
  Serial.println(F("[ENCODER] Format: baseline_ch0 | baseline_ch1 | baseline_ch2 | spike_count"));
}

// ============================================================
//  Oblicz energię RMS ramki (wartość bezwzględna odchylenia)
// ============================================================
float compute_energy(int16_t *buf, uint16_t len, uint8_t downsample) {
  float sum = 0;
  uint16_t count = 0;
  int32_t dc = 0;

  for (uint16_t i = 0; i < len; i += downsample) {
    dc += buf[i];
    count++;
  }
  if (count == 0) return 0;
  float mean = (float)dc / count;

  for (uint16_t i = 0; i < len; i += downsample) {
    float diff = buf[i] - mean;
    sum += diff * diff;
  }
  return sqrt(sum / count);
}

// ============================================================
//  Wyślij pakiet spike przez SPI
//  Format: [HEADER][TS_H][TS_L][ENERGY][CHANNEL][CHECKSUM]
// ============================================================
void send_spike(uint8_t channel, uint8_t energy_u8) {
  uint16_t ts = (uint16_t)(millis() & 0xFFFF);
  uint8_t pkt[6] = {
    SPIKE_HEADER,
    (uint8_t)(ts >> 8),
    (uint8_t)(ts & 0xFF),
    energy_u8,
    channel,
    (uint8_t)(SPIKE_HEADER ^ (ts>>8) ^ (ts&0xFF) ^ energy_u8 ^ channel)
  };

  digitalWrite(SPI_CS_PIN, LOW);
  delayMicroseconds(2);
  for (uint8_t i = 0; i < 6; i++) SPI.transfer(pkt[i]);
  delayMicroseconds(2);
  digitalWrite(SPI_CS_PIN, HIGH);

  spike_counter++;

  // Debug LED: mignij przy spiku
  digitalWrite(LED_DEBUG, HIGH);
  // (nie delay — zgasi się w następnej iteracji)
}

// ============================================================
void loop() {
  uint32_t now_us = micros();

  // Próbkowanie z dokładną częstotliwością
  if (now_us - last_sample_us < sample_interval_us) return;
  last_sample_us = now_us;

  frame_buf[frame_idx++] = analogRead(MIC_PIN);
  digitalWrite(LED_DEBUG, LOW);

  if (frame_idx < FRAME_SIZE) return;
  frame_idx = 0;

  // === Przetworz ramkę ===
  for (uint8_t ch = 0; ch < N_CHANNELS; ch++) {
    uint8_t ds = 1 << ch;  // kanał 0: ds=1, kanał 1: ds=2, kanał 2: ds=4
    float energy = compute_energy(frame_buf, FRAME_SIZE, ds);

    // Adaptuj baseline (tylko gdy cicho)
    if (energy < baseline_energy[ch] * 2.0f) {
      baseline_energy[ch] = baseline_energy[ch] * (1.0f - BASELINE_ALPHA)
                            + energy * BASELINE_ALPHA;
    }

    float delta = energy - baseline_energy[ch];

    // Sprawdź próg — wygeneruj spike
    if (delta > ENERGY_THRESHOLD && energy > NOISE_FLOOR) {
      uint8_t energy_u8 = (uint8_t)min(255.0f, (delta / 512.0f) * 255.0f);
      send_spike(ch, energy_u8);
    }
  }

  // Co ~1s: debug przez Serial
  static uint32_t last_debug = 0;
  if (millis() - last_debug > 1000) {
    last_debug = millis();
    Serial.print(F("[DEBUG] baseline: "));
    for (uint8_t ch = 0; ch < N_CHANNELS; ch++) {
      Serial.print(baseline_energy[ch], 1);
      if (ch < N_CHANNELS-1) Serial.print(F(" | "));
    }
    Serial.print(F(" | spikes/s: "));
    Serial.println(spike_counter);
    spike_counter = 0;
  }
}
