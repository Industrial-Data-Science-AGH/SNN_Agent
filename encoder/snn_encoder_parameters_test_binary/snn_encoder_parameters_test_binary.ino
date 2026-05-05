// ============================================================
//  ULTRA-FAST SNN Encoder — Arduino Mega
//  Optymalizacja: Binarna transmisja + Fixed-Point
// ============================================================

#define BAUD_RATE 250000 
#define PIN_SPIKE 6

// Struktura binarna - 8 bajtów zamiast ~30 bajtów tekstu
struct __attribute__((packed)) Report {
  uint16_t peak;
  uint16_t mean;
  uint16_t std;
  uint16_t total_spikes;
} report;

// Zmienne do statystyk (Fixed-point zamiast float)
long sum_x = 0;
long sum_x2 = 0;
uint16_t count = 0;
uint16_t current_peak = 0;

// AFE i SNN
int last_raw = 512;
uint32_t currentISI_us = 0;
uint32_t lastSpikeTime = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(PIN_SPIKE, OUTPUT);
}

void loop() {
  // 1. Szybki odczyt binarny (jeśli czekają 2 bajty próbki)
  if (Serial.available() >= 2) {
    uint16_t raw = (Serial.read() << 8) | Serial.read();

    // Prostym filtrem HP jest różnica próbek (najszybsza możliwa opcja)
    int filtered = abs((int)raw - last_raw);
    last_raw = raw;

    // Statystyki (akumulacja)
    count++;
    sum_x += filtered;
    sum_x2 += (long)filtered * filtered;
    if (filtered > current_peak) current_peak = filtered;

    // Co 32 próbki (potęga 2 = szybkie dzielenie) wysyłamy raport
    if (count >= 32) {
      report.peak = current_peak;
      report.mean = sum_x >> 5; // Szybkie dzielenie przez 32
      
      // Przybliżone STD (wariancja uproszczona dla szybkości)
      long var = (sum_x2 >> 5) - ((long)report.mean * report.mean);
      report.std = (var > 0) ? sqrt(var) : 0;
      report.total_spikes = (uint16_t)0; // Tu opcjonalnie licznik

      // SNN Rate Coding
      uint16_t rate = 5 + (report.peak / 6); // Uproszczone mapowanie
      currentISI_us = (report.peak > 15) ? (1000000UL / rate) : 0;

      // WYSYŁKA BINARNA - błyskawiczna
      Serial.write((byte*)&report, sizeof(report));

      // Reset
      sum_x = sum_x2 = count = current_peak = 0;
    }
  }

  // Generowanie impulsów (bez delay!)
  if (currentISI_us > 0) {
    uint32_t now = micros();
    if (now - lastSpikeTime >= currentISI_us) {
      lastSpikeTime = now;
      PORTD |= (1 << 6);  // Szybki zapis do portu (Pin 6 na Mega to PORTH lub PORTD zależy od mapowania)
      // Dla uproszczenia zostajemy przy digitalWrite, ale bez delay:
      digitalWrite(PIN_SPIKE, HIGH);
      // Impuls wyłączy się w następnym cyklu lub po krótkim czasie
      digitalWrite(PIN_SPIKE, LOW);
    }
  }
}