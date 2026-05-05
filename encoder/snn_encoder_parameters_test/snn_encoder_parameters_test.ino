// ============================================================
//  ZOPTYMALIZOWANY SNN Encoder — Arduino Mega
//  Cel: Powrót do wydajności czasu rzeczywistego
// ============================================================

#define BAUD_RATE 250000 
#define PIN_SPIKE 6

// Statystyki kroczące (bez pętli for!)
float m_n = 0, m_old = 0, s_n = 0;
long n = 0;
float current_peak = 0;

// SNN & AFE
float hp_alpha = 0.85; 
float filtered_amp = 0;
float last_raw_amp = 0;
uint32_t currentISI_us = 0;
uint32_t lastSpikeTime = 0;
uint32_t spikeCount = 0;

void setup() {
  Serial.begin(BAUD_RATE);
  pinMode(PIN_SPIKE, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    if (input == "SYN") { Serial.println("READY"); return; }
    
    int raw = input.toInt();

    // 1. AFE (Filtr HP)
    filtered_amp = hp_alpha * (filtered_amp + raw - last_raw_amp);
    last_raw_amp = raw;
    float x = abs(filtered_amp);

    // 2. Szybkie statystyki kroczące (Algorytm Welforda)
    n++;
    if (n == 1) {
      m_old = m_n = x;
      s_n = 0;
      current_peak = x;
    } else {
      m_n = m_old + (x - m_old) / n;
      s_n = s_n + (x - m_old) * (x - m_n);
      m_old = m_n;
      if (x > current_peak) current_peak = x;
    }

    // Oblicz odchylenie co np. 10 próbek, żeby oszczędzić CPU (sqrt jest wolne)
    float std = (n > 1) ? sqrt(s_n / (n - 1)) : 0;

    // Resetuj statystyki co 100 próbek, aby zachować "lokalność" okna
    if (n >= 100) { n = 0; current_peak = 0; }

    // 3. SNN Logic
    float rate = 5 + (current_peak / 1023.0) * 195;
    currentISI_us = (current_peak > 10) ? (uint32_t)(1000000.0 / rate) : 0;

    // 4. Minimalistyczny raport (skrócone liczby do 1 miejsca po przecinku)
    Serial.print(current_peak, 1); Serial.print('\t');
    Serial.print(m_n, 1);          Serial.print('\t');
    Serial.print(std, 1);          Serial.print('\t');
    Serial.print(currentISI_us);   Serial.print('\t');
    Serial.println(spikeCount);
  }

  // Generowanie impulsów
  if (currentISI_us > 0) {
    uint32_t now = micros();
    if (now - lastSpikeTime >= currentISI_us) {
      lastSpikeTime = now;
      digitalWrite(PIN_SPIKE, HIGH);
      delayMicroseconds(100); 
      digitalWrite(PIN_SPIKE, LOW);
      spikeCount++;
    }
  }
}