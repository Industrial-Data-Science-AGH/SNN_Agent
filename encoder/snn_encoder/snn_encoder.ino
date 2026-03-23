/*
 * SNN Encoder - DUAL MODE (Rate Coding & Time-to-First-Spike)
 * Autor: AI Agent (na podstawie ustaleń zespołu)
 */

#define MIC_PIN A0
#define SPIKE_OUT_PIN 2

// Odkomentuj tylko jeden rodzaj enkodera:
//#define MODE_RATE_CODING
#define MODE_TTFS

// Parametry globalne
#define RC_WINDOW_MS 50
#define RC_NOISE_FLOOR 80

void setup() {
  Serial.begin(115200);
  pinMode(SPIKE_OUT_PIN, OUTPUT);
  digitalWrite(SPIKE_OUT_PIN, LOW);
  Serial.println("[ENCODER] Init SNN Hardware Encoder...");
  
#ifdef MODE_RATE_CODING
  Serial.println("Mode: Rate Coding");
#elif defined(MODE_TTFS)
  Serial.println("Mode: Time-to-First-Spike (TTFS)");
#endif
}

void loop() {
  int energy = readAmplitude(RC_WINDOW_MS);
  
#ifdef MODE_RATE_CODING
  // Generuje ciąg krótkich impulsów zależy od siły sygnału (Rate Coding)
  if (energy > RC_NOISE_FLOOR) {
    int num_spikes = map(energy, RC_NOISE_FLOOR, 1023, 1, 10);
    for(int i = 0; i < num_spikes; i++) {
      sendSpike();
      delay(2); // Przerwa między spike'ami w burscie
    }
    Serial.print("RC Spikes: ");
    Serial.println(num_spikes);
  }
#endif

#ifdef MODE_TTFS
  // Generuje mocny / szybki impuls gdy energia szybko rośnie. (TTFS) dobren na szkło.
  static int last_energy = 0;
  int delta = energy - last_energy;
  
  if (energy > RC_NOISE_FLOOR && delta > 30) {
    sendSpike();
    Serial.print("TTFS Spike, Energy: ");
    Serial.print(energy);
    Serial.print(", Delta: ");
    Serial.println(delta);
    
    // Blokada (refractory period) po bardzo mocnym uderzeniu
    delay(100); 
  }
  last_energy = energy;
#endif

  // Mały delay na czyszczenie pętli
  delay(10);
}

// Funkcja pomocnicza: Obliczanie amplitudy p-p w oknie czasowym
int readAmplitude(int window_ms) {
  unsigned long start_time = millis();
  int max_val = 0;
  int min_val = 1023;
  
  while (millis() - start_time < window_ms) {
    int val = analogRead(MIC_PIN);
    if (val > max_val) max_val = val;
    if (val < min_val) min_val = val;
  }
  return max_val - min_val;
}

// Generowanie impulsu dla PCB
void sendSpike() {
  digitalWrite(SPIKE_OUT_PIN, HIGH);
  delayMicroseconds(500); // 0.5ms długość napięciowego spike'a
  digitalWrite(SPIKE_OUT_PIN, LOW);
}
