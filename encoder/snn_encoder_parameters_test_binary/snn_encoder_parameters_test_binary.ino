// ============================================================
//  SNN Encoder - 3 Channel Tensor Generator (Peak, Mean, Std)
// ============================================================

#define PIN_SPIKE_CH0  6 // Peak Energy
#define PIN_SPIKE_CH1  7 // Mean Energy
#define PIN_SPIKE_CH2  8 // Std Energy (Burst detection)

#define STATS_WINDOW 15 // Okno statystyk (15 klatek po 10ms = 150ms kontekstu)
int energyHistory[STATS_WINDOW];
int hIdx = 0;

void setup() {
  Serial.begin(115200);
  pinMode(PIN_SPIKE_CH0, OUTPUT);
  pinMode(PIN_SPIKE_CH1, OUTPUT);
  pinMode(PIN_SPIKE_CH2, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    if (input == "SYN") { Serial.println("READY"); return; }
    
    int currentEnergy = input.toInt();

    // 1. Aktualizacja historii energii
    energyHistory[hIdx] = currentEnergy;
    hIdx = (hIdx + 1) % STATS_WINDOW;

    // 2. Obliczanie parametrów tensora wejściowego [Peak, Mean, Std]
    long sum = 0;
    int peak = 0;
    for(int i=0; i<STATS_WINDOW; i++) {
      sum += energyHistory[i];
      if(energyHistory[i] > peak) peak = energyHistory[i];
    }
    int mean = sum / STATS_WINDOW;

    long sqSum = 0;
    for(int i=0; i<STATS_WINDOW; i++) {
      long d = energyHistory[i] - mean;
      sqSum += d * d;
    }
    int std = sqrt(sqSum / STATS_WINDOW);

    // 3. Raportowanie do Pythona (3 parametry tensora)
    Serial.print(peak); Serial.print('\t');
    Serial.print(mean); Serial.print('\t');
    Serial.println(std);

    // 4. KODOWANIE RATE CODING (3 kanały fizyczne)
    // Kanał 0: Peak
    if (peak > 50) triggerSpike(PIN_SPIKE_CH0);
    // Kanał 1: Mean (tylko jeśli stabilne)
    if (mean > 30) triggerSpike(PIN_SPIKE_CH1);
    // Kanał 2: Std (detekcja gwałtownych zmian / szumu pękania)
    if (std > 20)  triggerSpike(PIN_SPIKE_CH2);
  }
}

void triggerSpike(int pin) {
  digitalWrite(pin, HIGH);
  delayMicroseconds(50); // Bardzo krótki impuls
  digitalWrite(pin, LOW);
}