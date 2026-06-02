/*
 * SNN Decoder - Odbieranie i wyświetlanie spike'ów w Serial Monitor
 */

#define NEURON_VOUT_PIN 3

volatile unsigned long spike_count = 0;
unsigned long last_report_time = 0;

void setup() {
  Serial.begin(115200);
  pinMode(NEURON_VOUT_PIN, INPUT); // Piny odbierające napięcie z outputu PCB SNN
  
  // Przerwanie na opadającym zboczu spike'a
  attachInterrupt(digitalPinToInterrupt(NEURON_VOUT_PIN), onSpikeDetected, FALLING);
  Serial.println("[DECODER] Started listening to SNN output...");
}

void loop() {
  // Raportowanie sumaryczne co sekundę do konsoli
  if (millis() - last_report_time >= 1000) {
    noInterrupts();
    unsigned long current_counts = spike_count;
    spike_count = 0;
    interrupts();
    
    if (current_counts > 0) {
      Serial.print("Spikes in last second: ");
      Serial.println(current_counts);
    }
    
    last_report_time = millis();
  }
}

void onSpikeDetected() {
  spike_count++;
}
