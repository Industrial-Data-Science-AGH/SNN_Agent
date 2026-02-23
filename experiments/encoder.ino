// experiments/encoder.ino
#include <SPI.h>

#define MIC_PIN A0
#define ENERGY_THRESHOLD 150
#define WINDOW_SIZE 256

struct SpikePacket {
  uint8_t header = 0xAE;
  uint16_t timestamp;
  uint8_t energy;
  uint8_t id = 0x00;
  uint8_t checksum;
};

volatile SpikePacket currentSpike;
volatile bool spikeReady = false;
volatile uint16_t sampleCounter = 0;
volatile long sumEnergy = 0;
volatile int sampleIdx = 0;

void setup() {
  Serial.begin(115200);
  pinMode(MIC_PIN, INPUT);
  pinMode(MISO, OUTPUT); 

  SPCR |= _BV(SPE);       
  SPCR &= ~_BV(MSTR);    
  SPI.attachInterrupt();  

  cli();                 
  TCCR1A = 0;            
  TCCR1B = 0;
  TCNT1  = 0;
  OCR1A = 249;            
  TCCR1B |= (1 << WGM12); 
  TCCR1B |= (1 << CS11);  
  TIMSK1 |= (1 << OCIE1A);
  sei();                 

  Serial.println("[ENCODER] System ready. Sampling at 8kHz.");
}

ISR(TIMER1_COMPA_vect) {
  int val = analogRead(MIC_PIN) - 512; 
  sumEnergy += abs(val);
  sampleCounter++;
  sampleIdx++;

  if (sampleIdx >= WINDOW_SIZE) {
    uint8_t avgEnergy = sumEnergy / WINDOW_SIZE;
    
    if (avgEnergy > ENERGY_THRESHOLD) {
      currentSpike.timestamp = sampleCounter;
      currentSpike.energy = avgEnergy;
      currentSpike.checksum = currentSpike.header ^ (currentSpike.timestamp & 0xFF) ^ 
                              (currentSpike.timestamp >> 8) ^ currentSpike.energy ^ currentSpike.id;
      spikeReady = true;
    }
    
    sumEnergy = 0;
    sampleIdx = 0;
  }
}

volatile uint8_t byteIdx = 0;
ISR(SPI_STC_vect) {
  uint8_t* ptr = (uint8_t*)&currentSpike;
  
  if (spikeReady) {
    SPDR = ptr[byteIdx];
    byteIdx++;
    
    if (byteIdx >= sizeof(SpikePacket)) {
      byteIdx = 0;
      spikeReady = false; 
    }
  } else {
    SPDR = 0x00; 
  }
}

void loop() {
  static unsigned long lastPrint = 0;
  if (millis() - lastPrint > 1000) {
    Serial.print("[DEBUG] Samples: "); Serial.print(sampleCounter);
    Serial.print(" | Last Energy: "); Serial.println(currentSpike.energy);
    lastPrint = millis();
  }
}
