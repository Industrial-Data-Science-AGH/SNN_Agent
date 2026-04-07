# QUICK START - ENCODER (Arduino)

## TL;DR
Twoje zadanie: zbudować encoder, który:
1. Czyta mikrofon analog (ADC na Arduino A0)
2. Detectuje anomalie w sygnale audio
3. Wysyła "spike'i" (zdarzenia) do RPi przez SPI

---

## HARDWARE - Co Potrzebujesz

```
Arduino Uno/Nano          Mikrofon (WM-64 lub SPU0414HR5H)
└─ Pin A0 ─────────────────── Mikrofon OUT (analog 0-5V)
└─ Pin 10 ─────────────────── RPi GPIO 8 (CS)
└─ Pin 11 ─────────────────── RPi GPIO 10 (MOSI)
└─ Pin 12 ─────────────────── RPi GPIO 9 (MISO)
└─ Pin 13 ─────────────────── RPi GPIO 11 (SCLK)
└─ GND ────────────────────── GND (wspólny!)
```

**Schematyk mikrofonu**:
```
VCC (3.3V) ──┐
             ├─ WM-64 microphone
GND ─────────┤
             └─ OUT → Arduino A0 (przez RC filter 100k/100nF)
```

---

## SOFTWARE - Upload

### 1. Pobierz Arduino IDE
https://www.arduino.cc/en/software

### 2. Skopiuj kod encodera
```
encoder_arduino.ino  →  Arduino IDE
```

### 3. Konfiguruj dla swoich pinów
```cpp
#define MIC_PIN A0              // Twój pin ADC
#define SPI_CS_PIN 10           // Twój CS pin
#define ENERGY_THRESHOLD 150    // Próg detekcji (dostraj później)
```

### 4. Upload
```
Tools > Board > Arduino Uno (lub Nano)
Tools > Port > /dev/ttyUSB0 (lub COM3)
Sketch > Upload (Ctrl+U)
```

---

## TESTING

### Serial Monitor
```
Arduino IDE: Tools > Serial Monitor
Speed: 115200 baud
```

Powinieneś zobaczyć:
```
[ENCODER] Audio Encoder initialized - waiting for spikes...
[DEBUG] Samples: 1234 | Baseline energy: 45 | Anomalies/sec: 2
[SPI TX] Energy: 165 @ T=01234
```

### Manualna kalibracja
Jeśli falszywe alarmy - zwiększ threshold:
```cpp
#define ENERGY_THRESHOLD 200  // było 150
```

---

## PARAMETRY DO WYREGULOWANIA

```cpp
#define SAMPLE_RATE 8000        // Hz - częstotliwość próbkowania
#define ENERGY_THRESHOLD 150    // Próg anomalii (wyżej = rzadziej reaguje)
#define NOISE_THRESHOLD 50      // Próg szumu tła
```

### Jak wyregulować threshold?

1. **Uruchom encoder w ciszy** (po 10 sekundach)
2. **Przeczytaj Baseline energy** z Serial Monitor
3. **Ustaw ENERGY_THRESHOLD = baseline * 3**

Przykład:
- Baseline w ciszy: 40
- ENERGY_THRESHOLD powinno być: ~120

---

## ARCHITEKTURA ENCODERA

```
Mikrofon (0-5V analog)
         │
         ▼
    ADC 8kHz (125µs)
         │
         ▼
  High-Pass Filter (usuwa DC)
         │
         ▼
   Rolling Buffer (256 samples)
         │
         ▼
  Autoencoder Anomaly Detector
  ├─ Oblicz energię RMS
  ├─ Porównaj z baseline'em
  └─ energy_delta > THRESHOLD → SPIKE!
         │
         ▼
  SPI Packet (0xAE header)
  [HEADER] [TIMESTAMP_H] [TIMESTAMP_L] [ENERGY] [ID] [CHECKSUM]
         │
         ▼
  SPI Master → Raspberry Pi
```

---

## SPIKE FORMAT (To co wysyłasz)

Każdy spike to 6 bajtów:

| Byte | Nazwa | Wartość | Znaczenie |
|------|-------|---------|-----------|
| 0 | HEADER | 0xAE | Marker zdarzenia |
| 1-2 | TIMESTAMP | 0x03E8 | Sample number (1000) |
| 3 | ENERGY | 0x95 | Energia anomalii (149/255) |
| 4 | SPIKE_ID | 0x00 | Zawsze 0 (1 neuron) |
| 5 | CHECKSUM | 0x4E | XOR(H^T^E^ID) |

---

## DEBUG - Jeśli Coś Nie Działa

### Problem: Nie widać spike'ów
**Przyczyna**: Mikrofon nie podłączony lub ADC czyta szum
**Rozwiązanie**:
```cpp
// Dodaj w setup():
while(1) {
  int val = analogRead(MIC_PIN);
  Serial.println(val);  // Powinno oscylować wokół 512
  delay(100);
}
```

### Problem: Zbyt wiele fake'owych spike'ów
**Przyczyna**: Threshold za niski
**Rozwiązanie**: Zwiększ ENERGY_THRESHOLD o 50

### Problem: Brak komunikacji SPI z RPi
**Przyczyna**: Loose wires, CS pin inny
**Rozwiązanie**:
- Sprawdź #define SPI_CS_PIN
- Sprawdź kabele (głównie GND!)

---

## CO ROBIĆ DALEJ?

1. **Skalibruj mikrofon** - sprawdź ADC wartości w ciszy
2. **Ustaw threshold** - detector powinien reagować na dźwięki >100dB
3. **Przetestuj SPI komunikacją** - użyj oscyloskopu na SCLK
4. **Potwierdzony Teraz skontaktuj się z kolegą (RPi)** - test całego systemu

---

## OSCYLOSKOP - Co Powinieneś Widzieć?

```
PIN 13 (SCLK):
  ┌─┐ ┌─┐ ┌─┐ ┌─┐ ┌─┐
  └─┘ └─┘ └─┘ └─┘ └─┘  (4 MHz clock)

PIN 10 (CS):
  ───┐         ┌──────  (LOW podczas transmisji)
     └─────────┘

PIN 11 (MOSI):
  ──┬─┬─┬─────────────  (spike packet bytes)
     
(każdy spike trwa ~12µs na SPI)
```

---

## CHEAT SHEET - Szybkie Komendy

```cpp
// Włącz LED dla każdego spike'a
digitalWrite(LED_DEBUG, HIGH);
delay(50);
digitalWrite(LED_DEBUG, LOW);

// Wydrukuj warto
ści do debugowania
Serial.print("Energy: ");
Serial.println(current_energy);

// Resetuj baseline
baseline_energy = current_energy;
```

---

## Pytania?
- Spike'i nie wysyłają → sprawdź SPI pinout
- Falszywe alarmy → zwiększ ENERGY_THRESHOLD
- Mikrofon cichy → sprawdź mikrofon pinout (VCC, GND, OUT)

**Powodzenia! 🚀**
