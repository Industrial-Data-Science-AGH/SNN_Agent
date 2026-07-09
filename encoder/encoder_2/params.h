#ifndef PARAMS_H
#define PARAMS_H

#define ENCODER_MODE RATE_CODING
#define RATE_CODING 1
#define TTFS 2

#define AUTO_CALIBRATE 0
#define HPF_ENABLED 0
#define ALPHA 0.99f

// PINOUT (Rozszerzony do 4 kanałów)
#define PIN_MIC A0
#define PIN_SPIKE_ZCR   5  // CH0: Zero Crossing Rate
#define PIN_SPIKE_DELTA 6  // CH1: Delta Energii
#define PIN_SPIKE_CREST 7  // CH2: Crest Factor
#define PIN_SPIKE_PEAKS 8  // CH3: Peak Counting
#define PIN_DEBUG_LED   13

#define FRAME_WINDOW_MS 20

const uint8_t SPIKE_PINS[4] = {PIN_SPIKE_ZCR, PIN_SPIKE_DELTA, PIN_SPIKE_CREST, PIN_SPIKE_PEAKS};

// Konfiguracja Rate Coding
#define RC_MIN_RATE_HZ 5 
#define RC_MAX_RATE_HZ 50 
#define RC_NOISE_FLOOR 0.05f 
#define TTFS_THRESHOLD 0.1f 

const float LP_ALPHA[4] = {1.0f, 0.4f, 0.4f, 1.0f};
#define TTFS_WINDOW_US 5000UL
#define SPIKE_WIDTH_US 15000UL

// Maksymalne wartości dla normalizacji (ustawione eksperymentalnie na bazie zakresu ADC 0-1023)
#define MAX_ZCR_VAL     40.0f   // Maksymalna liczba przejść w ramce 20ms
#define MAX_DELTA_VAL   300.0f  // Maksymalna różnica energii między ramkami
#define MAX_CREST_VAL   5.0f    // Maksymalny stosunek Peak/Mean
#define MAX_PEAKS_VAL   30.0f   // Maksymalna liczba lokalnych maksimów

const float MAX_VALS[4] = {MAX_ZCR_VAL, MAX_DELTA_VAL, MAX_CREST_VAL, MAX_PEAKS_VAL};

#endif