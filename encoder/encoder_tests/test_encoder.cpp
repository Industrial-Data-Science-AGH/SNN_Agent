#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <cstdint>

// ============================================================
//  1. MOCKOWANIE ŚRODOWISKA ARDUINO
// ============================================================

#define INPUT 0
#define OUTPUT 1
#define LOW 0
#define HIGH 1
#define A0 0

// Definicja min/max jako makra preprocesora chroni przed konfliktami z std::min/max
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

uint8_t ADCSRA = 0; 

unsigned long mock_millis = 0;
unsigned long mock_micros = 0;
int mock_analog_read_val = 450;
int pin_states[100] = {0};

bool delay_was_called = false;
unsigned long total_delay_time_us = 0;

unsigned long millis() { return mock_millis; }
unsigned long micros() { return mock_micros; }
int analogRead(int pin) { return mock_analog_read_val; }
void pinMode(int pin, int mode) {}
void digitalWrite(int pin, int val) { pin_states[pin] = val; }

// Przechwytywanie funkcji opóźniających w celu weryfikacji przerw w działaniu
void delayMicroseconds(unsigned int us) {
    delay_was_called = true;
    total_delay_time_us += us;
    mock_micros += us; 
}

#define F(x) x
struct MockSerial {
    void begin(unsigned long speed) {}
    void print(const char* s) {}
    void print(int n) {}
    void print(float f) {}
    void println(const char* s = "") {}
    void println(int n) {}
    void println(float f) {}
};
MockSerial Serial;

// ============================================================
//  2. INKLUZJA FAKTYCZNEGO KODU ENKODERA
//  Ścieżka zgodna z konfiguracją katalogów w projekcie
// ============================================================
#include "../snn_encoder_params_hpf/snn_encoder_params_hpf.ino"

// ============================================================
//  3. FUNKCJE TESTOWE I RESET STANU SPIEKA
// ============================================================

void reset_mock_env() {
    mock_millis = 0;
    mock_micros = 0;
    mock_analog_read_val = 450;
    delay_was_called = false;
    total_delay_time_us = 0;
    for(int i = 0; i < 100; i++) pin_states[i] = 0;
    
    // Reset stanu zmiennych globalnych zaimportowanych z pliku .ino
    frameStartMs = 0;
    maxAc = 0.0f;
    sumAc = 0.0f;
    sumSq = 0.0f;
    sampleCnt = 0;
    newFrameReady = false;
    hp_filtered = 0.0f;
    prev_raw = 450.0f;
    lastWindowStart = 0;
    
    for(int i = 0; i < 3; i++) {
        channelValues[i] = 0.0f;
        smoothedVals[i] = 0.0f;
        currISI_us[i] = 0;
        lastSpikeTime[i] = 0;
        ttfsSpiked[i] = false;
        spikesampleCnt[i] = 0;
    }
}

// Test 1: Sprawdzenie czy generowanie impulsu nie wymusza przerw w ciągłości (brak delay)
void test_heavy_blocking_detection() {
    reset_mock_env();
    std::cout << "[TEST] Uruchamianie oryginalnej funkcji fireSpike(0)...\n";
    
    // Wywołanie produkcyjnej funkcji z pliku .ino
    fireSpike(0); 

    assert(pin_states[6] == 0 && "Pin powinien zostać opuszczony na koniec funkcji");

    // Weryfikacja blokowania
    if (delay_was_called) {
        std::cerr << "-> BŁĄD: Wykryto blokowanie pętli! Czas przestoju: " << total_delay_time_us << " us.\n";
    }
    assert(delay_was_called == false && "KOD GENERUJE PRZERWY MILISEKUNDOWE (Użyto delayMicroseconds)");
    
    std::cout << "-> Test blokowania: ZALICZONY (Brak opóźnień)\n";
}

// Test 2: Sprawdzenie ciągłego przetwarzania i kalkulacji 3 metryk statystycznych
void test_continuous_generation_and_metrics() {
    reset_mock_env();
    
    // Wywołanie oryginalnej funkcji setup z pliku .ino
    setup(); 
    
    // Symulacja dostarczania danych audio przez 25 kroków (ponad okno FRAME_WINDOW_MS = 20)
    for (int i = 0; i < 25; i++) {
        mock_analog_read_val = 450 + (i % 2 == 0 ? 50 : -50); 
        
        // Wywołanie pętli adekwatnej do wybranego trybu
        #if ENCODER_MODE == RATE_CODING
          loopRateCoding(); 
        #else
          loopTTFS();
        #endif
        
        mock_millis++;
        mock_micros += 1000;
    }

    // Weryfikacja poprawności uzyskanych metryk statystycznych
    std::cout << "[METRYKI] Peak: " << channelValues[0] << ", Mean: " << channelValues[1] << ", Std: " << channelValues[2] << "\n";
    assert(channelValues[0] > 0.0f && "Brak kalkulacji Peak");
    assert(channelValues[1] > 0.0f && "Brak kalkulacji Mean");
    assert(channelValues[2] > 0.0f && "Brak kalkulacji Std");
    
    // Sprawdzenie, czy podczas pełnego cyklu pętli nie wywołano funkcji blokującej
    assert(delay_was_called == false && "Główna pętla została zablokowana podczas przetwarzania!");

    std::cout << "-> Test ciągłości i metryk: ZALICZONY\n";
}

// Punkt wejścia dla kompilatora g++
int main() {
    std::cout << "=== URUCHAMIANIE INTEGRACYJNYCH TESTÓW KODU .INO ===\n";
    
    test_heavy_blocking_detection(); 
    test_continuous_generation_and_metrics();
    
    std::cout << "=== WSZYSTKIE TESTY PLIKU .INO ZALICZONE ===\n";
    return 0;
}