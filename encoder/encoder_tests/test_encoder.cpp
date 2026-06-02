#include "mock_params.h"
//  2. INKLUZJA FAKTYCZNEGO KODU ENKODERA
// ============================================================
#include "../snn_encoder_params_hpf/snn_encoder_params_hpf.ino" // dekoder arudino

#define LOOP_LEN 1000

// ============================================================
// FUNKCJE TESTOWE I RESET STANU SPIKE'A
// ============================================================

void reset_mock_env() {
    mock_millis = 0;
    mock_micros = 0;
    mock_analog_read_val = 450;
    delay_was_called = false;
    total_delay_time_us = 0;
    for(int i = 0; i < PIN_CNT; i++) pin_states[i] = 0;
    
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
    
    // Wywołanie z pliku .ino
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
    
    // Wywołanie  setup z pliku .ino
    setup(); 
    
    // Symulacja dostarczania danych audio przez 25 kroków (ponad okno FRAME_WINDOW_MS = 20)
    for (int i = 0; i < LOOP_LEN; i++) {
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


int main() {
    std::cout << "=== URUCHAMIANIE INTEGRACYJNYCH TESTÓW KODU .INO ===\n";
    
    test_heavy_blocking_detection(); 
    test_continuous_generation_and_metrics();
    
    std::cout << "=== WSZYSTKIE TESTY PLIKU .INO ZALICZONE ===\n";
    return 0;
}