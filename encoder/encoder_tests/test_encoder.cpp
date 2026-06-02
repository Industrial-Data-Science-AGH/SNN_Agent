#include "mock_params.h"
#include "../snn_encoder_params_hpf/snn_encoder_params_hpf.ino" // dekoder arudino

#define LOOP_LEN 1000
#define EPS 0.01 // toleracja błedów

// ============================================================
// FUNKCJE TESTOWE I RESET STANU SPIKE'A
// ============================================================

void reset_mock_env() {
    mock_millis = 0;
    mock_micros = 0;
    mock_analog_read = 450;
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
    fireSpike(0); 

    assert(pin_states[6] == 1 && "Pin powinien być HIGH w trakcie SPIKE");

    mock_micros += SPIKE_WIDTH_US;

    updateSpike();

    assert(pin_states[6] == 0 && "Pin powinien zostać opuszczony na koniec funkcji");

    // Weryfikacja blokowania
    if (delay_was_called) {
        std::cerr << "-> BŁĄD: Wykryto blokowanie pętli! Czas przestoju: " << total_delay_time_us << " us.\n";
    }
    assert(delay_was_called == false && "KOD GENERUJE PRZERWY MILISEKUNDOWE (Użyto delayMicroseconds)");
    
    std::cout << "-> Test blokowania: ZALICZONY (Brak opóźnień)\n";
}

// Test 2: Sprawdzenie ciągłego przetwarzania i kalkulacji 3 metryk statystycznych
void test_continuous_generation_and_metrics(int mode) {

    reset_mock_env();
    setup(); 

    float cur_max = 0.0f, cur_sum = 0.0f, cur_sum_sq = 0.0f;
    float test_hp = 0.0f, test_prev = 450.0f;
    int test_sample_cnt = 0;
    
    float peak = 0.0f, mean = 0.0f, stdev = 0.0f;
    
    for (int i = 0; i < LOOP_LEN; i++) {
        float val = 450 + (i % 2 == 0 ? 50 : -50);
        mock_analog_read = (int)val;

        // Emulacja HPF (Zwięźle)
        test_hp = ALPHA * (test_hp + val - test_prev);
        test_prev = val;
        float hpf_val = std::abs(test_hp);

        // Accumulacja wartości z filtra HPF
        cur_max = max(hpf_val, cur_max);
        cur_sum += hpf_val;
        cur_sum_sq += hpf_val * hpf_val;
        test_sample_cnt++;

        // Wywołanie pętli adekwatnej do wybranego trybu
        if (mode == RATE_CODING) 
          loopRateCoding(); 
        else
          loopTTFS();
        
        // Zatrzaśnięcie oczekiwanych metryk dokładnie w momencie resetu ramki w .ino
        if (sampleCnt == 0) {
            peak = cur_max;
            mean = cur_sum / (float)test_sample_cnt;
            stdev = sqrt(cur_sum_sq / (float)test_sample_cnt);
            
            cur_max = 0.0f; cur_sum = 0.0f; cur_sum_sq = 0.0f;
            test_sample_cnt = 0;
        }

        mock_millis++;
        mock_micros += 1000;
    }

    // Weryfikacja poprawności uzyskanych metryk statystycznych
    std::cout << "[METRYKI] Peak: " << channelValues[0] << ", Mean: " << channelValues[1] << ", Std: " << channelValues[2] << "\n";
    std::cout << "[OCZEKIWANE] Peak: " << peak << ", Mean: " << mean << ", Std: " << stdev << "\n";
    
    assert(std::abs(channelValues[0] - peak) <= EPS && "Błąd Peak");
    assert(std::abs(channelValues[1] - mean) <= EPS && "Błąd Mean");
    assert(std::abs(channelValues[2] - stdev) <= EPS && "Błąd Std");
    
    // Sprawdzenie, czy podczas pełnego cyklu pętli nie wywołano funkcji blokującej
    assert(delay_was_called == false && "Główna pętla została zablokowana podczas przetwarzania!");

    std::cout << "-> Test ciągłości i metryk: ZALICZONY\n";
}

int main() {
    std::cout << "=== URUCHAMIANIE INTEGRACYJNYCH TESTÓW KODU .INO ===\n";
    
    test_heavy_blocking_detection(); 
    test_continuous_generation_and_metrics(RATE_CODING);
    test_continuous_generation_and_metrics(TTFS);
    
    std::cout << "=== WSZYSTKIE TESTY PLIKU .INO ZALICZONE ===\n";
    return 0;
}