#ifndef FEATURES_H
#define FEATURES_H

// Wspólne zmienne filtracji i czasu
float hp_filtered = 0.0f;
float prev_raw = 450.0f;
uint32_t frameStartMs = 0;
bool newFrameReady = false;
uint32_t globalSampleIdx = 0;

// Rejestry cech dla kanałów
float channelValues[4] = {0.0f, 0.0f, 0.0f, 0.0f};
float smoothedVals[4]  = {0.0f, 0.0f, 0.0f, 0.0f};

// ALGO 1: ZCR State
float prev_val_centered = 0.0f;
uint16_t zcr_count = 0;

// ALGO 2: Delta Energii State
float de_sum_val = 0.0f;
uint16_t de_sample_cnt = 0;
float prev_frame_mean = 512.0f; 

// ALGO 3: Crest Factor State
float cf_sum_val = 0.0f;
float cf_frame_max = 0.0f;
uint16_t cf_sample_cnt = 0;

// ALGO 4: Peak Counting State
float p_val = 0.0f;
float pp_val = 0.0f;
uint16_t local_peaks_count = 0;

// Stan generatorów szpilek (4 kanały)
uint32_t currISI_us[4]       = {0, 0, 0, 0};
uint32_t lastSpikeTime_us[4] = {0, 0, 0, 0};
bool     spikeActive[4]      = {false, false, false, false};
uint32_t spikeStartUs[4]     = {0, 0, 0, 0};

// TTFS State
bool     ttfsSpiked[4]       = {false, false, false, false};
uint32_t ttfsSpikeAt_us[4]   = {0, 0, 0, 0};  
uint32_t ttfsFrameStart_us   = 0;

#endif