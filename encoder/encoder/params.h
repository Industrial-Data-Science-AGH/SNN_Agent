#ifndef PARAMS
#define PARAMS



#define ENCODER_MODE RATE_CODING

#define RATE_CODING 1
#define TTFS 2

// czy kalibrujemy i obliczamy max vals
#define AUTO_CALIBRATE 0

// PINOUT
#define PIN_MIC A0
#define PIN_SPIKE_PEAK 6 // peak energy
#define PIN_SPIKE_MEAN 7 // mean energy
#define PIN_SPIKE_CV 8   // coeef of var
#define PIN_DEBUG_LED 13

// okno analizy audio
#define FRAME_WINDOW_MS 20

const uint8_t SPIKE_PINS[3] = {PIN_SPIKE_PEAK, PIN_SPIKE_MEAN, PIN_SPIKE_CV};

#define RC_MIN_RATE_HZ 5 // min czestotilowsc spike

// [ZMIANA DLA LUI] Obniżono z 200 Hz do 50 Hz. 
// Przy 50 Hz maksymalny okres (ISI) to 20ms. Pozwala to na pełny impuls 15ms + 5ms przerwy.
#define RC_MAX_RATE_HZ 50 

#define RC_NOISE_FLOOR 0.05f // ponizej 5% normalized cisza
#define TTFS_THRESHOLD 0.1f // 10%

// 1.0 = brak filtra wygladzenia
const float LP_ALPHA[3] = {1.0f, 0.4f, 0.4f};


// [ZMIANA DLA LUI] Zmniejszono z 20000us na 5000us (5ms).
// Maksymalne opóźnienie szpilki (5ms) + czas trwania szpilki (15ms) = 20ms.
// Dzięki temu szpilka TTFS zawsze zamknie się w obrębie bieżącej ramki i nie zablokuje kolejnej.
#define TTFS_WINDOW_US 5000UL

#if AUTO_CALIBRATE == 0
  // domyślne wartości dla mikrofonu ze wzmocnieniem i bez
  // teraz daje wartosci dla mikrofonu ze wzmaczniaczem
  // na podstawie średniej z kiklu dźwięków szkła
  #define STRENGTHENED 1
    #if STRENGTHENED
      #define MAX_PEAK_VAL   597.3f
      #define MAX_MEAN_VAL   563.8f
      #define MAX_STD_VAL    0.03f
    #else
      #define MAX_PEAK_VAL   556.6f 
      #define MAX_MEAN_VAL   519.2f
      #define MAX_STD_VAL    0.03f
    #endif
  
  const float MAX_VALS[3] = {MAX_PEAK_VAL, MAX_MEAN_VAL, MAX_STD_VAL};
#endif

// [ZMIANA DLA LUI] Zwiększono z 500us do 15000us (15ms), ponieważ LUI 
// ignoruje amplitudę (>1.8V), a siłę synapsy skaluje liniowo szerokością pulsu.
#define SPIKE_WIDTH_US 15000UL

#define HPF_ENABLED 0
#define ALPHA 0.99f

#endif