#ifndef WELLFORD
#define WELLFORD

// wellford online algo dla std
static float wf_mean = 0.0f;
static float wf_M2 = 0.0f;
static uint32_t wf_n = 0;

static float frameMax = 0.0f;
static float hp_filtered = 0.0f;
static float prev_raw = 450.0f;

static float channelValues[3] = {0.0f, 0.0f, 0.0f};
static float smoothedVals[3] = {0.0f, 0.0f, 0.0f};

static uint32_t frameStartMs    = 0;
static bool     newFrameReady   = false;

// Spike generation state
static uint32_t currISI_us[3]    = {0, 0, 0};
static uint32_t lastSpikeTime_us[3] = {0, 0, 0};
static bool     spikeActive[3]   = {false, false, false};
static uint32_t spikeStartUs[3]  = {0, 0, 0};

// TTFS state
static bool     ttfsSpiked[3]       = {false, false, false};
static uint32_t ttfsSpikeAt_us[3]   = {0, 0, 0};  
static uint32_t ttfsFrameStart_us   = 0;

#if AUTO_CALIBRATE
   static float cal_maxPeak = 0.0f;
   static float cal_maxMean = 0.0f;
   static float cal_maxCV   = 0.0f;
   static uint32_t calStartMs = 0;
   static bool calDone = false;
   #define CAL_DURATION_MS 10000
#endif

#endif