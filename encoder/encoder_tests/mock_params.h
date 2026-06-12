#ifndef MOCK_PARAMS

#define MOCK_PARAMS

#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <cstdint>
#include <unistd.h>

// ============================================================
//  1. MOCKOWANIE ŚRODOWISKA ARDUINO
// ============================================================

#define INPUT 0
#define OUTPUT 1
#define LOW 0
#define HIGH 1
#define A0 0
#define PIN_CNT 14

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))

uint8_t ADCSRA = 0; 

unsigned long mock_millis = 0;
unsigned long mock_micros = 0;
int mock_analog_read = 450;
int pin_states[PIN_CNT] = {0};

bool delay_was_called = false;
unsigned long total_delay_time_us = 0;

unsigned long millis() { return mock_millis; }
unsigned long micros() { return mock_micros; }
int analogRead(int pin) { return mock_analog_read; }
void pinMode(int pin, int mode) {}
void digitalWrite(int pin, int val) { pin_states[pin] = val; }

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

#endif