/*
 * mega_standin.ino — STAND-IN for the Uno encoder, for bridge tests only. This is NOT the encoder.
 *
 * Emits Uno serial protocol v1 (see rpi_agents/agent/serial_protocol.py) with synthetic spike masks so the
 * Raspberry Pi bridge can be exercised on a real UART before the real Uno firmware (K2) exists. The mask
 * pattern is the same as rpi_agents/agent/synthetic.py: 50 priming hops, then a burst of 8 hops every 100.
 * It has no microphone input, no pulse outputs and no scientific meaning. Results obtained with it are
 * stand-in results and must be labelled as such.
 *
 *   $B,<ver>,<build_id>,<fs_hz>,<hop>,<n_ch>,<pulse_us>,<chset>*<CRC8>
 *   $F,<seq>,<t_us>,<n>,<mask>,<flags>,<txdrop>*<CRC8>
 *
 * Commands over serial (one character, no newline needed):
 *   I  print the boot line again
 *   G  lose the next 10 hops silently (as if the cable dropped them; txdrop unchanged)
 *   T  lose the next 10 hops and report them in txdrop (as if the TX buffer overflowed)
 *   L  merge one hop into the next frame (that frame reports n=384 and seq jumps by 2)
 *
 * Board: Arduino Mega 2560 (any AVR with a hardware Serial works). Baud 115200.
 */

#include <Arduino.h>

static const uint32_t FS_HZ = 19231UL;
static const uint32_t HOP = 192UL;
static const uint8_t N_CH = 7;
static const uint32_t PULSE_US = 6000UL;
static const char BUILD_ID[] = "5717A0D0";  // identifies this stand-in, not a real encoder build
static const char CHSET[] = "swap";

static const uint32_t PRIMING_HOPS = 50;
static const uint32_t BURST_EVERY = 100;
static const uint32_t BURST_LEN = 8;
static const uint8_t BURST_MASK = 0b1110000;  // hf_lo, hf_hi, flux

static uint32_t t0;
static uint32_t hop_index = 0;
static uint16_t txdrop = 0;
static uint8_t lose_hops = 0;
static bool lose_reports_txdrop = false;
static uint8_t late_state = 0;  // 0 none, 1 skip the next hop, 2 next frame is merged

static uint8_t crc8(const char *s) {
  uint8_t crc = 0;
  while (*s) {
    crc ^= (uint8_t)*s++;
    for (uint8_t i = 0; i < 8; i++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  }
  return crc;
}

// Sends "$<body>*<CRC>\r\n". A droppable line is skipped, and counted, when the TX buffer lacks room.
static bool emit(const char *body, bool droppable) {
  uint8_t crc = crc8(body);
  if (droppable && Serial.availableForWrite() < (int)(strlen(body) + 6)) {
    txdrop++;
    return false;
  }
  Serial.write('$');
  Serial.write(body);
  Serial.write('*');
  Serial.write("0123456789ABCDEF"[crc >> 4]);
  Serial.write("0123456789ABCDEF"[crc & 0x0F]);
  Serial.write('\r');
  Serial.write('\n');
  return true;
}

static void boot_line() {
  char body[64];
  snprintf(body, sizeof body, "B,1,%s,%lu,%lu,%u,%lu,%s", BUILD_ID, (unsigned long)FS_HZ,
           (unsigned long)HOP, (unsigned)N_CH, (unsigned long)PULSE_US, CHSET);
  emit(body, false);
}

static void handle_commands() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'I') boot_line();
    else if (c == 'G') { lose_hops = 10; lose_reports_txdrop = false; }
    else if (c == 'T') { lose_hops = 10; lose_reports_txdrop = true; }
    else if (c == 'L') late_state = 1;
  }
}

void setup() {
  Serial.begin(115200);
  boot_line();
  t0 = micros();
}

void loop() {
  handle_commands();
  // hop k ends at t0 + round((k + 1) * HOP / FS): the same grid as synthetic.py
  uint32_t due = t0 + (uint32_t)(((uint64_t)(hop_index + 1) * HOP * 1000000ULL + FS_HZ / 2) / FS_HZ);
  if ((int32_t)(micros() - due) < 0) return;

  uint32_t seq = hop_index++;
  uint32_t now = micros();
  if (lose_hops) {
    lose_hops--;
    if (lose_reports_txdrop) txdrop++;
    return;
  }
  uint16_t n = HOP;
  if (late_state == 1) { late_state = 2; return; }
  if (late_state == 2) { late_state = 0; n = 2 * HOP; }

  uint8_t flags = seq < PRIMING_HOPS ? 1 : 0;
  uint8_t mask = (seq >= PRIMING_HOPS && (seq % BURST_EVERY) < BURST_LEN) ? BURST_MASK : 0;
  char body[64];
  snprintf(body, sizeof body, "F,%lu,%lu,%u,%02X,%X,%u", (unsigned long)seq, (unsigned long)now,
           (unsigned)n, (unsigned)mask, (unsigned)flags, (unsigned)txdrop);
  emit(body, true);
}
