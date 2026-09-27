#!/usr/bin/env bash
# build_fw.sh — kompiluje szkic .ino dla ATmega328P @16 MHz (Arduino Uno/Nano) bez Arduino IDE.
# Użycie:  tools/build_fw.sh <szkic.ino> <katalog_wyjsciowy> [-DFLAGA=1 ...]
# Wymaga: gcc-avr, avr-libc, binutils-avr oraz sklonowanego ArduinoCore-avr
#         (ARDUINO_CORE=/ścieżka/do/ArduinoCore-avr; domyślnie ~/ArduinoCore-avr lub /tmp/ArduinoCore-avr).
set -euo pipefail
INO="$1"; OUT="$2"; shift 2
EXTRA=("$@")
CORE="${ARDUINO_CORE:-}"
[ -z "$CORE" ] && for d in "$HOME/ArduinoCore-avr" /tmp/ArduinoCore-avr; do [ -d "$d" ] && CORE="$d" && break; done
[ -d "$CORE" ] || { echo "brak ArduinoCore-avr (ustaw ARDUINO_CORE)"; exit 1; }
mkdir -p "$OUT"
COMMON=(-mmcu=atmega328p -DF_CPU=16000000L -DARDUINO=10819 -DARDUINO_AVR_UNO -DARDUINO_ARCH_AVR
        -Os -w -ffunction-sections -fdata-sections -I"$CORE/cores/arduino" -I"$CORE/variants/standard")
CORELIB="$OUT/../core.a"
if [ ! -f "$CORELIB" ]; then
  echo "[build] kompilacja rdzenia Arduino -> core.a"
  TMP="$(mktemp -d)"
  for f in "$CORE"/cores/arduino/*.c;   do avr-gcc -c "${COMMON[@]}" -std=gnu11 "$f" -o "$TMP/$(basename "$f").o"; done
  for f in "$CORE"/cores/arduino/*.cpp; do avr-g++ -c "${COMMON[@]}" -std=gnu++11 -fno-exceptions -fno-threadsafe-statics "$f" -o "$TMP/$(basename "$f").o"; done
  avr-ar rcs "$CORELIB" "$TMP"/*.o
fi
# .ino -> .cpp (Arduino dokłada #include <Arduino.h>)
CPP="$OUT/sketch.cpp"
{ echo '#include <Arduino.h>'; cat "$INO"; } > "$CPP"
avr-g++ -c "${COMMON[@]}" "${EXTRA[@]}" -std=gnu++11 -fno-exceptions -fno-threadsafe-statics "$CPP" -o "$OUT/sketch.o"
avr-gcc "${COMMON[@]}" -Wl,--gc-sections -o "$OUT/fw.elf" "$OUT/sketch.o" "$CORELIB" -lm
avr-objcopy -O ihex -R .eeprom "$OUT/fw.elf" "$OUT/fw.hex"
avr-size -C --mcu=atmega328p "$OUT/fw.elf" | sed -n '1,12p' > "$OUT/size.txt"
avr-objdump -d -j .text "$OUT/fw.elf" > "$OUT/fw.asm"
echo "[build] OK -> $OUT/fw.elf"; cat "$OUT/size.txt"
