#!/usr/bin/env bash
# run_checks.sh — odtwarza wszystkie weryfikacje "bez płytki": łatki, identyczność baseline, cykle ISR, test twina, parytet w symulatorze.
# Użycie: tools/run_checks.sh <encoder_v2.ino_oryginalny> <encoder_twin.py_oryginalny>
set -uo pipefail
ORIG_INO="${1:?ścieżka do oryginalnego encoder_v2.ino}"; ORIG_TWIN="${2:?ścieżka do oryginalnego encoder_twin.py}"
HERE="$(cd "$(dirname "$0")" && pwd)"; KIT="$(dirname "$HERE")"; cd "$KIT"
[ -x tools/simharness ] || tools/build_simharness.sh
FAIL=0
echo "### 1. generowanie + identyczność baseline + cykle ISR"; tools/run_predictions.sh "$ORIG_INO" build || FAIL=1
python3 tools/make_swap_twin.py "$ORIG_TWIN" twin/encoder_twin_swap.py >/dev/null; cp "$ORIG_TWIN" twin/encoder_twin.py
echo; echo "### 2. twin: baseline == oryginał bit-w-bit, swap działa"; python3 tests/test_twin_patch.py 2>&1 | grep -v Warning | tail -4 || FAIL=1
echo; echo "### 3. parytet firmware<->twin w symulatorze (audio syntetyczne; na własnym pliku: tools/parity_test.py --wav ...)"
python3 tools/parity_test.py --synth --seed 1 --bg-level 0.02 --variant baseline 2>&1 | grep "===" 
python3 tools/parity_test.py --synth --seed 1 --bg-level 0.02 --variant swap_full --mob-thr 1.95 --ac-thr 0.31 2>&1 | grep "===" 
echo; [ $FAIL = 0 ] && echo "WSZYSTKO OK" || echo "BŁĘDY — patrz wyżej"
