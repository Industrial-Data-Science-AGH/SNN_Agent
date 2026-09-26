#!/usr/bin/env bash
# run_predictions.sh — buduje warianty firmware, liczy przewidywane cykle ISR (analiza statyczna) i zapisuje predictions.json
# oraz sprawdza, że łatka z flagami=0 jest BAJT-W-BAJT identyczna z oryginałem.
# Użycie: tools/run_predictions.sh <encoder_v2.ino_oryginalny> [katalog_build]
set -euo pipefail
ORIG="${1:?podaj ścieżkę do oryginalnego encoder_v2.ino}"; B="${2:-build}"
HERE="$(cd "$(dirname "$0")" && pwd)"; KIT="$(dirname "$HERE")"
mkdir -p "$B"
python3 "$HERE/make_swap_ino.py" "$ORIG" "$KIT/firmware/encoder_v2_swap.ino"
"$HERE/build_fw.sh" "$ORIG" "$B/orig" >/dev/null
"$HERE/build_fw.sh" "$KIT/firmware/encoder_v2_swap.ino" "$B/p0" >/dev/null
cmp "$B/orig/fw.hex" "$B/p0/fw.hex" && echo ">>> flagi=0: firmware IDENTYCZNY z oryginałem (łatka nic nie zmienia w baseline)"
declare -A V=( [baseline]="" [acc32]="-DENC_ACC32=1" [dcfix]="-DENC_DC_FIX=1" [parity]="-DENC_PARITY=1"
               [swap]="-DENC_SET_SWAP=1" [swap_acc32]="-DENC_SET_SWAP=1 -DENC_ACC32=1" [swap_full]="-DENC_SET_SWAP=1 -DENC_PARITY=1 -DENC_ACC32=1" )
for n in "${!V[@]}"; do
  "$HERE/build_fw.sh" "$KIT/firmware/encoder_v2_swap.ino" "$B/$n" ${V[$n]} >/dev/null
  python3 "$HERE/isr_cycles.py" "$B/$n/fw.asm" --label "$n" --json "$B/$n/isr.json" >/dev/null
done
python3 - "$B" << 'PY'
import json, sys, os, re
B = sys.argv[1]; out = {}
flags = {"baseline": "", "acc32": "-DENC_ACC32=1", "dcfix": "-DENC_DC_FIX=1", "parity": "-DENC_PARITY=1", "swap": "-DENC_SET_SWAP=1",
         "swap_acc32": "-DENC_SET_SWAP=1 -DENC_ACC32=1", "swap_full": "-DENC_SET_SWAP=1 -DENC_PARITY=1 -DENC_ACC32=1"}
for n, f in flags.items():
    r = json.load(open(f"{B}/{n}/isr.json"))
    sz = open(f"{B}/{n}/size.txt").read()
    prog = int(re.search(r"Program:\s+(\d+)", sz).group(1)); data = int(re.search(r"Data:\s+(\d+)", sz).group(1))
    out[n] = dict(flags=f, isr_cycles_min=r["total_min"], isr_cycles_max=r["total_max"], pushes=r["pushes"], helpers=r["helpers"],
                  flash_bytes=prog, ram_bytes=data, f_cpu=r["f_cpu"])
json.dump(out, open("predictions.json", "w"), indent=2)
print("zapisano predictions.json")
PY
python3 "$HERE/../esos_time_analysis_avr.py" --predictions predictions.json --table-only
