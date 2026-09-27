# encoder_v2_swap — przykładowa ramka serial (dla W1 / bridge)

**Status:** realny zrzut z fizycznego Arduino Uno, firmware `encoder_v2_swap.ino`
(build produkcyjny: `ENC_BENCH=0`, bez `-DENC_DEBUG_FEAT`), 115200 baud.
Wejście A0 **pływające** — nie był podłączony mikrofon. To jest celowe: ten
dokument ma potwierdzić **format protokołu**, nie zdolność wykrywania dźwięku.
Zero na wszystkich kanałach to poprawne, oczekiwane zachowanie dla ciszy/szumu
tła, nie błąd.

## Surowy output (Serial Monitor, bez modyfikacji)

```
# encoder_v2 dt=10ms pulse=6ms ch=peak,peak_cnt,cv,zcr,flux,hf_lo,hf_hi
frame,s0,s1,s2,s3,s4,s5,s6
52,0,0,0,0,0,0,0
53,0,0,0,0,0,0,0
54,0,0,0,0,0,0,0
55,0,0,0,0,0,0,0
56,0,0,0,0,0,0,0
57,0,0,0,0,0,0,0
58,0,0,0,0,0,0,0
59,0,0,0,0,0,0,0
60,0,0,0,0,0,0,0
61,0,0,0,0,0,0,0
62,0,0,0,0,0,0,0
63,0,0,0,0,0,0,0
64,0,0,0,0,0,0,0
65,0,0,0,0,0,0,0
```

## Uwagi dla parsera / bridge'a (W1)

1. **Linia nagłówkowa (`# encoder_v2 dt=...`) jest przestarzała — ignorować.**
   Wypisuje stare etykiety kanałów (`peak_cnt`, `cv`), mimo że ten build
   faktycznie liczy `hjorth_mobility`/`autocorr_lag1` (`ENC_SET_SWAP=1` na
   stałe w tym pliku). **Jedynym źródłem prawdy dla znaczenia kanałów jest
   zamrożony `EncoderProfile.channel_map`**, nie ten string diagnostyczny.
2. **Pierwsza ramka danych to `frame=52`, nie `0` ani `1`.** Ramki `0..51`
   (52 sztuki, ~0.52s) to wewnętrzny priming floor/MAD firmware — nic nie
   jest w tym czasie wypisywane na serial. To nie jest luka/utrata danych,
   tylko normalny rozruch. (Potwierdzone też niezależnie w kodzie — patrz
   `encoder_config.json → trigger_logic.prime_frames_actual_count`.)
3. **Kolumny `s0..s6` to bity spike'a w TEJ ramce** (0/1), nie wartości
   ciągłe cechy — kolejność zgodna z `channel_map` (`ch1=peak … ch7=hf_hi`).
4. **Brak realnego znacznika czasu.** `frame` to lokalny licznik ramek
   firmware, nie `source_monotonic_us`. Dodanie prawdziwego `timestamp`+`seq`
   to zadanie **K2**, nie jest jeszcze zaimplementowane — proszę nie zakładać
   `source_us = frame * 10ms` jako zamiennika (dryfuje, patrz "late frames"
   w RUNBOOK/measurements).
5. **Ta konkretna próbka nie zawiera ani jednego wystrzelonego kanału**
   (wejście pływające, brak sygnału). Jeśli potrzebny jest przykład z
   niezerowymi bitami do testów parsera, można dołączyć fragment z
   `parity_real.json` (realne audio, symulator) — tam kanały strzelają
   regularnie, ale to już nie jest zrzut z fizycznej płytki, tylko z symulacji
   cyklowej (`simavr`). Dać znać, jeśli to też przydatne.