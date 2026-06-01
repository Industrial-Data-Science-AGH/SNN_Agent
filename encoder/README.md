# SNN Spike Encoder — Arduino Mega

## Schemat podłączenia

```
MAX4466 (mikrofon)          Arduino Mega
──────────────────          ────────────
VCC  ──────────────────────  5V
GND  ──────────────────────  GND
OUT  ──────── R1(10kΩ) ────  A0
              │
             R2(10kΩ)
              │
             GND
```

> R1/R2 tworzą dzielnik napięcia jeśli mikrofon daje >3.3V.
> Dla MAX4466 zasilanego 5V sygnał wyjściowy to ~0.5V p-p, dzielnik nie jest potrzebny.

```
Arduino Mega                Neuron PCB
────────────                ──────────
PIN 6 ──── R_series(100Ω) ── VIN_neuron
GND  ──────────────────────  GND_neuron
```

R_series 100Ω chroni przed przetężeniem i ogranicza pojemność pasożytniczą kabla.

## Kalibracja — pierwsze uruchomienie

1. Wgraj kod z `ENCODER_MODE RATE_CODING`
2. Otwórz Serial Monitor @ 115200
3. W ciszy sprawdź kolumnę AMP — powinna być < RC_NOISE_FLOOR (40)
4. Klaśnij w pobliżu mikrofonu — AMP powinno skoczyć do >200
5. Jeśli AMP w ciszy > 60 → zwiększ RC_NOISE_FLOOR do AMP_cisza + 20

## Parametry do dostrojenia

| Parametr | Domyślna | Co zmienić jeśli... |
|---|---|---|
| RC_NOISE_FLOOR | 40 | Za dużo fałszywych spike'ów w ciszy → zwiększ |
| RC_MAX_RATE_HZ | 200 | Neuron PCB saturuje za szybko → zmniejsz |
| SPIKE_WIDTH_US | 500 | Neuron nie "widzi" spikes → zwiększ do 1000 |
| TTFS_THRESHOLD | 50 | Za wrażliwy/za mało wrażliwy → dostosuj |
| TTFS_WINDOW_MS | 20 | Okno za krótkie dla szkła → zwiększ do 30 |

## Kiedy używać Rate Coding vs TTFS

**Rate Coding** — lepszy gdy:
- Sygnał jest ciągły (np. klasyfikacja dźwięku środowiskowego)
- Chcesz stabilnej, przewidywalnej liczby spike'ów
- Prostsza implementacja enkodera w hardware

**TTFS** — lepszy dla detekcji szkła gdy:
- Zdarzenie jest krótkie i impulsowe (~50-200ms)
- Zależy ci na latencji (pierwsze spike → reakcja systemu)
- Masz już wytrenowany neuron pod TTFS (sprawdź jakie wagi daje sim_train)

## Format wyjścia Serial

```
[ms_od_startu]   [amplituda ADC]   [ISI_us lub delay_ms]   [suma spike'ów]
1234             287               5000                     42
```

## Znane ograniczenia

- `readAmplitude()` blokuje pętlę na RC_WINDOW_MS ms → nie używaj przerwań SPI równocześnie
  bez przepisania na timer ISR
- ADC prescaler=16 daje ~77kHz sampling ale gorzej odrzuca szumy niż domyślny 9.6kHz —
  jeśli masz dużo szumu, zmień z powrotem na prescaler=128 (ADCSRA |= 0x07)
- SPIKE_WIDTH_US=500 + rate 200Hz = duty cycle 10% — w porządku dla neuronu RC
