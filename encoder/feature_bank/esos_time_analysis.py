"""
esos_time_analysis.py
======================
Skrypt analityczny dla systemu wbudowanego (ESOS).
Oblicza dokładny czas wykonania każdej cechy zdigitalizowanego bliźniaka
na fizycznym mikrokontrolerze (Cortex-M4F) wykorzystując analityczne modele cykli.
"""

from digital_twin_encoder import CHANNEL_EXTRACTORS, McuComplexity

# Parametry sprzętowe i ramki
F_CPU = 64_000_000         # Przykładowy zegar 64 MHz (np. STM32, nRF52)
HOP_SAMPLES = 192          # Przybliżona ilość próbek na 10ms (19231Hz)
N_FFT_BINS = 97            # Ilość binów dla rfft(192)

# Lista cech o niskiej sile dyskryminacyjnej na podstawie analizy Cohena (< 0.1)
LOW_YIELD_FEATURES = ["kurtosis", "tkeo_mean", "peak", "peak_cnt"]


def get_shared_base_complexity(n: int) -> McuComplexity:
    # Blokada DC, uśrednianie, RMS, Var, ZCR
    # n*(add, mul) dla blokady, uśrednianie (add, div), RMS (mul, add, sqrt)
    return McuComplexity(add=4*n, mul=3*n, cmp=2*n, div=1, sqrt=1)


def get_shared_fft_complexity(n: int, n_fft: int) -> McuComplexity:
    # Okno Hanninga (muls), FFT radix-2 (~2560 operacji z uwagi na brak idealnej potęgi 2 i padding),
    # moduł widma (n_fft muls, adds, sqrts)
    return McuComplexity(add=2560+n_fft, mul=2560+n+n_fft, sqrt=n_fft)


def cycles_to_us(cycles: int, f_cpu: int) -> float:
    return (cycles / f_cpu) * 1_000_000


def analyze_mcu_time_consumption():
    print(f"=== Analiza czasu na Arduino (Cortex-M4F @ {F_CPU//1_000_000} MHz) ===")
    
    base_comp = get_shared_base_complexity(HOP_SAMPLES)
    base_cycles = base_comp.cycles_cortex_m4f()
    print(f"Baza (RMS, DC, zcr_pre): {cycles_to_us(base_cycles, F_CPU):.2f} us ({base_cycles} cykli)")
    
    fft_comp = get_shared_fft_complexity(HOP_SAMPLES, N_FFT_BINS)
    fft_cycles = fft_comp.cycles_cortex_m4f()
    print(f"Baza FFT (Okno + Trans.): {cycles_to_us(fft_cycles, F_CPU):.2f} us ({fft_cycles} cykli)")
    print("-" * 65)

    total_cycles = base_cycles + fft_cycles
    
    for name, feat in CHANNEL_EXTRACTORS.items():
        comp = feat.complexity(HOP_SAMPLES, N_FFT_BINS)
        cycles = comp.cycles_cortex_m4f()
        total_cycles += cycles
        us = cycles_to_us(cycles, F_CPU)
        
        suggestion = ""
        if name in LOW_YIELD_FEATURES:
            suggestion = " [Sugestia usunięcia: Nieopłacalny stosunek czasu do siły d]"
            
        flag = "[FFT] " if feat.is_spectral else "[Czas]"
        print(f"{flag:6s} {name:20s}: {us:>6.2f} us ({cycles:>5} cykli){suggestion}")

    print("-" * 65)
    print(f"CAŁKOWITY CZAS RAMKI: {cycles_to_us(total_cycles, F_CPU):.2f} us / {10000.0:.2f} us budżetu (10ms)")
    print(f"CPU Load: {(total_cycles / (F_CPU * 0.01)) * 100:.1f}%")

if __name__ == "__main__":
    analyze_mcu_time_consumption()