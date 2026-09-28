"""What actually fits on the ATmega328P, from measurements rather than opinion.

The project's claim is not only "the SNN detects glass". It is "the SNN detects
glass on hardware where a Fourier front end does not fit". That second half is a
statement about cycles, and it is the one a supervisor will push on, so it is
kept separate from the quality numbers and sourced separately.

Measured, by Kacper, on a real board
    ``encoder/features-improvement/measurements.json``, section ``board``:
    ATmega328P at 16 MHz, sampling at 19230.5 Hz, the current seven channel
    time domain encoder. The per sample ISR costs 590.5 cycles (36.91 us) and
    takes 70.97 % of the CPU; the per frame work costs 2885 us on average, in a
    frame that lasts 9983 us.

    Those two numbers together leave about eleven microseconds per frame. The
    chip is not nearly full, it is full.

Estimated, here, with the arithmetic shown
    The cost of an FFT or of Goertzel bins on that chip. These are labelled
    ``estimated`` everywhere they appear and they are never mixed into the
    measured rows. The assumptions are stated as ranges rather than a single
    flattering figure, and even the optimistic end of each range is far outside
    the budget, which is what makes the conclusion robust to being wrong about
    the constant.

Nothing in the comparison's quality numbers depends on this file. It answers a
different question: not "which is more accurate" but "which one can run".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MEASUREMENTS = Path("encoder/features-improvement/measurements.json")
F_CPU_HZ = 16_000_000
HOP_SAMPLES = 192

# A 16 bit fixed point radix-2 butterfly on AVR: four multiplies, six add or
# subtract, plus addressing and loop overhead. Published AVR FFT work lands in
# this band; the low end is generous towards Fourier on purpose.
BUTTERFLY_CYCLES = (60, 100)
BUTTERFLY_CYCLES_OPTIMISTIC = 30

# One Goertzel step is a multiply-accumulate and two adds in 16 bit, per sample
# per bin, plus the loop.
GOERTZEL_CYCLES_PER_SAMPLE = (10, 16)


@dataclass(frozen=True)
class Budget:
    """One frame of the encoder's timeline, in cycles."""

    frame_us: float
    isr_cpu_fraction: float
    frame_processing_us: float

    @property
    def frame_cycles(self) -> int:
        return round(self.frame_us * F_CPU_HZ / 1e6)

    @property
    def isr_cycles(self) -> int:
        return round(self.frame_cycles * self.isr_cpu_fraction)

    @property
    def processing_cycles(self) -> int:
        return round(self.frame_processing_us * F_CPU_HZ / 1e6)

    @property
    def free_cycles(self) -> int:
        return self.frame_cycles - self.isr_cycles - self.processing_cycles

    @property
    def utilisation(self) -> float:
        return 1.0 - self.free_cycles / self.frame_cycles


def measured_budget(path: Path = MEASUREMENTS) -> Budget:
    board = json.loads(path.read_text(encoding="utf-8"))["board"]["baseline"]
    return Budget(
        frame_us=board["p1"]["per_us_mean"],
        isr_cpu_fraction=board["isr_cpu_pct"] / 100.0,
        frame_processing_us=board["p1"]["proc_us_mean"],
    )


def fft_cycles(n: int) -> tuple[int, int, int]:
    """(optimistic, low, high) cycles for one real radix-2 FFT of size n."""
    butterflies = (n // 2) * (n.bit_length() - 1)
    return (
        butterflies * BUTTERFLY_CYCLES_OPTIMISTIC,
        butterflies * BUTTERFLY_CYCLES[0],
        butterflies * BUTTERFLY_CYCLES[1],
    )


def goertzel_cycles(bins: int, samples: int = HOP_SAMPLES) -> tuple[int, int]:
    return (
        bins * samples * GOERTZEL_CYCLES_PER_SAMPLE[0],
        bins * samples * GOERTZEL_CYCLES_PER_SAMPLE[1],
    )


def report(path: Path = MEASUREMENTS) -> dict:
    budget = measured_budget(path)
    fft256 = fft_cycles(256)
    fft512 = fft_cycles(512)
    goertzel6 = goertzel_cycles(6)
    free = budget.free_cycles
    # Replacing the current channels rather than adding to them frees the whole
    # per frame processing slot; that is the only way a spectral front end has
    # any budget at all, and it is exactly what the `mcu` feature set models.
    if_replacing = free + budget.processing_cycles
    return {
        "source": {
            "measured": str(path),
            "board": "ATmega328P @ 16 MHz",
            "note": "ISR and frame processing are measured; FFT and Goertzel costs are estimated here",
        },
        "frame": {
            "period_us": budget.frame_us,
            "cycles": budget.frame_cycles,
            "isr_cycles_measured": budget.isr_cycles,
            "processing_cycles_measured": budget.processing_cycles,
            "free_cycles": free,
            "utilisation_pct": round(budget.utilisation * 100, 2),
        },
        "estimated_cost_cycles": {
            "fft_256_optimistic": fft256[0],
            "fft_256_range": list(fft256[1:]),
            "fft_512_range": list(fft512[1:]),
            "goertzel_6_bins_range": list(goertzel6),
        },
        "verdict": {
            "free_cycles_as_is": free,
            "free_cycles_if_spectral_replaces_current_channels": if_replacing,
            "fft_256_fits_as_is": fft256[0] <= free,
            "fft_256_fits_if_replacing": fft256[0] <= if_replacing,
            "goertzel_6_fits_if_replacing": goertzel6[1] <= if_replacing,
        },
    }


def main() -> int:
    data = report()
    frame, cost, verdict = data["frame"], data["estimated_cost_cycles"], data["verdict"]
    print(f"ATmega328P @ 16 MHz, frame {frame['period_us']} us = {frame['cycles']} cycles")
    print(f"  measured ISR                 {frame['isr_cycles_measured']:>8} cycles")
    print(f"  measured frame processing    {frame['processing_cycles_measured']:>8} cycles")
    print(f"  free                         {frame['free_cycles']:>8} cycles   ({frame['utilisation_pct']} % used)")
    print()
    print(f"  estimated 256-pt FFT         {cost['fft_256_range'][0]:>8} .. {cost['fft_256_range'][1]} cycles")
    print(f"    (optimistic 30 c/butterfly {cost['fft_256_optimistic']:>8} cycles)")
    print(f"  estimated 6 Goertzel bins    {cost['goertzel_6_bins_range'][0]:>8} .. {cost['goertzel_6_bins_range'][1]} cycles")
    print()
    print(f"  fits as is:            FFT-256 {verdict['fft_256_fits_as_is']}")
    print(f"  if spectral REPLACES the current channels, budget is {verdict['free_cycles_if_spectral_replaces_current_channels']} cycles")
    print(f"    FFT-256 (optimistic) {verdict['fft_256_fits_if_replacing']}   6 Goertzel bins {verdict['goertzel_6_fits_if_replacing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
