#!/usr/bin/env python3
"""Syntetyczne audio do testów (NIE zastępuje prawdziwych nagrań): tło, dudnienie, 'szkło', mowa."""
import numpy as np
from scipy.signal import butter, sosfilt
import soundfile as sf

SR = 44100

def _bp(x, lo, hi):
    return sosfilt(butter(4, [lo, hi], btype="band", fs=SR, output="sos"), x)

def make(seconds=20.0, seed=0, bg_level=0.004):
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    t = np.arange(n) / SR
    y = bg_level * rng.normal(size=n)                          # tło (szum, niski poziom)
    y += 0.5 * bg_level * np.sin(2 * np.pi * 50 * t)          # brum
    def add(start, sig):
        i = int(start * SR); y[i:i + len(sig)] += sig[: max(0, n - i)]
    # dudnienie / łomot (niskopasmowe, głośne, nie-szkło)
    for st in (4.0, 11.5):
        tt = np.arange(int(0.6 * SR)) / SR
        add(st, 0.55 * np.sin(2 * np.pi * 90 * tt) * np.exp(-tt / 0.12))
    # 'szkło': szerokopasmowy transient + wysokie rezonanse 4-9 kHz, rozpad, kilka 'dzwonków'
    for st in (7.0, 15.0):
        tt = np.arange(int(1.2 * SR)) / SR
        burst = _bp(rng.normal(size=len(tt)), 3500, 12000) * np.exp(-tt / 0.18)
        ring = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt / d) for f, d in ((4300, .25), (6100, .18), (8800, .12)))
        add(st, 0.30 * burst / burst.std() * 0.25 + 0.10 * ring)
        for k in range(3):
            add(st + 0.25 + 0.2 * k, 0.05 * _bp(rng.normal(size=int(0.08 * SR)), 5000, 11000))
    # 'mowa': modulowany szum 200-3000 Hz
    seg = _bp(rng.normal(size=int(3.0 * SR)), 200, 3000)
    env = np.abs(np.sin(2 * np.pi * 3.5 * np.arange(len(seg)) / SR)) ** 1.5
    add(0.5, 0.12 * seg / seg.std() * env * 0.5)
    return np.clip(y, -1, 1).astype(np.float32)

def make_k2_vector(seed=0, bg_level=0.004):
    """Ustalony wektor dla K2 krok 2: cisza, impuls, sinus, nagla zmiana amplitudy, szklo
    (w tej kolejnosci, jako jeden ciagly plik). Nie zastepuje realnych nagran ESC-50 -
    to synteza specjalnie pod te 5 przypadkow z opisu zadania, ktorych brakowalo w make()."""
    rng = np.random.default_rng(seed)
    segments = []

    def seg(seconds):
        n = int(seconds * SR)
        return np.zeros(n, dtype=np.float64), n

    # 1) cisza: tylko szum kwantyzacji ADC, bez zadnego sygnalu
    y, n = seg(2.0)
    y += 1e-4 * rng.normal(size=n)
    segments.append(y)

    # 2) impuls: pojedynczy szerokopasmowy klik ~5ms (nie 'szklo' - bez rezonansow), potem cisza
    y, n = seg(1.0)
    tt = np.arange(int(0.005 * SR)) / SR
    click = _bp(rng.normal(size=len(tt)), 500, 15000) * np.exp(-tt / 0.001)
    click = 0.6 * click / (np.abs(click).max() + 1e-9)
    y[: len(click)] += click
    segments.append(y)

    # 3) sinus: ton ciagly 1 kHz przez 2s (bez obwiedni/zaniku, w odroznieniu od 'dudnienia' w make())
    n = int(2.0 * SR)
    tt = np.arange(n) / SR
    segments.append(0.2 * np.sin(2 * np.pi * 1000 * tt))

    # 4) nagla zmiana amplitudy: skok do stalego, glosnego poziomu bez narastania (test DC/floor trackera)
    y, n = seg(1.0)
    y[:] = bg_level * 20
    y += bg_level * 2 * rng.normal(size=n)
    segments.append(y)

    # 5) szklo: ten sam ksztalt transientu co w make(), jedno zdarzenie
    y, n = seg(1.5)
    tt = np.arange(int(1.2 * SR)) / SR
    burst = _bp(rng.normal(size=len(tt)), 3500, 12000) * np.exp(-tt / 0.18)
    ring = sum(np.sin(2 * np.pi * f * tt) * np.exp(-tt / d) for f, d in ((4300, .25), (6100, .18), (8800, .12)))
    y[: len(tt)] += 0.30 * burst / (burst.std() + 1e-9) * 0.25 + 0.10 * ring
    segments.append(y)

    out = np.concatenate(segments)
    return np.clip(out, -1, 1).astype(np.float32)


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synth.wav"
    which = sys.argv[2] if len(sys.argv) > 2 else "base"
    sf.write(out, make_k2_vector() if which == "k2" else make(), SR)
    print("zapisano", out)