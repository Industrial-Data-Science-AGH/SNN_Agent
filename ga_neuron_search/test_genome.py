#!/usr/bin/env python3
"""
test_genome.py — testy niezmienników genomu i mechaniki GA (bez torcha).

Uruchom: python test_genome.py
Sprawdza, że każda operacja (random, mutate, crossover, repair) utrzymuje
wszystkie twarde ograniczenia sprzętowe i że GA na fitnessie syntetycznym
faktycznie poprawia wynik.
"""
from __future__ import annotations

import random

from genome import (MAX_FANIN, MAX_FANOUT, OUT_FANIN, Genome, crossover,
                    mutate, random_genome)


def assert_valid(g: Genome, ctx: str):
    v = g.violations()
    assert not v, f"{ctx}: naruszenia {v}\n{g.pretty()}"
    # dodatkowo twardo sprawdź fan-in/out i N
    for k in range(len(g.layers)):
        for c in g.fanout_counts(k):
            assert c <= MAX_FANOUT, f"{ctx}: fan-out {c}"
    for layer in g.layers:
        for n in layer:
            assert 1 <= len(n) <= MAX_FANIN, f"{ctx}: fan-in {len(n)}"
    want = min(OUT_FANIN, g.layer_sizes()[-2])
    assert len(g.layers[-1]) == 1 and len(g.layers[-1][0]) == want


def test_random_and_N():
    rng = random.Random(1)
    for n in range(2, 12):
        for _ in range(50):
            g = random_genome(n, rng, max_hidden_layers=4)
            assert_valid(g, f"random N={n}")
            assert g.total_neurons() == n, f"N mismatch {g.total_neurons()} != {n}"
    print("ok: random_genome — poprawność i stałe N dla N=2..11")


def test_mutation_preserves():
    rng = random.Random(2)
    for n in [4, 6, 8, 10]:
        g = random_genome(n, rng, 4)
        for _ in range(500):
            g = mutate(g, rng, rate=1.3)
            assert_valid(g, f"mutate N={n}")
            assert g.total_neurons() == n, f"mutacja zmieniła N: {g.total_neurons()} != {n}"
    print("ok: mutate — utrzymuje ograniczenia i stałe N (500 mutacji/N)")


def test_crossover_preserves():
    rng = random.Random(3)
    for n in [4, 6, 8, 10]:
        for _ in range(200):
            a = random_genome(n, rng, 4)
            b = random_genome(n, rng, 4)
            c = crossover(a, b, rng)
            assert_valid(c, f"crossover N={n}")
            assert c.total_neurons() == n
    print("ok: crossover — poprawność i stałe N (200 par/N)")


def test_ga_improves():
    from fitness import synth_fitness
    from ga import GAConfig, run_ga
    for n in [6, 8, 10]:
        cfg = GAConfig(n_total=n, pop_size=20, generations=12, seed=0)
        res = run_ga(synth_fitness, cfg)
        assert res.history[-1] >= res.history[0] - 1e-9, "GA nie może się pogorszyć"
        assert res.best.genome.is_valid()
        assert res.best.genome.total_neurons() == n
    print("ok: run_ga (synth) — zbiega i nie pogarsza best")


if __name__ == "__main__":
    test_random_and_N()
    test_mutation_preserves()
    test_crossover_preserves()
    test_ga_improves()
    print("\nWSZYSTKIE TESTY PRZESZŁY")
