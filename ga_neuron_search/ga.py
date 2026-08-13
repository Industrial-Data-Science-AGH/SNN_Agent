#!/usr/bin/env python3
"""
ga.py — silnik algorytmu genetycznego (single-objective, jeden run = jedno N).

Sweep po liczbie neuronów robimy na zewnątrz (run_search.py): dla każdego N
uruchamiamy osobny run GA z ustaloną liczbą neuronów, a potem porównujemy
najlepsze osobniki między N. Dzięki temu każdy run optymalizuje wyłącznie
OKABLOWANIE i układ warstw przy stałym budżecie sprzętowym.

Fitness jest wstrzykiwany jako callable(Genome) -> float (im więcej tym lepiej),
więc silnik nie zależy od torcha i da się go testować na funkcji syntetycznej.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from genome import Genome, crossover, mutate, random_genome

FitnessFn = Callable[[Genome], float]


@dataclass
class GAConfig:
    n_total: int                     # neurony w tym runie (ukryte + decyzyjny)
    pop_size: int = 24
    generations: int = 15
    elite: int = 3                   # ilu najlepszych przechodzi bez zmian
    tournament: int = 3
    crossover_p: float = 0.6
    mutation_rate: float = 1.2       # >1 => czasem 2 operatory
    max_hidden_layers: int = 4
    seed: int = 0
    patience: int = 6                # gen. bez poprawy -> stop


@dataclass
class Individual:
    genome: Genome
    fitness: float = float("-nan")


@dataclass
class GAResult:
    n_total: int
    best: Individual
    history: List[float] = field(default_factory=list)  # best fitness / generację
    evaluated: int = 0


def _tournament_select(pop: List[Individual], k: int, rng: random.Random) -> Individual:
    return max(rng.sample(pop, k), key=lambda ind: ind.fitness)


def run_ga(fitness: FitnessFn, cfg: GAConfig,
           log: Optional[Callable[[str], None]] = None) -> GAResult:
    rng = random.Random(cfg.seed)
    log = log or (lambda s: None)
    cache: Dict[str, float] = {}
    evaluated = 0

    def evaluate(g: Genome) -> float:
        nonlocal evaluated
        key = g.key()
        if key not in cache:
            cache[key] = fitness(g)
            evaluated += 1
        return cache[key]

    # populacja startowa
    pop: List[Individual] = []
    tries = 0
    while len(pop) < cfg.pop_size and tries < cfg.pop_size * 20:
        tries += 1
        g = random_genome(cfg.n_total, rng, cfg.max_hidden_layers)
        if g.is_valid():
            pop.append(Individual(g, evaluate(g)))
    if not pop:
        raise RuntimeError(f"nie udało się zbudować populacji dla N={cfg.n_total}")

    pop.sort(key=lambda i: i.fitness, reverse=True)
    best = pop[0]
    history = [best.fitness]
    log(f"[N={cfg.n_total}] gen 0  best={best.fitness:.4f}  {best.genome.layer_sizes()}")

    since = 0
    for gen in range(1, cfg.generations + 1):
        nxt: List[Individual] = pop[: cfg.elite]  # elityzm
        while len(nxt) < cfg.pop_size:
            if rng.random() < cfg.crossover_p and len(pop) >= 2:
                pa = _tournament_select(pop, cfg.tournament, rng)
                pb = _tournament_select(pop, cfg.tournament, rng)
                child = crossover(pa.genome, pb.genome, rng)
            else:
                pa = _tournament_select(pop, cfg.tournament, rng)
                child = pa.genome
            child = mutate(child, rng, cfg.mutation_rate)
            if not child.is_valid():
                continue
            nxt.append(Individual(child, evaluate(child)))

        nxt.sort(key=lambda i: i.fitness, reverse=True)
        pop = nxt
        if pop[0].fitness > best.fitness + 1e-9:
            best = pop[0]
            since = 0
        else:
            since += 1
        history.append(best.fitness)
        log(f"[N={cfg.n_total}] gen {gen}  best={best.fitness:.4f}  "
            f"gen_best={pop[0].fitness:.4f}  {best.genome.layer_sizes()}  eval={evaluated}")
        if since >= cfg.patience:
            log(f"[N={cfg.n_total}] early stop (brak poprawy przez {cfg.patience} gen.)")
            break

    return GAResult(cfg.n_total, best, history, evaluated)
