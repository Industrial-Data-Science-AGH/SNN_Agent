#!/usr/bin/env python3
"""
ga.py — silnik algorytmu genetycznego (single-objective, jeden run = jedno N).

Fitness jest callable(Genome[, budget]) -> float (więcej = lepiej). Opcjonalny
argument `budget` (0..1) skaluje koszt oceny (liczbę epok) — używany przez
successive-halving przy inicjalizacji populacji. Fitness bez tego argumentu
(np. synth) jest wywoływany po staremu.

Higiena (#4): fitness nigdy nie jest NaN/inf — niepoprawne wartości są sprowadzane
do -inf, żeby nie psuły sortowania i selekcji.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from genome import Genome, crossover, mutate, random_genome

FitnessFn = Callable[..., float]
NEG_INF = float("-inf")


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
    # successive-halving na starcie: oceń screen_mult*pop losowych osobników
    # tanim budżetem, zatrzymaj najlepsze pop_size, dopiero je oceń pełnym.
    screen_mult: int = 1             # 1 = wyłączone
    screen_budget: float = 0.34      # ułamek pełnego budżetu na screening


@dataclass
class Individual:
    genome: Genome
    fitness: float = NEG_INF         # #4: nigdy NaN


@dataclass
class GAResult:
    n_total: int
    best: Individual
    history: List[float] = field(default_factory=list)
    evaluated: int = 0


def _finite(x) -> float:
    """Sprowadź NaN/inf/None do -inf (higiena selekcji)."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return NEG_INF
    return x if math.isfinite(x) else NEG_INF


def _tournament_select(pop: List[Individual], k: int, rng: random.Random) -> Individual:
    return max(rng.sample(pop, k), key=lambda ind: ind.fitness)


def run_ga(fitness: FitnessFn, cfg: GAConfig,
           log: Optional[Callable[[str], None]] = None) -> GAResult:
    rng = random.Random(cfg.seed)
    log = log or (lambda s: None)
    cache: Dict[str, float] = {}
    evaluated = 0

    def _call(g: Genome, budget: float) -> float:
        # wołaj fitness(g, budget) jeśli obsługuje budżet, inaczej fitness(g)
        try:
            return _finite(fitness(g, budget))
        except TypeError:
            return _finite(fitness(g))

    def evaluate(g: Genome, budget: float = 1.0) -> float:
        nonlocal evaluated
        key = f"{g.key()}@{budget:.2f}"
        if key not in cache:
            cache[key] = _call(g, budget)
            evaluated += 1
        return cache[key]

    def _rand_valid() -> Genome:
        for _ in range(50):
            g = random_genome(cfg.n_total, rng, cfg.max_hidden_layers)
            if g.is_valid():
                return g
        raise RuntimeError(f"nie udało się zbudować genomu dla N={cfg.n_total}")

    # populacja startowa (z opcjonalnym screeningiem)
    if cfg.screen_mult > 1:
        pool = [_rand_valid() for _ in range(cfg.screen_mult * cfg.pop_size)]
        scored = sorted(pool, key=lambda g: evaluate(g, cfg.screen_budget), reverse=True)
        survivors = scored[: cfg.pop_size]
        log(f"[N={cfg.n_total}] screening {len(pool)} osobników @budżet "
            f"{cfg.screen_budget:.2f} -> zostaje {len(survivors)}")
        pop = [Individual(g, evaluate(g, 1.0)) for g in survivors]
    else:
        pop = [Individual(g := _rand_valid(), evaluate(g, 1.0))
               for _ in range(cfg.pop_size)]

    pop.sort(key=lambda i: i.fitness, reverse=True)
    best = pop[0]
    history = [best.fitness]
    log(f"[N={cfg.n_total}] gen 0  best={best.fitness:.4f}  {best.genome.layer_sizes()}")

    since = 0
    for gen in range(1, cfg.generations + 1):
        nxt: List[Individual] = pop[: cfg.elite]
        while len(nxt) < cfg.pop_size:
            if rng.random() < cfg.crossover_p and len(pop) >= 2:
                pa = _tournament_select(pop, cfg.tournament, rng)
                pb = _tournament_select(pop, cfg.tournament, rng)
                child = crossover(pa.genome, pb.genome, rng)
            else:
                child = _tournament_select(pop, cfg.tournament, rng).genome
            child = mutate(child, rng, cfg.mutation_rate)
            if not child.is_valid():
                continue
            nxt.append(Individual(child, evaluate(child, 1.0)))

        nxt.sort(key=lambda i: i.fitness, reverse=True)
        pop = nxt
        if pop[0].fitness > best.fitness + 1e-9:
            best, since = pop[0], 0
        else:
            since += 1
        history.append(best.fitness)
        log(f"[N={cfg.n_total}] gen {gen}  best={best.fitness:.4f}  "
            f"gen_best={pop[0].fitness:.4f}  {best.genome.layer_sizes()}  eval={evaluated}")
        if since >= cfg.patience:
            log(f"[N={cfg.n_total}] early stop (brak poprawy przez {cfg.patience} gen.)")
            break

    return GAResult(cfg.n_total, best, history, evaluated)
