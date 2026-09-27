#!/usr/bin/env python3
"""
ga.py — silnik algorytmu genetycznego (single-objective, jeden run = jedno N).

Fitness jest callable(Genome[, budget]) -> float (więcej = lepiej). Opcjonalny
argument `budget` (0..1) skaluje koszt oceny (liczbę epok) — używany przez
successive-halving przy inicjalizacji populacji. Fitness bez tego argumentu
(np. synth) jest wywoływany po staremu.

Higiena (#4): fitness nigdy nie jest NaN/inf — niepoprawne wartości są sprowadzane
do -inf, żeby nie psuły sortowania i selekcji.

CHECKPOINT/RESUME (M0 krok 4, dodane 25.09.2026 -- Marcel):
run_ga przyjmuje opcjonalny `checkpoint_path`. Gdy podany:
  - PRZED startem: jeśli plik istnieje, wznawiamy z jego stanu zamiast budować
    nową populację od zera.
  - PO KAŻDEJ zakończonej generacji (i po zbudowaniu populacji startowej,
    czyli "generacji 0"): zapisujemy stan atomowo (plik tymczasowy + os.replace).
  - PO ZAKOŃCZENIU (koniec generations albo early stop): checkpoint jest
    usuwany -- zakończony eksperyment nie musi być już wznawialny.

Kluczowa własność: checkpointujemy WYŁĄCZNIE na granicy generacji, nigdy w
jej trakcie. Jeśli proces padnie w połowie generacji G (np. w trakcie
ewaluacji dzieci), wszystkie losowania rng wykonane w tej generacji (tournament,
crossover, mutate) przepadają razem z nią -- ale stan rng zapisany w
checkpointcie to dokładnie stan SPRZED tych losowań (koniec generacji G-1).
Wznowienie odtwarza więc generację G identycznie jak w bezprzerwowym biegu
(te same wywołania rng w tej samej kolejności) -> wynik końcowy (best,
history, evaluated) jest identyczny bit-w-bit między biegiem przerwanym i
nieprzerwanym, o ile fitness jest deterministyczny względem genomu (patrz
test w smoke_test_ga_resume.py).

Co jest w checkpointcie: cfg (do walidacji zgodności), numer ostatniej
ukończonej generacji, populacja (genomy + fitness), best, history, evaluated,
since (licznik cierpliwości), cache fitnessu (tylko wpisy dokończone -- nigdy
"w locie"/None) i pełny stan random.Random (getstate/setstate). Format to
JSON -- dla czytelności i przenośności między maszynami (Linux dev -> Mac
M5 Max), kosztem nieco większego pliku niż pickle.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field
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


# ============================================================ checkpoint/resume

def _rng_state_to_json(rng: random.Random) -> list:
    version, internal_state, gauss_next = rng.getstate()
    return [version, list(internal_state), gauss_next]


def _rng_state_from_json(state: list) -> tuple:
    version, internal_state, gauss_next = state
    return (version, tuple(internal_state), gauss_next)


def _individual_to_json(ind: Individual) -> dict:
    return {"layers": ind.genome.layers, "fitness": ind.fitness}


def _individual_from_json(d: dict) -> Individual:
    return Individual(Genome.from_dict({"layers": d["layers"]}), d["fitness"])


def _save_checkpoint(path: str, *, cfg: GAConfig, gen: int, pop: List[Individual],
                      best: Individual, history: List[float], evaluated: int,
                      since: int, cache: Dict[str, float], rng: random.Random) -> None:
    """Zapis atomowy: piszemy do pliku tymczasowego, potem os.replace (rename
    jest atomowy na tym samym systemie plików) -- proces przerwany W TRAKCIE
    zapisu zostawia poprzedni, kompletny checkpoint nietknięty, nigdy pół-plik."""
    data = {
        "format_version": 1,
        "cfg": asdict(cfg),
        "gen": gen,  # numer OSTATNIEJ w pełni ukończonej generacji (0 = tylko init)
        "pop": [_individual_to_json(ind) for ind in pop],
        "best": _individual_to_json(best),
        "history": list(history),
        "evaluated": evaluated,
        "since": since,
        # tylko dokończone wpisy -- "w locie" (None) nie mogą tu wejść, bo
        # zapisujemy wyłącznie na granicy generacji, gdy _eval_many już wróciło
        "cache": dict(cache),
        "rng_state": _rng_state_to_json(rng),
    }
    tmp_path = f"{path}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _load_checkpoint(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint_summary(path: str) -> dict:
    """Pomocnicze dla raportu M0 -- podgląd checkpointu bez wznawiania GA."""
    data = _load_checkpoint(path)
    return {
        "gen": data["gen"],
        "best_fitness": data["best"]["fitness"],
        "evaluated": data["evaluated"],
        "since": data["since"],
        "pop_size_w_checkpointcie": len(data["pop"]),
    }


def run_ga(fitness: FitnessFn, cfg: GAConfig,
           log: Optional[Callable[[str], None]] = None,
           checkpoint_path: Optional[str] = None) -> GAResult:
    log = log or (lambda s: None)

    def _call(g: Genome, budget: float) -> float:
        # wołaj fitness(g, budget) jeśli obsługuje budżet, inaczej fitness(g)
        try:
            return _finite(fitness(g, budget))
        except TypeError:
            return _finite(fitness(g))

    resumed = checkpoint_path is not None and os.path.exists(checkpoint_path)

    if resumed:
        data = _load_checkpoint(checkpoint_path)
        on_disk_cfg = data["cfg"]
        this_cfg = asdict(cfg)
        if on_disk_cfg != this_cfg:
            diff = {k: (on_disk_cfg.get(k), this_cfg.get(k))
                    for k in this_cfg if on_disk_cfg.get(k) != this_cfg.get(k)}
            raise ValueError(
                f"[BŁĄD] Checkpoint {checkpoint_path} nie pasuje do podanego GAConfig "
                f"(inny eksperyment?) -- różnice (on_disk, podane): {diff}"
            )
        rng = random.Random()
        rng.setstate(_rng_state_from_json(data["rng_state"]))
        cache: Dict[str, float] = dict(data["cache"])
        evaluated = data["evaluated"]
        pop = [_individual_from_json(d) for d in data["pop"]]
        best = _individual_from_json(data["best"])
        history = list(data["history"])
        since = data["since"]
        start_gen = data["gen"] + 1
        log(f"[N={cfg.n_total}] WZNOWIONO z checkpointu {checkpoint_path}: "
            f"ostatnia ukończona gen={data['gen']}, best={best.fitness:.4f}, "
            f"evaluated={evaluated} -- kontynuacja od gen {start_gen}")
    else:
        rng = random.Random(cfg.seed)
        cache = {}
        evaluated = 0
        since = 0
        start_gen = 1

    def _eval_many(genomes: List[Genome], budget: float) -> List[float]:
        """Oceń partię genomów jednym wywołaniem fitness.batch (gdy dostępne).

        Dedup przez cache topologii `key@budget`. Wewnątrz partii powtórzony
        genom jest oznaczany jako "w trakcie" (None) i nie jest liczony 2x.
        Kolejność wyników = kolejność wejścia — GA pozostaje deterministyczny:
        te same genomy -> te same fitnessy co w wersji sekwencyjnej (szczegóły w README).
        """
        nonlocal evaluated
        keys = [f"{g.key()}@{budget:.2f}" for g in genomes]
        fresh = []
        # POPRAWKA 1: Natychmiastowe oznaczanie w cache, co naprawia dedup wewnątrz partii
        for key, g in zip(keys, genomes):
            if key not in cache:
                cache[key] = None  # in-flight (dedup natychmiastowy)
                fresh.append((key, g))

        if fresh:
            try:
                if hasattr(fitness, "batch"):
                    vals = fitness.batch([g for _, g in fresh], budget)
                else:
                    vals = [_call(g, budget) for _, g in fresh]

                for (key, _), v in zip(fresh, vals):
                    cache[key], evaluated = v, evaluated + 1

            except Exception:
                # POPRAWKA 2: Czyszczenie wpisów in-flight (None) z cache w razie błędu ewaluacji,
                # aby uniknąć TypeError przy późniejszym sortowaniu.
                for key, _ in fresh:
                    if cache.get(key) is None:
                        del cache[key]
                raise

        return [cache[k] for k in keys]

    def evaluate(g: Genome, budget: float = 1.0) -> float:
        return _eval_many([g], budget)[0]

    def _rand_valid() -> Genome:
        for _ in range(50):
            g = random_genome(cfg.n_total, rng, cfg.max_hidden_layers)
            if g.is_valid():
                return g
        raise RuntimeError(f"nie udało się zbudować genomu dla N={cfg.n_total}")

    if not resumed:
        # populacja startowa (z opcjonalnym screeningiem)
        if cfg.screen_mult > 1:
            pool = [_rand_valid() for _ in range(cfg.screen_mult * cfg.pop_size)]
            sc_fits = _eval_many(pool, cfg.screen_budget)
            scored = sorted(zip(pool, sc_fits), key=lambda x: x[1], reverse=True)
            survivors = [g for g, _ in scored[: cfg.pop_size]]
            log(f"[N={cfg.n_total}] screening {len(pool)} osobników @budżet "
                f"{cfg.screen_budget:.2f} -> zostaje {len(survivors)}")
            pop = [Individual(g, f) for g, f in zip(survivors, _eval_many(survivors, 1.0))]
        else:
            init = [_rand_valid() for _ in range(cfg.pop_size)]
            pop = [Individual(g, f) for g, f in zip(init, _eval_many(init, 1.0))]

        pop.sort(key=lambda i: i.fitness, reverse=True)
        best = pop[0]
        history = [best.fitness]
        log(f"[N={cfg.n_total}] gen 0  best={best.fitness:.4f}  {best.genome.layer_sizes()}")

        if checkpoint_path:
            _save_checkpoint(checkpoint_path, cfg=cfg, gen=0, pop=pop, best=best,
                              history=history, evaluated=evaluated, since=since, cache=cache, rng=rng)

    for gen in range(start_gen, cfg.generations + 1):
        children: List[Genome] = []
        while len(children) < cfg.pop_size - cfg.elite:
            if rng.random() < cfg.crossover_p and len(pop) >= 2:
                pa = _tournament_select(pop, cfg.tournament, rng)
                pb = _tournament_select(pop, cfg.tournament, rng)
                child = crossover(pa.genome, pb.genome, rng)
            else:
                child = _tournament_select(pop, cfg.tournament, rng).genome
            child = mutate(child, rng, cfg.mutation_rate)
            if not child.is_valid():
                continue
            children.append(child)
        nxt: List[Individual] = pop[: cfg.elite]
        nxt += [Individual(g, f) for g, f in zip(children, _eval_many(children, 1.0))]

        nxt.sort(key=lambda i: i.fitness, reverse=True)
        pop = nxt
        if pop[0].fitness > best.fitness + 1e-9:
            best, since = pop[0], 0
        else:
            since += 1
        history.append(best.fitness)
        log(f"[N={cfg.n_total}] gen {gen}  best={best.fitness:.4f}  "
            f"gen_best={pop[0].fitness:.4f}  {best.genome.layer_sizes()}  eval={evaluated}")

        if checkpoint_path:
            _save_checkpoint(checkpoint_path, cfg=cfg, gen=gen, pop=pop, best=best,
                              history=history, evaluated=evaluated, since=since, cache=cache, rng=rng)

        if since >= cfg.patience:
            log(f"[N={cfg.n_total}] early stop (brak poprawy przez {cfg.patience} gen.)")
            break

    if checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)  # eksperyment ukończony -- nic już nie ma do wznawiania

    return GAResult(cfg.n_total, best, history, evaluated)