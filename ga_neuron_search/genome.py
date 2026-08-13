#!/usr/bin/env python3
"""
genome.py — reprezentacja topologii SNN dla algorytmu genetycznego.

Genom opisuje POŁĄCZENIA sieci Lu.i (nie wagi — te uczy backprop w fitnessie).
Model jest warstwowy, ale liczba warstw JEST częścią genomu i podlega ewolucji.

Warstwy:
    - warstwa 0  = cechy encodera (źródła), rozmiar = N_FEATURES, nie liczą się do N.
    - warstwy 1..L-1 = ukryte neurony Lu.i.
    - warstwa L (ostatnia) = 1 neuron decyzyjny, dokładnie 3 wejścia.

Twarde ograniczenia sprzętowe płytki Lu.i (pilnowane przy każdej operacji):
    * fan-in  <= 3  — neuron ma 3 styki wejściowe,
    * fan-out <= 3  — neuron/cecha zasila <= 3 kolejne styki,
    * DAG            — połączenia tylko warstwa L -> L+1 (brak cykli z definicji),
    * suma neuronów ukrytych + decyzyjny == N (stałe na jeden run GA),
    * neuron decyzyjny ma dokładnie 3 wejścia,
    * (miękko) każda cecha wykorzystana >= 1 raz.

Nazwy węzłów warstwy L numerujemy 0..n_L-1; wejście neuronu to indeks w warstwie L-1.

Ten plik NIE importuje torcha — jest czystym opisem struktury, dzięki czemu GA
i testy działają bez zależności ML.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from typing import List, Optional

# Kanały encodera (musi zgadzać się z CHANNELS w snn_hw_pipeline.py)
N_FEATURES = 7
FEATURE_NAMES = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]

MAX_FANIN = 3
MAX_FANOUT = 3
OUT_FANIN = 3  # neuron decyzyjny ma dokładnie tyle wejść


@dataclass
class Genome:
    """Topologia warstwowa.

    layers[k] to lista neuronów warstwy k+1 (warstwa 0 = cechy jest domyślna).
    Każdy neuron = lista indeksów wejść w warstwie POPRZEDNIEJ, długość 1..3.

    Przykład (odpowiednik obecnego wariantu wąskiego 7->4->3->1):
        layers = [
            [[0,1,5],[2,3,6],[0,4,5],[1,4,6]],   # H: 4 neurony, wejścia = cechy
            [[0,1,2],[1,2,3],[2,3,0]],           # G: 3 neurony, wejścia = H
            [[0,1,2]],                            # D: 1 neuron, wejścia = G
        ]
    """
    layers: List[List[List[int]]] = field(default_factory=list)

    # ---------------------------------------------------------------- rozmiary
    def layer_sizes(self) -> List[int]:
        """Rozmiary warstw łącznie z warstwą cech na początku."""
        return [N_FEATURES] + [len(l) for l in self.layers]

    def n_hidden_layers(self) -> int:
        return max(0, len(self.layers) - 1)  # ostatnia to decyzyjna

    def total_neurons(self) -> int:
        """Wszystkie neurony Lu.i (ukryte + decyzyjny). Bez cech."""
        return sum(len(l) for l in self.layers)

    # ---------------------------------------------------------------- fan-out
    def fanout_counts(self, k: int) -> List[int]:
        """Ile razy każdy węzeł warstwy k jest używany jako wejście w warstwie k+1.

        k=0 -> cechy; k>=1 -> neurony warstwy layers[k-1].
        """
        size = self.layer_sizes()[k]
        counts = [0] * size
        if k >= len(self.layers):
            return counts
        for neuron in self.layers[k]:
            for src in neuron:
                if 0 <= src < size:
                    counts[src] += 1
        return counts

    # ---------------------------------------------------------------- walidacja
    def violations(self) -> List[str]:
        """Lista naruszeń ograniczeń. Pusta == genom poprawny sprzętowo."""
        errs: List[str] = []
        if not self.layers:
            return ["brak warstw"]

        sizes = self.layer_sizes()
        for k, layer in enumerate(self.layers):
            prev = sizes[k]  # rozmiar warstwy poprzedniej
            for i, neuron in enumerate(layer):
                if not (1 <= len(neuron) <= MAX_FANIN):
                    errs.append(f"L{k+1}n{i}: fan-in {len(neuron)} spoza [1,{MAX_FANIN}]")
                if len(set(neuron)) != len(neuron):
                    errs.append(f"L{k+1}n{i}: zduplikowane wejście {neuron}")
                for src in neuron:
                    if not (0 <= src < prev):
                        errs.append(f"L{k+1}n{i}: wejście {src} poza zakresem [0,{prev})")

        # fan-out każdej warstwy (cechy + ukryte, bez ostatniej bo nie ma następnej)
        for k in range(len(self.layers)):
            for node, c in enumerate(self.fanout_counts(k)):
                if c > MAX_FANOUT:
                    errs.append(f"L{k} węzeł {node}: fan-out {c} > {MAX_FANOUT}")

        # neuron decyzyjny — fan-in = min(3, rozmiar ostatniej warstwy ukrytej),
        # bo bez zdublowanych wejść nie da się wpiąć 3 styków z <3 neuronów
        want = min(OUT_FANIN, sizes[-2]) if len(sizes) >= 2 else OUT_FANIN
        if len(self.layers[-1]) != 1:
            errs.append(f"ostatnia warstwa ma {len(self.layers[-1])} neuronów, oczekiwano 1")
        elif len(self.layers[-1][0]) != want:
            errs.append(f"neuron decyzyjny: fan-in {len(self.layers[-1][0])} != {want}")

        return errs

    def is_valid(self) -> bool:
        return not self.violations()

    def features_used(self) -> set:
        used = set()
        if self.layers:
            for neuron in self.layers[0]:
                used.update(neuron)
        return used

    # ---------------------------------------------------------------- hash / io
    def key(self) -> str:
        """Stabilny hash topologii — klucz cache fitnessu."""
        blob = json.dumps(self.layers, separators=(",", ":"))
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        return {"layers": self.layers,
                "total_neurons": self.total_neurons(),
                "hidden_layers": self.n_hidden_layers(),
                "layer_sizes": self.layer_sizes()}

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        return cls(layers=[[list(n) for n in layer] for layer in d["layers"]])

    def copy(self) -> "Genome":
        return Genome(layers=[[list(n) for n in layer] for layer in self.layers])

    def pretty(self) -> str:
        s = " -> ".join(str(x) for x in self.layer_sizes())
        lines = [f"topologia {s}  (N={self.total_neurons()}, warstw ukrytych={self.n_hidden_layers()})"]
        for k, layer in enumerate(self.layers):
            tag = "D" if k == len(self.layers) - 1 else f"H{k+1}"
            lines.append(f"  {tag}: " + ", ".join(str(n) for n in layer))
        return "\n".join(lines)


# ==================================================================== budowanie

def sizes_feasible(hidden_sizes: List[int]) -> bool:
    """Czy ciąg rozmiarów warstw ukrytych da się okablować przy fan-out<=3.

    Warstwa o rozmiarze s ma 3*s styków wyjściowych, a każdy neuron następnej
    warstwy potrzebuje >=1 wejścia => n_{k+1} <= 3*n_k (n_0 = N_FEATURES).
    """
    prev = N_FEATURES
    for s in hidden_sizes:
        if s < 1 or s > MAX_FANOUT * prev:
            return False
        prev = s
    return True


def _gen_hidden_sizes(n_hidden: int, parts: int, rng: random.Random) -> List[int]:
    """Losowy podział n_hidden neuronów na `parts` warstw z zachowaniem
    n_{k+1} <= 3*n_k. Zwraca listę rozmiarów (może mieć < parts warstw, gdy
    feasibility na to nie pozwala)."""
    parts = max(1, min(parts, n_hidden))
    for _ in range(200):
        # losowe cięcia, potem walidacja feasibility
        if parts == 1:
            sizes = [n_hidden]
        else:
            cuts = sorted(rng.sample(range(1, n_hidden), parts - 1)) if n_hidden > parts \
                else list(range(1, n_hidden))
            sizes, prev = [], 0
            for c in cuts:
                sizes.append(c - prev); prev = c
            sizes.append(n_hidden - prev)
        if sizes_feasible(sizes):
            return sizes
    # awaryjnie: jedna warstwa (zawsze feasible dla N<=21)
    return [n_hidden]


def _wire_neuron(prev_size: int, fanout_left: List[int], rng: random.Random,
                 exact: Optional[int] = None) -> List[int]:
    """Wylosuj 1..3 (lub `exact`) wejść z węzłów, którym został budżet fan-out."""
    k = exact if exact is not None else rng.randint(1, min(MAX_FANIN, prev_size))
    k = min(k, prev_size)
    avail = [i for i in range(prev_size) if fanout_left[i] > 0]
    if len(avail) < k:  # zabrakło budżetu — dobierz z pełnych (repair i tak poprawi)
        avail = list(range(prev_size))
    pick = rng.sample(avail, k)
    for p in pick:
        fanout_left[p] -= 1
    return sorted(pick)


def random_genome(n_total: int, rng: random.Random,
                  max_hidden_layers: int = 3) -> Genome:
    """Losowy poprawny genom o dokładnie n_total neuronach (ukryte + decyzyjny).

    Ostatni neuron to decyzyjny (fan-in=3), więc ukrytych jest n_total-1.
    """
    assert n_total >= 2, "potrzeba >=2 neuronów (min. 1 ukryty + decyzyjny)"
    n_hidden = n_total - 1

    # ile warstw ukrytych — ograniczone liczbą neuronów
    max_L = max(1, min(max_hidden_layers, n_hidden))
    n_layers = rng.randint(1, max_L)
    sizes = _gen_hidden_sizes(n_hidden, n_layers, rng)  # rozmiary warstw ukrytych

    layers: List[List[List[int]]] = []
    prev_size = N_FEATURES
    for li, size in enumerate(sizes):
        fanout_left = [MAX_FANOUT] * prev_size
        layer = [_wire_neuron(prev_size, fanout_left, rng) for _ in range(size)]
        layers.append(layer)
        prev_size = size

    # warstwa decyzyjna: 1 neuron, dokładnie 3 wejścia z ostatniej warstwy ukrytej
    fanout_left = [MAX_FANOUT] * prev_size
    dec = _wire_neuron(prev_size, fanout_left, rng, exact=min(OUT_FANIN, prev_size))
    layers.append([dec])

    g = Genome(layers=layers)
    repair(g, rng)
    return g


# ==================================================================== repair

def repair(g: Genome, rng: random.Random) -> Genome:
    """Napraw genom w miejscu tak, by spełniał wszystkie twarde ograniczenia.

    Kolejność: zakresy wejść -> fan-in -> fan-out -> warstwa decyzyjna ->
    pokrycie cech. Repair jest idempotentny na poprawnym genomie.
    """
    if not g.layers:
        g.layers = [[[0, 1, 2]]]  # awaryjny minimalny genom (wymaga potem sanacji N)

    # 0) feasibility rozmiarów: n_{k+1} <= 3*n_k. Gdy operator strukturalny
    #    (split/move) zepsuł proporcje, spłaszcz warstwy ukryte do jednej —
    #    zachowuje liczbę neuronów N, tylko traci głębokość.
    hidden_sizes = [len(l) for l in g.layers[:-1]]
    if hidden_sizes and not sizes_feasible(hidden_sizes):
        flat = [n for layer in g.layers[:-1] for n in layer]
        g.layers = [flat, g.layers[-1]]

    sizes = g.layer_sizes()

    # 1) wejścia w zakresie + brak duplikatów + fan-in w [1,3]
    for k, layer in enumerate(g.layers):
        prev = sizes[k]
        for neuron in layer:
            fixed = [s for s in dict.fromkeys(neuron) if 0 <= s < prev]
            if not fixed:
                fixed = [rng.randrange(prev)]
            if len(fixed) > MAX_FANIN:
                fixed = rng.sample(fixed, MAX_FANIN)
            neuron[:] = sorted(fixed)

    # 2) fan-out <= 3 dla każdej warstwy źródłowej
    for k in range(len(g.layers)):
        _enforce_fanout(g, k, rng)

    # 3) warstwa decyzyjna: dokładnie 1 neuron z 3 wejściami
    _fix_decision(g, rng)

    # 4) pokrycie cech (miękkie): wepnij nieużytą cechę zamiast losowej w L1
    _cover_features(g, rng)

    return g


def _enforce_fanout(g: Genome, k: int, rng: random.Random) -> None:
    if k >= len(g.layers):
        return
    prev_size = g.layer_sizes()[k]
    counts = g.fanout_counts(k)
    over = [i for i, c in enumerate(counts) if c > MAX_FANOUT]
    if not over:
        return
    # dla przeciążonych węzłów przepnij nadmiarowe użycia na węzły z zapasem
    for src in over:
        # zbierz (neuron) używające src
        users = [neuron for neuron in g.layers[k] if src in neuron]
        while counts[src] > MAX_FANOUT and users:
            neuron = users.pop()
            spare = [j for j in range(prev_size)
                     if counts[j] < MAX_FANOUT and j not in neuron]
            if not spare:
                # brak miejsca — po prostu usuń wejście (fan-in >=1 utrzymany)
                if len(neuron) > 1:
                    neuron.remove(src)
                    counts[src] -= 1
                continue
            tgt = rng.choice(spare)
            neuron[neuron.index(src)] = tgt
            neuron.sort()
            counts[src] -= 1
            counts[tgt] += 1


def _fix_decision(g: Genome, rng: random.Random) -> None:
    prev_size = g.layer_sizes()[-2] if len(g.layers) >= 1 else N_FEATURES
    # scal ewentualne nadmiarowe neurony ostatniej warstwy do jednego
    if len(g.layers[-1]) != 1:
        g.layers[-1] = [g.layers[-1][0] if g.layers[-1] else [0]]
    dec = g.layers[-1][0]
    k = min(OUT_FANIN, prev_size)
    dec = [s for s in dict.fromkeys(dec) if 0 <= s < prev_size]
    while len(dec) < k:
        cand = [i for i in range(prev_size) if i not in dec]
        dec.append(rng.choice(cand))
    if len(dec) > k:
        dec = rng.sample(dec, k)
    g.layers[-1][0] = sorted(dec)


def _cover_features(g: Genome, rng: random.Random) -> None:
    if not g.layers:
        return
    used = g.features_used()
    missing = [f for f in range(N_FEATURES) if f not in used]
    if not missing:
        return
    counts = g.fanout_counts(0)
    for f in missing:
        # znajdź neuron w L1, gdzie można podmienić wejście o wysokim fan-out
        for neuron in g.layers[0]:
            swappable = [s for s in neuron if counts[s] > 1]
            if swappable and counts[f] < MAX_FANOUT:
                old = swappable[0]
                neuron[neuron.index(old)] = f
                neuron.sort()
                counts[old] -= 1
                counts[f] += 1
                break


# ==================================================================== mutacje

def mut_rewire(g: Genome, rng: random.Random) -> None:
    """Zmień jedno wejście losowego neuronu."""
    k = rng.randrange(len(g.layers))
    prev = g.layer_sizes()[k]
    neuron = rng.choice(g.layers[k])
    idx = rng.randrange(len(neuron))
    cand = [i for i in range(prev) if i not in neuron]
    if cand:
        neuron[idx] = rng.choice(cand)
        neuron.sort()


def mut_fanin(g: Genome, rng: random.Random) -> None:
    """Dodaj lub usuń jedno wejście neuronu ukrytego (fan-in w [1,3])."""
    if len(g.layers) < 2:
        return
    k = rng.randrange(len(g.layers) - 1)  # nie ruszamy decyzyjnego (stały fan-in)
    prev = g.layer_sizes()[k]
    neuron = rng.choice(g.layers[k])
    if len(neuron) < MAX_FANIN and rng.random() < 0.5:
        cand = [i for i in range(prev) if i not in neuron]
        if cand:
            neuron.append(rng.choice(cand))
            neuron.sort()
    elif len(neuron) > 1:
        neuron.pop(rng.randrange(len(neuron)))


def mut_move_neuron(g: Genome, rng: random.Random) -> None:
    """Przenieś neuron między sąsiednimi warstwami ukrytymi (N bez zmian)."""
    hid = len(g.layers) - 1
    if hid < 2:
        return
    src = rng.randrange(hid)
    dst = src + (1 if (src == 0 or rng.random() < 0.5) else -1)
    dst = max(0, min(hid - 1, dst))
    if src == dst or len(g.layers[src]) <= 1:
        return
    neuron = g.layers[src].pop(rng.randrange(len(g.layers[src])))
    # nowe wejścia z warstwy poprzedzającej dst
    prev = g.layer_sizes()[dst]
    k = min(len(neuron), prev)
    g.layers[dst].append(sorted(rng.sample(range(prev), max(1, k))))


def mut_split_layer(g: Genome, rng: random.Random) -> None:
    """Rozbij warstwę ukrytą na dwie (liczba warstw rośnie, N bez zmian)."""
    hid = len(g.layers) - 1
    if hid < 1:
        return
    k = rng.randrange(hid)
    if len(g.layers[k]) < 2:
        return
    layer = g.layers[k]
    cut = rng.randint(1, len(layer) - 1)
    first, second = layer[:cut], layer[cut:]
    g.layers[k] = first
    # neurony `second` przepinamy tak, by brały wejścia z `first`
    prev = len(first)
    rewired = [sorted(rng.sample(range(prev), rng.randint(1, min(MAX_FANIN, prev))))
               for _ in second]
    g.layers.insert(k + 1, rewired)


def mut_merge_layers(g: Genome, rng: random.Random) -> None:
    """Scal dwie sąsiednie warstwy ukryte (liczba warstw maleje, N bez zmian)."""
    hid = len(g.layers) - 1
    if hid < 2:
        return
    k = rng.randrange(hid - 1)
    merged = g.layers[k] + g.layers[k + 1]
    # neurony z drugiej scalanej warstwy brały wejścia z pierwszej -> przepnij
    del g.layers[k + 1]
    g.layers[k] = merged
    prev = g.layer_sizes()[k]
    for neuron in g.layers[k]:
        neuron[:] = [s for s in neuron if s < prev] or [rng.randrange(prev)]


MUTATIONS = [
    (mut_rewire, 0.40),
    (mut_fanin, 0.20),
    (mut_move_neuron, 0.15),
    (mut_split_layer, 0.15),
    (mut_merge_layers, 0.10),
]


def mutate(g: Genome, rng: random.Random, rate: float = 1.0) -> Genome:
    """Zastosuj 1+ operatorów mutacji, potem repair. Zwraca nowy genom."""
    child = g.copy()
    n_ops = 2 if (rate > 1 and rng.random() < (rate - 1.0)) else 1
    for _ in range(n_ops):
        r, acc = rng.random(), 0.0
        for op, p in MUTATIONS:
            acc += p
            if r <= acc:
                op(child, rng)
                break
    repair(child, rng)
    return child


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Struktura (liczba/rozmiary warstw) z rodzica A, okablowanie mieszane z A i B.

    Dla każdego neuronu losowo dziedziczymy wejścia z A albo B (gdy zgodne
    wymiarowo), potem repair. Utrzymuje N, bo N obu rodziców jest równe.
    """
    child = a.copy()
    for k, layer in enumerate(child.layers):
        if k >= len(b.layers):
            continue
        for i, neuron in enumerate(layer):
            if i < len(b.layers[k]) and rng.random() < 0.5:
                prev = child.layer_sizes()[k]
                cand = [s for s in b.layers[k][i] if 0 <= s < prev]
                if cand:
                    neuron[:] = sorted(dict.fromkeys(cand))
    repair(child, rng)
    return child
