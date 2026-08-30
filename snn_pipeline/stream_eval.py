#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
stream_eval.py — WSPÓLNA metryka strumieniowa: recall przy ustalonym budżecie
fałszywych alarmów na godzinę (FA/h), z rozbiciem po `kind` i przedziałem ufności
bootstrapem po `group_id`.

Po co: to jest liczba, która mówi, czy urządzenie jest przydatne (ile przeoczeń
przy dopuszczalnej liczbie fałszywych pobudek/h), a nie ramkowe F1 czy clip_f1.
Dotąd FA/h liczył tylko eval_stream.py — offline, po treningu, i nic tego nie
używało do WYBORU. Ten moduł jest importowany z trzech stron:
  * eval_stream.py            (raport offline),
  * snn_hw_pipeline.py train  (selekcja checkpointu),
  * ga_neuron_search/fitness  (fitness GA),
żeby optymalizowane było to samo, co decyduje o przydatności.

Moduł jest czysto-numpyowy i NIE zależy od modelu: każda strona liczy ciągi
spików neuronu decyzyjnego D po swojemu (LuiNet / GenomeNet), a tu podaje już
gotowe `trains` (lista binarnych wektorów, jeden na klip) + metadane klipów.

Definicje:
  * reguła dekodera (k, w): alarm, gdy padnie >= k spików D w oknie w ramek,
    z refrakcją po alarmie (jak licznik na J4 płytki D). Patrz `count_alarms`.
  * FA/h = fałszywe alarmy na klipach TŁA / godziny tła. Liczymy też per `kind`.
  * recall = ułamek klipów pozytywnych, na których padł >= 1 alarm.
  * recall @ budżet B: najwyższy recall spośród reguł, przy których KAŻDY `kind`
    tła mieści się w budżecie B (nie tylko suma). Gdy żadna reguła się nie mieści
    -> 0.0 (nie da się pracować w budżecie).
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# 1 ramka = DT sekund (spójne z snn_hw_pipeline.DT). Zostawione jako argument,
# ale domyślnie 10 ms, bo tyle trwa ramka enkodera (HOP_SAMPLES / FS_HZ).
DEFAULT_DT = 0.01
DEFAULT_REFRAC = 500          # 5 s martwej strefy po alarmie (budzi reaktor)

# Siatka reguł (k spików, w ramek) do wykreślenia krzywej FA/h<->recall.
# Od najczulszej (1:1) do bardzo ostrych (serie w wąskim oknie) — im ostrzej,
# tym mniej FA/h i mniej recall. `recall_at_fa_budget` wybiera z tego punkt pracy.
DEFAULT_RULES: Tuple[Tuple[int, int], ...] = (
    (1, 1), (1, 50), (1, 100), (1, 250),
    (2, 50), (2, 100), (2, 250), (2, 500),
    (3, 100), (3, 250), (3, 500),
    (4, 250), (4, 500),
)

DEFAULT_BUDGETS: Tuple[float, ...] = (1.0, 6.0)   # FA/h


# ============================================================ dekoder

def count_alarms(s: np.ndarray, k: int, w: int, refrac: int = DEFAULT_REFRAC) -> int:
    """Ile razy reguła 'k spików D w oknie w ramek' odpala (z refrakcją po alarmie).

    Przeniesione 1:1 z eval_stream.py, jedyne źródło tej logiki.
    """
    times = np.where(s)[0]
    alarms, i = 0, 0
    while i + k - 1 < len(times):
        if times[i + k - 1] - times[i] < w:
            alarms += 1
            i = int(np.searchsorted(times, times[i + k - 1] + refrac))
        else:
            i += 1
    return alarms


# ============================================================ metadane klipów

@dataclass
class ClipMeta:
    """Metadane jednego klipu potrzebne do metryki strumieniowej."""
    label: int                 # 1 = pozytyw (szkło), 0 = tło
    kind: str                  # positive | stationary | loud_event | speech | animal
    group_id: str              # jednostka podziału (bootstrap po grupach)
    n_frames: int              # długość klipu w ramkach (do godzin tła)


def load_files_meta(split_dir: str) -> Dict[str, ClipMeta]:
    """Wczytaj files.csv artefaktu spikowego -> {nazwa_csv: ClipMeta(bez n_frames)}.

    files.csv (schemat buildera v2): filepath,label,kind,source,group_id,csv.
    n_frames dolicza się osobno przy czytaniu ciągów spików.
    """
    path = os.path.join(split_dir, "files.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"brak files.csv w {split_dir} — artefakt bez metadanych kind/group_id "
            f"(zbuduj zbiór builderem v2)")
    meta: Dict[str, ClipMeta] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            label = 1 if r.get("label") == "positive" else 0
            meta[r["csv"]] = ClipMeta(label=label, kind=r.get("kind", "?"),
                                      group_id=r.get("group_id", r["csv"]), n_frames=0)
    return meta


# ============================================================ rdzeń metryki

@dataclass
class OperatingPoint:
    budget_fa_h: float
    rule: Optional[Tuple[int, int]]     # (k, w) wybranej reguły; None = nie da się
    recall: float
    fa_h_total: float
    fa_h_by_kind: Dict[str, float] = field(default_factory=dict)
    feasible: bool = True               # czy jakakolwiek reguła zmieściła się w budżecie
    recall_ci: Optional[Tuple[float, float]] = None   # bootstrap po group_id


def _alarms_matrix(trains: Sequence[np.ndarray], rules, refrac) -> np.ndarray:
    """[n_clips, n_rules] liczba alarmów każdego klipu przy każdej regule."""
    out = np.zeros((len(trains), len(rules)), dtype=np.int32)
    for ci, t in enumerate(trains):
        for ri, (k, w) in enumerate(rules):
            out[ci, ri] = count_alarms(t, k, w, refrac)
    return out


def _rates_for_rule(alarms_col, labels, kinds, hours_by_kind, neg_hours):
    """Dla jednej reguły: recall, FA/h total, FA/h per kind."""
    pos = labels == 1
    neg = labels == 0
    recall = float((alarms_col[pos] > 0).mean()) if pos.any() else 0.0
    fa_total = float(alarms_col[neg].sum())
    fa_h_total = fa_total / max(neg_hours, 1e-9)
    fa_h_by_kind = {}
    for kk, hrs in hours_by_kind.items():
        m = neg & (kinds == kk)
        fa_h_by_kind[kk] = float(alarms_col[m].sum()) / max(hrs, 1e-9)
    return recall, fa_h_total, fa_h_by_kind


def _recall_at_budget(alarms, labels, kinds, hours_by_kind, neg_hours, rules,
                      budget) -> OperatingPoint:
    """Najwyższy recall spośród reguł, przy których KAŻDY kind tła <= budget."""
    best = OperatingPoint(budget_fa_h=budget, rule=None, recall=0.0,
                          fa_h_total=0.0, feasible=False)
    for ri, rule in enumerate(rules):
        rec, fa_h_tot, fa_h_kind = _rates_for_rule(
            alarms[:, ri], labels, kinds, hours_by_kind, neg_hours)
        within = fa_h_tot <= budget and all(v <= budget for v in fa_h_kind.values())
        if within and rec >= best.recall:
            best = OperatingPoint(budget_fa_h=budget, rule=rule, recall=rec,
                                  fa_h_total=fa_h_tot, fa_h_by_kind=fa_h_kind,
                                  feasible=True)
    return best


def stream_report(trains: Sequence[np.ndarray], labels, kinds, groups,
                  n_frames=None, dt: float = DEFAULT_DT,
                  rules=DEFAULT_RULES, budgets=DEFAULT_BUDGETS,
                  refrac: int = DEFAULT_REFRAC, n_boot: int = 500,
                  seed: int = 0) -> Dict[float, OperatingPoint]:
    """Pełny raport: dla każdego budżetu FA/h -> punkt pracy (recall, reguła,
    FA/h per kind) + bootstrap CI recall po group_id.

    trains   : lista binarnych ciągów spików D (jeden na klip)
    labels   : [n] 0/1
    kinds    : [n] str (positive/stationary/loud_event/speech/animal)
    groups   : [n] str (group_id — jednostka bootstrapu)
    n_frames : [n] długości klipów; gdy None, brane z len(train)
    """
    labels = np.asarray(labels).astype(int)
    kinds = np.asarray(kinds).astype(object)
    groups = np.asarray(groups).astype(object)
    if n_frames is None:
        n_frames = np.array([len(t) for t in trains], dtype=np.int64)
    else:
        n_frames = np.asarray(n_frames).astype(np.int64)

    neg = labels == 0
    neg_hours = float(n_frames[neg].sum()) * dt / 3600.0
    bg_kinds = sorted(set(kinds[neg].tolist()))
    hours_by_kind = {kk: float(n_frames[neg & (kinds == kk)].sum()) * dt / 3600.0
                     for kk in bg_kinds}

    alarms = _alarms_matrix(trains, rules, refrac)

    rng = np.random.default_rng(seed)
    uniq_groups = np.array(sorted(set(groups.tolist())), dtype=object)
    # mapowanie group -> indeksy klipów (do resamplingu po grupach)
    gidx = {g: np.where(groups == g)[0] for g in uniq_groups}

    out: Dict[float, OperatingPoint] = {}
    for B in budgets:
        op = _recall_at_budget(alarms, labels, kinds, hours_by_kind, neg_hours,
                               rules, B)
        # bootstrap po group_id: losuj grupy ze zwracaniem, przelicz recall@B
        if n_boot > 0 and len(uniq_groups) > 1:
            boot = np.empty(n_boot, dtype=np.float64)
            for b in range(n_boot):
                pick = rng.choice(len(uniq_groups), size=len(uniq_groups), replace=True)
                idx = np.concatenate([gidx[uniq_groups[j]] for j in pick])
                bl, bk, bn = labels[idx], kinds[idx], n_frames[idx]
                bneg = bl == 0
                bneg_h = float(bn[bneg].sum()) * dt / 3600.0
                bh_kind = {kk: float(bn[bneg & (bk == kk)].sum()) * dt / 3600.0
                           for kk in bg_kinds}
                bop = _recall_at_budget(alarms[idx], bl, bk, bh_kind, bneg_h,
                                        rules, B)
                boot[b] = bop.recall
            op.recall_ci = (float(np.percentile(boot, 2.5)),
                            float(np.percentile(boot, 97.5)))
        out[B] = op
    return out


def primary_recall(report: Dict[float, OperatingPoint], budget: float) -> float:
    """Skalar do maksymalizacji przez GA / do selekcji checkpointu."""
    op = report.get(budget)
    return op.recall if op is not None else 0.0


def report_to_dict(report: Dict[float, OperatingPoint]) -> dict:
    """Raport -> serializowalny dict (do zapisu w hw_config.json / wyniki GA).
    To jest ta liczba, której dotąd NIGDZIE nie zapisywano."""
    out = {}
    for B, op in report.items():
        out[f"{B:g}_fa_h"] = {
            "recall": round(op.recall, 4),
            "recall_ci": [round(c, 4) for c in op.recall_ci] if op.recall_ci else None,
            "rule_k_w": list(op.rule) if op.rule else None,
            "fa_h_total": round(op.fa_h_total, 3),
            "fa_h_by_kind": {k: round(v, 3) for k, v in op.fa_h_by_kind.items()},
            "feasible": op.feasible,
        }
    return out


def format_report(report: Dict[float, OperatingPoint]) -> str:
    lines = []
    for B in sorted(report):
        op = report[B]
        ci = f" CI[{op.recall_ci[0]:.3f},{op.recall_ci[1]:.3f}]" if op.recall_ci else ""
        rule = f"k={op.rule[0]} w={op.rule[1]}" if op.rule else "BRAK (niewykonalne)"
        lines.append(f"  @ {B:g} FA/h: recall {op.recall:.3f}{ci}  "
                     f"[{rule}]  FA/h_total {op.fa_h_total:.2f}"
                     f"{'' if op.feasible else '  (budzet nieosiagalny)'}")
        for kk, v in sorted(op.fa_h_by_kind.items()):
            lines.append(f"        {kk:12s} FA/h {v:.2f}")
    return "\n".join(lines)


# ============================================================ self-test

if __name__ == "__main__":
    # syntetyczny sanity: pozytywy strzelają serie, tło rzadko/pojedynczo
    rng = np.random.default_rng(0)
    trains, labels, kinds, groups = [], [], [], []
    T = 200
    for g in range(40):
        # pozytyw: seria 3 spików blisko siebie
        s = np.zeros(T, np.uint8); s[[50, 53, 57]] = 1
        trains.append(s); labels.append(1); kinds.append("positive"); groups.append(f"pos_{g}")
    for g in range(60):
        s = np.zeros(T, np.uint8)
        if rng.random() < 0.3:
            s[rng.integers(0, T)] = 1     # pojedynczy przypadkowy spik tła
        trains.append(s); labels.append(0)
        kinds.append(rng.choice(["stationary", "loud_event", "speech", "animal"]))
        groups.append(f"neg_{g}")
    rep = stream_report(trains, labels, kinds, groups, dt=0.01, n_boot=200)
    print("[self-test stream_eval]")
    print(format_report(rep))
    print(f"primary recall @6 FA/h = {primary_recall(rep, 6.0):.3f}")
