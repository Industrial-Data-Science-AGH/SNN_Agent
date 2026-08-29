#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_dataset.py — sprawdza, czy wersja zbioru jest tym, za co się podaje.

Kończy się kodem != 0, gdy którakolwiek kontrola KRYTYCZNA nie przejdzie.
To jest brama: nie trenujemy na zbiorze, który tego nie przechodzi.

Kontrole krytyczne (BŁĄD):
    K1  schemat manifestu — komplet kolumn, dozwolone wartości
    K2  brakujące pliki
    K3  przeciek: ta sama GRUPA w dwóch splitach
    K4  rozjazd sum kontrolnych (plik zmieniony po zbudowaniu wersji)
    K5  duplikaty treści (ta sama sha256) rozrzucone po różnych splitach
    K6  błędne audio — nieodczytywalne albo poza dozwolonym zakresem
    K7  za mało GRUP pozytywnych w teście

Kontrole ostrzegawcze (OSTRZEŻENIE, nie blokują):
    O1  duplikaty treści wewnątrz jednego splitu
    O2  niezbalansowanie klas poza widełkami
    O3  brak któregoś `kind` w którymś splicie
    O4  nagrania na licencji niekomercyjnej

Użycie:
    python snn_pipeline/validate_dataset.py --version v1.0.0
    python snn_pipeline/validate_dataset.py --version v1.0.0 --quick   # bez sha256
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset_contract import (
    KINDS,
    LABELS,
    MANIFEST_COLUMNS,
    MAX_CHANNELS,
    MAX_DURATION_S,
    MIN_DURATION_S,
    MIN_SAMPLE_RATE,
    NONCOMMERCIAL_LICENSES,
    SPLITS,
    is_valid_version,
    sha256_of,
    version_dir,
)

MIN_POSITIVE_GROUPS_TEST = 12
BALANCE_RANGE = (0.01, 0.60)     # dopuszczalny udział pozytywów w splicie


class Report:
    def __init__(self) -> None:
        self.errors: List[str] = []
        self.warns: List[str] = []

    def error(self, code: str, msg: str) -> None:
        self.errors.append(f"[{code}] {msg}")
        print(f"  BŁĄD       [{code}] {msg}")

    def warn(self, code: str, msg: str) -> None:
        self.warns.append(f"[{code}] {msg}")
        print(f"  OSTRZEŻENIE [{code}] {msg}")

    def ok(self, code: str, msg: str) -> None:
        print(f"  ok         [{code}] {msg}")


def load_manifest(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--version", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="pomiń liczenie sha256 i czytanie nagłówków audio")
    ap.add_argument("--min-positive-groups-test", type=int, default=MIN_POSITIVE_GROUPS_TEST)
    args = ap.parse_args()

    if not is_valid_version(args.version):
        sys.exit(f"[BŁĄD] '{args.version}' nie jest wersją semantyczną (vX.Y.Z)")

    repo_root = args.repo_root.resolve()
    vdir = version_dir(repo_root, args.version)
    man_path = vdir / "manifest.csv"
    if not man_path.exists():
        sys.exit(f"[BŁĄD] nie znaleziono {man_path}")

    rows = load_manifest(man_path)
    r = Report()
    print(f"\nWalidacja {args.version} — {len(rows)} rekordów z {man_path}\n")

    # ---------------------------------------------------------------- K1 schemat
    missing_cols = [c for c in MANIFEST_COLUMNS if c not in (rows[0] if rows else {})]
    if missing_cols:
        r.error("K1", f"brak kolumn w manifeście: {missing_cols}")
    else:
        bad = Counter()
        for row in rows:
            if row["label"] not in LABELS:
                bad["label"] += 1
            if row["kind"] not in KINDS:
                bad["kind"] += 1
            if row["split"] not in SPLITS:
                bad["split"] += 1
            if (row["kind"] == "positive") != (row["label"] == "positive"):
                bad["label<->kind"] += 1
        if bad:
            r.error("K1", f"niedozwolone wartości: {dict(bad)}")
        else:
            r.ok("K1", "schemat i wartości w porządku")

    ids = Counter(row["id"] for row in rows)
    dup_ids = [i for i, n in ids.items() if n > 1]
    if dup_ids:
        r.error("K1", f"powtórzone ID: {len(dup_ids)} (np. {dup_ids[:3]})")

    # ---------------------------------------------------------------- K2 pliki
    missing = [row["filepath"] for row in rows if not (repo_root / row["filepath"]).exists()]
    if missing:
        r.error("K2", f"brakuje {len(missing)} plików (np. {missing[:3]})")
    else:
        r.ok("K2", "wszystkie pliki na miejscu")

    # ---------------------------------------------------------------- K3 przeciek
    groups: Dict[str, set] = defaultdict(set)
    for row in rows:
        groups[row["group_id"]].add(row["split"])
    leaked = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
    if leaked:
        r.error("K3", f"PRZECIEK — {len(leaked)} grup w więcej niż jednym splicie "
                      f"(np. {list(leaked.items())[:3]})")
    else:
        r.ok("K3", f"brak przecieku — {len(groups)} grup, każda w jednym splicie")

    # ---------------------------------------------------------------- K7 grupy poz.
    pos_groups = {sp: {row["group_id"] for row in rows
                       if row["split"] == sp and row["label"] == "positive"}
                  for sp in SPLITS}
    n_test = len(pos_groups["test"])
    if n_test < args.min_positive_groups_test:
        r.error("K7", f"tylko {n_test} grup pozytywnych w teście "
                      f"(minimum {args.min_positive_groups_test}) — metryka byłaby "
                      f"obarczona ogromnym błędem próbkowania")
    else:
        r.ok("K7", f"grupy pozytywne: train={len(pos_groups['train'])} "
                   f"val={len(pos_groups['val'])} test={n_test}")

    # ---------------------------------------------------------------- K4/K5/K6
    if args.quick:
        print("  pominięto   [K4/K5/K6] tryb --quick")
    else:
        changed, unreadable, bad_audio = [], [], []
        by_hash: Dict[str, List[dict]] = defaultdict(list)
        for i, row in enumerate(rows, 1):
            p = repo_root / row["filepath"]
            if not p.exists():
                continue
            digest = sha256_of(p)
            by_hash[digest].append(row)
            if digest != row["sha256"]:
                changed.append(row["filepath"])
            try:
                info = sf.info(str(p))
                if (info.samplerate < MIN_SAMPLE_RATE or info.channels > MAX_CHANNELS
                        or not (MIN_DURATION_S <= info.duration <= MAX_DURATION_S)):
                    bad_audio.append(
                        f"{row['filepath']} (sr={info.samplerate}, ch={info.channels}, "
                        f"{info.duration:.2f}s)")
            except Exception as exc:
                unreadable.append(f"{row['filepath']}: {exc}")
            if i % 2000 == 0:
                print(f"  … sprawdzono {i}/{len(rows)}", flush=True)

        if changed:
            r.error("K4", f"{len(changed)} plików ma inną sumę kontrolną niż w manifeście "
                          f"(np. {changed[:3]}) — zbiór NIE jest tą wersją")
        else:
            r.ok("K4", "wszystkie sumy kontrolne zgodne")

        cross, inner = [], []
        for digest, group in by_hash.items():
            if len(group) < 2:
                continue
            (cross if len({g["split"] for g in group}) > 1 else inner).append(group)
        if cross:
            r.error("K5", f"{len(cross)} zestawów identycznych plików rozrzuconych "
                          f"po różnych splitach (np. "
                          f"{[g['filepath'] for g in cross[0]][:2]})")
        else:
            r.ok("K5", "brak identycznych plików między splitami")
        if inner:
            r.warn("O1", f"{len(inner)} zestawów identycznych plików wewnątrz splitów "
                         f"— rozważ deduplikację")

        if unreadable or bad_audio:
            if unreadable:
                r.error("K6", f"{len(unreadable)} plików nie do odczytania "
                              f"(np. {unreadable[:2]})")
            if bad_audio:
                r.error("K6", f"{len(bad_audio)} plików poza dozwolonym zakresem audio "
                              f"(np. {bad_audio[:2]})")
        else:
            r.ok("K6", f"audio w porządku (sr ≥ {MIN_SAMPLE_RATE}, ch ≤ {MAX_CHANNELS}, "
                       f"{MIN_DURATION_S}–{MAX_DURATION_S}s)")

    # ---------------------------------------------------------------- ostrzeżenia
    for sp in SPLITS:
        rs = [row for row in rows if row["split"] == sp]
        if not rs:
            r.warn("O2", f"split {sp} jest pusty")
            continue
        frac = sum(1 for row in rs if row["label"] == "positive") / len(rs)
        if not (BALANCE_RANGE[0] <= frac <= BALANCE_RANGE[1]):
            r.warn("O2", f"{sp}: pozytywy stanowią {100*frac:.1f}% — poza widełkami "
                         f"{100*BALANCE_RANGE[0]:.0f}–{100*BALANCE_RANGE[1]:.0f}%")
        kinds_here = {row["kind"] for row in rs}
        for k in KINDS:
            if k not in kinds_here:
                r.warn("O3", f"{sp}: brak nagrań rodzaju `{k}`")

    nc = [row for row in rows if row["license"] in NONCOMMERCIAL_LICENSES]
    if nc:
        r.warn("O4", f"{len(nc)} nagrań na licencji niekomercyjnej — do produktu "
                     f"trzeba je odfiltrować")

    # ---------------------------------------------------------------- podsumowanie
    print()
    if r.errors:
        print(f"WYNIK: NIEZDANE — {len(r.errors)} błędów, {len(r.warns)} ostrzeżeń")
        sys.exit(1)
    print(f"WYNIK: ZDANE — 0 błędów, {len(r.warns)} ostrzeżeń")


if __name__ == "__main__":
    main()
