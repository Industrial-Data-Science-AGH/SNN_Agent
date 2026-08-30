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
    K8  podklasa VOICe niezgodna z katalogiem ekstrakcji
    K9  artefakt pochodny ma inną etykietę niż manifest, na który się powołuje

Kontrole ostrzegawcze (OSTRZEŻENIE, nie blokują):
    O1  duplikaty treści wewnątrz jednego splitu
    O2  niezbalansowanie klas poza widełkami
    O3  brak któregoś `kind` w którymś splicie
    O4  nagrania na licencji niekomercyjnej
    O5  artefakt deklaruje tę wersję, ale nie ma files.csv (niesprawdzalny)

Użycie:
    python snn_pipeline/validate_dataset.py --version v1.0.0
    python snn_pipeline/validate_dataset.py --version v1.0.0 --quick   # bez sha256
"""

from __future__ import annotations

import argparse
import csv
import json
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
MIN_EFFECTIVE_GROUPS_TEST = 12   # min. grup niosących 90% pozytywów testu (K10, koncentracja)
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
    ap.add_argument("--min-effective-groups-test", type=int, default=MIN_EFFECTIVE_GROUPS_TEST,
                    help="min. grup pokrywających 90%% pozytywów testu (kontrola koncentracji, K10)")
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

    # ------------------------------------------------- K10 KONCENTRACJA grup poz.
    # Sama LICZBA grup (K7) przechodzi trywialnie, gdy garstka grup niesie prawie
    # wszystkie pozytywy (np. 32/59 grup = 94% pozytywów). Liczymy EFEKTYWNĄ
    # różnorodność: ile grup potrzeba, by pokryć 90% pozytywnych KLIPÓW testu.
    pos_test_counts: Dict[str, int] = defaultdict(int)
    for row in rows:
        if row["split"] == "test" and row["label"] == "positive":
            pos_test_counts[row["group_id"]] += 1
    total_pos = sum(pos_test_counts.values())
    if total_pos == 0:
        r.error("K10", "brak pozytywnych klipów w teście — nie ma czego mierzyć")
    else:
        counts = sorted(pos_test_counts.values(), reverse=True)
        cum, n_eff = 0, 0
        for c in counts:
            cum += c; n_eff += 1
            if cum >= 0.90 * total_pos:
                break
        top_share = 100.0 * counts[0] / total_pos
        if n_eff < args.min_effective_groups_test:
            r.error("K10", f"koncentracja: 90% pozytywów testu niesie tylko {n_eff} "
                           f"grup (min {args.min_effective_groups_test}; z "
                           f"{len(pos_test_counts)} wszystkich), największa {top_share:.0f}% "
                           f"— metryka rozstrzygana przez garstkę nagrań")
        else:
            r.ok("K10", f"rozproszenie grup OK: 90% pozytywów w {n_eff}/"
                        f"{len(pos_test_counts)} grupach, największa {top_share:.0f}%")

    # k8 etykieta vs katalog

    FOLDER_SUBCLASS = {"glass": {"glassbreak"},
                       "hard_negative": {"gunshot", "babycry"}}
    bad_folder = []
    for row in rows:
        if row["source"] != "voice":
            continue
        parts = Path(row["filepath"]).parts
        folder = next((p for p in parts if p in FOLDER_SUBCLASS), None)
        if folder and row["subclass"] not in FOLDER_SUBCLASS[folder]:
            bad_folder.append(f"{row['filepath']} -> {row['subclass']}")
    if bad_folder:
        r.error("K8", f"{len(bad_folder)} klipów VOICe ma podklasę niezgodną "
                      f"z katalogiem ekstrakcji (np. {bad_folder[0]})")
    else:
        r.ok("K8", "podklasa VOICe zgodna z katalogiem ekstrakcji")

    # K9 artefakty pochodne vs ten manifest
    by_path = {row["filepath"]: row["label"] for row in rows}
    drift, checked, artifacts = [], 0, 0
    for files_csv in sorted(repo_root.glob("*/**/files.csv")):
        chan = files_csv.with_name("channels.json")
        if not chan.exists():
            continue
        prov = json.loads(chan.read_text(encoding="utf-8")).get("provenance", {})
        if prov.get("dataset_version") != args.version:
            continue
        artifacts += 1
        with files_csv.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                man = by_path.get(row["filepath"])
                if man is None:
                    continue
                checked += 1
                if man != row["label"]:
                    drift.append(f"{files_csv.parent.name}:{row['filepath']}")
    if drift:
        r.error("K9", f"{len(drift)} plików ma w artefakcie inną etykietę niż "
                      f"w manifeście {args.version} (np. {drift[0]})")
    else:
        r.ok("K9", f"artefakty powołujące się na tę wersję mają zgodne etykiety "
                   f"({checked} plików w {artifacts} artefaktach)")

    
    unverifiable = []
    for chan in sorted(repo_root.glob("*/**/channels.json")):
        prov = json.loads(chan.read_text(encoding="utf-8")).get("provenance", {})
        if prov.get("dataset_version") == args.version and \
                not chan.with_name("files.csv").exists():
            unverifiable.append(str(chan.relative_to(repo_root)))
    if unverifiable:
        r.warn("O5", f"{len(unverifiable)} artefaktów deklaruje {args.version}, "
                     f"ale nie ma files.csv — nie da się sprawdzić ich etykiet "
                     f"(np. {unverifiable[0]}); przebuduj je")

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
