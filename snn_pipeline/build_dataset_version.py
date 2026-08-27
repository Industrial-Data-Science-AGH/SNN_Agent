#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_dataset_version.py — buduje JEDNĄ, NIEZMIENNĄ wersję głównego zbioru SNN.

Wynik trafia do `dataset/versions/<wersja>/` i składa się z:
    manifest.csv   — jeden wiersz na nagranie, schemat w dataset_contract.py
    dataset.json   — metadane wersji (ziarno, commit, inwentarz źródeł)
    stats.md       — statystyki per split, gotowe do wklejenia w raport

Wersji NIE WOLNO nadpisywać — jeśli katalog istnieje, skrypt odmawia pracy.
Chodzi o to, żeby „wytrenowane na v1.0.0" zawsze znaczyło to samo.

Podział train/val/test idzie po GRUPACH (nagraniach źródłowych), nigdy po
plikach, i jest ZAPISANY w manifeście jako kolumna — nie odtwarzany z ziarna.
Ziarno jest tylko po to, żeby dało się powtórzyć budowę; źródłem prawdy o
podziale jest plik.

Użycie:
    python snn_pipeline/build_dataset_version.py --version v1.0.0
    python snn_pipeline/build_dataset_version.py --version v1.0.0 --dry-run
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import ESC50_GLASS_BREAKING_CLASS
from dataset_contract import (
    DATASEC_KIND,
    ESC50_KIND,
    MANIFEST_COLUMNS,
    SOURCE_LICENSE,
    VOICE_KIND,
    group_id_for,
    is_valid_version,
    make_id,
    sha256_of,
    version_dir,
)

VOICE_INTERVAL_RE = re.compile(r"_(synthetic_\d+)_(\d+\.\d+)-(\d+\.\d+)\.wav$")


FOLDER_SUBCLASSES: Dict[str, tuple] = {
    "glass": ("glassbreak",),
    "hard_negative": ("gunshot", "babycry"),
}

VOICE_EXTRACT_PAD_S = 0.30


# =============================================================================
# ZBIERANIE REKORDÓW
# =============================================================================

def _base_record(repo_root: Path, wav: Path, source: str, subclass: str,
                 kind: str) -> Optional[dict]:
    rel = str(wav.relative_to(repo_root))
    try:
        info = sf.info(str(wav))
    except Exception as exc:                      # plik uszkodzony albo nie-audio
        print(f"[WARN] nie mogę odczytać {rel}: {exc}")
        return None
    return {
        "id": make_id(source, rel),
        "filepath": rel,
        "sha256": sha256_of(wav),
        "bytes": wav.stat().st_size,
        "label": "positive" if kind == "positive" else "negative",
        "kind": kind,
        "source": source,
        "subclass": subclass,
        "group_id": group_id_for(source, wav),
        "split": "",                              # uzupełniane później
        "duration_s": round(info.duration, 4),
        "sample_rate": info.samplerate,
        "channels": info.channels,
        "subtype": info.subtype,
        "license": SOURCE_LICENSE[source],
    }


def collect_esc50(repo_root: Path) -> List[dict]:
    audio = repo_root / "data" / "ESC-50-master" / "audio"
    if not audio.exists():
        print(f"[WARN] ESC-50 nie znaleziono w {audio} — pomijam.")
        return []
    meta = {}
    meta_csv = repo_root / "data" / "ESC-50-master" / "meta" / "esc50.csv"
    with meta_csv.open() as fh:
        for row in csv.DictReader(fh):
            meta[row["filename"]] = (int(row["target"]), row["category"])
    out = []
    for wav in sorted(audio.glob("*.wav")):
        target, category = meta.get(wav.name, (-1, "unknown"))
        kind = "positive" if target == ESC50_GLASS_BREAKING_CLASS else \
            ESC50_KIND.get(target, "loud_event")
        rec = _base_record(repo_root, wav, "esc50", category, kind)
        if rec:
            out.append(rec)
    print(f"[INFO] ESC-50: {len(out)} nagrań")
    return out


def collect_datasec(repo_root: Path) -> List[dict]:
    base = repo_root / "dataset" / "datasec" / "PT_DATASET_250314"
    if not base.exists():
        print(f"[WARN] DataSEC nie znaleziono w {base} — pomijam.")
        return []
    out, unknown = [], set()
    for class_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        kind = DATASEC_KIND.get(class_dir.name)
        if kind is None:
            unknown.add(class_dir.name)
            kind = "loud_event"
        for wav in sorted(class_dir.rglob("*.wav")):
            rec = _base_record(repo_root, wav, "datasec", class_dir.name, kind)
            if rec:
                out.append(rec)
    if unknown:
        print(f"[WARN] klasy DataSEC bez wpisu w taksonomii (przyjęto loud_event): "
              f"{sorted(unknown)}")
    print(f"[INFO] DataSEC: {len(out)} nagrań")
    return out


def _voice_annotations(repo_root: Path) -> Dict[str, List[tuple]]:
    """Adnotacje VOICe: mix -> [(start, koniec, etykieta)]."""
    ann_dir = repo_root / "dataset" / "clean" / "annotation"
    if not ann_dir.exists():
        return {}
    out = {}
    for txt in sorted(ann_dir.glob("*.txt")):
        rows = []
        for line in txt.read_text().splitlines():
            parts = line.strip().split("\t")
            if len(parts) == 3:
                rows.append((float(parts[0]), float(parts[1]), parts[2]))
        out[txt.stem] = rows
    return out


def collect_voice(repo_root: Path) -> List[dict]:
    """Wycinki z VOICe, pocięte wcześniej przez voice_extract.py.

    Nie tniemy ich ponownie: `voice_extracted/` powstało z `dataset/clean` i
    jest w repo.

    KLASA pochodzi z katalogu ekstrakcji i nie podlega negocjacji. Adnotacje
    rozstrzygają wyłącznie PODKLASĘ w obrębie katalogu (gunshot vs babycry),
    żeby nie wrzucać wszystkiego do jednego worka „hard_negative".

    Strażnik kontaminacji działa ASYMETRYCZNIE, tak jak w
    build_combined_dataset.py: odrzucamy negatyw, w którego oknie jest szkło
    (etykieta byłaby fałszywa i uczyłaby sieć tłumienia na szkle), ale
    zachowujemy pozytyw, w którego oknie jest wystrzał — on nadal ZAWIERA
    szkło, jest tylko trudniejszy. VOICe to miksy polifoniczne: przy pad 0.30 s
    3961 z 4444 zdarzeń glassbreak nachodzi na inne zdarzenie, więc symetryczny
    strażnik zostawiłby 483 pozytywy zamiast 4444.
    """
    base = repo_root / "architecture_14_neurons_patryk_09_07" / "voice_extracted"
    if not base.exists():
        print(f"[WARN] voice_extracted nie znaleziono w {base} — pomijam.")
        return []
    ann = _voice_annotations(repo_root)
    if not ann:
        print("[WARN] brak dataset/clean/annotation — podklasy VOICe będą przybliżone.")

    out, resolved, guessed, dropped = [], 0, 0, 0
    for folder, allowed in FOLDER_SUBCLASSES.items():
        d = base / folder
        if not d.exists():
            continue
        for wav in sorted(d.glob("*.wav")):
            subclass = allowed[0]
            m = VOICE_INTERVAL_RE.search(wav.name)
            if m and m.group(1) in ann:
                mix, s, e = m.group(1), float(m.group(2)), float(m.group(3))
                if folder == "hard_negative":
                    foreign = [(a_s, a_e) for a_s, a_e, a_lab in ann[mix]
                               if a_lab not in allowed]
                    ps, pe = s - VOICE_EXTRACT_PAD_S, e + VOICE_EXTRACT_PAD_S
                    if any(min(pe, a_e) > max(ps, a_s) for a_s, a_e in foreign):
                        dropped += 1
                        continue

                if len(allowed) > 1:
                    best, best_ov = None, 0.0
                    for a_s, a_e, a_lab in ann[mix]:
                        if a_lab not in allowed:   # nigdy nie wychodzimy poza katalog
                            continue
                        ov = min(e, a_e) - max(s, a_s)
                        if ov > best_ov:
                            best, best_ov = a_lab, ov
                    if best:
                        subclass, resolved = best, resolved + 1
                    else:
                        guessed += 1
                else:
                    resolved += 1        # glass/ nie wymaga rozstrzygania
            else:
                guessed += 1

            rec = _base_record(repo_root, wav, "voice", subclass,
                               VOICE_KIND[subclass])
            if rec:
                out.append(rec)

    print(f"[INFO] VOICe: {len(out)} wycinków "
          f"(podklasa z adnotacji: {resolved}, przybliżona: {guessed}, "
          f"odrzucone jako skażone szkłem: {dropped})")
    return out


# =============================================================================
# PODZIAŁ
# =============================================================================

def unify_groups_by_content(records: List[dict]) -> int:
    """Scala grupy, które zawierają BAJT W BAJT ten sam plik.

    Powód, którego nie dało się przewidzieć bez pomiaru: DataSEC zawiera pliki
    przekopiowane z ESC-50 (np. `Birds/Birds-002.wav` == `1-34497-A-14.wav`).
    Reguła grupy jest z definicji per źródło, więc to samo nagranie pod dwiema
    nazwami dostawało dwie grupy i mogło trafić do dwóch splitów naraz — czyli
    przeciek, którego samo grupowanie nie łapie.

    Rozwiązanie: identyczna zawartość ZNACZY to samo nagranie. Grupy połączone
    wspólnym plikiem scalamy w jedną (union-find), a scalona grupa dostaje
    deterministyczne id `dup_<hash>`, żeby wynik nie zależał od kolejności.
    """
    parent: Dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    by_hash: Dict[str, List[str]] = defaultdict(list)
    for r in records:
        find(r["group_id"])
        by_hash[r["sha256"]].append(r["group_id"])

    merged = 0
    for gs in by_hash.values():
        uniq = set(gs)
        if len(uniq) > 1:
            merged += 1
            first = sorted(uniq)[0]
            for g in uniq:
                union(first, g)

    members: Dict[str, set] = defaultdict(set)
    for g in list(parent):
        members[find(g)].add(g)

    remap = {}
    for root, group_set in members.items():
        if len(group_set) > 1:
            h = hashlib.sha1("|".join(sorted(group_set)).encode()).hexdigest()[:10]
            for g in group_set:
                remap[g] = f"dup_{h}"
    for r in records:
        if r["group_id"] in remap:
            r["group_id"] = remap[r["group_id"]]

    if merged:
        print(f"[INFO] scalono {len(remap)} grup w {len(set(remap.values()))} "
              f"(identyczna zawartość w różnych źródłach) — {merged} zestawów duplikatów")
    return merged


def _voice_official_split(repo_root: Path) -> Dict[str, str]:
    """Opublikowany podział VOICe: {"synthetic_001": "train", ...}.

    VOICe rozprowadza własne listy miksów (69/69/69, parami rozłączne, suma 207).
    Użycie ich czyni wyniki porównywalnymi z pracami na tym zbiorze i podnosi
    liczbę niezależnych nagrań w teście z 32 do 69. Cena: proporcje VOICe to
    33/33/33, nie 70/15/15 — decyzja zapisana w stats.md.

    Zwraca {} jeśli list nie ma; wtedy VOICe idzie zwykłą ścieżką GroupShuffleSplit.
    """
    src = repo_root / "dataset" / "clean" / "source"
    files = {
        "train": src / "synthetic_source_training.txt",
        "val": src / "synthetic_source_validation.txt",
        "test": src / "synthetic_source_test.txt",
    }
    if not all(f.exists() for f in files.values()):
        return {}
    out: Dict[str, str] = {}
    for split, f in files.items():
        for line in f.read_text(encoding="utf-8").splitlines():
            mix = line.strip()
            if not mix:
                continue
            mix = mix[:-4] if mix.endswith(".wav") else mix
            if mix in out:
                print(f"[WARN] {mix} występuje w dwóch listach VOICe "
                      f"({out[mix]} i {split}) — biorę pierwszą")
                continue
            out[mix] = split
    print(f"[INFO] VOICe: oficjalny podział z dataset/clean/source ({len(out)} miksów)")
    return out


def assign_splits(records: List[dict], val_frac: float, test_frac: float,
                  seed: int, repo_root: Path) -> None:
    """Podział po GRUPACH, osobno w obrębie każdego źródła.

    Osobno per źródło, żeby proporcje 70/15/15 trzymały się w każdym z nich —
    inaczej jedno duże źródło zdominowałoby losowanie i któreś ze źródeł mogłoby
    zniknąć z testu. Grupy nigdy nie są dzielone.
    """
    # Grupa może po scaleniu obejmować DWA źródła (identyczny plik w ESC-50
    # i DataSEC). Przypisujemy ją wtedy w całości do jednego wiadra, wybranego
    # deterministycznie — inaczej dzielilibyśmy ją dwa razy i przeciek by wrócił.
    group_source: Dict[str, str] = {}
    for r in sorted(records, key=lambda x: (x["group_id"], x["source"], x["filepath"])):
        group_source.setdefault(r["group_id"], r["source"])

    by_source: Dict[str, List[dict]] = defaultdict(list)
    for r in records:
        by_source[group_source[r["group_id"]]].append(r)


    official = _voice_official_split(repo_root)
    if official:
        voice = by_source.pop("voice", [])
        unmatched = Counter()
        for r in voice:
            mix = r["group_id"].removeprefix("voice_")
            split = official.get(mix)
            if split is None:
                unmatched[r["group_id"]] += 1
                split = "train"
            r["split"] = split
        counts = Counter(r["split"] for r in voice)
        print(f"[INFO] VOICe: podział oficjalny — train={counts['train']} "
              f"val={counts['val']} test={counts['test']} plików")
        if unmatched:
            print(f"[WARN] {sum(unmatched.values())} plików VOICe w "
                  f"{len(unmatched)} grupach spoza oficjalnych list -> train "
                  f"(np. {next(iter(unmatched))})")

    for source, recs in sorted(by_source.items()):
        groups = [r["group_id"] for r in recs]
        idx = list(range(len(recs)))
        n_groups = len(set(groups))
        if n_groups < 3:
            for r in recs:
                r["split"] = "train"
            print(f"[WARN] {source}: tylko {n_groups} grup — wszystko do train")
            continue

        gss = GroupShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed)
        rest_i, test_i = next(gss.split(idx, groups=groups))
        rest_groups = [groups[i] for i in rest_i]
        rel_val = val_frac / (1.0 - test_frac)
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=seed)
        tr_i, val_i = next(gss2.split(rest_i, groups=rest_groups))

        for i in test_i:
            recs[i]["split"] = "test"
        for j in val_i:
            recs[rest_i[j]]["split"] = "val"
        for j in tr_i:
            recs[rest_i[j]]["split"] = "train"


# =============================================================================
# RAPORT
# =============================================================================

def build_stats(records: List[dict], version: str, seed: int) -> str:
    L = [f"# Statystyki zbioru `{version}`", ""]
    L.append(f"Wygenerowane {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}, "
             f"ziarno podziału `{seed}`.")
    L.append("")

    total_h = sum(r["duration_s"] for r in records) / 3600
    L.append(f"Razem **{len(records)} nagrań**, **{len({r['group_id'] for r in records})} "
             f"grup źródłowych**, **{total_h:.1f} h** audio.")
    L.append("")

    L.append("## Podział")
    L.append("")
    L.append("| split | nagrania | grupy | czas [h] | pozytywne | grupy pozytywne |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for sp in ("train", "val", "test"):
        rs = [r for r in records if r["split"] == sp]
        pos = [r for r in rs if r["label"] == "positive"]
        L.append(f"| {sp} | {len(rs)} | {len({r['group_id'] for r in rs})} | "
                 f"{sum(r['duration_s'] for r in rs)/3600:.2f} | "
                 f"{len(pos)} ({100*len(pos)/max(len(rs),1):.1f}%) | "
                 f"**{len({r['group_id'] for r in pos})}** |")
    L.append("")
    L.append("Kolumna *grupy pozytywne* jest ważniejsza niż liczba plików: to ona mówi, "
             "na ilu **niezależnych nagraniach** liczona jest metryka.")
    L.append("")

    L.append("## Rodzaj dźwięku (`kind`) — podstawa raportu fałszywych alarmów")
    L.append("")
    L.append("| kind | nagrania | czas [h] | train | val | test |")
    L.append("|---|---:|---:|---:|---:|---:|")
    for kind in ("positive", "stationary", "loud_event", "speech", "animal"):
        rs = [r for r in records if r["kind"] == kind]
        if not rs:
            continue
        c = Counter(r["split"] for r in rs)
        L.append(f"| `{kind}` | {len(rs)} | {sum(r['duration_s'] for r in rs)/3600:.2f} | "
                 f"{c['train']} | {c['val']} | {c['test']} |")
    L.append("")

    L.append("## Źródła i licencje")
    L.append("")
    L.append("| źródło | nagrania | grupy | czas [h] | licencja |")
    L.append("|---|---:|---:|---:|---|")
    for src in sorted({r["source"] for r in records}):
        rs = [r for r in records if r["source"] == src]
        L.append(f"| {src} | {len(rs)} | {len({r['group_id'] for r in rs})} | "
                 f"{sum(r['duration_s'] for r in rs)/3600:.2f} | {SOURCE_LICENSE[src]} |")
    L.append("")
    nc = [r for r in records if r["license"] in ("CC BY-NC 3.0",)]
    if nc:
        L.append(f"> **Uwaga licencyjna:** {len(nc)} nagrań jest na licencji niekomercyjnej. "
                 f"Do zbioru treningowego modelu przeznaczonego do produktu trzeba je odfiltrować "
                 f"(`license != 'CC BY-NC 3.0'`).")
        L.append("")

    L.append("## Parametry audio")
    L.append("")
    sr = Counter(r["sample_rate"] for r in records)
    ch = Counter(r["channels"] for r in records)
    st = Counter(r["subtype"] for r in records)
    L.append(f"- częstotliwości: {dict(sr.most_common())}")
    L.append(f"- kanały: {dict(ch.most_common())}")
    L.append(f"- format próbki: {dict(st.most_common())}")
    d = sorted(r["duration_s"] for r in records)
    L.append(f"- długość [s]: min {d[0]:.2f}, p50 {d[len(d)//2]:.2f}, "
             f"p95 {d[int(len(d)*0.95)]:.2f}, max {d[-1]:.2f}")
    L.append("")

    L.append("## Zmiany względem v1.0.0")
    L.append("")
    L.append("**Naprawiony błąd etykietowania VOICe (issue #34).** W v1.0.0 `collect_voice` "
             "wyprowadzało etykietę regułą maksymalnego pokrycia po WSZYSTKICH adnotacjach "
             "miksu i ignorowało katalog ekstrakcji. Interwał w nazwie pliku JEST interwałem "
             "jednej adnotacji, więc każde dłuższe zdarzenie, które go zawiera, remisowało "
             "lub wygrywało. Skutek: 975 babycry + 437 gunshot z katalogu `glass/` miało "
             "etykietę `negative` (1412 = 31.8% wyciętego szkła), a 241 klipów "
             "z `hard_negative/` etykietę `positive`. Teraz polaryzacja klasy pochodzi "
             "z katalogu ekstrakcji, a reguła pokrycia rozstrzyga wyłącznie gunshot vs babycry.")
    L.append("")
    L.append("**Polifonia.** VOICe to miksy nakładających się zdarzeń: przy pad 0.30 s "
             "3961 z 4444 zdarzeń glassbreak nachodzi na gunshot/babycry, a 3261 z 4444 "
             "wycinków hard_negative nachodzi na glassbreak. Przyjęto strażnika "
             "**asymetrycznego** (jak w `build_combined_dataset.py`, zgubionego przy "
             "przepisywaniu): odrzucamy skażony negatyw, bo jego etykieta byłaby fałszywa "
             "i uczyłaby sieć tłumienia na szkle; zachowujemy skażony pozytyw, bo on nadal "
             "ZAWIERA szkło — jest trudniejszy, nie błędny. Symetryczny strażnik zostawiłby "
             "483 pozytywy zamiast 4444. Twardych negatywów VOICe: 1183 zamiast 4444.")
    L.append("")
    L.append("**Podział VOICe** wzięty z opublikowanych list `dataset/clean/source/*.txt` "
             "(69/69/69 miksów, parami rozłączne). Podnosi liczbę niezależnych nagrań "
             "pozytywnych w teście z 59 do 96, kosztem proporcji: VOICe dzieli się 33/33/33, "
             "a pozostałe źródła 70/15/15, więc udział pozytywów w val/test jest wyższy "
             "niż w train. Metryką wdrożeniową jest FA/h liczone na godzinach tła, "
             "nie odsetek klipów, więc ta nierównowaga nie zniekształca celu.")
    L.append("")

    L.append("## Znane ograniczenia")
    L.append("")
    L.append("- Klasa pozytywna jest zdominowana przez wycinki VOICe pochodzące z 207 miksów; "
             "liczba **niezależnych** nagrań szkła jest o rząd wielkości mniejsza niż liczba plików.")
    L.append("- Udział pozytywów różni się między splitami (train ~40%, val/test ~46-47%) "
             "przez różne proporcje podziału VOICe i pozostałych źródeł. Progi kalibrowane "
             "na val trzeba przenosić na deployment ostrożnie.")
    L.append("- Tło stacjonarne pochodzi z ciągłych klas DataSEC i ESC-50, nie z nagrań "
             "z docelowego pomieszczenia. Fałszywe alarmy *w ciszy* są więc mierzone na "
             "zastępniku, nie na realnym tle instalacji.")
    L.append("- Audio nie jest transkodowane ani normalizowane — parametry są tylko mierzone "
             "i walidowane. Konwersję robi enkoder przy odczycie.")
    L.append("- ESC-50 jest na licencji niekomercyjnej.")
    return "\n".join(L) + "\n"


# =============================================================================
# MAIN
# =============================================================================

def git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                       text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent.parent)
    ap.add_argument("--version", required=True, help="np. v1.0.0")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true", help="policz i pokaż, nie zapisuj")
    args = ap.parse_args()

    if not is_valid_version(args.version):
        sys.exit(f"[BŁĄD] '{args.version}' nie jest wersją semantyczną (oczekiwano vX.Y.Z)")

    repo_root = args.repo_root.resolve()
    out_dir = version_dir(repo_root, args.version)
    if out_dir.exists() and not args.dry_run:
        sys.exit(f"[BŁĄD] {out_dir} już istnieje. Wersji NIE nadpisujemy — "
                 f"wydaj nowy numer.")

    records: List[dict] = []
    records += collect_esc50(repo_root)
    records += collect_datasec(repo_root)
    records += collect_voice(repo_root)
    if not records:
        sys.exit("[BŁĄD] nie zebrano ani jednego nagrania — sprawdź ścieżki źródeł.")

    unify_groups_by_content(records)
    assign_splits(records, args.val_frac, args.test_frac, args.seed, repo_root)
    voice_official = bool(_voice_official_split(repo_root))
    records.sort(key=lambda r: (r["source"], r["filepath"]))

    stats = build_stats(records, args.version, args.seed)
    print()
    print(stats)

    if args.dry_run:
        print("[dry-run] nic nie zapisano.")
        return

    out_dir.mkdir(parents=True, exist_ok=False)
    man = out_dir / "manifest.csv"
    with man.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_COLUMNS)
        w.writeheader()
        w.writerows(records)

    meta = {
        "version": args.version,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git_commit(repo_root),
        "split": {"val_frac": args.val_frac, "test_frac": args.test_frac,
                  "seed": args.seed, "grouped_by": "group_id, osobno per source",
                  # VOICe nie podlega losowaniu, jeśli są opublikowane listy —
                  # bez tego wpisu metadane kłamią o proporcjach tego źródła.
                  "voice_official_split": voice_official,
                  "voice_split_note": "VOICe: 69/69/69 z dataset/clean/source "
                                      "(33/33/33), pozostałe źródła wg val_frac/test_frac"
                                      if voice_official else "brak list — VOICe losowane"},
        "counts": {
            "records": len(records),
            "groups": len({r["group_id"] for r in records}),
            "positive": sum(1 for r in records if r["label"] == "positive"),
            "hours": round(sum(r["duration_s"] for r in records) / 3600, 3),
        },
        "sources": {s: sum(1 for r in records if r["source"] == s)
                    for s in sorted({r["source"] for r in records})},
        "manifest_sha256": sha256_of(man),
    }
    (out_dir / "dataset.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (out_dir / "stats.md").write_text(stats, encoding="utf-8")

    print(f"[ok] zapisano {out_dir}")
    print(f"     manifest.csv  ({len(records)} wierszy, sha256 {meta['manifest_sha256'][:16]}…)")
    print(f"     dataset.json  stats.md")
    print()
    print(f"Zwaliduj:  python snn_pipeline/validate_dataset.py --version {args.version}")


if __name__ == "__main__":
    main()
