"""Etykieta klipu VOICe musi zgadzać się z katalogiem, z którego został wycięty.

Wykrywa błąd #34: collect_voice wyprowadzało etykietę regułą maksymalnego
pokrycia po WSZYSTKICH adnotacjach miksu, więc dłuższe zdarzenie zawierające
wycinek przerzucało go do drugiej klasy. Zmierzone na v1.0.0: 975 babycry +
437 gunshot z katalogu glass/ jako negative (1412 = 31.8% wyciętego szkła)
i 241 klipów z hard_negative/ jako positive.

Test jest parametryzowany wersją, żeby dało się pokazać różnicę:
    pytest snn_pipeline/tests/test_voice_labels.py --dataset-version v1.0.0   # czerwony
    pytest snn_pipeline/tests/test_voice_labels.py                            # v2.0.0, zielony
"""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ALLOWED = {"glass": {"glassbreak"}, "hard_negative": {"gunshot", "babycry"}}
DEFAULT_VERSION = "v2.0.0"


@pytest.fixture(scope="module")
def rows(request):
    version = request.config.getoption("--dataset-version", default=DEFAULT_VERSION)
    path = REPO / "dataset" / "versions" / version / "manifest.csv"
    if not path.exists():
        pytest.skip(f"brak {path}")
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _folder(filepath: str):
    return next((p for p in Path(filepath).parts if p in ALLOWED), None)


def test_voice_subclass_matches_extraction_folder(rows):
    bad = []
    for row in rows:
        if row["source"] != "voice":
            continue
        folder = _folder(row["filepath"])
        if folder and row["subclass"] not in ALLOWED[folder]:
            bad.append((row["filepath"], row["subclass"]))
    assert not bad, f"{len(bad)} klipów z podklasą niezgodną z katalogiem, np. {bad[:3]}"


def test_no_glass_clip_labeled_negative(rows):
    bad = [r["filepath"] for r in rows
           if r["source"] == "voice" and _folder(r["filepath"]) == "glass"
           and r["label"] != "positive"]
    assert not bad, f"{len(bad)} klipów z glass/ oznaczonych jako negative, np. {bad[:3]}"


def test_no_hard_negative_labeled_positive(rows):
    bad = [r["filepath"] for r in rows
           if r["source"] == "voice" and _folder(r["filepath"]) == "hard_negative"
           and r["label"] != "negative"]
    assert not bad, f"{len(bad)} klipów z hard_negative/ jako positive, np. {bad[:3]}"


def test_groups_do_not_span_splits(rows):
    seen = {}
    leaked = set()
    for r in rows:
        seen.setdefault(r["group_id"], set()).add(r["split"])
    leaked = {g for g, s in seen.items() if len(s) > 1}
    assert not leaked, f"{len(leaked)} grup w więcej niż jednym splicie, np. {sorted(leaked)[:3]}"
