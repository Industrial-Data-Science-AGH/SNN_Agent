"""Artefakt spike'owy musi zgadzać się z manifestem, na który się powołuje.

Trzy rzeczy, które przeżyły w repo miesiące i których nikt nie łapał:

1. `spikes_manifest7` miał 194 z 194 miksów VOICe obecnych w teście również
   w treningu. Każda metryka policzona na tym artefakcie była liczona na
   przecieku, łącznie z `hw7_config.json`, z którego wzięto nastawy trymerów.
2. Wszystkie 40 plików ESC-50 z prefiksem `glass_` w `spikes_manifest7` to
   target 38 = `clock_tick`, nie 39 = `glass_breaking`. Błąd naprawiono
   w builderze zbioru, ale artefaktu spikowego nigdy nie przebudowano.
3. Pozytywy i negatywy były kodowane jednym ciągłym stanem enkodera, ale
   posortowane po etykiecie: najpierw wszystkie negatywy, potem wszystkie
   pozytywy. Stan enkodera (floor/MAD) był więc skorelowany z klasą.

Domyślnie sprawdzany jest `spikes_v2`; innym katalogiem:
    pytest snn_pipeline/tests/test_spike_artifact.py --spikes-dir <sciezka>
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def artifact(request):
    d = REPO / request.config.getoption("--spikes-dir")
    if not d.exists():
        pytest.skip(f"brak artefaktu {d}; zbuduj go encoder_twin.py build-manifest")
    rows = {}
    for split in SPLITS:
        f = d / split / "files.csv"
        if not f.exists():
            pytest.skip(f"brak {f}; artefakt zbudowany starą wersją buildera")
        with f.open(newline="", encoding="utf-8") as fh:
            rows[split] = list(csv.DictReader(fh))
    return d, rows


@pytest.fixture(scope="module")
def manifest(artifact):
    import json
    d, _ = artifact
    prov = json.loads((d / "train" / "channels.json").read_text(encoding="utf-8"))
    version = prov["provenance"]["dataset_version"]
    path = REPO / "dataset" / "versions" / version / "manifest.csv"
    if not path.exists():
        pytest.skip(f"artefakt powołuje się na {version}, którego nie ma na dysku")
    with path.open(newline="", encoding="utf-8") as fh:
        return version, list(csv.DictReader(fh))


def test_grupy_nie_przekraczaja_splitow(artifact):
    """Żadna grupa źródłowa nie może być w dwóch splitach naraz."""
    _, rows = artifact
    gdzie = defaultdict(set)
    for split in SPLITS:
        for r in rows[split]:
            gdzie[r["group_id"]].add(split)
    przeciek = {g: s for g, s in gdzie.items() if len(s) > 1}
    assert not przeciek, (
        f"{len(przeciek)} grup w więcej niż jednym splicie, np. "
        + ", ".join(f"{g}: {sorted(przeciek[g])}" for g in sorted(przeciek)[:3]))


def test_brak_clock_tick_w_pozytywach(artifact):
    """ESC-50 target 38 to clock_tick, 39 to glass_breaking."""
    _, rows = artifact
    zle = [r["filepath"] for split in SPLITS for r in rows[split]
           if r["label"] == "positive" and r["source"] == "esc50"
           and not re.search(r"-39\.wav$", r["filepath"])]
    assert not zle, f"{len(zle)} plików ESC-50 spoza target 39 w klasie pozytywnej: {zle[:3]}"


def test_etykiety_zgodne_z_manifestem(artifact, manifest):
    """Etykieta w artefakcie == etykieta w manifeście (to samo, co K9)."""
    _, rows = artifact
    version, man_rows = manifest
    man = {r["filepath"]: r["label"] for r in man_rows}
    drift = [r["filepath"] for split in SPLITS for r in rows[split]
             if man.get(r["filepath"], r["label"]) != r["label"]]
    assert not drift, f"{len(drift)} plików ma etykietę inną niż w {version}: {drift[:3]}"


def test_split_zgodny_z_manifestem(artifact, manifest):
    """Plik trafił do tego splitu, który przypisał mu manifest."""
    _, rows = artifact
    version, man_rows = manifest
    man = {r["filepath"]: r["split"] for r in man_rows}
    zle = [(r["filepath"], split) for split in SPLITS for r in rows[split]
           if man.get(r["filepath"], split) != split]
    assert not zle, f"{len(zle)} plików w innym splicie niż w {version}: {zle[:3]}"


def test_klasy_przeplecione_w_strumieniu(artifact):
    """Kolejność kodowania nie może korelować z etykietą.

    Enkoder ma jeden ciągły stan przez cały zbiór, więc gdy negatywy idą przed
    pozytywami, stan floor/MAD niesie informację o klasie. Sprawdzamy średnią
    pozycję każdej klasy w strumieniu: obie mają wyjść blisko 0.5.
    """
    _, rows = artifact
    for split in SPLITS:
        n = len(rows[split])
        if n < 20:
            continue
        srednia = {}
        for etykieta in ("positive", "negative"):
            poz = [i / (n - 1) for i, r in enumerate(rows[split])
                   if r["label"] == etykieta]
            if poz:
                srednia[etykieta] = sum(poz) / len(poz)
        for etykieta, v in srednia.items():
            assert 0.40 <= v <= 0.60, (
                f"{split}: {etykieta} ma średnią pozycję {v:.3f} w strumieniu "
                f"(oczekiwane ~0.5); klasy nie są przeplecione")


def test_kazdy_plik_ma_swoj_csv(artifact):
    """files.csv wymienia dokładnie te CSV-y, które leżą w katalogu."""
    d, rows = artifact
    for split in SPLITS:
        na_dysku = {p.name for p in (d / split).glob("*.csv")} - {"files.csv"}
        w_spisie = {r["csv"] for r in rows[split]}
        assert na_dysku == w_spisie, (
            f"{split}: {len(na_dysku - w_spisie)} plików bez wpisu, "
            f"{len(w_spisie - na_dysku)} wpisów bez pliku")


def test_liczby_zgadzaja_sie_z_manifestem(artifact, manifest):
    """Liczba klipów per split/klasa == manifest minus pliki zużyte na warmup."""
    _, rows = artifact
    version, man_rows = manifest
    art = Counter((split, r["label"]) for split in SPLITS for r in rows[split])
    man = Counter((r["split"], r["label"]) for r in man_rows)
    braki = {k: man[k] - art[k] for k in man if man[k] != art[k]}
    nadmiar = {k: -v for k, v in braki.items() if v < 0}
    assert not nadmiar, f"artefakt ma więcej klipów niż {version}: {nadmiar}"
    # Ubytki są dopuszczalne tylko z dwóch znanych powodów: pliki zużyte na warmup
    # floora (odrzucane z premedytacją) i pliki za krótkie na jedną ramkę.
    assert all(v < 50 for v in braki.values()), \
        f"za duży ubytek klipów wobec {version}: {braki}"
