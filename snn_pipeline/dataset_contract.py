# -*- coding: utf-8 -*-
"""
dataset_contract.py — jedno miejsce, w którym zdefiniowany jest KONTRAKT DANYCH.

Wszystko, co poniżej, jest wspólne dla buildera (`build_dataset_version.py`)
i walidatora (`validate_dataset.py`). Jeśli coś zmieniasz tutaj, zmieniasz
definicję zbioru — i musisz wydać nową WERSJĘ, bo stare artefakty przestają
być porównywalne.

Trzy rzeczy, które ten plik ustala:

  1. SCHEMAT manifestu — jakie kolumny ma każdy rekord.
  2. REGUŁA GRUPY — co znaczy „jedno nagranie źródłowe" dla każdego źródła.
     To jest jedyna obrona przed przeciekiem train/test i najważniejsza
     rzecz w całym pliku.
  3. TAKSONOMIA negatywów — czym różni się ciche tło od głośnego zdarzenia.
     Bez tego nie da się raportować fałszywych alarmów w podziale, który
     ma sens dla bramy zawsze-czuwającej.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# =============================================================================
# WERSJONOWANIE
# =============================================================================
# Semantyczne: MAJOR.MINOR.PATCH
#   MAJOR — zmienia się skład zbioru albo podział (stare wyniki nieporównywalne)
#   MINOR — dochodzą rekordy, podział istniejących bez zmian
#   PATCH — poprawki metadanych, żaden bajt audio się nie zmienia
# Wersji NIE WOLNO nadpisywać. Builder odmawia zapisu do istniejącego katalogu.
VERSION_RE = re.compile(r"^v\d+\.\d+\.\d+$")
VERSIONS_DIR = "dataset/versions"

# =============================================================================
# SCHEMAT MANIFESTU
# =============================================================================
MANIFEST_COLUMNS: List[str] = [
    "id",             # stabilne ID, wyliczone ze źródła i ścieżki (patrz make_id)
    "filepath",       # ścieżka względna do repo_root
    "sha256",         # suma kontrolna zawartości pliku
    "bytes",          # rozmiar w bajtach
    "label",          # positive | negative
    "kind",           # positive | stationary | loud_event | speech | animal
    "source",         # esc50 | datasec | voice
    "subclass",       # oryginalna klasa w źródle
    "group_id",       # NAGRANIE ŹRÓDŁOWE — jednostka podziału
    "split",          # train | val | test
    "duration_s",
    "sample_rate",
    "channels",
    "subtype",        # format próbki wg soundfile, np. PCM_16
    "license",
]

LABELS = ("positive", "negative")
KINDS = ("positive", "stationary", "loud_event", "speech", "animal")
SPLITS = ("train", "val", "test")

# =============================================================================
# LICENCJE
# =============================================================================
# CC BY-NC oznacza ZAKAZ użycia komercyjnego. Trzymamy to w manifeście, żeby
# dało się jednym zapytaniem zbudować podzbiór nadający się do produktu.
SOURCE_LICENSE: Dict[str, str] = {
    "esc50": "CC BY-NC 3.0",
    "datasec": "CC BY 4.0",
    "voice": "Other (Attribution)",
}
NONCOMMERCIAL_LICENSES = ("CC BY-NC 3.0",)

# =============================================================================
# AUDIO — co uznajemy za poprawne
# =============================================================================
# ŚWIADOMA DECYZJA: NIE transkodujemy i NIE normalizujemy amplitudy plików
# źródłowych. Powody:
#   * enkoder i tak resampluje do 19231 Hz (tyle ma ADC Arduino), więc drugi
#     resampling tylko traciłby jakość;
#   * `wav_to_adc_codes()` normalizuje szczytowo KAŻDY plik osobno, więc
#     wypalenie normalizacji w zbiorze skasowałoby bezwzględną głośność na
#     zawsze — a to jest cecha, którą realne urządzenie o stałym gainie widzi;
#   * kopia 11 GB audio w drugim formacie to koszt bez zysku.
# Zamiast tego: mierzymy parametry każdego pliku i walidator odrzuca to, co
# wypada poza zakres. Konwersja jest zadaniem czytającego, nie zbioru.
MIN_SAMPLE_RATE = 16000
MAX_CHANNELS = 2
MIN_DURATION_S = 0.15
MAX_DURATION_S = 600.0

# =============================================================================
# REGUŁA GRUPY — „co jest jednym nagraniem źródłowym"
# =============================================================================
# Grupa to jednostka podziału. Wszystkie pliki z jednej grupy trafiają do tego
# samego splitu, bo dzielą akustykę, tło i często tę samą próbkę zdarzenia.
# Podział po PLIKACH zamiast po grupach jest dokładnie tym błędem, który
# sprawił, że poprzednie metryki mierzyły pamięć zamiast umiejętności.

# Freesound: `Cos.<idUploadu>__<autor>__<tytul>.wav`. Jeden upload bywa pocięty
# na kilkanaście plików (396289 dał 16, 483590 dwanaście).
_FREESOUND_RE = re.compile(r"\.(\d{4,})__")
# ESC-50: `<fold>-<clipId>-<take>-<target>.wav`. Ujęcia A i B to TO SAMO nagranie.
_ESC50_RE = re.compile(r"^(\d+)-(\d+)-([A-Z])-(\d+)$")
# voice_extracted: `voiceglass_00000_<mix>_<start>-<koniec>.wav`
_VOICE_RE = re.compile(r"_(synthetic_\d+)_")


def group_id_for(source: str, path: Path) -> str:
    """Identyfikator NAGRANIA ŹRÓDŁOWEGO, z którego pochodzi plik.

    Reguła jest inna dla każdego źródła i to jest celowe — patrz komentarze
    przy wyrażeniach regularnych wyżej.
    """
    name, stem = path.name, path.stem

    if source == "esc50":
        m = _ESC50_RE.match(stem)
        # fold + clipId; take (A/B) CELOWO pomijamy — to dwa wycinki jednego nagrania
        return f"esc50_{m.group(1)}_{m.group(2)}" if m else f"esc50_{stem}"

    if source == "voice":
        m = _VOICE_RE.search(name)
        return f"voice_{m.group(1)}" if m else f"voice_{stem}"

    if source == "datasec":
        # DataSEC nazywa pliki `<Klasa>-NNN.wav` i każdy jest osobnym, wyciętym
        # przez autorów zdarzeniem — sprawdzone: 3226 plików = 3226 grup.
        # Końcówki `-NNN` NIE obcinamy: to numer nagrania w klasie, a nie numer
        # fragmentu jednego nagrania. Obcięcie zlepiłoby 109 niezależnych nagrań
        # szkła w jedną grupę i zmarnowało cały zbiór.
        m = _FREESOUND_RE.search(name)   # zapas, gdyby przyszła wersja użyła Freesound
        return f"datasec_fs{m.group(1)}" if m else f"datasec_{stem}"

    m = _FREESOUND_RE.search(name)
    return f"{source}_fs{m.group(1)}" if m else f"{source}_{stem}"


# =============================================================================
# TAKSONOMIA — czym jest dany dźwięk dla NASZEGO zadania
# =============================================================================
# `kind` odpowiada na pytanie, na które sam `label` nie odpowiada: JAKI to jest
# negatyw. Raport fałszywych alarmów rozbity na ciche tło / głośne zdarzenie /
# mowę jest główną metryką bramy zawsze-czuwającej, a dziś powstaje heurystyką
# na nazwach plików w `eval_stream.py`. Tu przenosimy to do danych.
#
# `stationary` = ciągłe tło bez ostrego ataku — zastępuje zbiór `notebooks/dataset`,
# który okazał się archiwalną pozostałością bez unikalnej zawartości.

ESC50_KIND: Dict[int, str] = {}
for _t in (10, 11, 12, 13, 16, 17, 18, 28, 35, 36, 38, 7):
    ESC50_KIND[_t] = "stationary"       # deszcz, fale, ogień, świerszcze, wiatr,
                                        # lanie wody, spłuczka, chrapanie, pralka,
                                        # odkurzacz, tykanie zegara, owady
for _t in (19, 22, 25, 30, 31, 32, 33, 34, 37, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 15, 27, 29):
    ESC50_KIND[_t] = "loud_event"       # burza, klaskanie, kroki, pukanie, myszka,
                                        # klawiatura, drzwi, puszka, budzik, śmigłowiec,
                                        # piła, syrena, klakson, silnik, pociąg, dzwony,
                                        # samolot, fajerwerki, piła ręczna, krople, mycie zębów, picie
for _t in (20, 21, 23, 24, 26):
    ESC50_KIND[_t] = "speech"           # płacz, kichanie, oddech, kaszel, śmiech
for _t in (0, 1, 2, 3, 4, 5, 6, 8, 9, 14):
    ESC50_KIND[_t] = "animal"

DATASEC_KIND: Dict[str, str] = {
    "Glass breaking": "positive",
    "Voices": "speech",
    # ciągła sceneria — nasze zastępstwo za tło stacjonarne
    "Wind turbine": "stationary",
    "Vehicle idling": "stationary",
    "Vacuum cleaner fan and hairdryer": "stationary",
    "Cicadas and crickets": "stationary",
    # głośne zdarzenia — najtrudniejsze negatywy
    "Thunder fireworks and gunshot": "loud_event",
    "Workshop": "loud_event",
    "Vehicle pass-by": "loud_event",
    "Lawn mower brush cutter and olive shaker": "loud_event",
    "Propeller aircrafts": "loud_event",
    "Sirens and alarms": "loud_event",
    "Jet aircrafts": "loud_event",
    "Music": "loud_event",
    "Train": "loud_event",
    "Bells": "loud_event",
    "Horn": "loud_event",
    # zwierzęta
    "Crows seagulls and magpies": "animal",
    "Dog barkings and howlings": "animal",
    "Birds": "animal",
    "Cat fights and moans": "animal",
    "Chicken coop": "animal",
}

VOICE_KIND: Dict[str, str] = {
    "glassbreak": "positive",
    "gunshot": "loud_event",
    "babycry": "speech",
}


# =============================================================================
# NARZĘDZIA
# =============================================================================

def make_id(source: str, relpath: str) -> str:
    """Stabilne ID rekordu.

    Liczone ze ŚCIEŻKI, nie z zawartości — dzięki temu przetrwa rekompresję
    albo zmianę formatu pliku, a jednocześnie jest identyczne na każdej
    maszynie. Za zgodność zawartości odpowiada osobna kolumna `sha256`.
    """
    h = hashlib.sha1(f"{source}|{relpath}".encode("utf-8")).hexdigest()
    return f"{source[:3]}_{h[:12]}"


def sha256_of(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def is_valid_version(v: str) -> bool:
    return bool(VERSION_RE.match(v))


def version_dir(repo_root: Path, version: str) -> Path:
    return repo_root / VERSIONS_DIR / version
