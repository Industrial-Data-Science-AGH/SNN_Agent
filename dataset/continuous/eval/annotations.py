"""
annotations.py — parser adnotacji VOICe i ekstrakcja zdarzeń glassbreak.

Format źródłowy (dataset/clean/clean/annotation/synthetic_XXX.txt), TSV bez
nagłówka, jedna linia = jedno zdarzenie w miksie:

    start_s<TAB>end_s<TAB>klasa

gdzie klasa in {glassbreak, gunshot, babycry}. Zdarzenia w obrębie jednego
pliku NACHODZĄ NA SIEBIE — to jest miks VOICe, nie osobne klipy (patrz
dyskusja w prompt.txt / stats.md: 3961/4444 zdarzeń glassbreak nachodzi na
gunshot/babycry przy pad 0.30 s).

Ten moduł dostarcza wyłącznie logikę "które fragmenty audio, oznaczone jako
glassbreak, wolno wyciąć jako zdarzenie testowe" — nie robi żadnego I/O na
audio.
"""
from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from typing import Literal

GlassbreakMode = Literal["clean", "background"]

GLASSBREAK = "glassbreak"
KNOWN_CLASSES = {"glassbreak", "gunshot", "babycry"}


@dataclass(frozen=True)
class AnnotatedEvent:
    start_s: float
    end_s: float
    label: str

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def overlaps(self, other: "AnnotatedEvent") -> bool:
        return self.start_s < other.end_s and other.start_s < self.end_s


@dataclass(frozen=True)
class GlassClip:
    """Jeden kandydujący fragment glassbreak w obrębie jednego pliku źródłowego."""
    source_stem: str          # np. "synthetic_001" (bez rozszerzenia)
    start_s: float
    end_s: float
    is_contaminated: bool     # czy nachodzi na gunshot/babycry w oryginale
    overlapping_labels: tuple[str, ...]

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s


def parse_annotation_file(path: str) -> list[AnnotatedEvent]:
    """Parsuje jeden plik synthetic_XXX.txt. Puste/białe linie są pomijane."""
    events: list[AnnotatedEvent] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != 3:
                # tolerujemy też separator spacjami wielokrotnymi, na wypadek
                # kopii pliku z innym whitespace'em
                parts = line.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{path}:{lineno}: oczekiwano 3 kolumn (start\\tend\\tklasa), "
                    f"dostano {parts!r}"
                )
            start_s, end_s, label = parts
            try:
                start_f, end_f = float(start_s), float(end_s)
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: nie-liczbowy czas: {parts!r}") from e
            if end_f <= start_f:
                raise ValueError(
                    f"{path}:{lineno}: end <= start ({start_f} .. {end_f})"
                )
            if label not in KNOWN_CLASSES:
                raise ValueError(
                    f"{path}:{lineno}: nieznana klasa {label!r}, "
                    f"oczekiwano jednej z {sorted(KNOWN_CLASSES)}"
                )
            events.append(AnnotatedEvent(start_f, end_f, label))
    return events


def _stem_from_annotation_path(path: str) -> str:
    # dataset/clean/clean/annotation/synthetic_001.txt -> synthetic_001
    return os.path.splitext(os.path.basename(path))[0]


def extract_glass_clips(
    annotation_path: str,
    mode: GlassbreakMode = "clean",
) -> list[GlassClip]:
    """Zwraca kandydujące fragmenty glassbreak z jednego pliku adnotacji.

    mode="clean":      tylko interwały glassbreak, które w oryginalnym miksie
                        NIE nachodzą czasowo na żadne zdarzenie gunshot/babycry.
                        Bezpieczniejsze dla ewaluacji: to co wstrzykujemy do
                        strumienia jest jednoznacznie "samo szkło", więc
                        recall/latency mierzą reakcję na szkło, a nie na
                        przypadkowy współwystępujący dźwięk.
    mode="background":  wszystkie interwały glassbreak, niezależnie od nakładek
                        (analogicznie do reguły "skażony pozytyw zachowujemy"
                        z dataset/versions/v2.0.0/stats.md). Zdarzenie testowe
                        może zawierać pod spodem gunshot/babycry — to jest
                        świadomie trudniejszy, bardziej realistyczny wariant.

    W obu trybach zwracany jest WYŁĄCZNIE natywny interwał czasowy zdarzenia
    glassbreak z adnotacji (bez paddingu — padding jest sprawą etapu miksowania
    w stream_builder, nie ekstrakcji).
    """
    events = parse_annotation_file(annotation_path)
    stem = _stem_from_annotation_path(annotation_path)
    glass_events = [e for e in events if e.label == GLASSBREAK]
    others = [e for e in events if e.label != GLASSBREAK]

    clips: list[GlassClip] = []
    for g in glass_events:
        overlapping = tuple(sorted({o.label for o in others if g.overlaps(o)}))
        contaminated = len(overlapping) > 0
        if mode == "clean" and contaminated:
            continue
        clips.append(
            GlassClip(
                source_stem=stem,
                start_s=g.start_s,
                end_s=g.end_s,
                is_contaminated=contaminated,
                overlapping_labels=overlapping,
            )
        )
    return clips


def collect_glass_clips(
    annotation_dir: str,
    allowed_stems: set[str] | None,
    mode: GlassbreakMode = "clean",
) -> list[GlassClip]:
    """Zbiera kandydujące klipy glassbreak z całego katalogu adnotacji.

    allowed_stems: jeśli podane, ograniczamy się do plików, których stem
    (np. "synthetic_007") znajduje się w tym zbiorze — to jest miejsce, przez
    które wchodzi wybór source/target split (patrz stream_builder.py, gdzie
    lista dozwolonych plików jest argumentem CLI, nie wbudowaną regułą).
    """
    paths = sorted(glob.glob(os.path.join(annotation_dir, "synthetic_*.txt")))
    if not paths:
        raise FileNotFoundError(
            f"brak plików adnotacji synthetic_*.txt w {annotation_dir}"
        )

    all_clips: list[GlassClip] = []
    for p in paths:
        stem = _stem_from_annotation_path(p)
        if allowed_stems is not None and stem not in allowed_stems:
            continue
        all_clips.extend(extract_glass_clips(p, mode=mode))

    if not all_clips:
        raise RuntimeError(
            f"0 kandydujących klipów glassbreak (mode={mode}) w {annotation_dir} "
            f"po zawężeniu do {len(allowed_stems) if allowed_stems is not None else 'wszystkich'} "
            f"plików źródłowych. Sprawdź --glassbreak-mode i listę dozwolonych plików."
        )
    return all_clips


def read_stem_list(path: str) -> set[str]:
    """Czyta listę nazw plików wav (jedna na linię, np. synthetic_007.wav)
    z pliku source_*.txt / target_*.txt i zwraca zbiór stemów bez rozszerzenia.
    """
    stems: set[str] = set()
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            stems.add(os.path.splitext(os.path.basename(line))[0])
    if not stems:
        raise ValueError(f"{path}: lista jest pusta")
    return stems
