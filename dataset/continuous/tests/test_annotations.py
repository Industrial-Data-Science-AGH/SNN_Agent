"""
test_annotations.py — testy parsera adnotacji i ekstrakcji zdarzeń glassbreak
(clean vs background mode), na podstawie fikstury zbudowanej z fragmentu
prawdziwej adnotacji synthetic_001.txt podanej w konwersacji.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from eval.annotations import (
    collect_glass_clips,
    extract_glass_clips,
    parse_annotation_file,
    read_stem_list,
)

# Fragment prawdziwej adnotacji synthetic_001 (pierwsze linie z konwersacji).
# Zawiera zarówno glassbreak nachodzący na inne klasy (4.00-5.36 nachodzi na
# gunshot 3.00-10.04), jak i potencjalnie "czyste" wystąpienia.
SAMPLE_ANNOTATION = """1.00\t4.40\tbabycry
3.00\t3.78\tgunshot
4.00\t5.36\tglassbreak
5.78\t6.20\tgunshot
6.36\t6.92\tglassbreak
7.20\t10.04\tgunshot
7.40\t9.28\tbabycry
8.92\t9.30\tglassbreak
100.04\t101.42\tglassbreak
"""
# Ostatnia linia (100.04-101.42) jest celowo daleko od wszystkiego innego -> czysta.


@pytest.fixture()
def annotation_path(tmp_path):
    p = tmp_path / "synthetic_001.txt"
    p.write_text(SAMPLE_ANNOTATION, encoding="utf-8")
    return str(p)


def test_parse_annotation_file_counts_all_rows(annotation_path):
    events = parse_annotation_file(annotation_path)
    assert len(events) == 9
    assert events[0].label == "babycry"
    assert events[2].label == "glassbreak"


def test_clean_mode_excludes_contaminated(annotation_path):
    clips = extract_glass_clips(annotation_path, mode="clean")
    # 3 glassbreak w danych: 4.00-5.36 (nachodzi na gunshot), 6.36-6.92
    # (nachodzi na gunshot 5.78-6.20? sprawdzmy: 6.36 > 6.20 -> NIE nachodzi;
    # ale nachodzi na babycry 1.00-4.40? nie, 6.36>4.40; nachodzi na gunshot
    # 7.20-10.04? 6.92<7.20 nie), 8.92-9.30 (nachodzi na gunshot 7.20-10.04 TAK
    # i babycry 7.40-9.28 TAK), 100.04-101.42 (izolowany, czysty)
    # 4.00-5.36 nachodzi na babycry 1.00-4.40 (4.00 < 4.40) -> skażony
    # 6.36-6.92 -> nie nachodzi na nic (gunshot 5.78-6.20 skończył się, gunshot
    #   7.20-10.04 jeszcze nie zaczął, babycry 7.40-9.28 jeszcze nie) -> czysty
    # 8.92-9.30 nachodzi na gunshot 7.20-10.04 i babycry 7.40-9.28 -> skażony
    # 100.04-101.42 -> izolowany -> czysty
    all_bg = extract_glass_clips(annotation_path, mode="background")
    by_start = {round(c.start_s, 2): c for c in all_bg}
    assert by_start[4.00].is_contaminated
    assert by_start[8.92].is_contaminated
    assert not by_start[6.36].is_contaminated
    assert not by_start[100.04].is_contaminated
    # w clean-mode zwracamy tylko nieskażone
    assert all(not c.is_contaminated for c in clips)
    clean_starts = {round(c.start_s, 2) for c in clips}
    assert 4.00 not in clean_starts
    assert 8.92 not in clean_starts


def test_background_mode_includes_all_glassbreak(annotation_path):
    clips = extract_glass_clips(annotation_path, mode="background")
    assert len(clips) == 4  # wszystkie glassbreak z fikstury
    contaminated_count = sum(1 for c in clips if c.is_contaminated)
    assert contaminated_count >= 2  # co najmniej 4.00 i 8.92


def test_overlapping_labels_recorded(annotation_path):
    clips = extract_glass_clips(annotation_path, mode="background")
    by_start = {round(c.start_s, 2): c for c in clips}
    assert "babycry" in by_start[4.00].overlapping_labels  # 1.00-4.40 nachodzi na 4.00-5.36
    assert set(by_start[8.92].overlapping_labels) == {"gunshot", "babycry"}
    assert by_start[100.04].overlapping_labels == ()


def test_malformed_line_raises(tmp_path):
    p = tmp_path / "synthetic_002.txt"
    p.write_text("1.00\t2.00\tnotaclass\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_annotation_file(str(p))


def test_end_before_start_raises(tmp_path):
    p = tmp_path / "synthetic_003.txt"
    p.write_text("5.00\t2.00\tglassbreak\n", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_annotation_file(str(p))


def test_read_stem_list(tmp_path):
    p = tmp_path / "synthetic_target_test.txt"
    p.write_text("synthetic_014.wav\nsynthetic_027.wav\n\nsynthetic_014.wav\n", encoding="utf-8")
    stems = read_stem_list(str(p))
    assert stems == {"synthetic_014", "synthetic_027"}


def test_collect_glass_clips_filters_by_allowed_stems(tmp_path):
    (tmp_path / "synthetic_001.txt").write_text(SAMPLE_ANNOTATION, encoding="utf-8")
    (tmp_path / "synthetic_002.txt").write_text(
        "50.00\t51.00\tglassbreak\n", encoding="utf-8"
    )
    clips_all = collect_glass_clips(str(tmp_path), allowed_stems=None, mode="background")
    stems_all = {c.source_stem for c in clips_all}
    assert stems_all == {"synthetic_001", "synthetic_002"}

    clips_filtered = collect_glass_clips(
        str(tmp_path), allowed_stems={"synthetic_002"}, mode="background"
    )
    assert all(c.source_stem == "synthetic_002" for c in clips_filtered)


def test_collect_glass_clips_empty_dir_raises(tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        collect_glass_clips(str(empty_dir), allowed_stems=None, mode="clean")
