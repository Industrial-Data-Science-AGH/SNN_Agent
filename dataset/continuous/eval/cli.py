#!/usr/bin/env python3
"""
cli.py — generator ciągłego datasetu ewaluacyjnego z dokładnie 5 zdarzeniami
rozbicia szkła (zadanie "Kacper" w master pipeline Marcela).

K3: każdy --seed/--seeds generuje PARĘ (continuous-val, continuous-test) —
nie pojedynczy strumień. Val jest budowany pierwszy i jego wybory (tło ESC-50
po group_id, pliki źródłowe szkła VOICe po source_stem) są wykluczane z puli
testu — patrz eval.stream_builder.build_val_test_pair. Wynik dla seeda N:

    continuous_eval_seedN_val.wav   + continuous_eval_seedN_val.manifest.json
    continuous_eval_seedN_test.wav  + continuous_eval_seedN_test.manifest.json

Przykłady:

    python -m continuous_eval.cli \\
        --glass-annotation-dir dataset/clean/clean/annotation \\
        --glass-audio-root dataset/clean/clean/audio \\
        --glass-allowed-stems dataset/clean/clean/target/synthetic_target_test.txt \\
        --train-stems-files dataset/clean/clean/source/synthetic_source_training.txt \\
                             dataset/clean/clean/source/synthetic_source_validation.txt \\
        --train-manifest dataset/versions/v2.0.0/manifest.csv \\
        --background-dir data/ESC-50-master/audio \\
        --seeds 42 43 44 \\
        --out-dir out/

Uwaga o rozłączności danych: ten skrypt NIE sprawdza automatycznie, czy pliki
wskazane przez --glass-annotation-dir / --background-dir pokrywają się z
danymi treningowymi POZA tym, co robią --train-stems-files / --train-manifest.
Odpowiedzialność za podanie właściwych ścieżek spoczywa na wywołującym.
"""
from __future__ import annotations

import argparse
import os
import sys

from .annotations import collect_glass_clips, read_stem_list, check_eval_train_overlap
from .audio_io import AudioStandard, write_audio
from .manifest import build_manifest_dict, write_manifest
from .stream_builder import (
    GENERATOR_VERSION, N_EVENTS, build_val_test_pair, validate_val_test_disjoint,
)
from .validate import validate_pair, ValidationError


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generator pary continuous-val/continuous-test (5 zdarzeń szkła każdy)."
    )
    p.add_argument("--glass-annotation-dir", required=True,
                    help="katalog z synthetic_XXX.txt (adnotacje VOICe)")
    p.add_argument("--glass-audio-root", required=True,
                    help="katalog z synthetic_XXX.wav odpowiadającymi adnotacjom")
    p.add_argument("--glass-allowed-stems", default=None,
                    help="opcjonalna lista dozwolonych plików źródłowych szkła "
                         "(np. dataset/clean/clean/target/synthetic_target_test.txt). "
                         "Bez tej flagi używane są WSZYSTKIE pliki w "
                         "--glass-annotation-dir.")
    p.add_argument("--train-stems-files", nargs="+", default=None,
                   help="listy plików treningowych do sprawdzenia rozłączności "
                        "(np. source_training.txt source_validation.txt). "
                        "Jeśli którykolwiek stem wystąpi też w puli szkła — ValueError.")
    p.add_argument("--train-manifest", required=True,
                   help="ścieżka do manifest.csv (np. dataset/versions/v2.0.0/manifest.csv). "
                        "group_id ze split=train (a nie tylko train) są wykluczone z tła — "
                        "patrz eval.annotations._load_train_group_ids.")
    p.add_argument("--glassbreak-mode", choices=["clean", "background"], default="clean",
                    help="clean (domyślnie): tylko zdarzenia glassbreak bez nakładki "
                         "na gunshot/babycry. background: dopuszcza nakładki.")
    p.add_argument("--background-dir", action="append", required=True, dest="background_dirs",
                    help="katalog z plikami .wav do użycia jako tło (można podać wielokrotnie)")
    p.add_argument("--duration-s", type=float, default=600.0,
                    help="długość KAŻDEGO z dwóch strumieni w sekundach (domyślnie 600 = 10 min)")
    p.add_argument("--min-gap-s", type=float, default=10.0,
                    help="minimalny odstęp między zdarzeniami szkła")
    p.add_argument("--end-margin-s", type=float, default=10.0,
                    help="margines od końca strumienia, w którym zdarzenia nie mogą się kończyć")
    p.add_argument("--warmup-s", type=float, default=30.0,
                   help="czas rozgrzewki (samo tło) na początku strumienia, "
                        "wykluczony z liczenia FA/h")
    p.add_argument("--event-gain-db-min", type=float, default=-3.0)
    p.add_argument("--event-gain-db-max", type=float, default=3.0)
    p.add_argument("--out-dir", required=True, help="katalog wyjściowy")
    p.add_argument("--out-prefix", default="continuous_eval",
                    help="prefiks nazwy pliku wyjściowego (domyślnie continuous_eval)")

    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, help="pojedynczy seed nadrzędny")
    seed_group.add_argument("--seeds", type=int, nargs="+", help="wiele seedów nadrzędnych naraz")

    p.add_argument("--skip-validate", action="store_true",
                    help="pomiń automatyczną walidację po wygenerowaniu (niezalecane)")
    return p


def _write_one(args, stream, *, seed: int, role: str, parent_seed: int, overlap_check: dict) -> tuple[str, str]:
    audio_name = f"{args.out_prefix}_seed{parent_seed}_{role}.wav"
    manifest_name = f"{args.out_prefix}_seed{parent_seed}_{role}.manifest.json"
    audio_path = os.path.join(args.out_dir, audio_name)
    manifest_path = os.path.join(args.out_dir, manifest_name)

    write_audio(audio_path, stream.audio, AudioStandard())

    manifest = build_manifest_dict(
        stream=stream,
        audio_path=audio_path,
        seed=seed,
        role=role,
        parent_seed=parent_seed,
        glassbreak_mode=args.glassbreak_mode,
        min_gap_s=args.min_gap_s,
        warmup_s=args.warmup_s,
        end_margin_s=args.end_margin_s,
        event_gain_db_range=(args.event_gain_db_min, args.event_gain_db_max),
        background_dirs=args.background_dirs,
        glass_audio_root=args.glass_audio_root,
        glass_allowed_stems_files=args.glass_allowed_stems,
        overlap_check=overlap_check,
    )
    write_manifest(manifest, manifest_path)

    print(f"[ok] seed={parent_seed} role={role} -> {audio_path}")
    print(f"     manifest -> {manifest_path}")
    for e in manifest["events"]:
        tag = " (skażone: " + ",".join(e["overlapping_labels"]) + ")" if e["overlapping_labels"] else ""
        print(f"     event[{e['index']}] {e['start_s']:.2f}s - {e['end_s']:.2f}s "
              f"<- {e['source_stem']}{tag}")

    if not args.skip_validate:
        try:
            validate_pair(audio_path, manifest_path)
            print("     walidacja: OK")
        except ValidationError as e:
            print(f"[fail] walidacja nie przeszła ({role}): {e}", file=sys.stderr)
            sys.exit(1)

    return audio_path, manifest_path


def generate_one(args, seed: int) -> None:
    allowed_stems = None
    if args.glass_allowed_stems:
        allowed_stems = read_stem_list(args.glass_allowed_stems)

    glass_clips = collect_glass_clips(
        args.glass_annotation_dir, allowed_stems, mode=args.glassbreak_mode
    )
    # K3: val+test razem potrzebują 2x N_EVENTS unikalnych source_stem (rozłącznych) —
    # samo len < N_EVENTS przepuściłoby dalej i wywaliło się dopiero w build_val_test_pair
    # z mniej czytelnym komunikatem, więc sprawdzamy z grubsza już tutaj.
    if len(glass_clips) < 2 * N_EVENTS:
        print(
            f"[fail] tylko {len(glass_clips)} kandydujących klipów glassbreak "
            f"(mode={args.glassbreak_mode}), potrzeba >= {2 * N_EVENTS} (val+test rozłącznie)",
            file=sys.stderr,
        )
        sys.exit(1)

    eval_stems = {c.source_stem for c in glass_clips}
    overlap_report = check_eval_train_overlap(
        eval_stems, args.train_stems_files or [], bg_group_ids=None, train_manifest_csv=None,
    )

    try:
        pair = build_val_test_pair(
            seed=seed,
            duration_s=args.duration_s,
            glass_clips=glass_clips,
            audio_root_for_glass=args.glass_audio_root,
            background_dirs=args.background_dirs,
            train_manifest_csv=args.train_manifest,
            min_gap_s=args.min_gap_s,
            warmup_s=args.warmup_s,
            end_margin_s=args.end_margin_s,
            event_gain_db_range=(args.event_gain_db_min, args.event_gain_db_max),
        )
    except RuntimeError as e:
        print(f"[fail] seed={seed}: {e}", file=sys.stderr)
        sys.exit(1)

    validate_val_test_disjoint(pair)  # niezależna kontrola z zewnątrz, po zbudowaniu obu

    os.makedirs(args.out_dir, exist_ok=True)

    val_overlap = dict(overlap_report)
    val_overlap["background"] = {"note": "group_id z train wykluczone na etapie doboru tła "
                                          "(collect_background_pool), nie tylko raportowane"}
    _write_one(args, pair.val, seed=pair.val_seed, role="val", parent_seed=seed,
               overlap_check=val_overlap)

    test_overlap = dict(overlap_report)
    test_overlap["background"] = {
        "note": "group_id z train ORAZ z continuous-val (ten sam --seed) wykluczone",
        "excluded_val_group_ids": sorted(pair.val_background_group_ids),
        "excluded_val_glass_source_stems": sorted(pair.val_glass_source_stems),
    }
    _write_one(args, pair.test, seed=pair.test_seed, role="test", parent_seed=seed,
               overlap_check=test_overlap)


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]

    print(f"[generator] wersja {GENERATOR_VERSION}, tryb glassbreak={args.glassbreak_mode}, "
          f"{len(seeds)} par(y) val+test, {args.duration_s:.0f}s każdy strumień")

    for seed in seeds:
        generate_one(args, seed)


if __name__ == "__main__":
    main()