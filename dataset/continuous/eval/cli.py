#!/usr/bin/env python3
"""
cli.py — generator ciągłego datasetu ewaluacyjnego z dokładnie 5 zdarzeniami
rozbicia szkła (zadanie "Kacper" w master pipeline Marcela).

Przykłady:

    # jeden 10-minutowy strumień, tło z ESC-50, szkło z VOICe (target_test)
    python -m continuous_eval.cli \\
        --glass-annotation-dir dataset/clean/clean/annotation \\
        --glass-audio-root dataset/clean/clean/audio \\
        --glass-allowed-stems dataset/clean/clean/target/synthetic_target_test.txt \\
        --background-dir data/ESC-50-master/audio \\
        --seed 42 \\
        --out-dir out/

    # 3 warianty z różnymi seedami, jak wymaga zakres zadania
    python -m continuous_eval.cli \\
        --glass-annotation-dir dataset/clean/clean/annotation \\
        --glass-audio-root dataset/clean/clean/audio \\
        --glass-allowed-stems dataset/clean/clean/target/synthetic_target_test.txt \\
        --background-dir data/ESC-50-master/audio \\
        --seeds 42 43 44 \\
        --out-dir out/

Uwaga o rozłączności danych: ten skrypt NIE sprawdza automatycznie, czy pliki
wskazane przez --glass-annotation-dir / --background-dir pokrywają się z
danymi treningowymi. Odpowiedzialność za podanie właściwej (rozłącznej z
treningiem) ścieżki spoczywa na wywołującym — patrz README, sekcja
"Co generator świadomie NIE robi".
"""
from __future__ import annotations

import argparse
import os
import sys

from .annotations import collect_glass_clips, read_stem_list
from .audio_io import AudioStandard, write_audio
from .manifest import build_manifest_dict, write_manifest
from .stream_builder import GENERATOR_VERSION, N_EVENTS, build_stream, collect_background_pool
from .validate import validate_pair, ValidationError


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generator ciągłego datasetu ewaluacyjnego z 5 zdarzeniami szkła."
    )
    p.add_argument("--glass-annotation-dir", required=True,
                    help="katalog z synthetic_XXX.txt (adnotacje VOICe)")
    p.add_argument("--glass-audio-root", required=True,
                    help="katalog z synthetic_XXX.wav odpowiadającymi adnotacjom")
    p.add_argument("--glass-allowed-stems", default=None,
                    help="opcjonalna lista dozwolonych plików źródłowych szkła "
                         "(np. dataset/clean/clean/target/synthetic_target_test.txt). "
                         "Bez tej flagi używane są WSZYSTKIE pliki w "
                         "--glass-annotation-dir — patrz README o rozłączności danych.")
    p.add_argument("--glassbreak-mode", choices=["clean", "background"], default="clean",
                    help="clean (domyślnie): tylko zdarzenia glassbreak bez nakładki "
                         "na gunshot/babycry. background: dopuszcza nakładki "
                         "(zdarzenie testowe może zawierać dodatkowy dźwięk w tle).")
    p.add_argument("--background-dir", action="append", required=True, dest="background_dirs",
                    help="katalog z plikami .wav do użycia jako tło (można podać "
                         "wielokrotnie, np. ESC-50 i DataSEC naraz)")
    p.add_argument("--duration-s", type=float, default=600.0,
                    help="długość strumienia w sekundach (domyślnie 600 = 10 min)")
    p.add_argument("--min-gap-s", type=float, default=2.0,
                    help="minimalny odstęp między zdarzeniami szkła")
    p.add_argument("--edge-margin-s", type=float, default=1.0,
                    help="margines od początku/końca strumienia, w którym "
                         "zdarzenia nie mogą się zaczynać/kończyć")
    p.add_argument("--event-gain-db-min", type=float, default=-3.0)
    p.add_argument("--event-gain-db-max", type=float, default=3.0)
    p.add_argument("--out-dir", required=True, help="katalog wyjściowy")
    p.add_argument("--out-prefix", default="continuous_eval",
                    help="prefiks nazwy pliku wyjściowego (domyślnie continuous_eval)")

    seed_group = p.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int, help="pojedynczy seed")
    seed_group.add_argument("--seeds", type=int, nargs="+", help="wiele seedów naraz")

    p.add_argument("--skip-validate", action="store_true",
                    help="pomiń automatyczną walidację po wygenerowaniu (niezalecane)")
    return p


def generate_one(args, seed: int) -> tuple[str, str]:
    allowed_stems = None
    if args.glass_allowed_stems:
        allowed_stems = read_stem_list(args.glass_allowed_stems)

    glass_clips = collect_glass_clips(
        args.glass_annotation_dir, allowed_stems, mode=args.glassbreak_mode
    )
    if len(glass_clips) < N_EVENTS:
        print(
            f"[fail] tylko {len(glass_clips)} kandydujących klipów glassbreak "
            f"(mode={args.glassbreak_mode}), potrzeba >= {N_EVENTS}",
            file=sys.stderr,
        )
        sys.exit(1)

    background_pool = collect_background_pool(args.background_dirs)

    standard = AudioStandard()
    stream = build_stream(
        duration_s=args.duration_s,
        glass_clips=glass_clips,
        audio_root_for_glass=args.glass_audio_root,
        background_pool=background_pool,
        seed=seed,
        min_gap_s=args.min_gap_s,
        edge_margin_s=args.edge_margin_s,
        event_gain_db_range=(args.event_gain_db_min, args.event_gain_db_max),
        standard=standard,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    audio_name = f"{args.out_prefix}_seed{seed}.wav"
    manifest_name = f"{args.out_prefix}_seed{seed}.manifest.json"
    audio_path = os.path.join(args.out_dir, audio_name)
    manifest_path = os.path.join(args.out_dir, manifest_name)

    write_audio(audio_path, stream.audio, standard)

    manifest = build_manifest_dict(
        stream=stream,
        audio_path=audio_path,
        seed=seed,
        glassbreak_mode=args.glassbreak_mode,
        min_gap_s=args.min_gap_s,
        edge_margin_s=args.edge_margin_s,
        event_gain_db_range=(args.event_gain_db_min, args.event_gain_db_max),
        background_dirs=args.background_dirs,
        glass_audio_root=args.glass_audio_root,
        glass_allowed_stems_file=args.glass_allowed_stems,
    )
    write_manifest(manifest, manifest_path)

    print(f"[ok] seed={seed} -> {audio_path}")
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
            print(f"[fail] walidacja nie przeszła: {e}", file=sys.stderr)
            sys.exit(1)

    return audio_path, manifest_path


def main() -> None:
    args = build_arg_parser().parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]

    print(f"[generator] wersja {GENERATOR_VERSION}, tryb glassbreak={args.glassbreak_mode}, "
          f"{len(seeds)} wariant(ów), {args.duration_s:.0f}s każdy")

    for seed in seeds:
        generate_one(args, seed)


if __name__ == "__main__":
    main()
