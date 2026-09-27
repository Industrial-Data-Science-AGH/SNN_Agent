#!/usr/bin/env python3
"""
freeze_m1.py -- M1 punkt 1: zamrozenie train/val/test, wersji enkodera i celu
wyboru modelu na walidacji, z zachowaniem lineage source/group_id.

Uruchamiac z katalogu master_pipeline/ (tak jak pipeline.py):

    python3 freeze_m1.py --config config.json \
        --encoder-profile ../contracts/fixtures/encoder-profile.json \
        --out freeze_manifest.json

Nie wymysla zadnych hashy -- liczy je z realnych plikow na dysku. Jesli
encoder-profile.json juz ma wlasny `config_sha256`, ten skrypt go tylko
przepisuje (nie liczy drugi raz) -- Kacper juz go policzyl, nie ma po co
duplikowac logiki hashowania w dwoch miejscach.
"""
import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone


def sha256_of_file(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_dataset_dir(root: str) -> dict:
    """Hashuje CALA zawartosc katalogu (rekurencyjnie): kazdy plik po jego
    sciezce wzglednej + sha256 tresci, posortowane, potem zhashowane razem.
    Deterministyczne bez wzgledu na kolejnosc listowania systemu plikow.
    Zwraca {"n_files": int, "manifest_sha256": str, "per_file": {...}} --
    per_file trzymamy, zeby przy sporze o "co sie zmienilo" nie zgadywac."""
    if not os.path.isdir(root):
        raise FileNotFoundError(f"katalog datasetu nie istnieje: {root}")

    entries = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            full = os.path.join(dirpath, fn)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            entries.append((rel, full))
    entries.sort(key=lambda x: x[0])

    if not entries:
        raise FileNotFoundError(f"katalog datasetu jest pusty: {root}")

    per_file = {}
    combined = hashlib.sha256()
    for rel, full in entries:
        file_hash = sha256_of_file(full)
        per_file[rel] = file_hash
        combined.update(rel.encode("utf-8"))
        combined.update(b"\0")
        combined.update(file_hash.encode("ascii"))
        combined.update(b"\n")

    return {
        "path": root,
        "n_files": len(entries),
        "manifest_sha256": combined.hexdigest(),
        "per_file_sha256": per_file,
    }


def load_encoder_hash(profile_path: str) -> dict:
    """Czyta contracts/fixtures/encoder-profile.json. Jesli ma config_sha256,
    przepisuje go 1:1 (Kacper juz go policzyl -- patrz notatka w docstringu
    modulu). Jesli go brakuje, liczy sha256 calego pliku jako fallback i
    jawnie to oznacza, zeby nikt nie pomyslal ze to ten sam hash co u Kacpra."""
    if not os.path.exists(profile_path):
        raise FileNotFoundError(
            f"nie znaleziono encoder-profile.json: {profile_path} -- "
            "podaj poprawna sciezke przez --encoder-profile"
        )
    with open(profile_path, "r", encoding="utf-8") as f:
        profile = json.load(f)

    result = {
        "profile_path": profile_path,
        "profile_id": profile.get("profile_id"),
        "encoder_variant": profile.get("encoder_variant"),
    }

    if "config_sha256" in profile:
        result["encoder_hash"] = profile["config_sha256"]
        result["encoder_hash_source"] = "config_sha256 z encoder-profile.json (obliczony przez Kacpra)"
    else:
        result["encoder_hash"] = sha256_of_file(profile_path)
        result["encoder_hash_source"] = (
            "FALLBACK: sha256 calego pliku (encoder-profile.json nie mial pola "
            "config_sha256) -- NIE jest to ten sam hash co ewentualny hash "
            "Kacpra, potwierdz zanim uzyjesz produkcyjnie"
        )

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--encoder-profile", default="../contracts/fixtures/encoder-profile.json")
    ap.add_argument("--out", default="freeze_manifest.json")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from pipeline_config import PipelineConfig

    config = PipelineConfig.from_json(args.config)
    project_root = os.path.dirname(os.getcwd())

    print("[freeze] Hashuje train/val/test...")
    train_abs = os.path.join(project_root, config.data.train)
    val_abs = os.path.join(project_root, config.data.val)
    test_abs = os.path.join(project_root, config.data.test)

    splits = {
        "train": hash_dataset_dir(train_abs),
        "val": hash_dataset_dir(val_abs),
        "test": hash_dataset_dir(test_abs),
    }
    for name, info in splits.items():
        print(f"  {name}: {info['n_files']} plikow, manifest_sha256={info['manifest_sha256'][:16]}...")

    print("[freeze] Czytam encoder-profile...")
    encoder = load_encoder_hash(args.encoder_profile)
    print(f"  encoder_hash={encoder['encoder_hash'][:16]}... "
          f"(profile_id={encoder['profile_id']}, variant={encoder['encoder_variant']})")

    freeze_manifest = {
        "freeze_schema_version": "1.0",
        "milestone": "M1 punkt 1",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": config.seed,

        "splits": {
            name: {
                "path": info["path"],
                "n_files": info["n_files"],
                "manifest_sha256": info["manifest_sha256"],
                # per_file_sha256 celowo pominiete tutaj (za duze na przegladanie
                # recznie) -- trzymane w osobnym pliku obok, patrz --out + '.files.json'
            }
            for name, info in splits.items()
        },

        "encoder": {
            "profile_id": encoder["profile_id"],
            "encoder_variant": encoder["encoder_variant"],
            "encoder_hash": encoder["encoder_hash"],
            "encoder_hash_source": encoder["encoder_hash_source"],
            "note": (
                "profile_id 'encoder_v3' i encoder_variant 'encoder_v2_swap.ino' "
                "to TEN SAM zamrozony wariant (potwierdzone przez Kacpra "
                "25.09.2026) -- 'v3' to wewnetrzny numer wersji configu, nie "
                "osobny firmware."
            ),
        },

        "selection_target": {
            "metric": config.ga.fitness_metric,
            "evaluated_on": "val",
            "note": (
                "Cel wyboru modelu na walidacji (M1 punkt 1). Zgodny z "
                "GAConfig.fitness_metric po poprawce literowki "
                "'recal_fa' -> 'recall_fa' (25.09.2026, Marcel)."
            ),
        },
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(freeze_manifest, f, indent=2, ensure_ascii=False)

    files_out = args.out.rsplit(".", 1)[0] + ".files.json"
    with open(files_out, "w", encoding="utf-8") as f:
        json.dump({name: info["per_file_sha256"] for name, info in splits.items()}, f, indent=2)

    print(f"\n[freeze] Zapisano: {args.out}")
    print(f"[freeze] Zapisano (per-plik, do audytu): {files_out}")
    print("[freeze] M1 punkt 1: gotowe do commitu -- to jest artefakt, ktory")
    print("         'pozwala odtworzyc selekcje' zgodnie z kryterium odbioru M1.")


if __name__ == "__main__":
    main()
    