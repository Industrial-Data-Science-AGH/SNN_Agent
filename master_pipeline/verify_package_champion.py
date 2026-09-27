#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_champion_package.py -- M5 punkt 2: "Sprawdzić powtórne załadowanie w
czystym procesie i identyczne wyjście."

Ma być odpalany jako OSOBNY proces (nie import), świeżo: bez żadnego stanu
pozostałego po package_champion.py w tym samym interpreterze. Robi dwie
niezależne rzeczy:

1. Integralność: liczy sha256 KAŻDEGO pliku wymienionego w
   package_manifest.json i porównuje z tym, co manifest deklaruje -- wykrywa
   uszkodzenie/podmianę plików między spakowaniem a wczytaniem u Patryka/
   Wiktora.
2. Identyczne wyjście: ładuje checkpoint od zera (Genome.from_dict ->
   GenomeNet -> set_quantize(True) -> load_state_dict -- ta sama ścieżka co
   `_load_champion_model` w continuous_eval.py), przepuszcza DOKŁADNIE to
   samo wejście co golden_replay.json, i porównuje wyjście BIT-DO-BITU z
   zapisanym `golden_replay_output.json`. Różnica tutaj oznacza, że
   checkpoint/kod modelu nie odtwarza się deterministycznie -- pakiet NIE
   powinien iść dalej, dopóki się to nie wyjaśni.

Kod wyjścia: 0 = OK, 1 = jakakolwiek niezgodność.

Użycie:
    python3 verify_champion_package.py --package-dir packages/champion_XXXX
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List


def sha256_of_file(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_integrity(package_dir: str, manifest: Dict[str, Any]) -> List[str]:
    errors = []
    for artifact in manifest["artifacts"]:
        path = os.path.join(package_dir, artifact["filename"])
        if not os.path.exists(path):
            errors.append(f"BRAK PLIKU: {artifact['filename']}")
            continue
        actual = sha256_of_file(path)
        if actual != artifact["sha256"]:
            errors.append(
                f"SHA256 NIEZGODNY: {artifact['filename']} "
                f"(oczekiwano {artifact['sha256'][:16]}..., jest {actual[:16]}...)"
            )
    return errors


def verify_replay(package_dir: str) -> List[str]:
    errors = []
    golden_path = os.path.join(package_dir, "golden_replay.json")
    with open(golden_path, "r", encoding="utf-8") as f:
        golden = json.load(f)

    checkpoint_path = os.path.join(package_dir, "champion_checkpoint.pt")
    actual_ckpt_sha = sha256_of_file(checkpoint_path)
    if actual_ckpt_sha != golden["checkpoint_sha256"]:
        errors.append(
            f"checkpoint_sha256 w golden_replay.json nie zgadza się z plikiem "
            f"champion_checkpoint.pt na dysku (oczekiwano {golden['checkpoint_sha256'][:16]}..., "
            f"jest {actual_ckpt_sha[:16]}...) -- weryfikacja wyjścia poniżej byłaby bez sensu, przerywam."
        )
        return errors

    with open(os.path.join(package_dir, golden["input_file"]), "r", encoding="utf-8") as f:
        input_blob = json.load(f)
    with open(os.path.join(package_dir, golden["output_file"]), "r", encoding="utf-8") as f:
        expected_output_blob = json.load(f)

    try:
        import numpy as np
        import torch
    except ImportError as e:
        errors.append(f"Brak numpy/torch w tym środowisku ({e}) -- nie mogę zweryfikować "
                      f"ponownego załadowania modelu, tylko integralność plików.")
        return errors

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    arch_dir = os.path.join(project_root, "architecture_14_neurons_patryk_09_07")
    for p in (project_root, os.path.join(project_root, "ga_neuron_search"), arch_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    from ga_neuron_search.genome import Genome
    import net

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    g = Genome.from_dict(ckpt["topology"])
    model = net.GenomeNet(g, hw=None, quantize=False)
    model.set_quantize(True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    spikes_in = np.array(input_blob["data"], dtype=np.float32)
    assert list(spikes_in.shape) == input_blob["shape"]

    with torch.no_grad():
        x = torch.from_numpy(spikes_in).unsqueeze(0)
        so = model(x)["so"][0, :, 0].detach().cpu().numpy()
    actual_output = (so > 0.5).astype(np.uint8)

    expected_output = np.array(expected_output_blob["data"], dtype=np.uint8)
    if actual_output.shape != expected_output.shape:
        errors.append(
            f"KSZTAŁT WYJŚCIA NIEZGODNY: oczekiwano {expected_output.shape}, "
            f"jest {actual_output.shape} -- model po ponownym załadowaniu zachowuje "
            f"się inaczej (inna architektura/wersja net.py?)."
        )
        return errors

    if not np.array_equal(actual_output, expected_output):
        n_diff = int(np.sum(actual_output != expected_output))
        errors.append(
            f"WYJŚCIE NIE JEST IDENTYCZNE po ponownym załadowaniu: {n_diff}/"
            f"{expected_output.size} ramek różni się. Ponowne załadowanie w czystym "
            f"procesie NIE odtwarza tego samego championa -- NIE przekazuj tego "
            f"pakietu dalej, dopóki się to nie wyjaśni (możliwe przyczyny: "
            f"niedeterministyczna kwantyzacja, brak model.eval(), inna wersja net.py)."
        )
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package-dir", required=True)
    args = ap.parse_args()

    manifest_path = os.path.join(args.package_dir, "package_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[VERIFY] [BŁĄD] Brak package_manifest.json w {args.package_dir}")
        sys.exit(1)
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"[VERIFY] Sprawdzam integralność {len(manifest['artifacts'])} artefaktów...")
    errors = verify_integrity(args.package_dir, manifest)
    if errors:
        for e in errors:
            print(f"[VERIFY] [BŁĄD] {e}")
        print(f"\n[VERIFY] {len(errors)} błędów integralności -- przerywam przed testem replay.")
        sys.exit(1)
    print("[VERIFY] Integralność OK (wszystkie sha256 się zgadzają).")

    print("[VERIFY] Ładuję checkpoint od zera i porównuję golden replay...")
    replay_errors = verify_replay(args.package_dir)
    if replay_errors:
        for e in replay_errors:
            print(f"[VERIFY] [BŁĄD] {e}")
        sys.exit(1)

    print("[VERIFY] Golden replay OK -- ponowne załadowanie w czystym procesie daje "
          "identyczne wyjście.")
    print("[VERIFY] WSZYSTKO OK.")


if __name__ == "__main__":
    main()
    