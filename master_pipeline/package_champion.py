#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
package_champion.py -- M5: "Przekazać pakiet championa".
D3. Zależności: M4, K3.

UWAGA: treść zależności "P1" z Twojej wiadomości nie została mi podana w tej
rozmowie -- ten skrypt NIE uwzględnia żadnych wymagań specyficznych dla P1,
bo ich nie znam. Jeśli P1 nakłada dodatkowy format/pole na pakiet, dopisz je
osobno.

KONTEKST / DECYZJE:

1. CO WCHODZI DO PAKIETU (punkt 1 M4/M5): checkpoint, manifest.json biegu,
   config.json biegu, topologia (genom, odczytany z checkpointu, osobno jako
   JSON do audytu bez torch), weights (sam checkpoint JUŻ zawiera state_dict
   -- osobno generuję tylko `weights_manifest.json`, listę tensorów z
   kształtem/dtype/sha256 KAŻDEGO tensora, żeby dało się zweryfikować
   integralność wag bez ładowania torch), decoder (reguła (k,w) -- patrz
   punkt 2), calibration_status (patrz punkt 3), golden_replay (patrz punkt 4).

2. DECODER: dokładnie jak w M4 (continuous_eval.py) -- reguła (k,w) NIE jest
   nigdzie trwale zapisana przez GA/eksport hardware'owy (champion.py:
   report_to_dict nigdy nie jest wołane). Jeśli bieg championa miał już
   policzony etap `continuous_eval` (M4), czytam `decoder_rule` STAMTĄD
   (metrics.continuous_eval.decoder_rule w manifest.json) -- nie liczę
   drugi raz tego samego. Jeśli tego etapu nie było, odtwarzam regułę na
   żywo DOKŁADNIE tymi samymi funkcjami co continuous_eval.py
   (`_load_champion_model`, `_build_real_fitness`, `_recover_operating_rule`)
   zamiast duplikować logikę w drugim miejscu.

3. CALIBRATION STATUS: ten projekt liczy `gain` (wzmocnienie enkodera)
   metodą percentylową na DIGITAL TWIN enkodera (encoder_twin.py), nie na
   fizycznej płytce Lu.i. Nie ma tu żadnego kroku, który by to zwalidował
   na prawdziwym ADC. Dlatego `calibration_status.json` jawnie i wprost
   mówi: model NIE został skalibrowany na sprzęcie, a `gain`/`gain_source`
   to wyłącznie parametry cyfrowego bliźniaka. To bezpośrednio realizuje
   kryterium odbioru M5: "plik JSON nie obiecuje fizycznej kompatybilności
   bez kalibracji" -- ta obietnica jest explicite ZAPRZECZONA w pliku,
   zamiast być przez milczenie sugerowana.

4. GOLDEN REPLAY (punkt 2 M5: "Sprawdzić powtórne załadowanie w czystym
   procesie i identyczne wyjście"): generuję DETERMINISTYCZNE syntetyczne
   wejście (ziarno stałe, niezależne od stanu torch/numpy w reszcie
   programu), przepuszczam przez model DOKŁADNIE tą samą ścieżką co
   `_run_model_get_d_spikes` w continuous_eval.py (jeden forward, bez
   cięcia na okna, `(so > 0.5).astype(uint8)`), i zapisuję wejście+wyjście.
   Osobny skrypt `verify_champion_package.py` odpala się jako NOWY proces
   (subprocess, nie import w tym samym interpreterze) i sprawdza, że
   ponowne załadowanie checkpointu daje BIT-DO-BITU identyczne wyjście --
   to jedyny sposób, żeby "czysty proces" znaczyło coś więcej niż "ten sam
   interpreter, który już ma wszystko w pamięci".

5. DUŻE ARTEFAKTY / STORAGE (punkt 2 M5: "Duże artefakty przekazać przez
   uzgodniony storage, nie zwykły commit binariów"): ten skrypt NIE ma
   dostępu do żadnego uzgodnionego storage (nie wiem, czy to S3/GCS/coś
   wewnętrznego) -- nie zgaduję. Zamiast tego dzieli artefakty na
   `small_files_for_git` (JSON-y, tekst) i `large_files_for_storage`
   (checkpoint .pt, dowolny plik > LARGE_FILE_THRESHOLD_BYTES), z sha256
   OBU grup w jednym `package_manifest.json` -- PR może więc opisać duże
   pliki hashem, nie treścią, a Ty wklejasz je ręcznie tam, gdzie już to
   robicie z innymi artefaktami tej wielkości.

6. PR #47 / squash / M5 punkt 3: to jest dyscyplina procesowa (branch
   hygiene), nie coś, co da się wymusić z tego skryptu bez dostępu do
   Twojego repo/GitHuba -- opisane tylko w komunikacie końcowym jako
   przypomnienie, nie zautomatyzowane.

Użycie:
    python3 package_champion.py --champion-json ../runs/champion.json \
        --out-dir ../packages/champion_<hash>_<data>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

LARGE_FILE_THRESHOLD_BYTES = 5 * 1024 * 1024  # 5 MiB -- checkpoint .pt zwykle > tego
GOLDEN_SEED = 20260927  # stałe, niezależne od reszty programu -- powtarzalne w kazdym procesie
GOLDEN_N_FRAMES = 64


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ensure_import_paths(project_root: str) -> None:
    """Identyczne wstawienie sys.path co run_continuous_eval_stage w
    continuous_eval.py -- bez tego `import net`/`from ga_neuron_search.genome
    import Genome` nie znajdą modułów niezależnie od tego, skąd ten skrypt
    jest odpalony."""
    this_dir = os.path.dirname(os.path.abspath(__file__))
    arch_dir = os.path.join(project_root, "architecture_14_neurons_patryk_09_07")
    for p in (this_dir, project_root, os.path.join(project_root, "ga_neuron_search"), arch_dir):
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256_of_file(path: str, buf_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_json(path: str, obj: Any) -> None:
    """Ta sama gwarancja co RunTracker/champion.py -- pakiet jest artefaktem
    referencyjnym, polowiczny zapis nie moze go po cichu uszkodzic."""
    tmp_path = path + f".tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


# ============================================================ ładowanie źródeł

def _load_champion_source(champion_json: Optional[str], run_dir_override: Optional[str],
                          checkpoint_override: Optional[str]) -> Tuple[str, str, Dict[str, Any]]:
    """Zwraca (run_dir, checkpoint_path, champion_entry). `champion_entry` to
    słownik `champion` z champion.json (może być pusty {}, jeśli podano
    ręczne override'y zamiast champion.json -- np. do zapakowania konkretnego
    runu bez ponownego odpalania champion.py)."""
    if run_dir_override:
        run_dir = run_dir_override
        checkpoint_path = checkpoint_override or os.path.join(run_dir, "winner_checkpoint.pt")
        return run_dir, checkpoint_path, {}

    if not champion_json:
        raise ValueError("Podaj --champion-json ALBO --run-dir (i opcjonalnie --checkpoint-path).")

    with open(champion_json, "r", encoding="utf-8") as f:
        report = json.load(f)

    if report.get("status") != "ok" or not report.get("champion"):
        raise RuntimeError(
            f"[PACKAGE] {champion_json}: status={report.get('status')!r} -- champion.py "
            f"nie wybrał zwycięzcy (patrz champion.json.reason). Nie ma czego pakować."
        )

    champion = report["champion"]
    run_dir = champion["run_dir"]
    checkpoint_path = checkpoint_override or champion.get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError(
            f"[PACKAGE] champion.json nie ma checkpoint_path dla {run_dir} -- "
            f"podaj ręcznie przez --checkpoint-path."
        )
    return run_dir, checkpoint_path, champion


def _load_run_manifest(run_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"[PACKAGE] brak manifest.json w {run_dir}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_config_path(run_dir: str) -> Optional[str]:
    config_path = os.path.join(run_dir, "config.json")
    return config_path if os.path.exists(config_path) else None


# ============================================================ decoder (odtworzenie j.w. M4)

def _get_decoder_rule(manifest: Dict[str, Any], checkpoint_path: str, config_path: Optional[str],
                      project_root: str, device: str) -> Dict[str, Any]:
    ce = manifest.get("metrics", {}).get("continuous_eval", {}).get("decoder_rule")
    if ce is not None:
        print("[PACKAGE] decoder_rule: odczytana z metrics.continuous_eval (M4 już policzone dla tego biegu).")
        result = dict(ce)
        result["source"] = "metrics.continuous_eval.decoder_rule (istniejący wynik M4)"
        return result

    print("[PACKAGE] decoder_rule: brak wyniku M4 dla tego biegu -- odtwarzam na żywo "
          "(RealFitness.stream_recall), tak samo jak continuous_eval.py.")
    if config_path is None:
        raise RuntimeError(
            "[PACKAGE] Nie mogę odtworzyć reguły decydera na żywo: brak config.json obok "
            "manifest.json tego biegu (potrzebny do zbudowania RealFitness identycznie jak "
            "ga_runner.py). Uruchom najpierw etap continuous-eval (M4) dla tego championa, "
            "albo dostarcz config.json."
        )
    # Import leniwy i identyczny jak w continuous_eval.py -- nie duplikuję logiki,
    # tylko reużywam tych samych funkcji na tym samym source of truth. Ładujemy
    # config przez PipelineConfig.from_json (jak ga_runner.py), zamiast zgadywać
    # konstruktor z surowego dict -- unika rozjazdu ze strukturą zagnieżdżonych
    # dataclassów (DataConfig/GAConfig/TrainConfig).
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pipeline_config import PipelineConfig
    from continuous_eval import (
        _load_champion_model, _build_real_fitness, _recover_operating_rule,
        OperatingRuleInfeasible,
    )

    config = PipelineConfig.from_json(config_path)
    model, g = _load_champion_model(checkpoint_path, device)
    rf, stream_budget = _build_real_fitness(config, device, project_root)
    try:
        (k, w), refrac = _recover_operating_rule(rf, model, g, stream_budget)
    except OperatingRuleInfeasible as e:
        # M3 odbior: "jesli budzet jest nieosiagalny, raport mowi infeasible" --
        # NIE blokujemy calego pakietu z tego powodu. Checkpoint/topologia/wagi/
        # calibration/golden-replay sa uzyteczne (i weryfikowalne) niezaleznie od
        # tego, czy istnieje reguła alarmu mieszcząca się w budżecie 6 FA/h --
        # Patryk/Wiktor nadal dostaja dokladnie oceniony, wczytywalny model.
        # `package_champion.main()` odczytuje ten status i konczy kodem 2, tak
        # samo jak champion.py/continuous_eval.py, zeby caller mogl odroznic
        # "spakowane, ale operacyjnie infeasible" od bledu.
        print(f"[PACKAGE] decoder_rule: INFEASIBLE -- {e}")
        return {
            "status": "infeasible",
            "budget_fa_h": stream_budget,
            "best_recall_at_budget": e.best_recall_at_budget,
            "reason": str(e),
            "source": "odtworzone na żywo przez package_champion.py (RealFitness.stream_recall) "
                      "-- ten bieg nie miał jeszcze policzonego etapu continuous_eval (M4); "
                      "żadna reguła (k,w) nie mieści budżetu FA/h na val.",
        }
    return {
        "k": k, "w": w, "refrac_frames": refrac, "budget_fa_h": stream_budget,
        "source": "odtworzone na żywo przez package_champion.py (RealFitness.stream_recall) "
                  "-- ten bieg nie miał jeszcze policzonego etapu continuous_eval (M4)",
    }


# ============================================================ calibration status

def _get_calibration_status(manifest: Dict[str, Any]) -> Dict[str, Any]:
    ce = manifest.get("metrics", {}).get("continuous_eval", {})
    gain = ce.get("gain")
    gain_source = ce.get("gain_source")
    return {
        "gain": gain,
        "gain_source": gain_source,
        "physically_calibrated_on_hardware": False,
        "calibration_method": (
            "Wzmocnienie (`gain`) liczone jest percentylowo (compute_global_gain, "
            "percentile=99.9) na DIGITAL TWIN enkodera (encoder_twin.py), na podstawie "
            "nagrań treningowych -- NIE na fizycznym ADC płytki Lu.i."
        ),
        "note": (
            "TEN PLIK NIE OBIECUJE FIZYCZNEJ KOMPATYBILNOŚCI ZE SPRZĘTEM. Model był "
            "oceniony wyłącznie w cyfrowym bliźniaku. Przed uruchomieniem na fizycznej "
            "płytce Lu.i wymagana jest osobna kalibracja na rzeczywistym torze ADC "
            "(poziom szumu, offset, rzeczywista charakterystyka mikrofonu/wzmacniacza) "
            "-- brak takiej kalibracji nie jest tu zakładany ani ukrywany milczeniem."
        ),
    }


# ============================================================ topologia / wagi

def _export_topology_and_weights(checkpoint_path: str, out_dir: str) -> Tuple[str, str]:
    """Zwraca (topology_json_path, weights_manifest_json_path). Nie duplikuje
    checkpointu -- tylko wyciąga z niego topologię do czytelnego JSON-a (do
    audytu bez torch) i listę tensorów wag z ich sha256 (do integralności bez
    ładowania modelu)."""
    import torch

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    topology_path = os.path.join(out_dir, "topology.json")
    _atomic_write_json(topology_path, {
        "topology_source": "checkpoint['topology'] (Genome.to_dict())",
        "genome": ckpt["topology"],
    })

    state_dict = ckpt["model"]
    tensors = {}
    for name, tensor in state_dict.items():
        raw = tensor.detach().cpu().numpy().tobytes()
        tensors[name] = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "n_params": int(tensor.numel()),
            "sha256": sha256_of_bytes(raw),
        }
    weights_manifest_path = os.path.join(out_dir, "weights_manifest.json")
    _atomic_write_json(weights_manifest_path, {
        "note": (
            "Same wagi (state_dict) NIE są tu duplikowane -- są w checkpoint_path "
            "(sekcja artifacts w package_manifest.json). To jest lista kontrolna: "
            "kształt/dtype/sha256 KAŻDEGO tensora, żeby dało się zweryfikować "
            "integralność wag bez ładowania torch/GenomeNet."
        ),
        "n_tensors": len(tensors),
        "total_params": sum(t["n_params"] for t in tensors.values()),
        "tensors": tensors,
    })
    return topology_path, weights_manifest_path


# ============================================================ golden replay

def _build_golden_replay(checkpoint_path: str, device: str, out_dir: str) -> str:
    """Deterministyczne wejście -> jeden forward -> zapisane wejście+wyjście.
    Ścieżka inferencji identyczna z `_run_model_get_d_spikes` w
    continuous_eval.py (jeden forward całego 'strumienia', bez cięcia na
    okna) -- to jest dokładnie to, co champion realnie robi w continuous eval,
    nie osobna, wymyślona na potrzeby testu ścieżka."""
    import numpy as np
    import torch

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from continuous_eval import _load_champion_model

    model, g = _load_champion_model(checkpoint_path, device)
    ch_in = g.layer_sizes()[0]

    rng = np.random.default_rng(GOLDEN_SEED)
    spikes_in = (rng.random((GOLDEN_N_FRAMES, ch_in)) < 0.15).astype(np.float32)

    with torch.no_grad():
        x = torch.from_numpy(spikes_in).unsqueeze(0).to(device)
        so = model(x)["so"][0, :, 0].detach().cpu().numpy()
    output = (so > 0.5).astype(np.uint8)

    input_path = os.path.join(out_dir, "golden_replay_input.json")
    output_path = os.path.join(out_dir, "golden_replay_output.json")
    _atomic_write_json(input_path, {"shape": list(spikes_in.shape), "dtype": "float32",
                                    "data": spikes_in.tolist()})
    _atomic_write_json(output_path, {"shape": list(output.shape), "dtype": "uint8",
                                     "data": output.tolist()})

    golden_path = os.path.join(out_dir, "golden_replay.json")
    _atomic_write_json(golden_path, {
        "purpose": (
            "M5 punkt 2: 'Sprawdzić powtórne załadowanie w czystym procesie i "
            "identyczne wyjście.' Ten plik wiąże konkretny checkpoint_sha256 z "
            "konkretnym deterministycznym wejściem/wyjściem -- verify_champion_"
            "package.py odpala się jako NOWY proces, ładuje checkpoint od zera, "
            "przepuszcza to samo wejście i porównuje wyjście bit-do-bitu."
        ),
        "seed": GOLDEN_SEED,
        "n_frames": GOLDEN_N_FRAMES,
        "ch_in": ch_in,
        "checkpoint_sha256": sha256_of_file(checkpoint_path),
        "input_file": os.path.basename(input_path),
        "output_file": os.path.basename(output_path),
        "input_sha256": sha256_of_file(input_path),
        "output_sha256": sha256_of_file(output_path),
    })
    return golden_path


# ============================================================ pakowanie

def package_champion(champion_json: Optional[str], out_dir: str,
                     run_dir_override: Optional[str] = None,
                     checkpoint_override: Optional[str] = None,
                     device: str = "cpu") -> str:
    os.makedirs(out_dir, exist_ok=True)
    project_root = _project_root()
    _ensure_import_paths(project_root)

    run_dir, checkpoint_path, champion_entry = _load_champion_source(
        champion_json, run_dir_override, checkpoint_override)
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(project_root, checkpoint_path)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"[PACKAGE] checkpoint nie istnieje: {checkpoint_path}")

    manifest = _load_run_manifest(run_dir)
    config_path = _find_run_config_path(run_dir)

    print(f"[PACKAGE] Źródło: run_dir={run_dir}, checkpoint={checkpoint_path}")

    # 1) checkpoint -- kopiowany do pakietu (duży plik, patrz punkt 5 docstringu)
    checkpoint_dest = os.path.join(out_dir, "champion_checkpoint.pt")
    shutil.copy2(checkpoint_path, checkpoint_dest)

    # 2) manifest + config biegu -- kopiowane jak są (małe, do gita)
    manifest_dest = os.path.join(out_dir, "run_manifest.json")
    _atomic_write_json(manifest_dest, manifest)
    config_dest = None
    if config_path is not None:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        config_dest = os.path.join(out_dir, "run_config.json")
        _atomic_write_json(config_dest, config_dict)

    # 3) topologia + wagi (audyt)
    topology_path, weights_manifest_path = _export_topology_and_weights(checkpoint_dest, out_dir)

    # 4) decoder
    decoder_rule = _get_decoder_rule(manifest, checkpoint_dest, config_path, project_root, device)
    decoder_path = os.path.join(out_dir, "decoder.json")
    _atomic_write_json(decoder_path, decoder_rule)

    # 5) calibration status
    calibration = _get_calibration_status(manifest)
    calibration_path = os.path.join(out_dir, "calibration_status.json")
    _atomic_write_json(calibration_path, calibration)

    # 6) golden replay
    golden_path = _build_golden_replay(checkpoint_dest, device, out_dir)

    # 7) opis ograniczeń do artykułu (M5 "Przekazanie") -- fakty, które znam,
    #    reszta to jawne TODO dla Marcela, nie zgadywanie treści naukowej.
    ce = manifest.get("metrics", {}).get("continuous_eval", {})
    limitations_path = os.path.join(out_dir, "limitations_for_article.json")
    _atomic_write_json(limitations_path, {
        "model_checkpoint_sha256": sha256_of_file(checkpoint_dest),
        "decoder_rule": decoder_rule,
        "recall_mean": ce.get("recall_mean"),
        "recall_min": ce.get("recall_min"),
        "fa_per_hour_mean": ce.get("fa_per_hour_mean"),
        "fa_per_hour_ci_pooled": ce.get("fa_per_hour_ci_pooled"),
        "recall_delta_vs_historical_clip": ce.get("recall_delta_vs_historical_clip"),
        "known_limitations": [
            "Model NIE był walidowany na fizycznym sprzęcie Lu.i -- tylko w cyfrowym "
            "bliźniaku enkodera (patrz calibration_status.json).",
            "Reguła decydera (k,w) jest odtwarzana algorytmicznie, nie jest zapisana "
            "w oryginalnym pipeline'ie GA (champion.py nigdy nie persystuje "
            "report_to_dict) -- odtworzenie zależy od stabilności RealFitness.stream_recall.",
            f"recall_tolerance_s użyty w M4={ce.get('recall_tolerance_s')!r} -- decyzja "
            "domyślna (0.0), niepotwierdzona formalnie przez Marcela.",
            "Przedział ufności FA/h jest Poissonowski (Garwood) -- zakłada niezależne, "
            "rzadkie zdarzenia FA w czasie; nie modeluje ewentualnej korelacji FA "
            "(np. wspólne źródło szumu w danym nagraniu).",
        ],
        "TODO_dla_Marcela": [
            "Uzupełnić narrację naukową (dlaczego ta architektura, porównanie do "
            "baseline'u, itp.) -- ten plik podaje tylko fakty liczbowe.",
        ],
    })

    # ---- manifest pakietu: SHA256 KAŻDEGO artefaktu + podział na małe/duże
    artifact_paths = [checkpoint_dest, manifest_dest, topology_path, weights_manifest_path,
                      decoder_path, calibration_path, golden_path,
                      os.path.join(out_dir, "golden_replay_input.json"),
                      os.path.join(out_dir, "golden_replay_output.json"),
                      limitations_path]
    if config_dest:
        artifact_paths.append(config_dest)

    artifacts = []
    for p in artifact_paths:
        size = os.path.getsize(p)
        artifacts.append({
            "filename": os.path.basename(p),
            "sha256": sha256_of_file(p),
            "size_bytes": size,
            "storage": "large_storage" if size > LARGE_FILE_THRESHOLD_BYTES else "git",
        })

    package_manifest = {
        "package_schema_version": "1.0",
        "milestone": "M5",
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run_dir": run_dir,
        "source_checkpoint_path": checkpoint_path,
        "champion_selection_summary": champion_entry,
        "artifacts": artifacts,
        "small_files_for_git": [a["filename"] for a in artifacts if a["storage"] == "git"],
        "large_files_for_storage": [a["filename"] for a in artifacts if a["storage"] == "large_storage"],
        "physical_compatibility_note": (
            "Ten pakiet opisuje model DOKŁADNIE tak, jak został oceniony (checkpoint + "
            "topologia + reguła decyzyjna + golden replay). NIE gwarantuje fizycznej "
            "kompatybilności z płytką Lu.i bez przeprowadzenia kalibracji na sprzęcie "
            "-- patrz calibration_status.json."
        ),
        # M3 odbior: "jesli budzet jest nieosiagalny, raport mowi infeasible" --
        # przeniesione tu z decoder.json, zeby main() moglo dac kod wyjscia 2 bez
        # ponownego parsowania osobnego pliku. Pakiet i tak jest kompletny
        # (checkpoint/topologia/wagi/calibration/golden-replay) -- infeasible
        # dotyczy tylko reguly alarmu, nie samego modelu.
        "decoder_status": decoder_rule.get("status", "ok"),
        "process_reminder": (
            "PR kierować do master (po poprawkach). Po jego squash kolejny etap "
            "zaczynać z aktualnego master, żeby nie powielać starej historii "
            "(M5 punkt 3) -- to dyscyplina procesowa niezautomatyzowana tu."
        ),
    }
    package_manifest_path = os.path.join(out_dir, "package_manifest.json")
    _atomic_write_json(package_manifest_path, package_manifest)

    print(f"[PACKAGE] Zapisano manifest pakietu: {package_manifest_path}")
    for a in artifacts:
        print(f"  [{a['storage']:>13}] {a['filename']:<32} sha256={a['sha256'][:16]}... "
              f"({a['size_bytes']} B)")
    if package_manifest["decoder_status"] == "infeasible":
        print("[PACKAGE] UWAGA: decoder_status=infeasible -- pakiet jest kompletny "
              "(model wczytywalny i zweryfikowany), ale zaden (k,w) nie miesci "
              "budzetu FA/h na val. Patrz decoder.json.")

    return out_dir, package_manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--champion-json", default=None, help="np. runs/champion.json (z champion.py)")
    ap.add_argument("--run-dir", default=None, help="Alternatywa do --champion-json: konkretny runs/run_XXXX")
    ap.add_argument("--checkpoint-path", default=None, help="Nadpisuje ścieżkę checkpointu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-verify", action="store_true",
                    help="Nie odpalaj weryfikacji w czystym procesie po spakowaniu (odradzane).")
    args = ap.parse_args()

    out_dir, package_manifest = package_champion(args.champion_json, args.out_dir, args.run_dir,
                                                 args.checkpoint_path, args.device)

    if not args.skip_verify:
        # Nazwa pliku na dysku to verify_package_champion.py (nie
        # verify_champion_package.py) -- bez tej poprawki main() zawsze
        # padał tu na FileNotFoundError przy domyślnym --skip-verify=False,
        # bo subprocess.run dostawał ścieżkę do nieistniejącego pliku.
        verify_script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "verify_package_champion.py")
        print(f"\n[PACKAGE] Uruchamiam weryfikację w NOWYM procesie: {verify_script}")
        result = subprocess.run([sys.executable, verify_script, "--package-dir", out_dir])
        if result.returncode != 0:
            print("\n[PACKAGE] [BŁĄD] Weryfikacja w czystym procesie NIE powiodła się -- "
                  "pakiet zapisany, ale NIE wysyłaj go dalej, dopóki to się nie wyjaśni.")
            sys.exit(1)

    print(f"\n[PACKAGE] Gotowe: {out_dir}")
    print("[PACKAGE] Pamiętaj: duże pliki (large_files_for_storage w package_manifest.json) "
          "NIE commitować do gita -- uzgodniony storage. PR do master, potem squash, "
          "kolejny etap z aktualnego master (M5 punkt 3).")

    if package_manifest["decoder_status"] == "infeasible":
        # Ten sam kod wyjscia co champion.py/continuous_eval.py (2, nie 0/1):
        # pakiet jest zapisany i poprawny, ale operacyjnie infeasible przy
        # obecnym budzecie FA/h -- caller (CI, skrypt) moze to odroznic od
        # bledu pakowania (kod 1) bez parsowania stdout.
        sys.exit(2)


if __name__ == "__main__":
    main()
    