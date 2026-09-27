import argparse
import glob
import sys
import multiprocessing
import os
import json

from pipeline_config import PipelineConfig
from hardware import get_device, resolve_workers
from tracker import RunTracker
from ga_runner import run_ga_stage, run_ext_evaluation_stage, run_final_evaluation_stage, run_hardware_export_stage
from continuous_eval import run_continuous_eval_stage

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline dla optymalizacji SNN (Lu.i)")

    # Globalne flagi
    parser.add_argument("--config", type=str, default="config.json", help="Ścieżka do pliku konfiguracyjnego JSON")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Wybór akceleratora")
    parser.add_argument("--workers", type=str, default="auto", help="Liczba workerów (int) lub 'auto' (benchmark)")
    parser.add_argument("--resume", type=str, default=None, help="Ścieżka do przerwanego katalogu run_... w celu wznowienia eksperymentu")

    # M4 (27.09.2026, Marcel): flagi Etapu 5 / komendy `continuous-eval`. Globalne
    # (nie tylko na subkomendzie continuous-eval), zeby dzialaly tez wewnatrz
    # `run-all`, gdzie Etap 5 jest wolany automatycznie po Etapie 3/4.
    parser.add_argument("--dataset-manifest-csv", type=str, default=None,
                        help="dataset/versions/vX.Y.Z/manifest.csv -- do świeżego "
                             "przeliczenia global_gain w ciągłej ewaluacji, jeśli "
                             "brak zamrożonego global_gain.json (patrz "
                             "continuous_eval.py, punkt 3 docstringu)")
    parser.add_argument("--gain-file", type=str, default=None,
                        help="jawna ścieżka do global_gain.json dla ciągłej ewaluacji "
                             "(domyślnie: szukane obok splitu train)")
    parser.add_argument("--recall-tolerance-s", type=float, default=0.0,
                        help="tolerancja okna recall w ciągłej ewaluacji (manifest.py: "
                             "'end_s + tolerancja' -- niedookreślone w źródle, "
                             "domyślnie 0.0; patrz continuous_eval.py punkt 4)")
    parser.add_argument("--out-csv-dir", type=str, default=None,
                        help="katalog na events.csv/false_alarms.csv z ciągłej "
                             "ewaluacji (M4: przekazanie do Karoliny/Andrzeja). "
                             "Domyślnie: <run_dir>/continuous_eval_csv")

    # Subkomendy do odpalania poszczególnych etapów lub całości
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run-all", help="Uruchamia pełny pipeline (GA -> Fine-Tuning -> Eval -> Hardware -> Continuous Eval)")
    subparsers.add_parser("train-ga", help="Uruchamia tylko etap algorytmu genetycznego (GA)")
    subparsers.add_parser("evaluate", help="Uruchamia ciągłą ewaluację na gotowym modelu")
    subparsers.add_parser("continuous-eval", help="Uruchamia tylko Etap 5 (ciągła ewaluacja 600s) na ukończonym --resume runie")

    return parser.parse_args()


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run_continuous_eval_if_available(config, tracker, args, stage_label: str = "ETAP 5/5") -> None:
    """M4 (27.09.2026, Marcel): Etap 5 jest CELOWO opcjonalny wewnątrz `run-all` --
    dataset/continuous/out zawiera audio, które NIE jest w gicie (patrz
    DataConfig.continuous_eval w pipeline_config.py: trzeba je odtworzyć
    komendą `dataset.continuous.eval.cli` na każdej maszynie z osobna).
    Maszyna bez odtworzonego datasetu ciągłego NIE powinna wywalać całego
    `run-all` -- pomijamy z jasnym komunikatem, tak jak inne opcjonalne braki
    w tym pliku (patrz load_hardware_profile w hardware.py). Brak checkpointu
    (Etap 3 się nie odbył) jest twardszym sygnałem, ale i tak tylko pomijamy
    Etap 5 -- reszta manifestu (Etapy 1-4) zostaje nienaruszona.

    Sprawdzamy `tracker.metrics` (NIE osobny argument `metrics_done`) --
    to jedyne autorytatywne źródło tego, co już policzono w tym runie,
    identycznie jak `log_metrics`/`update_manifest` w tracker.py."""
    if "continuous_eval" in tracker.metrics:
        print(f"[RESUME] Pomijam {stage_label} — ciągła ewaluacja 600s została już policzona.")
        return

    ckpt_path = tracker.metrics.get("continuous_test", {}).get("checkpoint_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[{stage_label}] Pomijam — brak checkpointu z Etapu 3 (continuous_test.checkpoint_path="
              f"{ckpt_path!r}). Uruchom najpierw Etap 3 (run_final_evaluation_stage) w tym samym runie.")
        return

    project_root = _project_root()
    continuous_dir = os.path.join(project_root, config.data.continuous_eval)
    manifest_paths = glob.glob(os.path.join(continuous_dir, "*.manifest.json"))
    if not manifest_paths:
        print(f"[{stage_label}] Pomijam — brak *.manifest.json w {continuous_dir}. "
              f"Dataset ciągły najwyraźniej nie został odtworzony na tej maszynie "
              f"(audio NIE jest w gicie -- patrz komenda w pipeline_config.py, "
              f"DataConfig.continuous_eval). Odtwórz go, albo uruchom ręcznie:\n"
              f"    python3 continuous_eval.py --checkpoint {ckpt_path} "
              f"--continuous-dir {continuous_dir} --dataset-manifest-csv <...>")
        return

    # M4 (27.09.2026, Marcel): domyslny katalog CSV to <run_dir>/continuous_eval_csv
    # -- tracker.get_run_dir() istnieje na prawdziwym RunTrackerze; nasz
    # _StandaloneTracker (uzywany tylko w testach) go nie ma, wiec wtedy po
    # prostu nie zapisujemy CSV automatycznie (trzeba --out-csv-dir jawnie).
    csv_dir = args.out_csv_dir
    if csv_dir is None and hasattr(tracker, "get_run_dir"):
        csv_dir = os.path.join(tracker.get_run_dir(), "continuous_eval_csv")

    print(f"\n>>> [{stage_label}] Ciągła ewaluacja 600s (checkpoint: {ckpt_path})...")
    run_continuous_eval_stage(
        config, tracker,
        checkpoint_path=ckpt_path,
        continuous_dir=continuous_dir,
        dataset_manifest_csv=args.dataset_manifest_csv,
        gain_file=args.gain_file,
        recall_tolerance_s=args.recall_tolerance_s,
        csv_dir=csv_dir,
    )


def main():
    args = parse_args()

    # 1. Ładowanie konfiguracji
    try:
        config = PipelineConfig.from_json(args.config)
        print(f"[INIT] Załadowano konfigurację z: {args.config}")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"[BŁĄD] Nie znaleziono pliku konfiguracyjnego: {args.config}. "
            "Upewnij się, że plik istnieje lub wskaż poprawną ścieżkę używając flagi --config."
        )

    # 2. Inicjalizacja sprzętu i workerów
    print("[INIT] Konfigurowanie środowiska...")
    device = get_device(args.device)
    workers_count, hw_benchmark = resolve_workers(args.workers, config=config, device=device)
    print(f"[INIT] Ustawiono urządzenie: {device.upper()} | Workery: {workers_count}")

    # 3. Uruchomienie Run Trackera
    tracker = RunTracker(
        config=config,
        device=device,
        workers=workers_count,
        hw_benchmark=hw_benchmark
    )

    # 3.5. LOGIKA WZNOWIENIA (RESUME)
    metrics_done = {}
    if args.resume:
        manifest_path = os.path.join(args.resume, "manifest.json")
        if not os.path.exists(manifest_path):
            print(f"[BŁĄD] Nie znaleziono pliku manifest.json w {args.resume}")
            sys.exit(1)

        print(f"[RESUME] Odtwarzanie stanu eksperymentu z: {args.resume}")
        with open(manifest_path, "r", encoding="utf-8") as f:
            old_manifest = json.load(f)

        metrics_done = old_manifest.get("metrics", {})

        # Podpinamy tracker pod stary katalog, żeby uniknąć tworzenia nowego folderu
        tracker.run_dir = args.resume
        if hasattr(tracker, "manifest_path"):
            tracker.manifest_path = manifest_path
        if hasattr(tracker, "metrics"):
            tracker.metrics = metrics_done
        if hasattr(tracker, "stage_times") and "stage_times" in old_manifest:
            tracker.stage_times = old_manifest["stage_times"]

    # 4. Routing komend
    if args.command == "run-all":
        # ETAP 1: Trening i poszukiwanie struktury (GA)
        if "ga_stage" in metrics_done and "best_topology" in metrics_done["ga_stage"]:
            print("[RESUME] Pomijam Etap 1 (GA) — optymalna topologia znajduje się już w manifeście.")
            best_topology = metrics_done["ga_stage"]["best_topology"]
        else:
            best_topology = run_ga_stage(config, tracker)

        # ETAP 2: Ewaluacja na rozszerzonym zbiorze spikes_ext
        if "spikes_ext_eval" in metrics_done:
            print("[RESUME] Pomijam Etap 2 — ewaluacja spikes_ext została już policzona.")
        else:
            run_ext_evaluation_stage(config, tracker, best_topology)

        # ETAP 3: Ewaluacja ciągła / testowa
        if "continuous_test" in metrics_done:
            print("[RESUME] Pomijam Etap 3 — ewaluacja testowa została już policzona.")
        else:
            run_final_evaluation_stage(config, tracker, best_topology)

        # ETAP 4: Eksport pod hardware
        if "hardware_export" in metrics_done:
            print("[RESUME] Pomijam Etap 4 — eksport sprzętowy został już wygenerowany.")
        else:
            run_hardware_export_stage(config, tracker, best_topology)

        # ETAP 5: Ciągła ewaluacja 600s (opcjonalna -- patrz docstring funkcji)
        _run_continuous_eval_if_available(config, tracker, args)

    elif args.command == "train-ga":
        if "ga_stage" in metrics_done:
            print("[RESUME] Etap GA już zakończony w tym runie.")
        else:
            run_ga_stage(config, tracker)

    elif args.command == "evaluate":
        print(f"\n>>> Startuję ewaluację na gotowym modelu z manifestu...")
        if "ga_stage" in metrics_done and "best_topology" in metrics_done["ga_stage"]:
            best_topology = metrics_done["ga_stage"]["best_topology"]
            run_final_evaluation_stage(config, tracker, best_topology)
        else:
            print("[BŁĄD] Samodzielna komenda 'evaluate' wymaga użycia flagi --resume wskazującej na ukończony run (brak best_topology).")
            sys.exit(1)

    elif args.command == "continuous-eval":
        # M4 (27.09.2026, Marcel): NOWA, samodzielna komenda -- celowo osobna
        # od `evaluate` (ta ostatnia zostaje przy swoim dotychczasowym
        # znaczeniu: zwykły test na klipach, nie zmieniam jej zachowania pod
        # nikim). Wymaga --resume runu z ukończonym Etapem 3 (checkpoint).
        if not args.resume:
            print("[BŁĄD] Komenda 'continuous-eval' wymaga --resume wskazującego na "
                  "ukończony run (potrzebny checkpoint z Etapu 3).")
            sys.exit(1)
        if "continuous_test" not in metrics_done:
            print("[BŁĄD] Run w --resume nie ma ukończonego Etapu 3 (continuous_test) "
                  "-- uruchom najpierw 'run-all' albo 'evaluate' na tym runie.")
            sys.exit(1)
        _run_continuous_eval_if_available(config, tracker, args, stage_label="CONTINUOUS-EVAL")

    # Zakończenie
    tracker.update_manifest(status="COMPLETED")
    print(f"\n[SUKCES] Pipeline zakończył pracę. Raport dostępny w: {tracker.get_run_dir()}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Dla kompatybilności z macOS i Windows
    main()
    