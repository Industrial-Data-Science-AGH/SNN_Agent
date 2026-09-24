import argparse
import sys
import multiprocessing
import os
import json

from pipeline_config import PipelineConfig
from hardware import get_device, resolve_workers
from tracker import RunTracker
from ga_runner import run_ga_stage, run_ext_evaluation_stage, run_final_evaluation_stage, run_hardware_export_stage

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline dla optymalizacji SNN (Lu.i)")
    
    # Globalne flagi
    parser.add_argument("--config", type=str, default="config.json", help="Ścieżka do pliku konfiguracyjnego JSON")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Wybór akceleratora")
    parser.add_argument("--workers", type=str, default="auto", help="Liczba workerów (int) lub 'auto' (benchmark)")
    parser.add_argument("--resume", type=str, default=None, help="Ścieżka do przerwanego katalogu run_... w celu wznowienia eksperymentu")
    
    # Subkomendy do odpalania poszczególnych etapów lub całości
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("run-all", help="Uruchamia pełny pipeline (GA -> Fine-Tuning -> Eval -> Hardware)")
    subparsers.add_parser("train-ga", help="Uruchamia tylko etap algorytmu genetycznego (GA)")
    subparsers.add_parser("evaluate", help="Uruchamia ciągłą ewaluację na gotowym modelu")
    
    return parser.parse_args()

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

    # Zakończenie
    tracker.update_manifest(status="COMPLETED")
    print(f"\n[SUKCES] Pipeline zakończył pracę. Raport dostępny w: {tracker.get_run_dir()}")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)  # Dla kompatybilności z macOS i Windows
    main()