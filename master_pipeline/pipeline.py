import argparse
import sys
import time

# Zależnie od tego, jak nazwałeś pliki, importujemy nasze klocki:
from config import PipelineConfig
from hardware import get_device, resolve_workers
from tracker import RunTracker

def parse_args():
    parser = argparse.ArgumentParser(description="Master Pipeline dla optymalizacji SNN (Lu.i)")
    
    # Globalne flagi (wymagania z ticketa)
    parser.add_argument("--config", type=str, default="config.json", help="Ścieżka do pliku konfiguracyjnego JSON")
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Wybór akceleratora")
    parser.add_argument("--workers", type=str, default="auto", help="Liczba workerów (int) lub 'auto' (benchmark)")
    
    # Subkomendy do odpalania poszczególnych etapów lub całości
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    subparsers.add_parser("run-all", help="Uruchamia pełny pipeline (GA -> Fine-Tuning -> Eval)")
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
        print(f"[INIT] Nie znaleziono {args.config}, generuję domyślną konfigurację...")
        config = PipelineConfig()
        # Zapisujemy ją, żeby użytkownik mógł ją później edytować
        config.to_json("config.json")

    # 2. Inicjalizacja sprzętu i workerów
    print("[INIT] Konfigurowanie środowiska...")
    device = get_device(args.device)
    workers_count, hw_benchmark = resolve_workers(args.workers)
    print(f"[INIT] Ustawiono urządzenie: {device.upper()} | Workery: {workers_count}")

    # 3. Uruchomienie Run Trackera
    tracker = RunTracker(
        config=config,
        device=device,
        workers=workers_count,
        hw_benchmark=hw_benchmark
    )

    # 4. Routing komend (tutaj w przyszłości podepniesz logikę ML)
    if args.command == "run-all":
        print(f"\n>>> [ETAP 1/3] Startuję pełny eksperyment GA...")
        # Symulacja pracy...
        time.sleep(1)
        tracker.log_stage_time("GA_search", 125.4)
        
        print(f">>> [ETAP 2/3] Ewaluacja spikes_ext...")
        tracker.log_metrics("spikes_ext", {"clip_f1": 0.88, "precision": 0.91})
        
    elif args.command == "train-ga":
        print(f"\n>>> Startuję tylko etap algorytmu genetycznego...")
        
    elif args.command == "evaluate":
        print(f"\n>>> Startuję ciągłą ewaluację...")

    # Zakończenie
    tracker.update_manifest(status="COMPLETED")
    print(f"\n[SUKCES] Pipeline zakończył pracę. Raport zapisano w: {tracker.get_run_dir()}")

if __name__ == "__main__":
    main()