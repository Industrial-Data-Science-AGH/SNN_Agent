# -*- coding: utf-8 -*-
"""
Run Pipeline — główny skrypt orkiestrujący wszystkie fazy pipeline SNN.

Fazy:
  A — Setup: download ESC-50, preprocessing, baseline metrics
  B — HAT: Hardware-Aware Training (50 epok, Gumbel-softmax annealing)
  C — QAT: Quantization-Aware Training (20 epok, mixed precision)
  D — Evaluation: benchmark table, thermal drift, power estimate
  E — Export: JSON/CSV/Arduino .h + HIL validation + MCP4151 table

Użycie:
    python snn_pipeline/run_pipeline.py --phase all
    python snn_pipeline/run_pipeline.py --phase A --data-limit 20
    python snn_pipeline/run_pipeline.py --phase B --epochs 5
"""

import argparse
import sys
import time
from pathlib import Path

import torch

# Dodaj parent dir do PYTHONPATH (żeby importy z snn_pipeline działały)
sys.path.insert(0, str(Path(__file__).parent.parent))

from snn_pipeline.config import (
    DEVICE,
    PATH_CONFIG,
    RANDOM_SEED,
    TRAIN_CONFIG,
    set_global_seed,
)


def phase_a(args) -> dict:
    """FAZA A — Setup i baseline.

    1. Pobiera ESC-50
    2. Buduje dataset z augmentacją
    3. Tworzy DataLoadery
    4. Oblicza baseline metrics (model z wagami domyślnymi)
    """
    from snn_pipeline.data_pipeline import build_dataset, get_dataloaders
    from snn_pipeline.evaluation import evaluate_model
    from snn_pipeline.snn_model import GlassBreakSNN

    print("\n" + "=" * 70)
    print("  FAZA A — SETUP & BASELINE")
    print("=" * 70)

    t0 = time.time()

    # 1. Buduj dataset
    datasets = build_dataset(
        encoding=args.encoding,
        n_timesteps=args.n_timesteps,
        data_limit=args.data_limit,
        augment=not args.no_augment,
    )
    loaders = get_dataloaders(datasets, batch_size=args.batch_size)

    # 2. Baseline model (wagi z symulacji, bez kwantyzacji)
    print("\n[FAZA A] Ewaluacja modelu baseline (wagi z symulacji)...")
    baseline_model = GlassBreakSNN(quantize_mode="none").to(DEVICE)
    baseline_results = evaluate_model(baseline_model, loaders["test"], label="Baseline (sim_train)")

    elapsed = time.time() - t0
    print(f"\n[FAZA A] Zakończona w {elapsed:.1f}s")

    # Zapis checkpointu
    torch.save({
        "baseline_results": baseline_results,
        "baseline_state_dict": baseline_model.state_dict(),
    }, PATH_CONFIG.checkpoint_dir / "phase_a.pt")

    return {
        "datasets": datasets,
        "loaders": loaders,
        "baseline_model": baseline_model,
        "baseline_results": baseline_results,
    }


def phase_b(args, phase_a_data: dict) -> dict:
    """FAZA B — Hardware-Aware Training.

    1. Inicjalizuje model z wagami baseline
    2. Trenuje z HAT (Gumbel-softmax, mismatch, threshold sweep)
    3. Zapisuje learning curves i tabelę wag
    """
    from snn_pipeline.evaluation import evaluate_model
    from snn_pipeline.hat_trainer import HATTrainer
    from snn_pipeline.snn_model import GlassBreakSNN

    print("\n" + "=" * 70)
    print("  FAZA B — HARDWARE-AWARE TRAINING (HAT)")
    print("=" * 70)

    t0 = time.time()

    loaders = phase_a_data["loaders"]

    # 1. Model z Gumbel quantization
    hat_model = GlassBreakSNN(quantize_mode="gumbel").to(DEVICE)

    # 2. HAT Training
    epochs = args.epochs if args.epochs else TRAIN_CONFIG.hat_epochs
    trainer = HATTrainer(hat_model, learning_rate=TRAIN_CONFIG.learning_rate)
    history = trainer.train(loaders["train"], loaders["val"], epochs=epochs)

    # 3. Learning curves
    trainer.plot_learning_curves()

    # 4. Tabela wag
    weight_table = trainer.get_weight_table()
    print(f"\n{weight_table}")

    # Zapisz tabelę
    with open(PATH_CONFIG.output_dir / "hat_weight_table.txt", "w", encoding="utf-8") as f:
        f.write(weight_table)

    # 5. Ewaluacja po HAT
    hat_results = evaluate_model(hat_model, loaders["test"], label="Po HAT")

    elapsed = time.time() - t0
    print(f"\n[FAZA B] Zakończona w {elapsed:.1f}s")

    torch.save({
        "hat_model_state_dict": hat_model.state_dict(),
        "hat_history": history,
        "hat_results": hat_results,
    }, PATH_CONFIG.checkpoint_dir / "phase_b.pt")

    return {
        **phase_a_data,
        "hat_model": hat_model,
        "hat_results": hat_results,
        "hat_history": history,
    }


def phase_c(args, phase_b_data: dict) -> dict:
    """FAZA C — Quantization-Aware Training + Sensitivity Analysis.

    1. Calibration (50 próbek glass_break)
    2. QAT fine-tuning (20 epok, mixed precision)
    3. Sensitivity analysis → heatmapa
    4. Identyfikacja kandydatów na MCP4151
    """
    from snn_pipeline.evaluation import evaluate_model
    from snn_pipeline.qat_trainer import QATTrainer
    from snn_pipeline.sensitivity import (
        generate_heatmap,
        identify_mcp4151_candidates,
        sensitivity_analysis,
    )

    print("\n" + "=" * 70)
    print("  FAZA C — QUANTIZATION-AWARE TRAINING (QAT)")
    print("=" * 70)

    t0 = time.time()

    loaders = phase_b_data["loaders"]
    qat_model = phase_b_data["hat_model"]  # Kontynuuj z modelem po HAT

    # 1. QAT Trainer
    qat_epochs = args.qat_epochs if args.qat_epochs else TRAIN_CONFIG.qat_epochs
    qat_trainer = QATTrainer(qat_model)

    # 2. Calibration
    qat_trainer.calibrate(loaders["train"])

    # 3. QAT Training
    qat_history = qat_trainer.train(loaders["train"], loaders["val"], epochs=qat_epochs)
    qat_trainer.plot_qat_curves()

    # 4. Ewaluacja po QAT
    qat_results = evaluate_model(qat_model, loaders["test"], label="Po HAT+QAT")

    # 5. Sensitivity Analysis
    print("\n[FAZA C] Analiza wrażliwości...")
    sensitivities = sensitivity_analysis(qat_model, loaders["test"])
    generate_heatmap(sensitivities)
    mcp_candidates = identify_mcp4151_candidates(sensitivities)

    elapsed = time.time() - t0
    print(f"\n[FAZA C] Zakończona w {elapsed:.1f}s")

    torch.save({
        "qat_model_state_dict": qat_model.state_dict(),
        "qat_history": qat_history,
        "qat_results": qat_results,
        "sensitivities": sensitivities,
        "mcp_candidates": mcp_candidates,
    }, PATH_CONFIG.checkpoint_dir / "phase_c.pt")

    return {
        **phase_b_data,
        "qat_model": qat_model,
        "qat_results": qat_results,
        "sensitivities": sensitivities,
        "mcp_candidates": mcp_candidates,
    }


def phase_d(args, phase_c_data: dict) -> dict:
    """FAZA D — Ewaluacja kompletna + benchmark.

    1. Benchmark table (Baseline vs HAT vs HAT+QAT)
    2. Thermal drift simulation (100×)
    3. Weight error histogram
    4. Power estimate
    """
    from snn_pipeline.evaluation import (
        benchmark_table,
        power_estimate,
        thermal_drift_simulation,
        weight_error_histogram,
    )

    print("\n" + "=" * 70)
    print("  FAZA D — EWALUACJA & BENCHMARK")
    print("=" * 70)

    t0 = time.time()

    loaders = phase_c_data["loaders"]
    qat_model = phase_c_data["qat_model"]

    # 1. Benchmark table
    all_results = [
        phase_c_data["baseline_results"],
        phase_c_data["hat_results"],
        phase_c_data["qat_results"],
    ]
    table = benchmark_table(all_results)

    # 2. Thermal drift simulation
    print("\n[FAZA D] Symulacja dryfu termicznego...")
    drift_results = thermal_drift_simulation(
        qat_model, loaders["test"],
        n_runs=TRAIN_CONFIG.thermal_drift_runs,
        drift_pct=TRAIN_CONFIG.thermal_drift_pct,
    )

    # 3. Weight error histogram
    weight_errors = weight_error_histogram(qat_model)

    # 4. Power estimate
    power = power_estimate(qat_model)

    elapsed = time.time() - t0
    print(f"\n[FAZA D] Zakończona w {elapsed:.1f}s")

    return {
        **phase_c_data,
        "benchmark_table": table,
        "drift_results": drift_results,
        "weight_errors": weight_errors,
        "power": power,
    }


def phase_e(args, phase_d_data: dict) -> dict:
    """FAZA E — Export i Deployment.

    1. Export JSON/CSV/Arduino .h
    2. HIL validation (100 scenariuszy)
    3. MCP4151 programming table
    """
    from snn_pipeline.export import (
        export_weights_csv,
        export_weights_json,
        generate_arduino_header,
    )
    from snn_pipeline.hil_validation import (
        generate_mcp4151_table,
        hil_simulation,
    )

    print("\n" + "=" * 70)
    print("  FAZA E — EXPORT & DEPLOYMENT")
    print("=" * 70)

    t0 = time.time()

    model = phase_d_data["qat_model"]
    loaders = phase_d_data["loaders"]
    qat_results = phase_d_data["qat_results"]

    # 1. Export wag
    export_weights_json(model, qat_results, version="hat_qat_v1")
    export_weights_csv(model)
    generate_arduino_header(model)

    # 2. HIL validation
    print("\n[FAZA E] Hardware-in-the-Loop validation...")
    hil_results = hil_simulation(model, loaders["test"])

    # 3. MCP4151 table
    mcp_candidates = phase_d_data.get("mcp_candidates", [])
    critical_names = [c["name"] if isinstance(c, dict) else c for c in mcp_candidates]
    if not critical_names:
        # Generuj tabelę dla wszystkich wag (na wszelki wypadek)
        critical_names = None
    generate_mcp4151_table(model, critical_synapses=critical_names)

    elapsed = time.time() - t0
    print(f"\n[FAZA E] Zakończona w {elapsed:.1f}s")

    # Final summary
    print("\n" + "=" * 70)
    print("  PIPELINE ZAKOŃCZONY POMYŚLNIE")
    print("=" * 70)
    print(f"  Output files:")
    for f in sorted(PATH_CONFIG.output_dir.glob("*")):
        print(f"    📄 {f.name}")
    print("=" * 70)

    return {**phase_d_data, "hil_results": hil_results}


def main():
    """Entry point — parsuje argumenty i uruchamia odpowiednie fazy."""
    parser = argparse.ArgumentParser(
        description="SNN HAT/QAT Pipeline — Glass Break Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady użycia:
  python snn_pipeline/run_pipeline.py --phase all               # Pełny pipeline
  python snn_pipeline/run_pipeline.py --phase A --data-limit 20 # Szybki test Phase A
  python snn_pipeline/run_pipeline.py --phase B --epochs 5      # HAT z 5 epokami
  python snn_pipeline/run_pipeline.py --phase all --quick        # Szybki smoke test
        """,
    )

    parser.add_argument(
        "--phase", type=str, default="all",
        choices=["A", "B", "C", "D", "E", "all"],
        help="Faza do uruchomienia (A/B/C/D/E/all). Default: all",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Liczba epok HAT (domyślnie: 50)")
    parser.add_argument("--qat-epochs", type=int, default=None, help="Liczba epok QAT (domyślnie: 20)")
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG.batch_size, help="Batch size")
    parser.add_argument("--encoding", type=str, default="ttfs", choices=["ttfs", "rate"],
                        help="Typ enkodowania spike'ów")
    parser.add_argument("--n-timesteps", type=int, default=100, help="Liczba kroków czasowych")
    parser.add_argument("--data-limit", type=int, default=None,
                        help="Limit plików audio (do szybkich testów)")
    parser.add_argument("--no-augment", action="store_true", help="Wyłącz augmentację danych")
    parser.add_argument("--quick", action="store_true",
                        help="Szybki tryb: data-limit=50, epochs=5, qat-epochs=3")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")

    args = parser.parse_args()

    # Quick mode
    if args.quick:
        args.data_limit = args.data_limit or 50
        args.epochs = args.epochs or 5
        args.qat_epochs = args.qat_epochs or 3

    # Seed
    set_global_seed(args.seed)

    print(f"\n{'#' * 70}")
    print(f"  SNN HARDWARE-AWARE TRAINING PIPELINE")
    print(f"  Phase: {args.phase} | Encoding: {args.encoding} | Device: {DEVICE}")
    print(f"  Seed: {args.seed} | Quick: {args.quick}")
    if args.data_limit:
        print(f"  Data limit: {args.data_limit} files")
    print(f"{'#' * 70}\n")

    t_total = time.time()

    # Run phases
    data = {}

    phases_to_run = ["A", "B", "C", "D", "E"] if args.phase == "all" else [args.phase]

    # Jeśli startujemy od fazy > A, trzeba załadować checkpoint
    if phases_to_run[0] != "A":
        data = _load_previous_phases(phases_to_run[0], args)

    for phase in phases_to_run:
        if phase == "A":
            data = phase_a(args)
        elif phase == "B":
            data = phase_b(args, data)
        elif phase == "C":
            data = phase_c(args, data)
        elif phase == "D":
            data = phase_d(args, data)
        elif phase == "E":
            data = phase_e(args, data)

    elapsed_total = time.time() - t_total
    print(f"\n[TOTAL] Pipeline zakończony w {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")


def _load_previous_phases(start_phase: str, args) -> dict:
    """Ładuje checkpointy z poprzednich faz.

    Args:
        start_phase: Faza od której startujemy.
        args: Argumenty CLI.

    Returns:
        Dane z poprzednich faz.
    """
    from snn_pipeline.data_pipeline import build_dataset, get_dataloaders
    from snn_pipeline.snn_model import GlassBreakSNN

    data = {}

    # Zawsze potrzebujemy danych
    print("[INFO] Ładuję dataset...")
    datasets = build_dataset(
        encoding=args.encoding,
        n_timesteps=args.n_timesteps,
        data_limit=args.data_limit,
        augment=not args.no_augment,
    )
    loaders = get_dataloaders(datasets, batch_size=args.batch_size)
    data["datasets"] = datasets
    data["loaders"] = loaders

    # Ładuj checkpoint Phase A
    ckpt_a = PATH_CONFIG.checkpoint_dir / "phase_a.pt"
    if ckpt_a.exists():
        loaded = torch.load(ckpt_a, map_location=DEVICE, weights_only=False)
        baseline_model = GlassBreakSNN(quantize_mode="none").to(DEVICE)
        baseline_model.load_state_dict(loaded["baseline_state_dict"])
        data["baseline_model"] = baseline_model
        data["baseline_results"] = loaded["baseline_results"]
    else:
        print("[WARN] Brak checkpointu Phase A — baseline nie będzie dostępny.")
        data["baseline_results"] = {"label": "Baseline (brak)", "precision": 0, "recall": 0,
                                     "f1": 0, "accuracy": 0, "fnr": 1, "weights": {}}

    # Ładuj checkpoint Phase B
    if start_phase in ("C", "D", "E"):
        ckpt_b = PATH_CONFIG.checkpoint_dir / "phase_b.pt"
        if ckpt_b.exists():
            loaded = torch.load(ckpt_b, map_location=DEVICE, weights_only=False)
            hat_model = GlassBreakSNN(quantize_mode="hat").to(DEVICE)
            hat_model.load_state_dict(loaded["hat_model_state_dict"])
            data["hat_model"] = hat_model
            data["hat_results"] = loaded["hat_results"]
        else:
            print("[ERROR] Brak checkpointu Phase B — uruchom Phase B najpierw.")
            sys.exit(1)

    # Ładuj checkpoint Phase C
    if start_phase in ("D", "E"):
        ckpt_c = PATH_CONFIG.checkpoint_dir / "phase_c.pt"
        if ckpt_c.exists():
            loaded = torch.load(ckpt_c, map_location=DEVICE, weights_only=False)
            qat_model = GlassBreakSNN(quantize_mode="hat").to(DEVICE)
            qat_model.load_state_dict(loaded["qat_model_state_dict"])
            data["qat_model"] = qat_model
            data["qat_results"] = loaded["qat_results"]
            data["sensitivities"] = loaded.get("sensitivities", {})
            data["mcp_candidates"] = loaded.get("mcp_candidates", [])
        else:
            print("[ERROR] Brak checkpointu Phase C — uruchom Phase C najpierw.")
            sys.exit(1)

    return data


if __name__ == "__main__":
    main()
