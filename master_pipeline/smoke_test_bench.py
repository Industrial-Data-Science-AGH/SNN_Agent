"""
Smoke test dla benchmark_winner_batch_sizes (krok 3 M0, "Dotrenowanie championa").

Wymaga załatanego winner.py (batch_size/num_samples jako parametry train_full)
i działającego środowiska ga_neuron_search -- tego samego, co wcześniejszy
smoke_test_benchmark.py.

Użycie (z katalogu master_pipeline/):
    python3 smoke_test_winner_bench.py --config config.json
"""
import argparse
import multiprocessing

from pipeline_config import PipelineConfig
from hardware import get_device, benchmark_winner_batch_sizes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--device", default="cpu", choices=["auto", "cpu", "cuda", "mps"])
    ap.add_argument("--batch-sizes", default="128,256")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--num-samples", type=int, default=256, help="Mała wartość na smoke test -- produkcyjnie 12000.")
    args = ap.parse_args()

    batch_sizes = tuple(int(x) for x in args.batch_sizes.split(","))

    config = PipelineConfig.from_json(args.config)
    device = get_device(args.device)
    print(f"[SMOKE] device={device}, batch_sizes={batch_sizes}, epochs={args.epochs}, "
          f"seeds={args.seeds}, num_samples={args.num_samples} (małe -- tylko test poprawności)")

    results = benchmark_winner_batch_sizes(
        config,
        device=device,
        batch_sizes=batch_sizes,
        epochs=args.epochs,
        seeds=args.seeds,
        num_samples=args.num_samples,
        repeats=1,
        warmup=False,
    )
    print("\n[SMOKE] Zwrócone wyniki:", results)
    print(
        "\n[SMOKE] Sprawdź:\n"
        "  1. Brak tracebacku przy imporcie winner (ImportError = brak łatki batch_size/num_samples\n"
        "     albo zły sys.path).\n"
        "  2. Log '[winner] plan: HAT 0..X, QAT X..Y' pokazuje się dla każdego batch_size.\n"
        "  3. Czas dla batch_size=256 nie jest absurdalnie inny niż dla 128 (rząd wielkości ten sam)."
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
    