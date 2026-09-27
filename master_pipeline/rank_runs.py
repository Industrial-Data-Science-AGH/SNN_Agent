import os
import json
import glob
import argparse
from typing import Dict

def count_neurons(topology: Dict) -> int:
    """Zlicza neurony na podstawie topologii genomu (sumuje długości warstw)."""
    if "layers" in topology:
        return sum(len(layer) for layer in topology["layers"])
    return 0

def main():
    parser = argparse.ArgumentParser(description="Globalny ranking wariantów SNN")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Katalog z wynikami (domyślnie: runs/)")
    args = parser.parse_args()

    manifest_files = glob.glob(os.path.join(args.runs_dir, "*", "manifest.json"))
    valid_runs = []

    for path in manifest_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
            
        # Odrzucamy przerwane eksperymenty
        if manifest.get("status") != "COMPLETED":
            continue
            
        metrics = manifest.get("metrics", {})
        ga_stage = metrics.get("ga_stage", {})
        best_topology = ga_stage.get("best_topology", {})
        
        # Weryfikacja ograniczeń sprzętowych z ticketa
        n_neurons = count_neurons(best_topology)
        if n_neurons > 10 or n_neurons == 0:
            continue
            
        # Ekstrakcja kluczowych danych ewaluacyjnych
        spikes_ext_eval = metrics.get("spikes_ext_eval", {})
        continuous_test = metrics.get("continuous_test", {})
        
        dataset_name = continuous_test.get("dataset", "brak_danych")
        test_f1 = continuous_test.get("test_clip_f1", 0.0)
        ext_f1 = spikes_ext_eval.get("ext_clip_f1", 0.0)
        latency = continuous_test.get("latency_sec", 0.0)
        
        run_name = os.path.basename(os.path.dirname(path))
        
        valid_runs.append({
            "run_name": run_name,
            "dataset": dataset_name,
            "neurons": n_neurons,
            "test_f1": test_f1,
            "ext_f1": ext_f1,
            "latency": latency
        })

    if not valid_runs:
        print("[RANKING] Brak ukończonych eksperymentów spełniających kryteria (<= 10 neuronów).")
        return

    # Grupowanie i izolacja wyników po datasecie testowym
    grouped_runs = {}
    for r in valid_runs:
        grouped_runs.setdefault(r["dataset"], []).append(r)
        
    for dataset, runs in grouped_runs.items():
        # Sortowanie po wynikach docelowego środowiska produkcyjnego (Etap 3)
        runs.sort(key=lambda x: x["test_f1"], reverse=True)
        
        print(f"\n=== RANKING DLA DATASETU: {dataset} ===")
        print(f"{'Eksperyment':<30} | {'Test F1':<10} | {'Ext F1':<10} | {'Neurony':<8} | {'Narzut (s)':<10}")
        print("-" * 77)
        for r in runs:
            print(f"{r['run_name']:<30} | {r['test_f1']:<10.4f} | {r['ext_f1']:<10.4f} | {r['neurons']:<8} | {r['latency']:<10.2f}")
        print("-" * 77)

if __name__ == "__main__":
    main()
    