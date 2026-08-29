#!/usr/bin/env python3
"""
plot_history.py — wykres krzywej uczenia GA (fitness vs pokolenie) do pokazania
zespołowi. Czyta wynikowy JSON z run_search.py i zapisuje PNG.

Użycie:
    python plot_history.py wyniki_demo.json        # -> wyniki_demo.png
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "ga_results.json"
    out = src.rsplit(".", 1)[0] + ".png"
    results = json.load(open(src, encoding="utf-8"))

    plt.figure(figsize=(8, 5))
    for r in sorted(results, key=lambda x: x["n_total"]):
        h = r["history"]
        sizes = "-".join(str(x) for x in r["topology"]["layer_sizes"])
        plt.plot(range(len(h)), h, marker="o",
                 label=f"N={r['n_total']}  (najlepsza: {sizes}, fit={r['fitness']:.3f})")
    plt.xlabel("pokolenie GA")
    plt.ylabel("najlepszy fitness (metryka clip-F1 / AP)")
    plt.title("Algorytm genetyczny topologii SNN — poprawa z pokolenia na pokolenie")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"zapisano wykres -> {out}")


if __name__ == "__main__":
    main()
