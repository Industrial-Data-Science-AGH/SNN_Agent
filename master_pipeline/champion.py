"""
M3 punkt 2 (26.09.2026, Marcel): wybor "championa" -- najlepszego kandydata
SPOSROD WIELU przebiegow pipeline.py (runs/run_*/), nie w obrebie jednego GA.

Reguła rankingu (uzgodniona z Marcelem):
    1. Odfiltruj przebiegi NIEZGODNE protokolem -- rozne source_commit/
       encoder_hash/dataset_manifest_hash nie sa ze soba porownywalne, bo
       moglyby np. mierzyc test na innym splicie albo innym enkoderze.
    2. Wsrod zgodnych: najpierw walidacja (median_key -- mediana z
       winner_seeds przebiegow na val, ta sama wielkosc, ktora winner.py uzywa
       do wyboru modelu), potem budzet FA (patrz nizej), a dopiero na koncu
       zlozonosc/energia jako tie-breaker.
    3. Jesli PRZY DANYM fitness_metric=="recall_fa" najlepszy kandydat ma
       recall_fa == 0.0 (w granicach epsilon), NIE wybieramy go cicho jako
       "zwyciezce" -- to sygnal, ze budzet FA/h (stream_budget, domyslnie
       6.0 z fitness.py) jest przy tej architekturze nieosiagalny (siec nie
       potrafi utrzymac FA/h ponizej budzetu przy JAKIMKOLWIEK progu decyzji,
       nie "martwa siec" -- patrz komentarz w RealFitness o recall_fa==0 jako
       legalnej podlodze). Raportujemy wtedy status="infeasible", zamiast
       wymuszac sukces.

UWAGA / ZALOZENIE DO POTWIERDZENIA (odpowiedz na pytanie o "wczesniej
zapisana regule" zlozonosci/energii):
    W GAConfig (pipeline_config.py) istnieja juz `feature_penalty` i
    `parsimony_eps`, ale to sa wagi kary w PROXY fitness wewnatrz samego GA
    (run_ga_stage) -- nie ma dzis osobnej, jawnie zapisanej formuly do
    porownywania GOTOWYCH zwyciezcow MIEDZY przebiegami. Ponizej uzywam
    najprostszej, monotonicznej reguly zgodnej z duchem parsimony_eps:
    energia/zlozonosc = (total_neurons, hidden_layers) rosnaco, z
    parsimony_eps jako progiem "praktycznie remisu" na metryce walidacyjnej
    (dwaj kandydaci w obrebie parsimony_eps na val sa traktowani jak remis i
    rozstrzygani zlozonoscia, a nie szumem <parsimony_eps na val). To jest
    NOWA definicja na potrzeby tego zadania, nie odtworzenie istniejacej
    formuly -- do potwierdzenia/poprawki, gdy Kacper/Wiktor beda mieli
    realny model energetyczny plytki (patrz comparison_matrix_manifest.json:
    "Estimated" / "Missing").

Uzycie:
    python3 champion.py --runs-dir runs [--reference-run runs/run_XXXX]
"""
import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

INFEASIBLE_EPS = 1e-9  # recall_fa <= tego = "budżet FA/h nieosiągalny", nie "0 z zaokrąglenia"


def _atomic_write_json(path: str, obj: Any) -> None:
    """Ta sama gwarancja co RunTracker._atomic_write_json (tracker.py) --
    champion.json jest jawnym stanem uzywanym pozniej (np. do wyboru, ktory
    checkpoint eksportowac na produkcje), wiec polowiczny zapis nie moze go
    uszkodzic."""
    tmp_path = path + f".tmp.{os.getpid()}"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


@dataclass
class Candidate:
    run_dir: str
    manifest: Dict[str, Any]
    fitness_metric: str

    # Wypełniane przez _extract()
    protocol: Tuple[Any, Any, Any] = field(default=None)  # (source_commit, encoder_hash, dataset_manifest_hash)
    val_median: Optional[float] = None
    test_score: Optional[float] = None
    total_neurons: Optional[int] = None
    hidden_layers: Optional[int] = None
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    incompatible_reason: Optional[str] = None

    def is_compatible(self) -> bool:
        return self.incompatible_reason is None

    def budget_met(self) -> Optional[bool]:
        """None = nie dotyczy (metryka inna niż recall_fa -- pojęcie budżetu
        FA/h nie ma tu zastosowania, więc nie blokujemy na "infeasible")."""
        if self.fitness_metric != "recall_fa" or self.val_median is None:
            return None
        return self.val_median > INFEASIBLE_EPS


def _extract(run_dir: str, manifest: Dict[str, Any], fitness_metric: str) -> Candidate:
    c = Candidate(run_dir=run_dir, manifest=manifest, fitness_metric=fitness_metric)

    hw = manifest.get("metrics", {}).get("hardware_export", {})
    ct = manifest.get("metrics", {}).get("continuous_test", {})

    c.protocol = (
        hw.get("source_commit", manifest.get("git_sha")),
        hw.get("encoder_hash"),
        json.dumps(hw.get("dataset_manifest_hash"), sort_keys=True) if hw.get("dataset_manifest_hash") else None,
    )
    if hw.get("encoder_hash") is None or hw.get("dataset_manifest_hash") is None:
        c.incompatible_reason = (
            "brak hardware_export.encoder_hash/dataset_manifest_hash w manifest.json "
            "(uruchom run_hardware_export_stage albo M1 freeze_m1.py) -- nie da się "
            "potwierdzić zgodności protokołu, więc kandydat jest pomijany, nie zgadywany."
        )
        return c

    winner_seeds = ct.get("winner_seeds")
    median_key = f"val_{fitness_metric}_median_of_{winner_seeds}_seeds" if winner_seeds else None
    c.val_median = ct.get(median_key) if median_key else None
    c.test_score = ct.get(f"test_{fitness_metric}")
    c.total_neurons = hw.get("total_neurons")
    c.hidden_layers = hw.get("hidden_layers")
    c.checkpoint_path = ct.get("checkpoint_path")
    c.checkpoint_sha256 = ct.get("checkpoint_sha256")

    if c.val_median is None or not isinstance(c.val_median, (int, float)) or not math.isfinite(c.val_median):
        c.incompatible_reason = (
            f"brak/nieskończona wartość {median_key!r} w metrics.continuous_test -- "
            "run_final_evaluation_stage prawdopodobnie się nie zakończył."
        )
    return c


def load_candidates(runs_dir: str, fitness_metric_filter: Optional[str] = None) -> List[Candidate]:
    candidates = []
    for manifest_path in sorted(glob.glob(os.path.join(runs_dir, "run_*", "manifest.json"))):
        run_dir = os.path.dirname(manifest_path)
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[CHAMPION] [POMIJAM] {manifest_path}: nieczytelny manifest ({exc}) -- "
                  f"prawdopodobnie przerwany run sprzed atomowego zapisu (tracker.py M3 punkt 3).")
            continue

        if manifest.get("status") != "COMPLETED":
            print(f"[CHAMPION] [POMIJAM] {run_dir}: status={manifest.get('status')!r} (nie COMPLETED)")
            continue

        fm = manifest.get("config", {}).get("ga", {}).get("fitness_metric")
        # config.json jest osobnym plikiem od manifest.json w tym repo (patrz
        # tracker.py _setup_dir) -- manifest.json go nie zawiera, więc czytamy
        # go z config.json obok, jeśli jest.
        config_path = os.path.join(run_dir, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                fm = cfg.get("ga", {}).get("fitness_metric", fm)
            except (json.JSONDecodeError, OSError):
                pass
        fm = fm or "recall_fa"

        if fitness_metric_filter and fm != fitness_metric_filter:
            print(f"[CHAMPION] [POMIJAM] {run_dir}: fitness_metric={fm!r} != {fitness_metric_filter!r} "
                  f"(różne metryki nie są ze sobą porównywalne).")
            continue

        candidates.append(_extract(run_dir, manifest, fm))
    return candidates


def select_champion(candidates: List[Candidate]) -> Dict[str, Any]:
    usable = [c for c in candidates if c.is_compatible()]
    excluded = [c for c in candidates if not c.is_compatible()]

    report: Dict[str, Any] = {
        "n_candidates_total": len(candidates),
        "n_excluded_incompatible": len(excluded),
        "excluded": [{"run_dir": c.run_dir, "reason": c.incompatible_reason} for c in excluded],
    }

    if not usable:
        report["status"] = "no_candidates"
        report["champion"] = None
        return report

    # Krok 1: zgodność protokołu -- grupujemy po (source_commit, encoder_hash,
    # dataset_manifest_hash) i bierzemy NAJLICZNIEJSZĄ grupę (= aktualny,
    # ustabilizowany protokół), zamiast milcząco mieszać przebiegi z różnych
    # commitów/enkoderów w jednym rankingu.
    groups: Dict[Tuple, List[Candidate]] = {}
    for c in usable:
        groups.setdefault(c.protocol, []).append(c)
    best_protocol = max(groups.keys(), key=lambda k: len(groups[k]))
    pool = groups[best_protocol]

    dropped_other_protocol = [c for c in usable if c.protocol != best_protocol]
    report["protocol_used"] = {
        "source_commit": best_protocol[0],
        "encoder_hash": best_protocol[1],
        "dataset_manifest_hash_json": best_protocol[2],
    }
    report["n_dropped_other_protocol"] = len(dropped_other_protocol)
    report["dropped_other_protocol"] = [c.run_dir for c in dropped_other_protocol]

    # Krok 2: budżet FA -- jeśli fitness_metric=="recall_fa" i WSZYSCY w puli
    # mają val_median <= epsilon, budżet jest strukturalnie nieosiągalny przy
    # tej architekturze/danych -- nie wybieramy "najmniej złego zera" jako
    # cichego zwycięzcy.
    metric = pool[0].fitness_metric
    if metric == "recall_fa":
        feasible = [c for c in pool if (c.budget_met() is True)]
        if not feasible:
            report["status"] = "infeasible"
            report["reason"] = (
                f"Wszyscy kandydaci zgodni protokołem (n={len(pool)}) mają "
                f"val_{metric}_median <= {INFEASIBLE_EPS} -- żaden nie osiąga "
                f"budżetu FA/h (stream_budget) przy JAKIMKOLWIEK progu decyzji. "
                f"To read-out o architekturze/danych, nie o pojedynczym runie -- "
                f"nie forsujemy sukcesu wyborem najmniej złego zera."
            )
            report["champion"] = None
            report["candidates_considered"] = [
                {"run_dir": c.run_dir, f"val_{metric}_median": c.val_median} for c in pool
            ]
            return report
        pool = feasible

    # Krok 3: ranking -- (1) walidacja malejąco, w obrębie parsimony_eps
    # traktowana jak remis, (2) w remisie: mniej neuronów, potem mniej warstw
    # ukrytych (proxy złożoności/energii -- patrz zastrzeżenie w docstringu
    # modułu, to nowa reguła do potwierdzenia).
    parsimony_eps = 0.02
    cfg_path = os.path.join(pool[0].run_dir, "config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                parsimony_eps = json.load(f).get("ga", {}).get("parsimony_eps", parsimony_eps)
        except (json.JSONDecodeError, OSError):
            pass

    best_val = max(c.val_median for c in pool)
    near_best = [c for c in pool if (best_val - c.val_median) <= parsimony_eps]

    def complexity_key(c: Candidate):
        # None (brak danych o topologii) trafia na koniec, nie na początek.
        n = c.total_neurons if c.total_neurons is not None else float("inf")
        h = c.hidden_layers if c.hidden_layers is not None else float("inf")
        return (n, h)

    ranked = sorted(near_best, key=lambda c: (-c.val_median, complexity_key(c)))
    champion = ranked[0]

    report["status"] = "ok"
    report["parsimony_eps_used"] = parsimony_eps
    report["n_in_near_best_tie"] = len(near_best)
    report["champion"] = {
        "run_dir": champion.run_dir,
        "fitness_metric": metric,
        f"val_{metric}_median": champion.val_median,
        f"test_{metric}": champion.test_score,
        "total_neurons": champion.total_neurons,
        "hidden_layers": champion.hidden_layers,
        "checkpoint_path": champion.checkpoint_path,
        "checkpoint_sha256": champion.checkpoint_sha256,
        "seed": champion.manifest.get("metrics", {}).get("continuous_test", {}).get("seed"),
    }
    report["ranking"] = [
        {"run_dir": c.run_dir, f"val_{metric}_median": c.val_median,
         "total_neurons": c.total_neurons, "hidden_layers": c.hidden_layers}
        for c in ranked
    ]
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--fitness-metric", default=None,
                    help="Ogranicz do przebiegów z tym config.ga.fitness_metric "
                         "(domyślnie: bez filtra -- różne metryki i tak trafią do "
                         "osobnych grup przy rankingu, ale filtr daje czytelniejszy log).")
    ap.add_argument("--out", default=None, help="Gdzie zapisać champion.json (domyślnie: <runs-dir>/champion.json).")
    args = ap.parse_args()

    candidates = load_candidates(args.runs_dir, args.fitness_metric)
    report = select_champion(candidates)

    out_path = args.out or os.path.join(args.runs_dir, "champion.json")
    _atomic_write_json(out_path, report)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\n[CHAMPION] status={report['status']} -- zapisano {out_path}")

    if report["status"] == "infeasible":
        sys.exit(2)  # kod wyjścia != 0/1, żeby CI/skrypty mogły odróżnić "infeasible" od zwykłego błędu
    if report["status"] == "no_candidates":
        sys.exit(1)


if __name__ == "__main__":
    main()
        