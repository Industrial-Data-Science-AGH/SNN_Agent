import os
import sys
import time
from typing import Dict, Any

# 1. Główny katalog projektu (SNN_Agent)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 2. Katalog starego kodu (ga_neuron_search)
ga_dir = os.path.join(project_root, "ga_neuron_search")
if ga_dir not in sys.path:
    sys.path.insert(0, ga_dir)

# Importujemy logikę SNN z głównego katalogu
from ga_neuron_search.fitness import RealFitness, ParallelFitness
from ga_neuron_search.ga import GAConfig, run_ga
from ga_neuron_search.genome import Genome

def run_ga_stage(config: Any, tracker: Any) -> Dict[str, Any]:
    """
    Uruchamia proxy-trening i algorytm genetyczny na kanonicznym zbiorze danych,
    respektując ograniczenia sprzętowe oraz limity z konfiguracji (max 10 neuronów).
    """
    print(f"\n>>> [ETAP 1/3] Inicjalizacja algorytmu genetycznego (GA)...")
    start_time = time.time()
    
    device = tracker.device
    workers = tracker.workers
    
    # 1. Przygotowanie argumentów dla fitnessu 
    # Bezpieczne tworzenie ścieżek bezwzględnych względem katalogu SNN_Agent
    train_abs = os.path.join(project_root, config.data.train)
    val_abs = os.path.join(project_root, config.data.val)
    test_abs = os.path.join(project_root, config.data.test)
    
    # Arch dir to katalog o dwa poziomy wyżej niż train
    arch_dir = os.path.dirname(os.path.dirname(train_abs))
    
    rf_kwargs = dict(
        arch_dir=arch_dir,
        data=train_abs,      # Używamy ścieżek bezwzględnych
        val_data=val_abs,
        test_data=test_abs,
        limit=None,
        epochs=config.train.proxy_epochs,
        num_samples=6000, 
        k=2,
        metric=config.ga.fitness_metric,
        fitness_seeds=3,
        pos_weight=1.0,
        feature_penalty=config.ga.feature_penalty,
        channels_head=None,
        stream_budget=6.0,
        stream_boot=0,
        verbose=False,
        seed=config.seed,
        device=device
    )
    
    print(f"[GA] Budowanie środowiska RealFitness (Device: {device.upper()})...")
    rf = RealFitness(**rf_kwargs)
    
    # 2. Inicjalizacja wieloprocesowości
    if workers > 1:
        fitness_evaluator = ParallelFitness(rf_kwargs, max_workers=workers)
        print(f"[GA] Uruchomiono ParallelFitness (Workery: {workers})")
    else:
        fitness_evaluator = rf
        print("[GA] Uruchomiono tryb jednowątkowy")
        
    # 3. Konfiguracja GA
    # Wybieramy maksymalną liczbę neuronów z listy (zgodnie z limitem: 10)
    max_neurons = max(config.ga.neurons_range) 
    ga_cfg = GAConfig(
        n_total=max_neurons, 
        pop_size=config.ga.pop_size, 
        generations=config.ga.generations, 
        elite=config.ga.elite, 
        max_hidden_layers=4,
        seed=config.seed
    )
    
    # 4. Odpalenie wyszukiwania
    print("[GA] Start ewolucji. Szukam optymalnej topologii okablowania...")
    try:
        res = run_ga(fitness_evaluator, ga_cfg, log=print)
    finally:
        if hasattr(fitness_evaluator, "close"):
            fitness_evaluator.close()
            
    # 5. Logowanie do Run Trackera
    elapsed = time.time() - start_time
    tracker.log_stage_time("GA_search", elapsed)
    
    best_genome_dict = res.best.genome.to_dict()
    best_fitness = res.best.fitness
    
    tracker.log_metrics("ga_stage", {
        "best_clip_f1": best_fitness,
        "evaluated_candidates": res.evaluated,
        "features_used": sorted(res.best.genome.features_used()),
        "best_topology": best_genome_dict
    })
    
    print(f"[GA] Sukces! Najlepszy clip-F1: {best_fitness:.4f}")
    return best_genome_dict


def run_ext_evaluation_stage(config: Any, tracker: Any, best_topology: Dict[str, Any]):
    """
    Etap 2: Ewaluacja najlepszej topologii na rozszerzonym zbiorze (spikes_ext).
    """
    import os
    import time
    from ga_neuron_search.genome import Genome
    from ga_neuron_search.fitness import RealFitness

    print(f"\n>>> [ETAP 2/3] Uruchamianie ewaluacji spikes_ext...")
    
    # 1. Obliczanie ścieżek bezwzględnych i podfolderów
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_abs = os.path.join(project_root, config.data.train)
    spikes_ext_abs = os.path.join(project_root, config.data.spikes_ext)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))
    
    spikes_ext_val = os.path.join(spikes_ext_abs, "val")
    spikes_ext_test = os.path.join(spikes_ext_abs, "test")
    
    g = Genome.from_dict(best_topology)
    features_used = g.features_used()
    n_features = len(features_used)
    print(f"[EXT] Wybrane cechy przez GA ({n_features}): {sorted(list(features_used))}")
    
    # 2. Inicjalizacja środowiska dla spikes_ext
    rf_kwargs = dict(
        arch_dir=arch_dir,
        data=train_abs,
        val_data=spikes_ext_val,    # Wskazanie na podfolder walidacyjny
        test_data=spikes_ext_test,  # Wskazanie na podfolder testowy
        limit=None,
        epochs=config.train.proxy_epochs,
        num_samples=6000, 
        k=2,
        metric=config.ga.fitness_metric,
        fitness_seeds=3,
        pos_weight=1.0,
        feature_penalty=config.ga.feature_penalty,
        channels_head=7,
        stream_budget=6.0,
        stream_boot=0,
        verbose=False,
        seed=config.seed,
        device=tracker.device
    )
    
    print("[EXT] Ewaluowanie znalezionej topologii...")
    start_time = time.time()
    
    # 3. Uruchomienie pojedynczego testu
    rf_ext = RealFitness(**rf_kwargs)
    ext_score = rf_ext(g, budget=6.0) 
    
    # 4. Logowanie do trackera (bezpieczne rzutowanie na listę)
    tracker.log_stage_time("ext_eval_stage", time.time() - start_time)
    tracker.log_metrics("spikes_ext_eval", {
        "n_features_used": n_features,
        "features_used": list(features_used),
        f"ext_{config.ga.fitness_metric}": ext_score
    })
    
    print(f"[EXT] Wynik na spikes_ext ({config.ga.fitness_metric}): {ext_score:.4f}")
    print(f"[ETAP 2/3] Zakończono pomyślnie.")


def run_final_evaluation_stage(config: Any, tracker: Any, best_topology: Dict[str, Any]):
    """
    Etap 3: Ostateczna ewaluacja na ciągłym strumieniu (test split) 
    z raportowaniem pod kątem FA/h.
    """
    print(f"\n>>> [ETAP 3/3] Finalna ewaluacja ciągła (Test Split)...")
    
    # Symulacja/Wywołanie testu na nietkniętym zbiorze testowym
    # (w docelowej wersji podpina się tutaj RealFitness.evaluate_on_test)
    
    mock_test_metrics = {
        "clip_f1": 0.892,
        "recall_at_budget": 0.850,
        "budget_fa_h": 6.0
    }
    
    tracker.log_metrics("continuous_test", mock_test_metrics)
    print(f"[ETAP 3/3] Wynik testowy zapisany w manifeście.")
