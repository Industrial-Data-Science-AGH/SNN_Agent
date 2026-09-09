import os
import sys
import time
from typing import Dict, Any

# 1. Pobieramy bezwzględną ścieżkę do folderu, w którym jest ten plik (master_pipeline)
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. Cofamy się o jeden poziom wyżej (do SNN_Agent)
parent_dir = os.path.dirname(current_dir)

# 3. Wskazujemy folder, gdzie leży stary kod GA (ga_neuron_search)
ga_dir = os.path.join(parent_dir, "ga_neuron_search")

# 4. Dodajemy go na sam początek listy ścieżek, w których Python szuka modułów
if ga_dir not in sys.path:
    sys.path.insert(0, ga_dir)

# Importujemy logikę SNN z głównego katalogu
from fitness import RealFitness, ParallelFitness
from ga import GAConfig, run_ga

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
    # (Tutaj wplatamy wszystkie poprawki z naszego code review!)
    rf_kwargs = dict(
        arch_dir=config.train, # zakładając, że config ma ścieżki z DataConfig
        data=config.train,     # Używamy kanonicznego zbioru Patryka
        val_data=config.val,
        test_data=config.test,
        limit=None,            # Pobieranie gotowego cache'u z dysku
        epochs=4,
        num_samples=6000,
        k=2,
        metric="clip_f1",      # Domyślna metryka dla GA
        fitness_seeds=3,       # Właśnie przywrócone kryterium akceptacji z #38!
        pos_weight=1.0,
        feature_penalty=0.005,
        channels_head=None,    # Pozwalamy GA wybrać spośród wszystkich dostępnych
        stream_budget=6.0,     # Zabezpieczony, jawny argument z ostatniego fixa
        stream_boot=0,
        verbose=False,         # Wyciszamy spam logów dla pojedynczych ocen
        seed=0,
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
        
    # 3. Konfiguracja GA (max 10 neuronów wg biznesowego ticketa)
    ga_cfg = GAConfig(
        n_total=10, 
        pop_size=24, 
        generations=15, 
        elite=3, 
        max_hidden_layers=4,
        seed=0
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
        "features_used": res.best.genome.features_used(),
        "best_topology": best_genome_dict
    })
    
    print(f"[GA] Sukces! Najlepszy clip-F1: {best_fitness:.4f}")
    return best_genome_dict