import os
import sys
import time
import json
from typing import Dict, Any
from tracker import SetEncoder

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
    print(f"\n>>> [ETAP 1/4] Inicjalizacja algorytmu genetycznego (GA)...")
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
        num_samples=config.train.num_samples, 
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
    print(f"\n>>> [ETAP 2/4] Uruchamianie ewaluacji spikes_ext...")
    
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
        num_samples=config.train.num_samples, 
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
    # POPRAWKA (M1 punkt 2, 25.09.2026 -- Marcel): `budget` w RealFitness.__call__
    # skaluje EPOKI (zakres 0..1, jak przy halvingu) -- to NIE jest stream_budget
    # (FA/h, osobny argument konstruktora, już ustawiony wyżej na 6.0). Pomylenie
    # tych dwóch "budgetów" (budget=6.0) trenowało 6x więcej epok niż
    # config.train.proxy_epochs mówi, po cichu. budget=1.0 = pełne proxy_epochs,
    # zgodnie z tym, czego reszta pipeline'u (np. run_ga_stage) i tak oczekuje.
    rf_ext = RealFitness(**rf_kwargs)
    ext_score = rf_ext(g, budget=1.0)
    
    # 4. Logowanie do trackera (bezpieczne rzutowanie na listę)
    tracker.log_stage_time("ext_eval_stage", time.time() - start_time)
    tracker.log_metrics("spikes_ext_eval", {
        "n_features_used": n_features,
        "features_used": sorted(features_used),
        f"ext_{config.ga.fitness_metric}": ext_score
    })
    
    print(f"[EXT] Wynik na spikes_ext ({config.ga.fitness_metric}): {ext_score:.4f}")
    print(f"[ETAP 2/4] Zakończono pomyślnie.")


def run_final_evaluation_stage(config: Any, tracker: Any, best_topology: Dict[str, Any]):
    """
    Etap 3: Ostateczna ewaluacja (Test Split / Continuous).
    Tymczasowo korzysta ze standardowego zbioru testowego. Gotowe do przepięcia na zbiór Kacpra.

    POPRAWKA (M1 punkt 2 -- "żaden checkpoint/seed/próg nie jest wybrany po
    końcowym teście", 25.09.2026, Marcel):

    Wcześniej val_data i test_data wskazywały na TEN SAM plik (test). RealFitness
    (fitness.py) NIE robi selekcji "po drodze" -- train_once trenuje przez
    ustaloną liczbę epok i raz na końcu woła eval_events(..., split="val");
    __call__ tylko uśrednia wynik po fitness_seeds (nie wybiera najlepszego
    seeda). Więc to nie był klasyczny cherry-picking po teście -- ale przez
    aliasing val=test kod czytał test pod etykietą "val" (myląco, i 3x w jednym
    wywołaniu, po fitness_seeds=3), co zaprasza przyszły błąd (ktoś doda
    prawdziwą selekcję "po val" nie wiedząc, że to w istocie test) i łamie
    zasadę "test dotykany raz, na końcu, wyłącznie do raportu".

    Teraz: val_data to prawdziwy, ODDZIELNY split walidacyjny (RealFitness go
    wymaga wewnętrznie, choć w tym etapie i tak nie wpływa na żadną decyzję --
    train_once nie ma early stoppingu). Wynik testowy liczony jest JAWNIE przez
    eval_events(model, split="test") na modelu ze świeżego treningu, osobno dla
    każdego seeda -- test czytany dokładnie raz na seed, wyłącznie do raportu,
    nigdy do żadnej decyzji (średnia po seedach nie jest selekcją: żaden seed
    nie jest odrzucany ani preferowany na podstawie wyniku na teście).

    Druga poprawka: usunięty błędny `budget=6.0` (patrz run_ext_evaluation_stage
    -- ten sam mixup `budget`/`stream_budget`, trenował 6x więcej epok niż
    config.train.proxy_epochs).
    """
    print(f"\n>>> [ETAP 3/4] Finalna ewaluacja ciągła...")

    # 1. Przygotowanie ścieżek
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_abs = os.path.join(project_root, config.data.train)
    val_abs = os.path.join(project_root, config.data.val)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))

    # UWAGA: Tutaj podmienić 'config.data.test' na 'config.data.continuous_kacper' - Dataset od Kacpra
    target_test_abs = os.path.join(project_root, config.data.test)

    g = Genome.from_dict(best_topology)

    # Prawdziwy, ROZDZIELONY val/test -- żadnego aliasingu.
    rf_kwargs = dict(
        arch_dir=arch_dir,
        data=train_abs,
        val_data=val_abs,           # prawdziwy split walidacyjny (config.data.val)
        test_data=target_test_abs,  # prawdziwy, nietknięty test -- czytany jawnie niżej
        limit=None,
        epochs=config.train.proxy_epochs,
        num_samples=config.train.num_samples,
        k=2,
        metric=config.ga.fitness_metric,
        fitness_seeds=3,
        pos_weight=1.0,
        feature_penalty=0.0,        # Na etapie testu nie karzemy już za cechy
        channels_head=7,            # Ograniczenie do oryginalnych 7 kanałów
        stream_budget=6.0,
        stream_boot=0,
        verbose=False,
        seed=config.seed,
        device=tracker.device
    )

    print(f"[TEST] Ładowanie modelu na strumień testowy: {target_test_abs}")
    start_time = time.time()

    rf_final = RealFitness(**rf_kwargs)

    # fitness_seeds świeżych treningów; TEST czytany jawnie i raz na seed,
    # wyłącznie do raportu -- średnia po seedach nie jest selekcją.
    metric_key = config.ga.fitness_metric
    test_scores = []
    for si in range(rf_final.fitness_seeds):
        _, _, _, model = rf_final.train_once(
            g, epochs=config.train.proxy_epochs, seed=config.seed + si, return_model=True
        )
        test_m = rf_final.eval_events(model, split="test")
        test_scores.append(test_m.get(metric_key, test_m.get("clip_f1", 0.0)))

    final_score = sum(test_scores) / len(test_scores)

    elapsed = time.time() - start_time
    tracker.log_stage_time("final_eval_stage", elapsed)

    # Zebranie metryk - obecnie wpadnie tu finalny wynik fitness,
    # ale struktura jest gotowa na przyjęcie pełnego słownika z FA/h i AP z logiki Kacpra
    real_test_metrics = {
        "dataset": "canonical_test_placeholder",
        f"test_{metric_key}": final_score,
        f"test_{metric_key}_per_seed": test_scores,
        "latency_sec": elapsed
    }

    tracker.log_metrics("continuous_test", real_test_metrics)
    print(f"[ETAP 3/4] Wynik testowy ({metric_key}: {final_score:.4f}, "
          f"per-seed: {[f'{s:.4f}' for s in test_scores]}) zapisany w manifeście.")


def run_hardware_export_stage(config: Any, tracker: Any, best_topology: Dict[str, Any]):
    """
    Etap 4: Eksport wyewoluowanej topologii i wag do formatu układu LUI.
    Dostosowane do dynamicznych warstw GenomeNet z wykorzystaniem stałych sprzętowych Patryka.
    """
    print(f"\n>>> [ETAP 4/4] Eksport konfiguracji sprzętowej LUI...")
    start_time = time.time()
    
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_abs = os.path.join(project_root, config.data.train)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))
    
    if arch_dir not in sys.path:
        sys.path.insert(0, arch_dir)
    
    # Uwaga: importowane lokalnie, ponieważ ścieżka do snn_hw_pipeline
    # zależy od config.data.train, znanego dopiero w runtime stąd możliwe podkreślenie.
    try:
        from snn_hw_pipeline import DT, V_TH, CHANNELS, W_DEADZONE, W_MAX, pulses_to_fire
    except ImportError as e:
        print(f"[BŁĄD] Nie można zaimportować snn_hw_pipeline: {e}")
        return

    def export_genome_config(model, path, extra=None):
        """Dynamiczny konwerter GenomeNet do LUI z obsługą zmiennej głębokości sieci."""
        cfg = {"dt_s": DT, "v_th": V_TH, "channels": CHANNELS, "boards": {}}
        if extra:
            cfg.update(extra)
            
        # Zamiast hardcodować [CHANNELS, model.H, model.G], wyciągamy nazwy dynamicznie z każdej warstwy
        pre_names = [CHANNELS] + [layer.names for layer in model.layers()[:-1]]
        
        for layer, pres in zip(model.layers(), pre_names):
            W = layer.weights().detach()
            vl, ts, tm = layer.v_leak().detach(), layer.tau_syn().detach(), layer.tau_mem().detach()
            
            for i, name in enumerate(layer.names):
                w = W[i]
                m = w.abs().max().item()
                if m < W_DEADZONE:
                    print(f"[!] {name}: wszystkie wagi w martwej strefie — neuron nieużywany")
                    continue
                    
                V_LEAK_MIN_HW = 0.20 * V_TH
                k_allow = (V_TH - V_LEAK_MIN_HW) / max(V_TH - vl[i].item(), 1e-3)
                k = min(W_MAX / m, k_allow)
                v_leak_hw = V_TH - k * (V_TH - vl[i].item())
                
                syn = []
                for j, pre in enumerate(pres):
                    if layer.mask[i, j] == 0:
                        continue
                    wij = w[j].item()
                    pot = 100.0 * abs(wij) * k / W_MAX
                    if pot < 5.0:
                        continue
                    syn.append({
                        "port": f"J{len(syn)+1}",
                        "from": pre,
                        "sign": "+" if wij >= 0 else "-",
                        "pot_pct": round(pot, 1),
                        "w_sim": round(wij, 4),
                        "pulses_to_fire_100Hz": pulses_to_fire(abs(wij) * k, ts[i].item(), tm[i].item(), v_leak_hw)
                    })
                    
                cfg["boards"][name] = {
                    "tau_syn_ms": round(1000 * ts[i].item(), 1),
                    "tau_mem_ms": round(1000 * tm[i].item(), 1),
                    "v_leak": round(v_leak_hw, 3),
                    "led_bar_pct": round(50.0 * v_leak_hw / V_TH, 1),
                    "scale_k": round(k, 3),
                    "synapses": syn,
                }
                
        with open(path, "w") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False, cls=SetEncoder)

    # 2. Inicjalizacja środowiska i pełny trening modelu pod eksport sprzętowy
    g = Genome.from_dict(best_topology)
    rf_export = RealFitness(
        arch_dir=arch_dir, 
        data=train_abs, 
        stream_budget=6.0,
        val_data=train_abs, 
        test_data=train_abs,
        epochs=config.train.proxy_epochs, 
        num_samples=config.train.num_samples, 
        metric=config.ga.fitness_metric, 
        channels_head=7, 
        device=tracker.device
    )
    
    print(f"[EXPORT] Trening finalnego modelu (Epoki: {config.train.proxy_epochs})...")
    metrics, _, _, final_model = rf_export.train_once(g, epochs=config.train.proxy_epochs, seed=config.seed, return_model=True)
    
    # 3. Wywołanie naszego dynamicznego konwertera
    export_dir = os.path.join(tracker.get_run_dir(), "hardware_export")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "hw_config.json")
    
    print(f"[EXPORT] Zrzucanie wag i topologii do {export_path}...")
    extra_meta = {
        "best_clip_f1": metrics.get(f"val_{config.ga.fitness_metric}", 0.0),
        "topology_manifest": best_topology
    }
    export_genome_config(final_model, export_path, extra=extra_meta)
    
    # 4. Finalizacja i logowanie
    tracker.log_stage_time("hardware_export", time.time() - start_time)
    tracker.log_metrics("hardware_export", {
        "pytorch_score": metrics.get(f"val_{config.ga.fitness_metric}", 0.0),
        "export_path": export_path
    })
    
    print(f"[ETAP 4/4] Zakończono pomyślnie. Artefakt gotowy do wdrożenia na płycie LUI.")
    