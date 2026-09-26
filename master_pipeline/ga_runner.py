import os
import sys
import time
import json
from typing import Dict, Any
from tracker import get_git_sha

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
        # M2 punkt 4 (25.09.2026, Marcel): było zahardkodowane fitness_seeds=3,
        # ignorując config.train.fitness_seeds (domyślnie 1 w TrainConfig) --
        # profil eksperymentu był po cichu ignorowany. UWAGA: to zmienia
        # faktyczne zachowanie na produkcyjne configi bez jawnego
        # fitness_seeds=3 (koszt/wariancja proxy-fitnessu w GA spadnie z 3
        # seedów do 1, chyba że config to nadpisze).
        fitness_seeds=config.train.fitness_seeds,
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
        fitness_seeds=config.train.fitness_seeds,  # M2 punkt 4 -- patrz run_ga_stage
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


def _sha256_of_file(path: str, buf_size: int = 1 << 20) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(buf_size), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_freeze_manifest(path: str = "freeze_manifest.json") -> Dict[str, Any]:
    """M2 punkt 3 (26.09.2026, Marcel): wspolne zrodlo prawdy dla lineage
    (encoder_hash, dataset hash), zamrozone w M1 punkt 1 przez freeze_m1.py.
    Brak pliku nie wywala pipeline'u -- zwraca puste pola z jawna notatka,
    zeby ktos nie pomyslal ze artefakt ma prawdziwy hash, ktorego nie ma."""
    if not os.path.exists(path):
        print(f"[LINEAGE] UWAGA: nie znaleziono {path} -- odpal freeze_m1.py "
              f"(M1 punkt 1) zanim to pojdzie do produkcji. Pola lineage "
              f"beda null.")
        return {
            "encoder_hash": None,
            "dataset_manifest_hash": {"train": None, "val": None, "test": None},
            "_source": None,
        }
    with open(path, "r", encoding="utf-8") as f:
        fm = json.load(f)
    return {
        "encoder_hash": fm.get("encoder", {}).get("encoder_hash"),
        "dataset_manifest_hash": {
            name: info.get("manifest_sha256")
            for name, info in fm.get("splits", {}).items()
        },
        "_source": path,
    }


def run_final_evaluation_stage(config: Any, tracker: Any, best_topology: Dict[str, Any]):
    """
    Etap 3: Trening zwycięskiej topologii + finalna ewaluacja testowa.

    POPRAWKA (M1 punkt 2, 25.09.2026, Marcel): val_data i test_data wskazywały
    kiedyś na TEN SAM plik (test) -- teraz to prawdziwe, rozdzielone splity.

    POPRAWKA (M2 punkt 2/3, 26.09.2026, Marcel): ten etap wcześniej miał
    WŁASNĄ reimplementację treningu (fitness_seeds osobnych train_once, każdy
    czytający test, uśrednionych na końcu) -- zduplikowaną i niespójną z tym,
    co robi run_hardware_export_stage (który trenował JESZCZE RAZ, od zera,
    innym seedem, z val_data=test_data=train_abs). To znaczyło, że wagi
    wyeksportowane do hw_config.json NIE były tymi samymi wagami, których
    wynik trafiał do raportu testowego.

    Teraz: korzystamy z winner.train_full (ten sam kod projektu, już napisany
    pod dokładnie ten cel) -- trenuje `winner_seeds` niezależnych przebiegów
    HAT->QAT, wybiera MEDIANĘ po metryce walidacyjnej (odporność na przypadek,
    nie "najlepszy z rzutu"), i czyta test DOKŁADNIE RAZ dla tego jednego
    wybranego modelu (zgodnie z zasadą "test dotykany raz, wyłącznie do
    raportu"). Zapisuje checkpoint na dysk -- run_hardware_export_stage
    wczytuje TEN SAM plik zamiast trenować ponownie.

    UWAGA: to zmienia sposób raportowania względem poprzedniej wersji -- nie
    ma już `test_{metric}_per_seed` (średnia z fitness_seeds odczytów testu).
    Zamiast tego jest jeden odczyt testu dla wybranego (medianowego) modelu,
    plus `val_{metric}_median_of_N_seeds` jako miara odporności na poziomie
    walidacji. To ściślej trzyma się zasady odbioru M1 (test czytany raz), ale
    to świadoma zmiana zakresu raportu -- flagowane tutaj, żeby nikt nie był
    zaskoczony brakiem per-seed testu w manifeście.
    """
    print(f"\n>>> [ETAP 3/4] Trening zwycięzcy (winner.train_full) + finalna ewaluacja...")

    # 1. Przygotowanie ścieżek
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_abs = os.path.join(project_root, config.data.train)
    val_abs = os.path.join(project_root, config.data.val)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))
    if arch_dir not in sys.path:
        sys.path.insert(0, arch_dir)

    # UWAGA: Tutaj podmienić 'config.data.test' na 'config.data.continuous_eval'
    # (patrz pipeline_config.py -- juz zbudowany, 26.09.2026) po potwierdzeniu
    # ze RealFitness/eval_events umie czytac format continuous (audio+manifest),
    # a nie tylko katalog splitu z files.csv.
    target_test_abs = os.path.join(project_root, config.data.test)

    g = Genome.from_dict(best_topology)

    decoder_k = 2  # dekoder: >=k spikow D w oknie = alarm (patrz winner.tune_k)

    # Prawdziwy, ROZDZIELONY val/test -- żadnego aliasingu.
    rf_kwargs = dict(
        arch_dir=arch_dir,
        data=train_abs,
        val_data=val_abs,           # prawdziwy split walidacyjny (config.data.val)
        test_data=target_test_abs,  # prawdziwy, nietknięty test -- czytany jawnie niżej
        limit=None,
        epochs=config.train.winner_epochs,
        num_samples=config.train.num_samples,
        k=decoder_k,
        metric=config.ga.fitness_metric,
        fitness_seeds=config.train.fitness_seeds,  # M2 punkt 4 -- patrz run_ga_stage
        pos_weight=1.0,
        feature_penalty=0.0,        # Na etapie testu nie karzemy już za cechy
        channels_head=7,            # Ograniczenie do oryginalnych 7 kanałów
        stream_budget=6.0,
        stream_boot=0,
        verbose=False,
        seed=config.seed,
        device=tracker.device
    )

    print(f"[WINNER] Trening pełny (winner.train_full, epoki={config.train.winner_epochs}, "
          f"winner_seeds={config.train.winner_seeds})...")
    start_time = time.time()

    rf_final = RealFitness(**rf_kwargs)

    from winner import train_full  # lokalny import -- wymaga arch_dir juz w sys.path

    ckpt_path = os.path.join(tracker.get_run_dir(), "winner_checkpoint.pt")
    metric_key = config.ga.fitness_metric

    best_model, median_m, final_m = train_full(
        rf_final, g,
        epochs=config.train.winner_epochs,
        hat_frac=config.train.hat_frac,
        lr=config.train.lr,
        pos_weight=1.0,
        seeds=config.train.winner_seeds,
        select_metric=metric_key,
        ckpt=ckpt_path,
        log=print,
    )

    elapsed = time.time() - start_time
    tracker.log_stage_time("final_eval_stage", elapsed)

    checkpoint_sha256 = _sha256_of_file(ckpt_path) if os.path.exists(ckpt_path) else None

    real_test_metrics = {
        # M2 punkt 1 (25.09.2026, Marcel): usunieta myląca etykieta
        # "canonical_test_placeholder" pozostała po buggu z aliasingiem
        # val=test (naprawionym w M1 punkt 2) -- teraz zapisujemy realną
        # ścieżkę splitu testowego, żeby manifest był wiarygodny.
        "dataset": config.data.test,
        f"test_{metric_key}": final_m.get(metric_key, final_m.get("clip_f1", 0.0)),
        # M2 punkt 2/3: jeden odczyt testu (dla modelu wybranego jako mediana
        # z winner_seeds przebiegow na val) -- nie usredniamy juz odczytow testu.
        f"val_{metric_key}_median_of_{config.train.winner_seeds}_seeds":
            median_m.get(metric_key, median_m.get("clip_f1", 0.0)),
        "winner_seeds": config.train.winner_seeds,
        "decoder_k": decoder_k,
        "latency_sec": elapsed,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": checkpoint_sha256,
        "seed": config.seed,
    }

    tracker.log_metrics("continuous_test", real_test_metrics)
    print(f"[ETAP 3/4] Test ({metric_key}): {real_test_metrics[f'test_{metric_key}']:.4f} "
          f"(checkpoint: {ckpt_path}, sha256={(checkpoint_sha256 or '')[:16]}...)")


def run_hardware_export_stage(config: Any, tracker: Any, best_topology: Dict[str, Any]):
    """
    Etap 4: Eksport wyewoluowanej topologii i wag do formatu układu LUI.

    POPRAWKA (M2 punkt 2/3, 26.09.2026, Marcel): ten etap trenował WŁASNY
    model od zera (inny seed, val_data=test_data=train_abs -- nawet nie
    prawdziwa walidacja) zamiast eksportować to, co run_final_evaluation_stage
    już wytrenowało i oceniło na teście. Wagi w hw_config.json NIE były więc
    tymi, których wynik trafiał do raportu.

    Teraz: eksport NIE trenuje niczego. Wczytuje checkpoint zapisany przez
    run_final_evaluation_stage (winner.train_full) i serializuje dokładnie
    te wagi. Test niezmienności: liczymy sha256 pliku checkpointu przed i po
    wywołaniu eksportu -- eksport jest czysto do odczytu, nigdy nie modyfikuje
    ani nie retrenuje modelu.
    """
    print(f"\n>>> [ETAP 4/4] Eksport konfiguracji sprzętowej LUI...")
    start_time = time.time()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_abs = os.path.join(project_root, config.data.train)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))

    if arch_dir not in sys.path:
        sys.path.insert(0, arch_dir)

    # 1. Znajdź checkpoint zapisany przez run_final_evaluation_stage. Wymaga,
    # zeby etap 3 uruchomil sie najpierw w tym samym runie (albo zeby jego
    # metryki zostaly odtworzone przez --resume z manifest.json) -- pipeline.py
    # zawsze woła stage 3 przed stage 4, wiec w normalnym przebiegu to zawsze
    # bedzie ustawione.
    final_metrics = tracker.metrics.get("continuous_test", {})
    ckpt_path = final_metrics.get("checkpoint_path")
    if not ckpt_path or not os.path.exists(ckpt_path):
        raise RuntimeError(
            f"[EXPORT] Brak checkpointu z run_final_evaluation_stage (szukano: "
            f"{ckpt_path!r}). M2 punkt 2: eksport nie trenuje juz wlasnego "
            f"modelu -- uruchom najpierw etap 3 (final evaluation) w tym samym "
            f"runie, albo wznow z --resume runu, ktory juz go ma."
        )

    checkpoint_sha256_before = _sha256_of_file(ckpt_path)

    # 2. Odtworzenie modelu z checkpointu -- ta sama konstrukcja co w
    # winner.train_full/fitness.py (net.GenomeNet(g, hw=None, quantize=False),
    # potem set_quantize(True) przed load_state_dict, zgodnie z konwencja
    # projektu w snn_hw_pipeline.py przy wczytywaniu ckpt).
    import torch
    import net
    from winner import export_genome_config as winner_export_genome_config
    from snn_hw_pipeline import CHANNELS

    print(f"[EXPORT] Ładowanie checkpointu: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=tracker.device)
    g = Genome.from_dict(ckpt.get("topology", best_topology))
    model = net.GenomeNet(g, hw=None, quantize=False).to(tracker.device)
    model.set_quantize(True)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Lineage: encoder_hash + dataset hashes z freeze_manifest.json (M1
    # punkt 1), zamiast trzymac osobna, mogaca sie rozjechac kopie.
    lineage = _load_freeze_manifest()

    export_dir = os.path.join(tracker.get_run_dir(), "hardware_export")
    os.makedirs(export_dir, exist_ok=True)
    export_path = os.path.join(export_dir, "hw_config.json")

    print(f"[EXPORT] Zrzucanie wag i topologii do {export_path}...")
    extra_meta = {
        "best_clip_f1": ckpt.get("metrics", {}).get(config.ga.fitness_metric, 0.0),
        "topology_manifest": best_topology,

        # M2 punkt 3 (26.09.2026, Marcel): lineage do odtworzenia dokladnie
        # tego artefaktu -- source_commit z tracker.get_git_sha() (ten sam,
        # ktory trafia do manifest.json), seed, hashe datasetow/enkodera z
        # freeze_manifest.json (M1 punkt 1), checkpoint_hash policzony PRZED
        # eksportem (test niezmiennosci nizej sprawdza ze sie nie zmienil),
        # decoder (prog k dekodera zdarzeniowego) i jednostki pol ponizej.
        "source_commit": get_git_sha(),
        "seed": config.seed,
        "encoder_hash": lineage["encoder_hash"],
        "dataset_manifest_hash": lineage["dataset_manifest_hash"],
        "lineage_source": lineage["_source"],
        "checkpoint_path": ckpt_path,
        "checkpoint_hash": checkpoint_sha256_before,
        "decoder": {"k": final_metrics.get("decoder_k", 2),
                    "note": ">= k spikow neuronu D w oknie = alarm (winner.tune_k)"},
        "units": {
            "dt_s": "s", "v_th": "V", "tau_syn_ms": "ms", "tau_mem_ms": "ms",
            "v_leak": "V", "led_bar_pct": "%", "pot_pct": "% zakresu trymera",
            "pulses_to_fire_100Hz": "liczba impulsow @ 100Hz drive",
        },

        # M1 punkt 1/3 (25.09.2026, Marcel): RAM/czas kompilacji dla Uno (ATmega328P)
        # są dziś zmierzone tylko przez `simavr` (symulator cyklowo-dokładny) --
        # Kacper nie ma obecnie możliwości uruchomienia tego na fizycznej płytce
        # (tylko on ma do niej dostęp, brak czasu). Decyzja: nie blokujemy M1 na
        # tym punkcie, ale jawnie oznaczamy wynik jako "Estimated", nie "Measured"
        # (ta sama konwencja co ekran Energy w planie UI), żeby ryzyko było widoczne
        # w artefakcie używanym do budowy fizycznej, a nie ciche.
        "platform_validation": {
            "uno_atmega328p": {
                "status": "Estimated",
                "source": "simavr (cycle-accurate simulator, nie fizyczny krzem)",
                "note": (
                    "RAM/czas kompilacji NIE potwierdzone na fizycznym ATmega328P. "
                    "Do zweryfikowania przez Kacpra, gdy będzie miał czas/dostęp do "
                    "płytki. Nie blokuje M1 -- traktować jako otwarte ryzyko."
                ),
            }
        },
    }
    winner_export_genome_config(model, export_path, channels=CHANNELS, extra=extra_meta)

    # 4. Test niezmienności: eksport jest czysto do odczytu -- checkpoint na
    # dysku (i wagi w modelu) nie mogły się zmienić w trakcie eksportu.
    checkpoint_sha256_after = _sha256_of_file(ckpt_path)
    if checkpoint_sha256_before != checkpoint_sha256_after:
        raise RuntimeError(
            "[EXPORT] Checkpoint zmienil sie w trakcie eksportu "
            f"({checkpoint_sha256_before} -> {checkpoint_sha256_after}) -- "
            "eksport NIE powinien modyfikowac ani trenowac modelu (M2 punkt 2/3)."
        )

    # 5. Finalizacja i logowanie
    tracker.log_stage_time("hardware_export", time.time() - start_time)
    tracker.log_metrics("hardware_export", {
        "pytorch_score": extra_meta["best_clip_f1"],
        "export_path": export_path,
        "checkpoint_hash": checkpoint_sha256_before,
        "checkpoint_hash_verified_unchanged": True,
    })

    print(f"[ETAP 4/4] Zakończono pomyślnie. Artefakt gotowy do wdrożenia na płycie LUI.")
    