#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
continuous_eval.py -- ciagla ewaluacja championa na strumieniu 600s
(dataset/continuous/out/*.manifest.json), zamiast na pocietych klipach ze
splitu test.

KONTEKST / DECYZJE PODJETE PRZY PISANIU TEGO SKRYPTU (do potwierdzenia z
Marcelem w M4 -- zaden z ponizszych punktow nie jest "odtworzeniem" istniejacej,
jawnie zapisanej reguly, bo taka reguła nigdzie nie istnieje):

1. DEKODER OPERACYJNY (k, w, refrac).
   `net.genome_eval_events`/`decoder_k` (zahardkodowane k=2 w CZTERECH
   miejscach: ga_runner.py x3, hardware.py x1 -- osobny, udokumentowany bug,
   komentarz w kazdym z nich mylnie odsyla do "winner.tune_k", ktore nigdy nie
   jest wolane) to INNA regula dekodera niz ta, ktora FAKTYCZNIE zdecydowala o
   championie. Champion jest wybierany przez `fitness_metric="recall_fa"`
   (freeze_manifest.json, pipeline_config.py) -- ta sciezka idzie przez
   `RealFitness.stream_recall` -> `stream_eval_torch.evaluate_stream` ->
   `stream_eval.stream_report`, ktora przeszukuje `DEFAULT_RULES` (pary k,w)
   i wybiera te, ktora miesci `stream_budget` (domyslnie 6.0 FA/h, identyczne
   we wszystkich etapach pipeline'u). `k=2` przekazywane do RealFitness NIE
   jest w ogole uzywane przez ta sciezke (potwierdzone czytajac fitness.py:
   `stream_recall`/`stream_report_test` nie przekazuja `k` do `evaluate_stream`).

   Ta wybrana (k,w) NIGDY nie jest zapisywana do manifest.json (potwierdzone
   czytajac champion.py -- `report_to_dict`, ktory serializowalby `rule_k_w`,
   nigdzie nie jest wolany w ga_runner.py). Wiec ten skrypt ODTWARZA selekcje
   wprost: woła `RealFitness.stream_recall(model, g)` na splicie VAL (tym
   samym, na ktorym GA/winner.py selekcjonowaly), i uzywa
   `report[stream_budget].rule` jako reguly operacyjnej. `refrac` jest zawsze
   `stream_eval.DEFAULT_REFRAC` = 500 ramek (5s), niezaleznie od reguly.

   Jesli zaden (k,w) nie miesci budzetu na val dla tego championa -- to jest
   "infeasible" w tym samym sensie co w champion.py (patrz jego docstring:
   recall_fa==0 to legalna podloga, nie blad) i skrypt PRZERYWA z jasnym
   komunikatem zamiast cicho zwracac zera.

2. ENKODER (audio -> kanaly spike'owe).
   Idzie przez `architecture_14_neurons_patryk_09_07/encoder_twin.py`
   (`encode_file`) -- TEN SAM kod, ktory zbudowal `spikes_v2`/`spikes_ext`
   (`build_manifest`/`build_dataset`). Ciagly strumien 600s jest juz JEDNYM
   plikiem .wav (`stream_builder.py`), wiec feedujemy go w calosci przez
   `encode_file` z JEDNYM swiezym `EncoderState()` -- floor/MAD stabilizuje
   sie SAM w pierwszych sekundach strumienia. To jest zgodne z projektem
   manifestu: `warmup_s` (domyslnie 30s) jest wlasnie po to, zeby wykluczyc
   z FA/h okres, w ktorym floor jeszcze nie jest ustabilizowany -- NIE
   rozgrzewamy enkodera osobno na zewnetrznym tle (w odroznieniu od
   `build_manifest`, ktory rozgrzewa wspolny stan PRZED zapisem CSV, bo tam
   kazdy plik jest osobnym krotkim klipem bez wlasnego warmupu).

3. GAIN.
   `global_gain.json` nie istnieje dla obecnego `spikes_v2` (zbudowany
   najwyrazniej starsza sciezka `build_dataset`, nie `build-manifest` --
   `git log --all` nic nie znalazl, na dysku tez brak). Bez zamrozonego
   wzmocnienia, ten skrypt PRZELICZA je swiezo przez
   `encoder_twin.compute_global_gain()` na plikach .wav ze splitu train
   (metoda "all-files", percentyl 99.9 -- domyslne w encoder_twin.py), zamiast
   cicho udawac, ze to ten sam gain, ktorego uzyto przy oryginalnym treningu.
   Zapisane w metrykach jako `gain_source: "recomputed_fresh_no_frozen_record_found"`.
   Wymaga listy plikow .wav splitu train (`--dataset-manifest-csv`, kolumny
   filepath+split z `dataset/versions/vX.Y.Z/manifest.csv`) -- `spikes_v2` sam
   w sobie (CSV z juz zakodowanymi kanalami) nie ma juz sciezek do surowego
   audio.

4. RECALL / LATENCY / FA-H.
   Dokladnie wg definicji w `dataset/continuous/eval/manifest.py` (docstring
   modulu, sekcja "Jak Marcel liczy metryki"):
     - recall zdarzenia i: alarm w oknie [start_s, end_s + tolerancja]
     - latency: czas miedzy start_s a PIERWSZYM alarmem w tym oknie
     - FA/h: alarmy POZA wszystkimi oknami zdarzen, liczone na odcinku
       [warmup_s, duration_s] (warmup wylaczony), podzielone przez
       (duration_s - warmup_s) / 3600
   `tolerancja` NIE zostala ustalona w zrodle (nazwany, ale niewypelniony
   placeholder w docstringu manifest.py -- nie ma jej w polu `config`
   manifestu ani nigdzie indziej). Ten skrypt przyjmuje domyslnie
   `tolerancja = 0.0` (scisle okno zdarzenia, bez naciagania) jako jawna,
   bezpieczna decyzje -- parametryzowane przez `--recall-tolerance-s`, do
   potwierdzenia/nadpisania przez Marcela. Dopasowanie alarm<->zdarzenie jest
   ONE-TO-ONE: alarm juz przypisany do zdarzenia (w kolejnosci chronologicznej)
   nie moze zostac ponownie przypisany do kolejnego zdarzenia, nawet gdy
   `tolerancja`>0 sprawia, ze dwa sasiednie okna zdarzen by sie nakladaly.

5. MANIFEST "KACPRA" (rozstrzygniete 27.09.2026, potwierdzone przez Marcela):
   sposob budowania (schemat manifestu, stream_builder.py, annotations.py)
   pochodzi z brancha Kacpra, ale SAM DATASET (audio + `*.manifest.json` w
   `dataset/continuous/out`) zbudowal Marcel samodzielnie. Wiec wejscie
   czytane przez ten skrypt (`dataset/continuous/out/*.manifest.json`,
   schemat z `dataset/continuous/eval/manifest.py`) JEST tym, o czym mowi
   zadanie M4 -- nie ma osobnego, innego manifestu od Kacpra do podmiany.

6. CI DLA FA/H I POROWNANIE DO METRYKI KLIPOWEJ (kryterium odbioru M4, punkt
   3: "Raportowac przedzial ufnosci FA/h i roznice do historycznej metryki na
   klipach"; "Przy zerowym FA wynik ma dodatnia gorna granice niepewnosci").
   FA/h to zliczenie zdarzen Poissona na skonczonej ekspozycji (godziny tla)
   -- CI liczone dokladna metoda Poissona (Garwood, przez `chi2.ppf`), NIE
   przyblizeniem normalnym (ktore przy FA=0 dalby CI=[0,0], czyli fałszywa
   "obietnice braku alarmow" -- dokladnie to, czego kryterium odbioru
   zabrania). Przy FA=0 gorna granica wychodzi > 0 (tzw. "rule of three" jest
   szczegolnym przypadkiem tej samej formuly). Liczone i per-strumien (w
   `per_seed[i]["fa_per_hour_ci"]`), i zbiorczo po wszystkich streamach
   (`fa_per_hour_ci_pooled`, sumujac zdarzenia i godziny ekspozycji -- NIE
   usredniajac CI, bo usrednianie przedzialow ufnosci jest statystycznie
   niepoprawne).
   Historyczna metryka klipowa: liczona SWIEZO na tym samym championie przez
   `RealFitness.eval_events(model, split="test")` (dokladnie ta funkcja, ktora
   liczyla `clip_recall`/`clip_fa_rate` dotychczas), z `k` = pierwszy element
   odtworzonej reguly operacyjnej (punkt 1) -- dla najuczciwszego porownania
   wspolnym mianownikiem. UWAGA JEDNOSTEK: `clip_fa_rate` to alarmy/klip, nie
   alarmy/h -- ten skrypt NIE odejmuje ich bezposrednio od `fa_per_hour`
   (rozne jednostki), tylko raportuje oba obok siebie plus deltę recall
   (`recall_mean - clip_recall`, ta sama jednostka, sensowna roznica).

7. KAZDY FA NA OSI CZASU (kryterium odbioru M4). Pelna lista znacznikow
   czasowych kazdego alarmu sklasyfikowanego jako FA jest w
   `per_seed[i]["fa_times_s"]` (sekundy od poczatku strumienia) -- oraz w
   `false_alarms.csv` przy zapisie CSV (patrz CLI/`--out-csv-dir`).

Uzycie (samodzielne, lub jako Etap 5 w pipeline.py):
    python3 continuous_eval.py --config config.json \
        --checkpoint runs/run_XXXX/winner_checkpoint.pt \
        --continuous-dir ../dataset/continuous/out \
        --dataset-manifest-csv ../dataset/versions/v2.0.0/manifest.csv \
        --out-csv-dir runs/run_XXXX/continuous_eval_csv

Przekazanie (M4, "Karolina dostaje CSV/JSON metryk; Andrzej identyczny zestaw
do hardware"): `--out`/`log_metrics` daje JSON, `--out-csv-dir` daje DWA CSV
(`events.csv` per-zdarzeniowy, `false_alarms.csv` per-alarm) -- ten SAM
komplet plikow idzie do obu odbiorcow, zeby nie bylo dwoch, moglych sie
rozjechac, wersji tej samej liczby.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================ statystyka FA/h

def _poisson_rate_ci(n_events: int, exposure_hours: float,
                     alpha: float = 0.05) -> Tuple[float, float]:
    """Dokladny, DWUSTRONNY (Garwood) przedzial ufnosci dla stopy procesu
    Poissona -- NIE przyblizenie normalne, ktore przy n_events=0 dalby [0,0]
    (falszywa 'obietnica braku alarmow', zakazana przez kryterium odbioru
    M4). Dla n_events=0 gorna granica tego dwustronnego CI wychodzi ~3.69/T
    (alpha/2=0.025 w gornym ogonie) -- pokrewna, ale NIE identyczna z
    popularna jednostronna 'rule of three' (~3.0/T, ktora uzywa alpha=0.05
    wprost, nie alpha/2); obie sa dodatnie i obie spelniaja kryterium odbioru
    ('gorna granica > 0'), wybieram dwustronna bo to standardowe znaczenie
    'przedzialu ufnosci' (ma tez sensowna dolna granice, nie tylko gorna).
    Wymaga scipy (juz uzywane w tym repo, np. przez encoder_twin.py:
    scipy.signal.lfilter)."""
    from scipy.stats import chi2

    if exposure_hours <= 0:
        return (0.0, float("inf"))
    lower = 0.0 if n_events == 0 else 0.5 * chi2.ppf(alpha / 2, 2 * n_events) / exposure_hours
    upper = 0.5 * chi2.ppf(1 - alpha / 2, 2 * (n_events + 1)) / exposure_hours
    return (float(lower), float(upper))


# ============================================================ gain

def _load_train_wav_paths(dataset_manifest_csv: Optional[str], root: str) -> List[str]:
    """Czyta kolumny filepath/split z dataset/versions/vX.Y.Z/manifest.csv
    (schemat dataset_contract.py) i zwraca sciezki bezwzgledne plikow train."""
    if not dataset_manifest_csv or not os.path.exists(dataset_manifest_csv):
        return []
    import csv
    paths = []
    with open(dataset_manifest_csv, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("split") == "train":
                paths.append(os.path.join(root, r["filepath"]))
    return paths


def _resolve_gain(train_wav_paths: List[str], spikes_train_dir: str,
                  percentile: float, method: str,
                  gain_file: Optional[str]) -> Tuple[float, str]:
    """Zwraca (gain, gain_source). Kolejnosc prob (patrz punkt 3 docstringu
    modulu):
      1. --gain-file jawnie podany i istnieje -> wczytaj.
      2. Obok spikes_train_dir (tam, gdzie `encoder_twin.py build-manifest`
         by go zapisal, gdyby spikes_v2 bylo zbudowane ta sciezka) ->
         wczytaj, jesli istnieje.
      3. Brak zamrozonego rekordu -> policz swiezo `compute_global_gain()` na
         `train_wav_paths` i oznacz jawnie jako
         "recomputed_fresh_no_frozen_record_found" -- NIE jest to gwarantowane
         byc bit-identyczne z gainem uzytym przy oryginalnym treningu."""
    from encoder_twin import compute_global_gain  # arch_dir juz w sys.path

    candidates = []
    if gain_file:
        candidates.append(gain_file)
    candidates.append(os.path.join(os.path.dirname(spikes_train_dir.rstrip("/")), "global_gain.json"))
    candidates.append(os.path.join(spikes_train_dir, "global_gain.json"))

    for path in candidates:
        if path and os.path.exists(path):
            try:
                cached = json.load(open(path, encoding="utf-8"))
                print(f"[GAIN] Wczytano zamrożone wzmocnienie {cached['gain']:.4f} z {path}")
                return float(cached["gain"]), f"frozen:{path}"
            except (json.JSONDecodeError, KeyError) as e:
                print(f"[GAIN] {path} nieczytelny ({e}) -- pomijam")

    if not train_wav_paths:
        raise RuntimeError(
            "[GAIN] Brak zamrożonego global_gain.json i brak listy plików train "
            "do świeżego przeliczenia -- podaj --dataset-manifest-csv (kolumny "
            "filepath,split z dataset/versions/vX.Y.Z/manifest.csv) albo --gain-file."
        )
    print(f"[GAIN] Brak zamrożonego wzmocnienia -- liczę świeżo z "
          f"{len(train_wav_paths)} plików train (percentyl {percentile}, metoda "
          f"{method}). UWAGA: to NIE jest gwarantowane być tym samym gainem, "
          f"którego użyto przy budowie obecnego spikes_v2 (M4: gain_source).")
    gain = compute_global_gain(train_wav_paths, percentile=percentile, method=method)
    return gain, "recomputed_fresh_no_frozen_record_found"


# ============================================================ enkodowanie strumienia

def _encode_continuous_audio(audio_path: str, gain: float) -> Tuple[np.ndarray, float]:
    """Koduje CAŁY ciągły plik .wav na kanały spike'owe JEDNYM wywołaniem
    `encode_file` z fresh `EncoderState()` (patrz punkt 2 docstringu modułu)."""
    from encoder_twin import encode_file, EncoderState, HOP_SAMPLES, FS_HZ

    spikes = encode_file(audio_path, gain=gain, state=EncoderState())
    frame_dt = HOP_SAMPLES / FS_HZ
    return spikes, frame_dt


def _run_model_get_d_spikes(model, spikes_in: np.ndarray, device: str) -> np.ndarray:
    """Jeden forward całego strumienia (bez cięcia na okna) -- membrana D musi
    płynąć ciągle przez cały strumień, analogicznie do `d_spike_trains` w
    `stream_eval_torch.py` (tam też jeden forward per klip; tu klip = cały
    strumień 600s)."""
    import torch
    with torch.no_grad():
        x = torch.from_numpy(spikes_in.astype(np.float32)).unsqueeze(0).to(device)
        so = model(x)["so"][0, :, 0].detach().cpu().numpy()
    return (so > 0.5).astype(np.uint8)


def _count_alarms_with_times(train_d: np.ndarray, k: int, w: int, refrac: int) -> List[int]:
    """Jak `stream_eval.count_alarms`, ale zwraca INDEKSY RAMEK alarmów (nie
    tylko licznik) -- potrzebne do recall/latency per-zdarzeniowego, którego
    `count_alarms` nie daje (on tylko liczy alarmy całego klipu). Logika
    identyczna 1:1, żeby FA/h tego skryptu było policzalne tą samą metodą co
    przy selekcji championa."""
    times = np.where(train_d)[0]
    alarms: List[int] = []
    i = 0
    while i + k - 1 < len(times):
        if times[i + k - 1] - times[i] < w:
            alarm_frame = int(times[i + k - 1])
            alarms.append(alarm_frame)
            i = int(np.searchsorted(times, alarm_frame + refrac))
        else:
            i += 1
    return alarms


# ============================================================ metryki wg manifest.py

def _score_manifest(manifest: dict, alarm_frames: List[int], frame_dt: float,
                    tolerance_s: float) -> Dict[str, Any]:
    """Recall/latency per zdarzenie + FA/h na tle, wg definicji w
    dataset/continuous/eval/manifest.py (sekcja "Jak Marcel liczy metryki").

    Dopasowanie alarm<->zdarzenie jest ONE-TO-ONE (M4 punkt 2): zdarzenia są
    przetwarzane chronologicznie, a alarm raz przypisany do zdarzenia jest
    wyjęty z puli i nie może posłużyć do wykrycia kolejnego -- inaczej przy
    `tolerance_s`>0 i blisko siebie leżących zdarzeniach jeden alarm mógłby
    fałszywie "wykryć" dwa zdarzenia naraz."""
    alarm_times_s = sorted(a * frame_dt for a in alarm_frames)
    warmup_s = manifest["config"]["warmup_s"]
    duration_s = manifest["audio"]["duration_s"]
    events = sorted(manifest["events"], key=lambda e: e["start_s"])

    used = [False] * len(alarm_times_s)
    per_event = []
    for ev in events:
        lo, hi = ev["start_s"], ev["end_s"] + tolerance_s
        hit_idx = next((i for i, t in enumerate(alarm_times_s)
                        if not used[i] and lo <= t <= hi), None)
        detected = hit_idx is not None
        latency_s = None
        if detected:
            used[hit_idx] = True
            latency_s = alarm_times_s[hit_idx] - ev["start_s"]
        per_event.append({
            "index": ev["index"], "start_s": ev["start_s"], "end_s": ev["end_s"],
            "detected": detected, "latency_s": latency_s,
            "is_contaminated": ev.get("is_contaminated", False),
        })

    # FA/h: alarmy poza WSZYSTKIMI oknami zdarzeń (niezależnie, czy zużytymi
    # do dopasowania one-to-one powyżej -- okno zdarzenia wyklucza z FA
    # KAŻDY alarm w nim leżący, nie tylko ten jeden dopasowany), na
    # [warmup_s, duration_s] -- warmup wyłączony z FA (floor jeszcze się
    # stabilizuje w tym okresie, patrz punkt 2 docstringu modułu).
    event_windows = [(ev["start_s"], ev["end_s"] + tolerance_s) for ev in events]
    fa_times_s = []
    for t in alarm_times_s:
        if t < warmup_s or t > duration_s:
            continue
        if any(lo <= t <= hi for lo, hi in event_windows):
            continue
        fa_times_s.append(t)
    fa_count = len(fa_times_s)
    background_hours = (duration_s - warmup_s) / 3600.0
    fa_per_hour = fa_count / max(background_hours, 1e-9)
    fa_per_hour_ci = _poisson_rate_ci(fa_count, background_hours)

    n_detected = sum(1 for e in per_event if e["detected"])
    recall = n_detected / max(len(per_event), 1)
    latencies = [e["latency_s"] for e in per_event if e["detected"]]

    return {
        "seed": manifest["seed"],
        "n_events": len(per_event),
        "n_detected": n_detected,
        "recall": recall,
        "mean_latency_s": float(np.mean(latencies)) if latencies else None,
        "fa_count": fa_count,
        "background_hours": background_hours,
        "fa_per_hour": fa_per_hour,
        "fa_per_hour_ci": {"lower": fa_per_hour_ci[0], "upper": fa_per_hour_ci[1],
                          "method": "poisson_exact_garwood", "alpha": 0.05},
        "fa_times_s": fa_times_s,
        "per_event": per_event,
    }


# ============================================================ championa + dekoder

def _load_champion_model(checkpoint_path: str, device: str):
    """Odtwarza model dokładnie jak `run_hardware_export_stage`
    (ga_runner.py): `net.GenomeNet(g, hw=None, quantize=False)` ->
    `set_quantize(True)` -> `load_state_dict` -> `eval()`."""
    import torch
    from ga_neuron_search.genome import Genome
    import net

    ckpt = torch.load(checkpoint_path, map_location=device)
    g = Genome.from_dict(ckpt["topology"])
    model = net.GenomeNet(g, hw=None, quantize=False).to(device)
    model.set_quantize(True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, g


def _build_real_fitness(config: Any, device: str, project_root: str):
    """Buduje `RealFitness` z DOKŁADNIE tymi samymi argumentami co
    `run_ga_stage`/`run_final_evaluation_stage` (ga_runner.py) -- ta sama
    instancja jest reużywana zarówno do odtworzenia reguły operacyjnej
    (`_recover_operating_rule`) jak i do świeżych metryk klipowych
    (`_historical_clip_metrics`), żeby nie ładować danych dwa razy."""
    from ga_neuron_search.fitness import RealFitness

    train_abs = os.path.join(project_root, config.data.train)
    val_abs = os.path.join(project_root, config.data.val)
    test_abs = os.path.join(project_root, config.data.test)
    arch_dir = os.path.dirname(os.path.dirname(train_abs))
    stream_budget = 6.0  # identyczne we wszystkich etapach (ga_runner.py, hardware.py)

    return RealFitness(
        arch_dir=arch_dir, data=train_abs, val_data=val_abs, test_data=test_abs,
        limit=None, epochs=config.train.proxy_epochs, num_samples=config.train.num_samples,
        k=2, metric=config.ga.fitness_metric, fitness_seeds=config.train.fitness_seeds,
        pos_weight=1.0, feature_penalty=config.ga.feature_penalty, channels_head=None,
        stream_budget=stream_budget, stream_boot=0, verbose=False, seed=config.seed,
        device=device,
    ), stream_budget


def _recover_operating_rule(rf, model, g, stream_budget: float) -> Tuple[Tuple[int, int], int]:
    """Odtwarza (k,w) operacyjne dokładnie tak, jak zdecydowało o championie
    `RealFitness.stream_recall` przy `fitness_metric="recall_fa"` (patrz punkt
    1 docstringu modułu) -- NIE czyta żadnego zapisanego pola, bo takiego pola
    nigdzie nie ma. Zwraca ((k, w), refrac_frames)."""
    from stream_eval import DEFAULT_REFRAC

    recall, report = rf.stream_recall(model, g)
    op = report.get(stream_budget)
    if op is None or op.rule is None:
        raise RuntimeError(
            f"[DECODER] stream_recall nie znalazł żadnej reguły (k,w) mieszczącej "
            f"się w budżecie {stream_budget} FA/h na val dla tego championa -- "
            f"budżet INFEASIBLE (analogicznie do champion.py status='infeasible', "
            f"patrz jego docstring o recall_fa==0 jako legalnej podłodze). "
            f"continuous_eval nie może policzyć sensownego FA/h/recall bez reguły "
            f"operacyjnej -- nie zgaduję zastępczej."
        )
    print(f"[DECODER] Odtworzona reguła operacyjna: k={op.rule[0]}, w={op.rule[1]} "
          f"(recall@val={recall:.3f} przy budżecie {stream_budget} FA/h, "
          f"refrac={DEFAULT_REFRAC} ramek)")
    return op.rule, DEFAULT_REFRAC


def _historical_clip_metrics(rf, model, k: int) -> Optional[Dict[str, Any]]:
    """M4 punkt 3: 'różnica do historycznej metryki na klipach'. Liczy
    ŚWIEŻO `clip_recall`/`clip_fa_rate`/`clip_f1` na tym samym championie
    przez `RealFitness.eval_events(model, split="test")` -- ta sama funkcja,
    która dotychczas raportowała metryki klipowe -- używając `k` z odtworzonej
    reguły operacyjnej (punkt 1 docstringu modułu), żeby porównanie miało
    wspólny mianownik zamiast dwóch niezależnie dobranych progów.

    Zwraca None (zamiast zgadywać) gdy RealFitness nie ma testowego splitu
    (brak `test_data` przy konstrukcji) -- lepiej jawnie brak porównania niż
    ciche 0.0."""
    try:
        return rf.eval_events(model, k=k, split="test")
    except (ValueError, FileNotFoundError) as e:
        print(f"[HISTORYCZNA METRYKA] Nie udało się policzyć clip-metrics "
              f"referencyjnych ({e}) -- porównanie do historycznej metryki "
              f"pominięte, nie zgaduję wartości.")
        return None


# ============================================================ etap glowny

def run_continuous_eval_stage(config: Any, tracker: Any, checkpoint_path: str,
                              continuous_dir: str,
                              dataset_manifest_csv: Optional[str] = None,
                              gain_file: Optional[str] = None,
                              recall_tolerance_s: float = 0.0,
                              csv_dir: Optional[str] = None) -> Dict[str, Any]:
    """Etap ciągłej ewaluacji: uruchamia championa (`checkpoint_path`) na
    każdym strumieniu 600s z `continuous_dir` (`*.manifest.json`), agreguje
    recall/latency/FA-h (z CI Poissona) po seedach, porównuje do świeżo
    policzonej historycznej metryki klipowej, loguje do trackera jako
    `"continuous_eval"` (obok istniejących `"continuous_test"` z
    `run_final_evaluation_stage` -- inna nazwa specjalnie, żeby nie nadpisać
    metryk klipowych test-splitu tą ciągłą oceną). Gdy `csv_dir` podany,
    zapisuje tam `events.csv` i `false_alarms.csv` (M4: "Karolina dostaje
    CSV/JSON metryk; Andrzej identyczny zestaw do hardware")."""
    print(f"\n>>> [CONTINUOUS EVAL] Ładowanie championa: {checkpoint_path}")
    project_root = _project_root()
    arch_dir = os.path.join(project_root, "architecture_14_neurons_patryk_09_07")
    for p in (project_root, os.path.join(project_root, "ga_neuron_search"), arch_dir):
        if p not in sys.path:
            sys.path.insert(0, p)

    device = tracker.device
    model, g = _load_champion_model(checkpoint_path, device)

    rf, stream_budget = _build_real_fitness(config, device, project_root)
    rule, refrac = _recover_operating_rule(rf, model, g, stream_budget)
    k, w = rule
    historical = _historical_clip_metrics(rf, model, k)

    train_abs = os.path.join(project_root, config.data.train)
    train_wav_paths = _load_train_wav_paths(dataset_manifest_csv, root=project_root)
    gain, gain_source = _resolve_gain(
        train_wav_paths, spikes_train_dir=train_abs,
        percentile=99.9, method="all-files", gain_file=gain_file,
    )

    manifest_paths = sorted(glob.glob(os.path.join(continuous_dir, "*.manifest.json")))
    if not manifest_paths:
        raise FileNotFoundError(
            f"[CONTINUOUS EVAL] brak *.manifest.json w {continuous_dir} -- odtwórz "
            f"strumienie komendą z DataConfig.continuous_eval (pipeline_config.py, "
            f"komentarz przy polu continuous_eval)."
        )

    per_seed_results = []
    for mpath in manifest_paths:
        manifest = json.load(open(mpath, encoding="utf-8"))
        audio_path = os.path.join(os.path.dirname(mpath), manifest["audio"]["path"])
        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"[CONTINUOUS EVAL] audio {audio_path} nie istnieje (manifest {mpath} "
                f"je referuje, ale audio NIE jest w gicie -- odtwórz komendą z "
                f"pipeline_config.py DataConfig.continuous_eval)."
            )
        expected_sha = manifest["audio"]["sha256"]
        actual_sha = hashlib.sha256(open(audio_path, "rb").read()).hexdigest()
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"[CONTINUOUS EVAL] {audio_path}: sha256 nie zgadza się z manifestem "
                f"({actual_sha[:16]}... != {expected_sha[:16]}...) -- audio zostało "
                f"zmienione albo to nie ten plik, dla którego zbudowano manifest."
            )

        print(f"[CONTINUOUS EVAL] seed={manifest['seed']}: kodowanie {audio_path}...")
        spikes_in, frame_dt = _encode_continuous_audio(audio_path, gain)
        train_d = _run_model_get_d_spikes(model, spikes_in, device)
        alarm_frames = _count_alarms_with_times(train_d, k, w, refrac)
        result = _score_manifest(manifest, alarm_frames, frame_dt, recall_tolerance_s)
        print(f"[CONTINUOUS EVAL] seed={manifest['seed']}: recall={result['recall']:.3f} "
              f"({result['n_detected']}/{result['n_events']}), "
              f"FA/h={result['fa_per_hour']:.3f}, "
              f"latency_srednia={result['mean_latency_s']}")
        per_seed_results.append(result)

    recalls = [r["recall"] for r in per_seed_results]
    fa_rates = [r["fa_per_hour"] for r in per_seed_results]
    all_latencies = [e["latency_s"] for r in per_seed_results for e in r["per_event"] if e["detected"]]

    # CI zbiorczy: PULOWANY po zdarzeniach/godzinach (nie średnia z CI per-seed
    # -- uśrednianie przedziałów ufności jest statystycznie niepoprawne;
    # patrz punkt 6 docstringu modułu).
    total_fa = sum(r["fa_count"] for r in per_seed_results)
    total_hours = sum(r["background_hours"] for r in per_seed_results)
    pooled_ci = _poisson_rate_ci(total_fa, total_hours)

    recall_delta_vs_historical = None
    if historical is not None and recalls:
        recall_delta_vs_historical = float(np.mean(recalls)) - historical["clip_recall"]

    summary = {
        "checkpoint_path": checkpoint_path,
        "decoder_rule": {
            "k": k, "w": w, "refrac_frames": refrac, "budget_fa_h": stream_budget,
            "note": ("Reguła odtworzona ponownym wywołaniem "
                     "RealFitness.stream_recall na val (fitness_metric="
                     f"{config.ga.fitness_metric!r}) -- NIE zapisana nigdzie w "
                     "oryginalnym manifest.json championa (patrz champion.py: "
                     "report_to_dict nigdy nie jest wołane)."),
        },
        "gain": gain, "gain_source": gain_source,
        "recall_tolerance_s": recall_tolerance_s,
        "recall_tolerance_note": ("'tolerancja' z docstringu manifest.py nie była "
                                  "ustalona w źródle -- to jawna decyzja tego "
                                  "skryptu (domyślnie 0.0), do potwierdzenia."),
        "n_streams": len(per_seed_results),
        "recall_mean": float(np.mean(recalls)) if recalls else None,
        "recall_min": float(np.min(recalls)) if recalls else None,
        "fa_per_hour_mean": float(np.mean(fa_rates)) if fa_rates else None,
        "fa_per_hour_max": float(np.max(fa_rates)) if fa_rates else None,
        "fa_per_hour_ci_pooled": {
            "lower": pooled_ci[0], "upper": pooled_ci[1],
            "method": "poisson_exact_garwood", "alpha": 0.05,
            "total_fa_count": total_fa, "total_background_hours": total_hours,
            "note": ("Pulowane po wszystkich strumieniach (suma zdarzen/godzin), "
                     "NIE srednia z CI per-seed -- usrednianie przedzialow "
                     "ufnosci jest statystycznie niepoprawne. Przy total_fa=0 "
                     "'upper' > 0: gorna granica niepewnosci, nie zero."),
        },
        "background_hours_total": total_hours,
        "mean_latency_s": float(np.mean(all_latencies)) if all_latencies else None,
        "historical_clip_metrics": historical,
        "recall_delta_vs_historical_clip": recall_delta_vs_historical,
        "historical_comparison_note": (
            "clip_fa_rate (historyczna) to alarmy/KLIP, fa_per_hour (ciagly) to "
            "alarmy/GODZINE -- rozne jednostki, NIE odejmowac bezposrednio. "
            "recall_delta_vs_historical_clip to jedyna bezposrednio porownywalna "
            "roznica (ta sama jednostka: ulamek wykrytych pozytywow)."
            if historical is not None else
            "Brak porownania -- RealFitness nie mial testowego splitu albo "
            "eval_events sie nie powiodlo (patrz log powyzej)."
        ),
        "per_seed": per_seed_results,
    }

    tracker.log_metrics("continuous_eval", summary)
    print(f"[CONTINUOUS EVAL] Zakończono: recall_mean={summary['recall_mean']}, "
          f"fa_per_hour_mean={summary['fa_per_hour_mean']}, "
          f"fa_per_hour_ci_pooled=[{pooled_ci[0]:.3f},{pooled_ci[1]:.3f}]")

    if csv_dir:
        _write_csv_outputs(csv_dir, per_seed_results)

    return summary


# ============================================================ eksport CSV (M4 punkt "przekazanie")

def _write_csv_outputs(csv_dir: str, per_seed_results: List[Dict[str, Any]]) -> None:
    """Zapisuje `events.csv` (jeden wiersz na zdarzenie) i `false_alarms.csv`
    (jeden wiersz na kazdy FA, z jego znacznikiem czasowym -- M4: 'kazdy FA da
    się wskazac na osi czasu'). Ten sam komplet plikow ma isc i do Karoliny, i
    do Andrzeja (M4, "identyczny zestaw"), zeby nie bylo dwoch wersji tej samej
    liczby."""
    import csv as _csv

    os.makedirs(csv_dir, exist_ok=True)

    events_path = os.path.join(csv_dir, "events.csv")
    with open(events_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["seed", "event_index", "start_s", "end_s", "detected",
                   "latency_s", "is_contaminated"])
        for r in per_seed_results:
            for e in r["per_event"]:
                w.writerow([r["seed"], e["index"], e["start_s"], e["end_s"],
                           int(e["detected"]), e["latency_s"] if e["latency_s"] is not None else "",
                           int(e["is_contaminated"])])

    fa_path = os.path.join(csv_dir, "false_alarms.csv")
    with open(fa_path, "w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(["seed", "alarm_time_s"])
        for r in per_seed_results:
            for t in r["fa_times_s"]:
                w.writerow([r["seed"], round(t, 4)])

    print(f"[CONTINUOUS EVAL] Zapisano CSV: {events_path}, {fa_path}")


# ============================================================ CLI samodzielne

class _StandaloneTracker:
    """Minimalny zamiennik RunTracker do samodzielnego CLI (bez pełnego
    run_dir/manifest.json) -- run_continuous_eval_stage potrzebuje tylko
    `.device` i `.log_metrics()`. Do integracji z prawdziwym pipeline'em
    (run-all/evaluate) należy przekazać prawdziwy RunTracker zamiast tego."""

    def __init__(self, device: str):
        self.device = device
        self.metrics: Dict[str, dict] = {}

    def log_metrics(self, name: str, m: dict) -> None:
        self.metrics[name] = m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--checkpoint", required=True,
                    help="winner_checkpoint.pt z run_final_evaluation_stage")
    ap.add_argument("--continuous-dir", default=None,
                    help="domyślnie: <project_root>/<config.data.continuous_eval>")
    ap.add_argument("--dataset-manifest-csv", default=None,
                    help="dataset/versions/vX.Y.Z/manifest.csv -- do świeżego "
                         "przeliczenia global_gain, jeśli brak zamrożonego pliku")
    ap.add_argument("--gain-file", default=None)
    ap.add_argument("--recall-tolerance-s", type=float, default=0.0,
                    help="tolerancja okna recall (manifest.py: 'end_s + "
                         "tolerancja' -- niedookreślone w źródle, domyślnie 0.0)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None, help="zapisz podsumowanie JSON do pliku")
    ap.add_argument("--out-csv-dir", default=None,
                    help="zapisz events.csv + false_alarms.csv do tego katalogu "
                         "(M4: komplet do przekazania Karolinie/Andrzejowi)")
    args = ap.parse_args()

    sys.path.insert(0, os.getcwd())
    from pipeline_config import PipelineConfig

    config = PipelineConfig.from_json(args.config)
    project_root = _project_root()
    continuous_dir = args.continuous_dir or os.path.join(project_root, config.data.continuous_eval)

    tracker = _StandaloneTracker(args.device)
    run_continuous_eval_stage(
        config, tracker, checkpoint_path=args.checkpoint,
        continuous_dir=continuous_dir,
        dataset_manifest_csv=args.dataset_manifest_csv,
        gain_file=args.gain_file,
        recall_tolerance_s=args.recall_tolerance_s,
        csv_dir=args.out_csv_dir,
    )

    summary = tracker.metrics["continuous_eval"]
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
        print(f"[CONTINUOUS EVAL] Zapisano: {args.out}")


if __name__ == "__main__":
    main()
    