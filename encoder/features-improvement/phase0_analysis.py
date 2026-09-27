#!/usr/bin/env python3
"""
phase0_analysis.py — Faza 0.3: czy hjorth_mobility i autocorr_lag1 wnoszą coś ponad obecne kanały,
i jakie progi BEZWZGLĘDNE mają dostać (kanały poziomowe, jak hf_lo/hf_hi)?

Uruchamiasz TY, na prawdziwych nagraniach z manifestu v2.0.0 (potrzebne audio; librosa, scikit-learn, pandas):
    python3 phase0_analysis.py --manifest dataset/versions/v2.0.0/manifest.csv --root . \
        --twin twin/encoder_twin_swap.py --jobs 8 --max-per-cell 400 --out phase0_results.json
    # --gain: produkcyjne globalne wzmocnienie (jeśli znasz, np. z channels.json zbioru spikes_v2/train).
    #         Domyślnie liczone z próbki plików train (percentyl 99.9, jak compute_global_gain).

Zasady metodologiczne (tak jak w zadaniu): cechy liczone tym samym front-endem co twin (resampling do 19231 Hz,
kody ADC, EMA DC); wybór/progi WYŁĄCZNIE na train, ocena na val; przedziały ufności bootstrapem PO group_id
(w klasie pozytywnej niezależnych grup jest ~100, nie tysiące plików). Stan enkodera: rozgrzany na tle
stacjonarnym (jak build_manifest), kopiowany per plik — przybliżenie strumienia produkcyjnego.

Wynik: phase0_results.json (+ czytelne podsumowanie). recommended_thresholds wpisz do
encoder_twin_swap.py (MOB_THR/AC_THR) i encoder_v2_swap.ino (-DMOB_THR=..f -DAC_THR=..f).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

FEATS = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_ratio", "mobility", "autocorr", "rms"]
SETS = {                                   # zestawy cech do porównania (hf_lo/hf_hi = ta sama cecha hf_ratio)
    "A_baseline":  ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_ratio"],
    "E_bez_pc_cv": ["peak", "zcr", "flux", "hf_ratio"],
    "B_swap":      ["peak", "zcr", "flux", "hf_ratio", "mobility", "autocorr"],
    "C_tylko_mob": ["peak", "zcr", "flux", "hf_ratio", "mobility"],
    "D_tylko_ac":  ["peak", "zcr", "flux", "hf_ratio", "autocorr"],
}
_TW = None
_BASE = None
_GAIN = 1.0


# ------------------------------------------------------------------ twin / worker
def load_twin(path):
    os.environ["ENCODER_CHANNEL_SET"] = "baseline"          # cechy bazowe + mobility/autocorr z _frame_features
    name = "encoder_twin_p0"
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(m)
    return m


def _state_to_dict(st):
    return dict(floor_v=st.floor_v.copy(), mad_v=st.mad_v.copy(), refrac=st.refrac.copy(), rms_prev=st.rms_prev,
                hf_rms_prev=st.hf_rms_prev, spike_thr=st.spike_thr, floors_primed=st.floors_primed, n_seen=st.n_seen)


def _dict_to_state(d):
    st = _TW.EncoderState()
    for k, v in d.items():
        setattr(st, k, copy.deepcopy(v))
    return st


def _init_worker(twin_path, base_dict, gain):
    global _TW, _BASE, _GAIN
    _TW = load_twin(twin_path)
    _BASE, _GAIN = base_dict, gain


def _process(args):
    """Jeden plik -> (cechy [n,9], bramka [n]). Ramki od 0 (stan rozgrzany, bez primingu)."""
    path, = args
    try:
        codes = _TW.wav_to_adc_codes(path, gain=_GAIN)
        ff = _TW._frame_features(_TW._remove_dc(codes))
        if ff["n_frames"] == 0:
            return None
        orig = _TW.wav_to_adc_codes
        _TW.wav_to_adc_codes = lambda *a, **k: codes                   # nie ładuj audio drugi raz
        try:
            _, feat, gate = _TW.encode_file(path, gain=_GAIN, state=_dict_to_state(_BASE), return_features=True)
        finally:
            _TW.wav_to_adc_codes = orig
        n = min(len(feat), ff["n_frames"])
        if n == 0:
            return None
        out = np.column_stack([feat[:n, 0], feat[:n, 1], feat[:n, 2], feat[:n, 3], feat[:n, 4], feat[:n, 5],
                               ff["mobility"][:n], ff["autocorr"][:n], ff["rms"][:n]]).astype(np.float32)
        return out, gate[:n]
    except Exception as e:                                            # jeden zły plik nie przerywa analizy
        print(f"[!] {path}: {e}", file=sys.stderr)
        return None


# ------------------------------------------------------------------ dane
def sample_files(df, split, max_per_cell, seed):
    d = df[df.split == split]
    parts = [g.sample(n=min(len(g), max_per_cell), random_state=seed) for _, g in d.groupby(["label", "kind"])]
    return pd.concat(parts).reset_index(drop=True)


def collect(df_files, root, pool):
    paths = [os.path.join(root, p) for p in df_files.filepath]
    res = list(pool.map(_process, [(p,) for p in paths], chunksize=4))
    X, Y, G, K, GATE, FID = [], [], [], [], [], []
    for i, (r, row) in enumerate(zip(res, df_files.itertuples())):
        if r is None:
            continue
        f, gate = r
        n = len(f)
        X.append(f); GATE.append(gate)
        Y.append(np.full(n, int(row.label == "positive"))); G.append(np.full(n, row.group_id, dtype=object))
        K.append(np.full(n, row.kind, dtype=object)); FID.append(np.full(n, i))
    return dict(X=np.vstack(X), y=np.concatenate(Y), group=np.concatenate(G), kind=np.concatenate(K),
                gate=np.concatenate(GATE), fid=np.concatenate(FID))


# ------------------------------------------------------------------ analizy
def cohens_d(x, y):
    a, b = x[y == 1], x[y == 0]
    sp = np.sqrt((a.var(ddof=1) * (len(a) - 1) + b.var(ddof=1) * (len(b) - 1)) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / (sp + 1e-12))


def clip_scores(score, fid, y):
    """max wyniku ramkowego w klipie -> (score_clip, y_clip, group_clip_index)."""
    ids = np.unique(fid)
    sc = np.array([score[fid == i].max() for i in ids])
    yy = np.array([y[fid == i][0] for i in ids])
    return ids, sc, yy


def auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def boot_delta(y, groups, s_a, s_b, n_boot, seed=0):
    """Sparowany bootstrap po group_id: rozkład AUC(s_b) - AUC(s_a) dla klipów."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by_g = {g: np.where(groups == g)[0] for g in ug}
    d = []
    for _ in range(n_boot):
        pick = rng.choice(ug, size=len(ug), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        d.append(auc(y[idx], s_b[idx]) - auc(y[idx], s_a[idx]))
    d = np.array(d)
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def fit_score(train, val, cols, model):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import QuantileTransformer
    from sklearn.pipeline import make_pipeline
    ix = [FEATS.index(c) for c in cols]
    rng = np.random.default_rng(0)
    sub = rng.choice(len(train["y"]), size=min(400_000, len(train["y"])), replace=False)
    Xt, yt = train["X"][sub][:, ix], train["y"][sub]
    if model == "LR":
        m = make_pipeline(QuantileTransformer(output_distribution="normal", n_quantiles=200, random_state=0),
                          LogisticRegression(max_iter=300, class_weight="balanced"))
    else:
        m = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, class_weight="balanced", random_state=0)
    m.fit(Xt, yt)
    return m.predict_proba(val["X"][:, ix])[:, 1]


def thresholds_analysis(tr, va, neg_target):
    """Progi bezwzględne mobility/autocorr na ramkach BRAMKOWANYCH (jak hf_lo/hf_hi w firmware)."""
    out = {}
    for name, ix in (("mobility", FEATS.index("mobility")), ("autocorr", FEATS.index("autocorr"))):
        d = cohens_d(tr["X"][:, ix], tr["y"])
        above = d > 0                                              # kierunek z znaku d na train
        g_tr = tr["gate"]
        neg = tr["X"][g_tr & (tr["y"] == 0), ix]
        pos = tr["X"][g_tr & (tr["y"] == 1), ix]
        qs = [0.90, 0.95, 0.97, 0.98, 0.99]
        rows = []
        for q in qs:
            thr = float(np.quantile(neg, q if above else 1 - q))
            fire = (lambda v: v > thr) if above else (lambda v: v < thr)
            r = dict(neg_quantile=q, thr=thr, pos_rate_train=float(fire(pos).mean()), neg_rate_train=float(fire(neg).mean()))
            gv = va["gate"]
            r["pos_rate_val"] = float(fire(va["X"][gv & (va["y"] == 1), ix]).mean())
            r["neg_rate_val"] = float(fire(va["X"][gv & (va["y"] == 0), ix]).mean())
            r["neg_rate_val_by_kind"] = {k: float(fire(va["X"][gv & (va["y"] == 0) & (va["kind"] == k), ix]).mean())
                                        for k in np.unique(va["kind"][va["y"] == 0]) if (gv & (va["kind"] == k)).any()}
            rows.append(r)
        pick = min(rows, key=lambda r: abs(r["neg_rate_train"] - neg_target))
        out[name] = dict(cohens_d_train=d, fire_when="powyżej progu" if above else "PONIŻEJ progu (FIRE_BELOW)",
                         grid=rows, recommended_thr=pick["thr"], recommended_at=pick)
    # odniesienie: obecne kanały hf_lo/hf_hi (progi z encoder_twin.py) na tych samych ramkach
    ix = FEATS.index("hf_ratio")
    ref = {}
    for nm, thr in (("hf_lo", 0.28), ("hf_hi", 0.35)):
        g = va["gate"]
        ref[nm] = dict(thr=thr, pos_rate_val=float((va["X"][g & (va["y"] == 1), ix] > thr).mean()),
                       neg_rate_val=float((va["X"][g & (va["y"] == 0), ix] > thr).mean()))
    out["reference_hf"] = ref
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("--twin", required=True, help="encoder_twin_swap.py (lub oryginalny encoder_twin.py z _frame_features)")
    ap.add_argument("--gain", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--max-per-cell", type=int, default=400, help="max plików na (label,kind) w splicie")
    ap.add_argument("--warmup-seconds", type=float, default=30.0)
    ap.add_argument("--neg-target", type=float, default=0.03, help="docelowy odsetek ramek negatywnych (bramkowanych), które kanał może odpalić")
    ap.add_argument("--boot", type=int, default=300)
    ap.add_argument("--no-gbm", action="store_true", help="tylko regresja logistyczna (szybciej)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="phase0_results.json")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    tw = load_twin(a.twin)
    if not hasattr(tw, "_frame_features"):
        sys.exit("ten twin nie ma _frame_features — użyj encoder_twin_swap.py")
    tr_files = sample_files(df, "train", a.max_per_cell, a.seed)
    va_files = sample_files(df, "val", a.max_per_cell, a.seed)
    print(f"[dane] train {len(tr_files)} plików, val {len(va_files)} plików (max {a.max_per_cell}/komórkę)")

    gain = a.gain
    if gain is None:
        sub = tr_files.sample(n=min(300, len(tr_files)), random_state=a.seed)
        gain = tw.compute_global_gain([os.path.join(a.root, p) for p in sub.filepath])
        print(f"[gain] policzone z {len(sub)} plików train: {gain:.4f} (podaj --gain z produkcji, jeśli znasz)")
    stat = tr_files[(tr_files.kind == "stationary") & (tr_files.label == "negative")]
    warm = sorted(os.path.join(a.root, p) for p in stat.filepath) or sorted(os.path.join(a.root, p) for p in tr_files[tr_files.label == "negative"].filepath)
    base_state, n_used = tw._warmup_state(warm, a.warmup_seconds, gain=gain)
    base = _state_to_dict(base_state)

    with ProcessPoolExecutor(a.jobs, initializer=_init_worker, initargs=(a.twin, base, gain)) as pool:
        print("[cechy] liczenie ramek train..."); TR = collect(tr_files, a.root, pool)
        print("[cechy] liczenie ramek val...");   VA = collect(va_files, a.root, pool)
    print(f"[ramki] train {len(TR['y'])} (poz {TR['y'].mean():.1%}), val {len(VA['y'])} (poz {VA['y'].mean():.1%})")

    res = dict(n_files=dict(train=len(tr_files), val=len(va_files)), gain=gain, frames=dict(train=int(len(TR['y'])), val=int(len(VA['y']))))

    # 1) Cohen's d (train)
    res["cohens_d_train"] = {f: cohens_d(TR["X"][:, i], TR["y"]) for i, f in enumerate(FEATS) if f != "rms"}
    print("\n=== 1. Cohen's d (train, wszystkie ramki; |d|>0.2 = coś widać) ===")
    for f, v in res["cohens_d_train"].items():
        print(f"  {f:10s} d={v:+.3f}")

    # 2) korelacje (Spearman, podpróba train)
    from scipy.stats import spearmanr
    rng = np.random.default_rng(a.seed)
    sub = rng.choice(len(TR["y"]), size=min(150_000, len(TR["y"])), replace=False)
    cols = ["peak", "cv", "zcr", "flux", "hf_ratio", "mobility", "autocorr"]
    C = spearmanr(TR["X"][sub][:, [FEATS.index(c) for c in cols]]).correlation
    res["spearman"] = {ci: {cj: float(C[i, j]) for j, cj in enumerate(cols)} for i, ci in enumerate(cols)}
    print("\n=== 2. Korelacja Spearmana (train) ===")
    print("            " + " ".join(f"{c:>8s}" for c in cols))
    for i, ci in enumerate(cols):
        print(f"  {ci:9s} " + " ".join(f"{C[i, j]:+8.2f}" for j in range(len(cols))))

    # 3) wartość przyrostowa: LR (+GBM), ramkowe i klipowe AUC na val, sparowany bootstrap po group_id
    print("\n=== 3. Wartość przyrostowa (uczone na train, oceniane na val; AUC klipowy = max ramkowego wyniku w klipie) ===")
    res["models"] = {}
    for mdl in (["LR"] if a.no_gbm else ["LR", "GBM"]):
        sc, clip = {}, {}
        for nm, cols_ in SETS.items():
            s = fit_score(TR, VA, cols_, mdl)
            ids, cs, cy = clip_scores(s, VA["fid"], VA["y"])
            cg = np.array([VA["group"][VA["fid"] == i][0] for i in ids])
            sc[nm] = s; clip[nm] = (cs, cy, cg)
        res["models"][mdl] = {"auc": {}, "delta": {}}
        print(f"\n  model {mdl}:")
        for nm in SETS:
            fa, ca = auc(VA["y"], sc[nm]), auc(clip[nm][1], clip[nm][0])
            res["models"][mdl]["auc"][nm] = dict(frame=fa, clip=ca)
            print(f"    {nm:12s} AUC ramkowy {fa:.3f}   AUC klipowy {ca:.3f}")
        for a_, b_ in (("A_baseline", "B_swap"), ("E_bez_pc_cv", "B_swap"), ("E_bez_pc_cv", "C_tylko_mob"), ("E_bez_pc_cv", "D_tylko_ac"), ("C_tylko_mob", "B_swap")):
            cs_a, cy, cg = clip[a_]; cs_b = clip[b_][0]
            m, lo, hi = boot_delta(cy, cg, cs_a, cs_b, a.boot, a.seed)
            res["models"][mdl]["delta"][f"{b_}-{a_}"] = dict(mean=m, ci95=[lo, hi])
            flag = "  <-- CI nad zerem" if lo > 0 else ("  <-- CI pod zerem" if hi < 0 else "  (CI obejmuje 0)")
            print(f"    ΔAUC klipowy {b_:11s} - {a_:11s} = {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]{flag}")

    # 4) progi bezwzględne
    print(f"\n=== 4. Progi bezwzględne na ramkach bramkowanych (cel: ~{100*a.neg_target:.0f}% odpaleń na negatywach; dobór na train, kontrola na val) ===")
    res["thresholds"] = thresholds_analysis(TR, VA, a.neg_target)
    for nm in ("mobility", "autocorr"):
        t = res["thresholds"][nm]
        print(f"\n  {nm} (d_train={t['cohens_d_train']:+.2f}; odpala {t['fire_when']}):")
        print(f"    {'kwantyl neg':>11s} {'próg':>9s} {'poz train':>10s} {'neg train':>10s} {'poz val':>8s} {'neg val':>8s}")
        for r in t["grid"]:
            mark = " <-- rekomendowany" if r["thr"] == t["recommended_thr"] else ""
            print(f"    {r['neg_quantile']:11.2f} {r['thr']:9.4f} {100*r['pos_rate_train']:9.1f}% {100*r['neg_rate_train']:9.1f}% "
                  f"{100*r['pos_rate_val']:7.1f}% {100*r['neg_rate_val']:7.1f}%{mark}")
    print("\n  odniesienie (obecne kanały, val): " + "; ".join(
        f"{k} próg {v['thr']}: poz {100*v['pos_rate_val']:.1f}% / neg {100*v['neg_rate_val']:.1f}%" for k, v in res["thresholds"]["reference_hf"].items()))

    res["recommended_thresholds"] = dict(
        mob_thr=res["thresholds"]["mobility"]["recommended_thr"], mob_fire_below=res["thresholds"]["mobility"]["fire_when"].startswith("PONIŻEJ"),
        ac_thr=res["thresholds"]["autocorr"]["recommended_thr"], ac_fire_below=res["thresholds"]["autocorr"]["fire_when"].startswith("PONIŻEJ"))
    print("\n>>> recommended_thresholds:", json.dumps(res["recommended_thresholds"]))
    print("    twin: ENCODER_MOB_THR / ENCODER_AC_THR (env) albo stałe MOB_THR/AC_THR; firmware: -DMOB_THR=<x>f -DAC_THR=<y>f"
          "; jeśli fire_below różni się od domyślnego (mob: powyżej, ac: poniżej), ustaw MOB_FIRE_BELOW/AC_FIRE_BELOW.")
    json.dump(res, open(a.out, "w"), indent=2, default=float)
    print("zapisano", a.out)


if __name__ == "__main__":
    main()
