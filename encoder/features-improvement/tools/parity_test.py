#!/usr/bin/env python3
"""
parity_test.py — kryterium 3: czy encoder_twin.py i encoder_v2.ino dają to samo na ustalonym pliku?

Metoda (bez płytki): PRAWDZIWY firmware (fw.elf zbudowany avr-gcc z tych samych flag co na płytkę) działa
w symulatorze cyklowym simavr; próbki ADC wstrzykiwane są tak samo jak w twinie — te same, CAŁKOWITE kody
10-bit (twin nie zaokrągla kodów, więc zaokrąglamy je jawnie i podajemy IDENTYCZNE do obu stron).
Porównujemy per ramka: (a) wartości cech, (b) bity spike'ów s0..s6.

Użycie:
    python3 tools/parity_test.py --wav plik.wav --gain 1.0 --variant baseline
    python3 tools/parity_test.py --wav plik.wav --variant swap --mob-thr 1.9 --ac-thr 0.3
    python3 tools/parity_test.py --synth --variant all          # zestaw testów na audio syntetycznym
Warianty firmware: baseline | dcfix | parity | swap | swap_full (swap + parity + acc32).
Wynik zapisywany do JSON (--json), np. do wklejenia w measurements.json/parity.
"""
from __future__ import annotations

import argparse, importlib.util, json, os, subprocess, sys, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KIT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)

VARIANTS = {
    "baseline":  dict(flags=[], twin="baseline"),
    "dcfix":     dict(flags=["-DENC_DC_FIX=1"], twin="baseline"),                       # tylko poprawka DC
    "parity":    dict(flags=["-DENC_PARITY=1"], twin="baseline"),                       # DC + HF round + EPS_FLOOR
    "swap":      dict(flags=["-DENC_SET_SWAP=1"], twin="swap"),                         # tylko wymiana kanałów
    "swap_full": dict(flags=["-DENC_SET_SWAP=1", "-DENC_PARITY=1", "-DENC_ACC32=1"], twin="swap"),
}
PRIME = 52    # ramki 0..51 to priming (firmware nic nie wypisuje) — jak w twinie


def load_twin(kind, path, env):
    os.environ["ENCODER_CHANNEL_SET"] = kind
    for k, v in env.items():
        os.environ[k] = str(v)
    name = f"twin_{kind}_{abs(hash(tuple(sorted(env.items()))))}"
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(m)
    return m


def build_fw(name, flags, out_root, ino):
    out = os.path.join(out_root, f"parity_{name}")
    cmd = [os.path.join(KIT, "tools", "build_fw.sh"), ino, out, "-DENC_DEBUG_FEAT=1", "-DENC_BAUD=2000000", *flags]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("kompilacja firmware nie powiodła się:\n" + r.stdout + r.stderr)
    return os.path.join(out, "fw.elf")


def run_sim(elf, codes_u16, wait=150000):
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        codes_u16.astype("<u2").tofile(f)
        cp = f.name
    r = subprocess.run([os.path.join(KIT, "tools", "simharness"), elf, cp, "--hop", "192", "--wait", str(wait)],
                       capture_output=True, text=True, timeout=1800)
    os.unlink(cp)
    isr = [l for l in r.stderr.splitlines() if l.startswith("ISR:")]
    rows = {}
    for line in r.stdout.replace("\x1b[32m", "").replace("\x1b[0m", "").splitlines():
        p = line.strip().split(",")
        if len(p) == 15 and p[0].isdigit():
            try:
                rows[int(p[0])] = ([float(v) for v in p[1:8]], [int(v) for v in p[8:15]])
            except ValueError:
                pass
    return rows, (isr[0] if isr else "")


def _stats(a, b):
    d = np.abs(a - b)
    rel = d / (np.abs(b) + 1e-3)
    return dict(n=int(len(a)), max_abs=float(d.max()) if len(d) else 0.0,
                median_rel=float(np.median(rel)) if len(d) else 0.0,
                p99_rel=float(np.percentile(rel, 99)) if len(d) else 0.0)


def compare(twin_mod, codes_u16, fw_rows, kind, event_rms):
    et = twin_mod
    codes = codes_u16.astype(np.float64)
    et.wav_to_adc_codes = lambda *a, **k: codes            # twin dostaje TE SAME całkowite kody
    x = et._remove_dc(codes)
    ff = et._frame_features(x)
    n = ff["n_frames"]
    rms = ff["rms"]
    flux = np.maximum(0.0, np.log1p(rms) - np.log1p(np.concatenate(([0.0], rms[:-1]))))
    if kind == "swap":
        names = ["peak", "mobility", "autocorr", "zcr", "flux", "hf_ratio", "hf_ratio"]
        tw = [ff["peak"], ff["mobility"], ff["autocorr"], ff["zcr"], flux, ff["hf_ratio"], ff["hf_ratio"]]
    else:
        names = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_ratio", "hf_ratio"]
        tw = [ff["peak"], None, ff["cv"], ff["zcr"], flux, ff["hf_ratio"], ff["hf_ratio"]]
    bits_tw = et.encode_file("dummy", gain=1.0)              # spike'i twina (ramki po primingu)
    frames = [k for k in range(PRIME, n) if k in fw_rows]
    evt = np.array([rms[k] >= event_rms for k in frames])    # ramki "zdarzeniowe" (głośne)
    res = {"frames_twin": int(n - PRIME), "frames_fw": len(frames), "event_rms": event_rms,
           "n_event_frames": int(evt.sum()), "features": {}, "channels": {}}
    for c in (0, 1, 2, 3, 4, 5):
        if tw[c] is None:
            continue
        a = np.array([fw_rows[k][0][c] for k in frames])
        b = np.array([tw[c][k] for k in frames])
        res["features"][names[c]] = dict(event=_stats(a[evt], b[evt]), background=_stats(a[~evt], b[~evt]))
    fb_all = np.array([fw_rows[k][1] for k in frames])
    tb_all = np.array([bits_tw[k - PRIME] for k in frames])
    for c in range(7):
        mis = fb_all[:, c] != tb_all[:, c]
        res["channels"][c] = dict(mismatch=int(mis.sum()), mismatch_event=int(mis[evt].sum()),
                                  fw_spikes=int(fb_all[:, c].sum()), twin_spikes=int(tb_all[:, c].sum()))
    anym = (fb_all != tb_all).any(axis=1)
    res["any_mismatch_frames"] = int(anym.sum())
    res["agree_pct"] = 100.0 * (1 - anym.mean()) if len(anym) else 0.0
    res["agree_pct_event"] = 100.0 * (1 - anym[evt].mean()) if evt.any() else float("nan")
    res["names"] = names
    return res


def report(variant, res, isr):
    print(f"\n=== {variant}: {res['frames_fw']}/{res['frames_twin']} ramek po primingu; zgodność spike'ów "
          f"(7 kanałów naraz): wszystkie {res['agree_pct']:.2f}%  |  ramki zdarzeniowe (rms>={res['event_rms']:g} LSB, "
          f"n={res['n_event_frames']}): {res['agree_pct_event']:.2f}% ===")
    print(f"  {'cecha':10s} | {'ZDARZENIA: mediana rel':>23s} {'p99 rel':>9s} {'max|Δ|':>9s} | {'TŁO: mediana rel':>17s} {'p99 rel':>9s}")
    for k, v in res["features"].items():
        e, b = v["event"], v["background"]
        print(f"  {k:10s} | {100*e['median_rel']:22.3f}% {100*e['p99_rel']:8.2f}% {e['max_abs']:9.4f} | "
              f"{100*b['median_rel']:16.3f}% {100*b['p99_rel']:8.2f}%")
    print("  spike'i per kanał [rozjazd(wszystkie)/rozjazd(zdarzenia) | firmware/twin]: " +
          "  ".join(f"s{c}:{v['mismatch']}/{v['mismatch_event']}|{v['fw_spikes']}/{v['twin_spikes']}" for c, v in res["channels"].items()))
    if isr:
        print("  " + isr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--synth", action="store_true", help="użyj audio syntetycznego (tools/synth_audio.py)")
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--variant", default="baseline", help="baseline|dcfix|swap|swap_full|all")
    ap.add_argument("--twin", default=os.path.join(KIT, "twin", "encoder_twin_swap.py"))
    ap.add_argument("--ino", default=os.path.join(KIT, "firmware", "encoder_v2_swap.ino"))
    ap.add_argument("--mob-thr", type=float, default=None)
    ap.add_argument("--ac-thr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bg-level", type=float, default=0.004, help="poziom tła w audio syntetycznym (amplituda, 1.0=pełna skala)")
    ap.add_argument("--event-rms", type=float, default=20.0, help="próg rms [LSB] dla ramek zdarzeniowych")
    ap.add_argument("--json")
    ap.add_argument("--workdir", default=os.path.join(KIT, "build"))
    a = ap.parse_args()

    import soundfile as sf
    if a.synth:
        import synth_audio as sa
        wav = os.path.join(a.workdir, f"synth_{a.seed}.wav")
        os.makedirs(a.workdir, exist_ok=True)
        sf.write(wav, sa.make(14.0, seed=a.seed, bg_level=a.bg_level), sa.SR)
    else:
        wav = a.wav
        if not wav:
            ap.error("podaj --wav albo --synth")
    variants = list(VARIANTS) if a.variant == "all" else [a.variant]
    out = {}
    for v in variants:
        cfg = VARIANTS[v]
        env, flags = {}, list(cfg["flags"])
        if cfg["twin"] == "swap":
            if a.mob_thr is not None:
                env["ENCODER_MOB_THR"] = a.mob_thr; flags.append(f"-DMOB_THR={a.mob_thr}f")
            if a.ac_thr is not None:
                env["ENCODER_AC_THR"] = a.ac_thr; flags.append(f"-DAC_THR={a.ac_thr}f")
        tw = load_twin(cfg["twin"], a.twin, env)
        codes = np.clip(np.round(tw.wav_to_adc_codes(wav, gain=a.gain)), 0, 1023).astype(np.uint16)
        elf = build_fw(v, flags, a.workdir, a.ino)
        rows, isr = run_sim(elf, codes)
        res = compare(tw, codes, rows, cfg["twin"], a.event_rms)
        res["isr_sim"] = isr
        report(v, res, isr)
        out[v] = res
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print("\nzapisano", a.json)


if __name__ == "__main__":
    main()
