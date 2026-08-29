#!/usr/bin/env python3
"""
snn_hw_pipeline.py — Delta Spike / Wake-Up AI
Trening SNN 6->4->3->1 z ograniczeniami sprzętowymi płytek Lu.i wpisanymi w graf.

Zasada przewodnia: NIE trenujemy gęstej sieci i nie przycinamy jej potem.
Maska łączności (fan-in <= 3) jest w modelu od pierwszej epoki, bo dokładnie tyle
wejść ma płytka. Przedostatnia warstwa ma 3 neurony == 3 wejścia neuronu decyzyjnego.

Model neuronu odwzorowuje Lu.i, a nie snn.Leaky:
    I[t] = a*I[t-1] + W @ s[t]
    V[t] = b*V[t-1] + (1-b)*(V_leak + I[t])
    s[t] = 1 gdy V[t] >= V_th ;  V[t] <- 0 po spiku (reset do zera)
Próg V_th = VDD/2 jest sprzętowo nieruchomy, więc jedyne, co możemy przesuwać,
to wagi i V_leak. Warunek odpalenia: sum(w * PSP) >= V_th - V_leak.

Harmonogram treningu (parametry idą 1:1 na PCB, stąd dwie fazy):
    faza HAT  (epoki 0..hat)   — pełna precyzja wag + wstrzykiwany szum sprzętowy
                                 (rozrzut trymera, tolerancja τ, rozdzielczość paska
                                 V_leak), żeby rozwiązanie było odporne na ręczną
                                 kalibrację, a nie tylko na idealne liczby.
    faza QAT  (epoki hat..end) — kwantyzacja STE do W_LEVELS działek trymera
                                 + martwa strefa; `best` resetowany, bo tylko
                                 skwantyzowany checkpoint da się wykręcić na płytce.
Znaki wag (przełączniki +/- na płytce) zamrażane pod koniec treningu.

Użycie:
    python snn_hw_pipeline.py train   --data ./spikes_csv --out hw_config.json
    python snn_hw_pipeline.py train   --data ./spikes_csv --hw-params hw_params.json --out hw_config.json
    python snn_hw_pipeline.py train   --data ./spikes_csv --limit 200 --epochs 8   # szybki smoke test
    python snn_hw_pipeline.py compare --sim sim.npz --hw hw_spikes.csv --layer H
"""

from __future__ import annotations

import argparse, glob, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler

# ============================================================ stałe sprzętowe

DT = 0.010  # s, == hop ramki encodera
V_TH = 1.0  # próg, sprzętowo VDD/2 — nie jest uczony
V_LEAK_MAX = 0.90 * V_TH  # wyżej neuron strzela samoczynnie
TAU_SYN_RANGE = (0.005, 0.220)  # C_syn = 10 uF, g 45 uS..inf
TAU_MEM_RANGE = (0.020, 2.200)  # C_mem = 22 uF, g 10 uS..inf
W_MAX = 1.60  # waga przy potencjometrze na pełnej skali
W_DEADZONE = 0.05  # poniżej tego nie da się ustawić trymerem
W_LEVELS = 20  # realna rozdzielczość ręki na trymerze (~5% skali)

# rozrzut sprzętowy wstrzykiwany w treningu (i w teście odporności):
SIGMA_W_HW = 0.5 * W_MAX / W_LEVELS  # ręka ustawia trymer z dokładnością ~pół działki
SIGMA_TAU_HW = 0.10  # tolerancja RC / odczyt τ z Fazy A (mnożnikowo)
SIGMA_VLEAK_HW = 0.02 * V_TH  # pasek LED ma skończoną rozdzielczość

# v3: 7 kanałów (crest wymieniona na hf_lo + hf_hi — patrz encoder_twin.py)
CHANNELS = ["peak", "peak_cnt", "cv", "zcr", "flux", "hf_lo", "hf_hi"]
CH_IN = len(CHANNELS)

# maski łączności: [dla każdego neuronu post] lista indeksów pre, max 3 (fan-in płytki).
# Indeksy: 0 peak, 1 peak_cnt, 2 cv, 3 zcr, 4 flux, 5 hf_lo, 6 hf_hi (kod termometrowy
# udziału energii HF). Każdy kanał trafia do >=1 płytki H; kanały widmowe (5,6)
# rozłożone tak, że każda płytka H dostaje jeden z nich zmieszany z cechami czasowymi —
# sieć od pierwszej epoki może uczyć się koniunkcji "głośne ORAZ wysokoczęstotliwościowe".
MASK_H = [
    [0, 1, 5],  # H0: peak + peak_cnt + hf_lo (amplituda transientu + treść HF)
    [2, 3, 6],  # H1: cv + zcr + hf_hi        (charakter widma + mocny HF)
    [0, 4, 5],  # H2: peak + flux + hf_lo     (atak + treść HF)
    [1, 4, 6],
]  # H3: peak_cnt + flux + hf_hi (mikro-szpilki + mocny HF)
MASK_G = [[0, 1, 2], [1, 2, 3], [2, 3, 0]]  # 4 -> 3
MASK_O = [[0, 1, 2]]  # 3 -> 1  (dokładnie 3 wejścia)

NAMES_H = ["H0", "H1", "H2", "H3"]
NAMES_G = ["G0", "G1", "G2"]
NAMES_O = ["D"]

# --- wariant SZEROKI (wide): H rozszerzona do 8 płytek, do wykorzystania 15 płytek ---
# G nadal 3 (D ma fan-in 3), więc dodatkowa pojemność idzie w H. 8 płytek H, każda
# fan-in 3, wszystkie 7 kanałów pokryte, kanały widmowe (5,6) rozłożone po połowie płytek.
MASK_H8 = [
    [0, 1, 5],  # H0: peak, peak_cnt, hf_lo
    [2, 3, 6],  # H1: cv, zcr, hf_hi
    [0, 4, 5],  # H2: peak, flux, hf_lo
    [1, 4, 6],  # H3: peak_cnt, flux, hf_hi
    [0, 3, 6],  # H4: peak, zcr, hf_hi
    [1, 2, 5],  # H5: peak_cnt, cv, hf_lo
    [3, 4, 5],  # H6: zcr, flux, hf_lo
    [2, 4, 6],
]  # H7: cv, flux, hf_hi
MASK_G8 = [
    [0, 1, 2],  # G0: H0,H1,H2
    [3, 4, 5],  # G1: H3,H4,H5
    [6, 7, 0],
]  # G2: H6,H7,H0  (8 wyjść H -> 9 slotów G, H0 współdzielone)
NAMES_H8 = ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"]


def topo(wide):
    """Zwraca (mask_h, n_h, names_h, mask_g) dla wariantu wąskiego/szerokiego."""
    if wide:
        return MASK_H8, 8, NAMES_H8, MASK_G8
    return MASK_H, 4, NAMES_H, MASK_G


def infer_wide(state_dict):
    """Czy checkpoint jest wariantem szerokim — po liczbie płytek H (H.w wiersze)."""
    return state_dict["H.w"].shape[0] == 8


def dense_mask(mask, n_pre, n_post):
    m = torch.zeros(n_post, n_pre)
    for i, pre in enumerate(mask):
        assert len(pre) <= 3, f"fan-in {len(pre)} > 3 — płytka tego nie zrobi"
        for j in pre:
            m[i, j] = 1.0
    return m


# ============================================================ surrogate gradient


class SpikeFn(torch.autograd.Function):
    """Heaviside w przód, pochodna arctan w tył."""

    ALPHA = 5.0

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x >= 0).float()

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + (SpikeFn.ALPHA * x) ** 2)
        return grad * sg


spike_fn = SpikeFn.apply


def quantize_ste(w):
    """Kwantyzacja wag do rozdzielczości trymera + martwa strefa. STE w tył."""
    step = W_MAX / W_LEVELS
    q = torch.round(w / step) * step
    q = torch.where(q.abs() < W_DEADZONE, torch.zeros_like(q), q)
    q = q.clamp(-W_MAX, W_MAX)
    return w + (q - w).detach()


def _map(p, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(p)


def _inv_map(v, lo, hi):
    z = min(max((v - lo) / (hi - lo), 1e-4), 1 - 1e-4)
    return math.log(z / (1 - z))


# ============================================================ warstwa


class LuiLayer(nn.Module):
    """Warstwa płytek Lu.i. tau/V_leak per neuron, wagi maskowane i kwantyzowane."""

    def __init__(self, n_pre, n_post, mask, names, hw=None, quantize=False):
        super().__init__()
        self.n_pre, self.n_post, self.names = n_pre, n_post, names
        self.quantize = quantize  # włączane dopiero w fazie QAT
        self.mismatch_active = False  # szum sprzętowy (HAT / test odporności)
        self.register_buffer("mask", dense_mask(mask, n_pre, n_post))

        w = torch.randn(n_post, n_pre) * 0.5
        self.w = nn.Parameter(w * self.mask)

        # jeśli mamy zmierzone parametry płytek — zamrażamy je zamiast uczyć
        self.hw_fixed = hw is not None
        if self.hw_fixed:
            ts = torch.tensor([hw[n]["tau_syn"] for n in names])
            tm = torch.tensor([hw[n]["tau_mem"] for n in names])
            self.register_buffer("tau_syn_f", ts)
            self.register_buffer("tau_mem_f", tm)
        else:
            # start: szybka synapsa, membrana ~150 ms. Zerowa inicjalizacja dawała
            # tau_mem ~1.1 s, przy którym beta ~0.99 i gradient przez 3 warstwy ginie.
            self.p_tsyn = nn.Parameter(
                torch.full((n_post,), _inv_map(0.030, *TAU_SYN_RANGE))
            )
            self.p_tmem = nn.Parameter(
                torch.full((n_post,), _inv_map(0.150, *TAU_MEM_RANGE))
            )

        self.p_vleak = nn.Parameter(
            torch.full((n_post,), _inv_map(0.4, 0.0, V_LEAK_MAX))
        )
        self.register_buffer("sign_ref", torch.ones(n_post, n_pre))
        self.register_buffer("sign_frozen", torch.zeros(1))

    # -- parametry fizyczne -------------------------------------------------
    def tau_syn(self):
        return self.tau_syn_f if self.hw_fixed else _map(self.p_tsyn, *TAU_SYN_RANGE)

    def tau_mem(self):
        return self.tau_mem_f if self.hw_fixed else _map(self.p_tmem, *TAU_MEM_RANGE)

    def v_leak(self):
        return _map(self.p_vleak, 0.0, V_LEAK_MAX)

    def weights(self):
        w = self.w * self.mask
        if bool(self.sign_frozen.item()):
            w = self.sign_ref * w.abs()
        return quantize_ste(w) if self.quantize else w

    def freeze_signs(self):
        with torch.no_grad():
            s = torch.sign(self.w * self.mask)
            s[s == 0] = 1.0
            self.sign_ref.copy_(s)
            self.sign_frozen.fill_(1.0)

    # -- dynamika -----------------------------------------------------------
    def forward(self, s_in):
        """s_in: [B, T, n_pre] -> (spikes [B,T,n_post], V [B,T,n_post])"""
        B, T, _ = s_in.shape
        ts, tm, vl, W = self.tau_syn(), self.tau_mem(), self.v_leak(), self.weights()

        if self.mismatch_active:
            # szum tylko na istniejących synapsach — losowy trymer na W=0 tworzyłby
            # połączenie, którego na płytce fizycznie nie ma
            W = W + (W.detach() != 0).float() * torch.randn_like(W) * SIGMA_W_HW
            ts = (ts * torch.exp(torch.randn_like(ts) * SIGMA_TAU_HW)).clamp(
                *TAU_SYN_RANGE
            )
            tm = (tm * torch.exp(torch.randn_like(tm) * SIGMA_TAU_HW)).clamp(
                *TAU_MEM_RANGE
            )
            vl = (vl + torch.randn_like(vl) * SIGMA_VLEAK_HW).clamp(0.0, V_LEAK_MAX)

        a = torch.exp(-DT / ts)
        b = torch.exp(-DT / tm)
        # jedno GEMM na całą sekwencję zamiast T małych wewnątrz pętli
        inj = F.linear(s_in, W)  # [B, T, n_post]

        I = torch.zeros(B, self.n_post, device=s_in.device)
        V = vl.expand(B, -1).clone()
        out_s, out_v = [], []

        for t in range(T):
            I = a * I + inj[:, t]
            V = b * V + (1.0 - b) * vl + I
            out_v.append(V)  # zapis PRZED resetem — stąd płynie gradient straty
            s = spike_fn(V - V_TH)
            V = V * (1.0 - s)  # reset do zera, jak na oscylogramie Lu.i
            out_s.append(s)
        return torch.stack(out_s, 1), torch.stack(out_v, 1)


class LuiNet(nn.Module):
    def __init__(self, hw=None, quantize=False, wide=False):
        super().__init__()
        mask_h, n_h, names_h, mask_g = topo(wide)
        self.wide = wide
        self.H = LuiLayer(CH_IN, n_h, mask_h, names_h, hw, quantize)
        self.G = LuiLayer(n_h, 3, mask_g, NAMES_G, hw, quantize)
        self.O = LuiLayer(3, 1, MASK_O, NAMES_O, hw, quantize)

    def forward(self, x):
        sh, _ = self.H(x)
        sg, _ = self.G(sh)
        so, vo = self.O(sg)
        return {"sh": sh, "sg": sg, "so": so, "vo": vo.squeeze(-1)}

    def layers(self):
        return [self.H, self.G, self.O]

    def freeze_signs(self):
        for l in self.layers():
            l.freeze_signs()

    def set_quantize(self, flag: bool):
        for l in self.layers():
            l.quantize = flag

    def set_mismatch(self, flag: bool):
        for l in self.layers():
            l.mismatch_active = flag


# ============================================================ dane


class SpikeClips(Dataset):
    """Każdy CSV = jeden klip. Kolumny: frame,s0..s5 (+ opcjonalnie label).
    Bez kolumny label: etykieta 1, jeśli w nazwie pliku jest 'glass'/'szklo'.

    Okna trzymane jako uint8 (spiki są 0/1), cache w .npz obok danych —
    parsowanie ~10k CSV trwa minuty, wczytanie cache sekundy. Każde okno
    pamięta indeks pliku źródłowego, żeby split walidacyjny szedł po PLIKACH:
    okna zachodzą na siebie (stride < T), więc split po oknach przecieka."""

    def __init__(self, root, T=200, stride=50, limit=None):
        files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        if not files:
            sys.exit(f"brak CSV w {root}")
        if limit:
            pos = [f for f in files if self._name_label(f)][:limit]
            neg = [f for f in files if not self._name_label(f)][:limit]
            files = sorted(pos + neg)

        cache = os.path.join(root, f"_cache_T{T}_s{stride}.npz")
        sig = (
            f"{len(files)}|{sum(os.path.getsize(f) for f in files)}|"
            f"{max(os.path.getmtime(f) for f in files):.0f}|{limit}"
        )
        if os.path.exists(cache):
            try:
                z = np.load(cache, allow_pickle=False)
                ok = str(z["sig"]) == sig
            except (EOFError, ValueError, OSError):
                ok = False  # cache ucięty (np. równoległy zapis) — przebuduj
            if ok:
                self.win, self.lab, self.fidx = z["win"], z["lab"], z["fidx"]
                self.file_lab = z["file_lab"]
                print(
                    f"[data] cache: {len(self.lab)} okien, pozytywnych "
                    f"{int(self.lab.sum())} ({100 * self.lab.mean():.1f}%)",
                    flush=True,
                )
                return

        wins, labs, fidxs, file_labs = [], [], [], []
        t0 = time.time()
        for fi, f in enumerate(files):
            with open(f) as fh:
                header = fh.readline().strip().split(",")
                arr = np.loadtxt(fh, delimiter=",", ndmin=2, dtype=np.float32)
            if arr.size == 0:
                arr = np.zeros((0, len(header)), dtype=np.float32)
            s = arr[:, 1 : 1 + CH_IN].astype(np.uint8)
            if "label" in header:
                y = float(arr[:, header.index("label")].max()) if len(arr) else 0.0
            else:
                y = float(self._name_label(f))
            file_labs.append(y)
            if len(s) < T:
                s = np.pad(s, ((0, T - len(s)), (0, 0)))
            for i in range(0, len(s) - T + 1, stride):
                wins.append(s[i : i + T])
                labs.append(y)
                fidxs.append(fi)
            if fi % 1000 == 999:
                print(
                    f"[data] {fi + 1}/{len(files)} plików ({time.time() - t0:.0f}s)",
                    flush=True,
                )

        self.win = np.stack(wins)
        self.lab = np.array(labs, dtype=np.float32)
        self.fidx = np.array(fidxs, dtype=np.int32)
        self.file_lab = np.array(file_labs, dtype=np.float32)
        # zapis atomowy: temp per-proces + rename, żeby równoległe treningi nigdy
        # nie czytały pliku w połowie zapisu (wcześniej dawało EOFError)
        tmp = f"{cache}.tmp{os.getpid()}"
        np.savez_compressed(
            tmp,
            win=self.win,
            lab=self.lab,
            fidx=self.fidx,
            file_lab=self.file_lab,
            sig=sig,
        )
        os.replace(tmp + ".npz" if not tmp.endswith(".npz") else tmp, cache)
        print(
            f"[data] {len(self.lab)} okien z {len(files)} plików, pozytywnych "
            f"{int(self.lab.sum())} ({100 * self.lab.mean():.1f}%), "
            f"cache -> {cache}",
            flush=True,
        )

    @staticmethod
    def _name_label(f):
        nm = os.path.basename(f).lower()
        return "glass" in nm or "szklo" in nm

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return torch.from_numpy(self.win[i]).float(), torch.tensor(self.lab[i])


def split_by_file(ds, val_frac=0.2, seed=0):
    """Stratyfikowany podział po plikach źródłowych (nie po oknach)."""
    rng = np.random.default_rng(seed)
    va_files = []
    for c in (0.0, 1.0):
        fs = np.where(ds.file_lab == c)[0]
        rng.shuffle(fs)
        va_files.extend(fs[: max(1, int(val_frac * len(fs)))])
    va_mask = np.isin(ds.fidx, va_files)
    return np.where(~va_mask)[0], np.where(va_mask)[0]


def make_sampler(labels, num_samples):
    """Balans klas + stały budżet okien na epokę: 10x więcej danych ma dawać
    więcej różnorodności między epokami, a nie 10x dłuższą epokę."""
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    w = (1.0 / np.maximum(counts, 1))[labels.astype(int)]
    return WeightedRandomSampler(
        torch.from_numpy(w).double(), min(num_samples, len(w)), replacement=True
    )


# ============================================================ strata


def loss_fn(out, y, pos_weight, rate_lo=0.02, rate_hi=0.30, margin_w=0.5, spk_w=0.0):
    # logit = jak wysoko membrana wyjściowa doszła w całym oknie.
    # Dokładnie to widzisz na 7. diodzie: "czy kiedykolwiek przekroczyła próg".
    vmax = out["vo"].max(dim=1).values
    logit = 6.0 * (vmax - V_TH)
    bce = F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=torch.tensor(pos_weight, device=y.device)
    )

    # margines na negatywach: tło ma nie podchodzić pod próg
    neg = (y < 0.5).float()
    margin = ((vmax - 0.80).clamp(min=0) ** 2 * neg).sum() / neg.sum().clamp(min=1)

    # neurony ukryte mają żyć: ani martwe, ani zasycone (analog nie strzela w kółko)
    reg = 0.0
    for k in ("sh", "sg"):
        r = out[k].mean(dim=(0, 1))
        reg = (
            reg
            + ((rate_lo - r).clamp(min=0) ** 2).sum()
            + ((r - rate_hi).clamp(min=0) ** 2).sum()
        )

    total = bce + margin_w * margin + 0.2 * reg

    # zliczanie spików D: reguła dekodera "k spików w oknie" jest dyskryminująca
    # tylko wtedy, gdy szkło daje SERIĘ spików, a tło żadnego. Sam vmax-BCE tego
    # nie wymusza — nagradza jedno przekroczenie progu i toleruje pojedyncze
    # spiki na tle (stąd setki alarmów/h w symulacji strumieniowej).
    if spk_w > 0:
        nspk = out["so"].sum(dim=(1, 2)).clamp(max=5.0)
        pos = (y > 0.5).float()
        neg = 1.0 - pos
        spk = ((2.0 - nspk).clamp(min=0) * pos).sum() / pos.sum().clamp(min=1) + (
            nspk * neg
        ).sum() / neg.sum().clamp(min=1)
        total = total + spk_w * spk

    return total, logit


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tp = fp = fn = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        vmax = model(x)["vo"].max(dim=1).values
        p = (vmax >= V_TH).float()
        tp += ((p == 1) & (y == 1)).sum().item()
        fp += ((p == 1) & (y == 0)).sum().item()
        fn += ((p == 0) & (y == 1)).sum().item()
    rec = tp / max(tp + fn, 1)
    pre = tp / max(tp + fp, 1)
    f1 = 2 * pre * rec / max(pre + rec, 1e-9)
    return {"recall": rec, "precision": pre, "f1": f1}


@torch.no_grad()
def evaluate_events(model, base, indices, dev, ks=(1, 2, 3), bs=256):
    """Metryki na poziomie KLIPÓW — to, co widać na żywym demie: czy nagranie
    szkła budzi system i ile klipów tła robi fałszywy alarm. Reguła decyzyjna:
    klip alarmuje, gdy którekolwiek okno ma >= k spików neuronu D. k > 1 może
    egzekwować dekoder (zliczanie impulsów z J4) bez zmian w analogu."""
    model.eval()
    indices = np.asarray(indices)
    nspk = np.zeros(len(indices))
    for lo in range(0, len(indices), bs):
        chunk = indices[lo : lo + bs]
        x = torch.from_numpy(base.win[chunk]).float().to(dev)
        nspk[lo : lo + len(chunk)] = model(x)["so"].sum(dim=(1, 2)).cpu().numpy()

    fidx, lab = base.fidx[indices], base.lab[indices]
    files = np.unique(fidx)
    f_lab = np.array([lab[fidx == f].max() for f in files])
    f_spk = np.array([nspk[fidx == f].max() for f in files])

    res = {}
    n_pos, n_neg = int((f_lab == 1).sum()), int((f_lab == 0).sum())
    for k in ks:
        det = f_spk >= k
        rec = float(det[f_lab == 1].mean()) if n_pos else 0.0
        fa = float(det[f_lab == 0].mean()) if n_neg else 0.0
        res[f"k{k}"] = {"clip_recall": round(rec, 4), "clip_fa_rate": round(fa, 4)}
        print(
            f"[zdarzenia] k={k}: wykryte {100 * rec:.1f}% klipów glass, "
            f"fałszywy alarm w {100 * fa:.1f}% klipów tła "
            f"({n_pos} glass / {n_neg} tła)",
            flush=True,
        )
    return res


@torch.no_grad()
def robustness_check(model, loader, dev, n_passes=5):
    """Monte Carlo pod ręczną kalibrację: F1 przy losowym rozrzucie trymerów,
    τ i V_leak (te same sigmy co w treningu HAT). Szum losowany per batch,
    więc jeden przebieg już uśrednia wiele realizacji płytek."""
    model.set_mismatch(True)
    f1s = [evaluate(model, loader, dev)["f1"] for _ in range(n_passes)]
    model.set_mismatch(False)
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_min": float(np.min(f1s)),
        "f1_passes": [round(f, 4) for f in f1s],
    }


@torch.no_grad()
def robustness_check_ensemble(model, loader, dev, n_passes=8, k_of=3, need=2):
    """Odporność z ENSEMBLE na neuronie decyzyjnym: warstwy H i G liczone raz
    (jeden zestaw fizycznych płytek, ze swoim rozrzutem), a neuron D powielony
    `k_of` razy z NIEZALEŻNYM rozrzutem trymerów, decyzja głosem `need`-z-`k_of`.
    To mierzy, czy uśrednienie rozrzutu 3 kopii D podnosi najgorszy przypadek F1
    (bo to rozrzut przy ręcznej kalibracji jest realnym ryzykiem)."""
    model.set_mismatch(True)
    f1s = []
    for _ in range(n_passes):
        tp = fp = fn = 0
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            sh, _ = model.H(x)  # H raz (z szumem tego przebiegu)
            sg, _ = model.G(sh)  # G raz
            votes = torch.zeros(x.shape[0], device=dev)
            for _c in range(k_of):  # D powielony, każdy z własnym szumem
                _, vo = model.O(sg)
                vmax = vo.squeeze(-1).max(dim=1).values
                votes += (vmax >= V_TH).float()
            p = (votes >= need).float()
            tp += ((p == 1) & (y == 1)).sum().item()
            fp += ((p == 1) & (y == 0)).sum().item()
            fn += ((p == 0) & (y == 1)).sum().item()
        rec = tp / max(tp + fn, 1)
        pre = tp / max(tp + fp, 1)
        f1s.append(2 * pre * rec / max(pre + rec, 1e-9))
    model.set_mismatch(False)
    return {
        "f1_mean": float(np.mean(f1s)),
        "f1_min": float(np.min(f1s)),
        "vote": f"{need}-z-{k_of}",
        "f1_passes": [round(f, 4) for f in f1s],
    }


# ============================================================ eksport pod trymery


def pulses_to_fire(w, tau_syn, tau_mem, v_leak, rate_hz=100.0, max_n=60):
    """Ile impulsów po `rate_hz` na JEDNEJ synapsie o wadze w potrzeba do spiku.
    To jest test binarny z Fazy C kalibracji — mierzalny, w odróżnieniu od 'kręć aż zamiga'."""
    if w <= 0:
        return None
    a, b = math.exp(-DT / tau_syn), math.exp(-DT / tau_mem)
    step = max(1, int(round((1.0 / rate_hz) / DT)))
    I, V = 0.0, v_leak
    for n in range(1, max_n + 1):
        for k in range(step):
            I = a * I + (w if k == 0 else 0.0)
            V = b * V + (1 - b) * v_leak + I
            if V >= V_TH:
                return n
    return None


def export_config(model, path, extra=None):
    cfg = {"dt_s": DT, "v_th": V_TH, "channels": CHANNELS, "boards": {}}
    if extra:
        cfg.update(extra)
    pre_names = [CHANNELS, model.H.names, model.G.names]  # auto: 4 lub 8 płytek H

    for layer, pres in zip(model.layers(), pre_names):
        W = layer.weights().detach()
        vl, ts, tm = (
            layer.v_leak().detach(),
            layer.tau_syn().detach(),
            layer.tau_mem().detach(),
        )

        for i, name in enumerate(layer.names):
            w = W[i]
            m = w.abs().max().item()
            if m < W_DEADZONE:
                print(
                    f"[!] {name}: wszystkie wagi w martwej strefie — neuron nieużywany"
                )
                continue

            # Skalujemy neuron tak, by najsilniejsza waga siedziała na pełnej skali
            # trymera, i kompensujemy zapasem do progu: V_leak' = V_th - k*(V_th - V_leak).
            # Niezmienniczość jest ścisła dla pierwszego spiku ze stanu spoczynku
            # (a to jest dokładnie zdarzenie, które wykrywamy). Reset-do-zera łamie ją
            # dla kolejnych spików w serii — stąd dolne ograniczenie na V_leak poniżej.
            V_LEAK_MIN_HW = 0.20 * V_TH  # pasek < 10% jest praktycznie nieustawialny
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
                    print(
                        f"[i] {name}.J{len(syn) + 1} ({pre}): pot {pot:.1f}% — poniżej "
                        f"rozdzielczości trymera, zostaw W na zerze (synapsa nieaktywna)"
                    )
                syn.append(
                    {
                        "port": f"J{len(syn) + 1}",
                        "from": pre,
                        "sign": "+" if wij >= 0 else "-",
                        "pot_pct": round(pot, 1),
                        "w_sim": round(wij, 4),
                        "pulses_to_fire_100Hz": pulses_to_fire(
                            abs(wij) * k, ts[i].item(), tm[i].item(), v_leak_hw
                        ),
                    }
                )

            cfg["boards"][name] = {
                "tau_syn_ms": round(1000 * ts[i].item(), 1),
                "tau_mem_ms": round(1000 * tm[i].item(), 1),
                "v_leak": round(v_leak_hw, 3),
                "led_bar_pct": round(50.0 * v_leak_hw / V_TH, 1),  # próg = 50% paska
                "scale_k": round(k, 3),
                "synapses": syn,
            }

    with open(path, "w") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    print(f"\n[export] {path}", flush=True)
    print(f"{'płytka':6} {'τ_syn':>8} {'τ_mem':>9} {'pasek LED':>10}  synapsy")
    for n, b in cfg["boards"].items():
        s = "  ".join(
            f"{x['port']}={x['from']}{x['sign']}{x['pot_pct']:.0f}%"
            f"(n*={x['pulses_to_fire_100Hz']})"
            for x in b["synapses"]
        )
        print(
            f"{n:6} {b['tau_syn_ms']:7.1f}ms {b['tau_mem_ms']:8.1f}ms {b['led_bar_pct']:9.1f}%  {s}"
        )


# ============================================================ trening


def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hw = json.load(open(args.hw_params)) if args.hw_params else None
    if hw:
        print(f"[hw] zamrażam zmierzone τ dla {len(hw)} płytek — trenuję wagi i V_leak")

    if args.val_data:
        # gotowy podział po plikach z manifestu (build-manifest w encoder_twin.py)
        ds_tr = SpikeClips(args.data, T=args.T, stride=args.stride, limit=args.limit)
        ds_va = SpikeClips(args.val_data, T=args.T, stride=args.stride)
        tr_ds, tr_lab = ds_tr, ds_tr.lab
        va_ds = ds_va
        va_base, va_indices = ds_va, np.arange(len(ds_va))
        print(
            f"[split] train {len(tr_lab)} okien / val {len(va_ds)} okien "
            f"(katalogi manifestu, val pos {100 * ds_va.lab.mean():.1f}%)",
            flush=True,
        )
    else:
        ds = SpikeClips(args.data, T=args.T, stride=args.stride, limit=args.limit)
        tr_idx, va_idx = split_by_file(ds, val_frac=0.2, seed=0)
        tr_ds, tr_lab = Subset(ds, tr_idx), ds.lab[tr_idx]
        va_ds = Subset(ds, va_idx)
        va_base, va_indices = ds, va_idx
        print(
            f"[split] train {len(tr_idx)} okien / val {len(va_idx)} okien "
            f"(po plikach, val pos {100 * ds.lab[va_idx].mean():.1f}%)",
            flush=True,
        )

    dl_tr = DataLoader(
        tr_ds, batch_size=args.bs, sampler=make_sampler(tr_lab, args.num_samples)
    )
    # per-epokowa walidacja na stałej próbce (pełna byłaby ~1/3 kosztu epoki);
    # pełny zbiór walidacyjny idzie dopiero na koniec i do testu odporności
    rng = np.random.default_rng(1)
    if len(va_ds) > args.val_cap:
        va_sub = rng.choice(len(va_ds), size=args.val_cap, replace=False)
        dl_va = DataLoader(Subset(va_ds, va_sub), batch_size=256)
    else:
        dl_va = DataLoader(va_ds, batch_size=256)
    dl_va_full = DataLoader(va_ds, batch_size=256)

    model = LuiNet(hw=hw, quantize=False, wide=args.wide).to(dev)
    if args.wide:
        print("[topo] wariant SZEROKI: H=8 płytek (7→8→3→1)", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    hat_epochs = 0 if args.no_quant else int(round(args.hat_frac * args.epochs))
    freeze_ep = int(0.8 * args.epochs)
    print(
        f"[plan] HAT (pełna precyzja + szum sprzętowy): epoki 0..{hat_epochs - 1}, "
        f"QAT (kwantyzacja {W_LEVELS} działek): {hat_epochs}..{args.epochs - 1}, "
        f"znaki zamrożone od {freeze_ep}, patience {args.patience}",
        flush=True,
    )

    log_f = open(args.log, "w", buffering=1)
    log_f.write("epoch,phase,loss,recall,precision,f1,lr,sec\n")

    best, since_best = -1.0, 0
    for ep in range(args.epochs):
        t0 = time.time()
        phase = "HAT" if ep < hat_epochs else "QAT"

        if ep == hat_epochs and not args.no_quant:
            model.set_quantize(True)
            # tylko skwantyzowany checkpoint da się przenieść na trymery —
            # reset best, żeby QAT nie przegrywał z niekwantyzowanym HAT
            best, since_best = -1.0, 0
            print(f"[ep {ep}] start QAT: kwantyzacja włączona, reset best", flush=True)
        if ep == freeze_ep:
            model.freeze_signs()
            print(
                f"[ep {ep}] znaki wag zamrożone (przełączniki +/- ustalone)", flush=True
            )

        model.train()
        model.set_mismatch(True)
        loss_sum, n_b = 0.0, 0
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            if args.aug:
                # augmentacja spike-trainów: gubienie pojedynczych spików (enkoder
                # na sprzęcie też nie jest deterministyczny) + wspólne przesunięcie
                # czasowe — zdarzenie nie zawsze zaczyna się w tej samej ramce okna
                x = x * (torch.rand_like(x) > 0.05).float()
                x = torch.roll(x, shifts=int(torch.randint(-3, 4, (1,))), dims=1)
            loss, _ = loss_fn(
                model(x), y, args.pos_weight, margin_w=args.margin_w, spk_w=args.spk_w
            )
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += loss.item()
            n_b += 1
        model.set_mismatch(False)
        sched.step()

        m = evaluate(model, dl_va, dev)
        # checkpoint po F1: recall+0.3*precision łapał się na zdegenerowanych
        # wczesnych epokach ("zgłaszaj wszystko"). F1 karze to wprost.
        if m["f1"] > best:
            best, since_best, tag = m["f1"], 0, " *"
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": ep,
                    "phase": phase,
                    "metrics": m,
                    "wide": args.wide,
                },
                args.ckpt,
            )
        else:
            since_best += 1
            tag = ""
        sec = time.time() - t0
        lr_now = sched.get_last_lr()[0]
        log_f.write(
            f"{ep},{phase},{loss_sum / max(n_b, 1):.4f},{m['recall']:.4f},"
            f"{m['precision']:.4f},{m['f1']:.4f},{lr_now:.5f},{sec:.1f}\n"
        )
        print(
            f"ep {ep:3d} [{phase}] loss {loss_sum / max(n_b, 1):.4f}  rec {m['recall']:.3f} "
            f"prec {m['precision']:.3f} f1 {m['f1']:.3f}  {sec:.0f}s{tag}",
            flush=True,
        )

        # early stopping liczony osobno w każdej fazie (reset przy starcie QAT);
        # w HAT nie zatrzymuje treningu, tylko skraca fazę
        if since_best >= args.patience:
            if phase == "HAT":
                print(
                    f"[ep {ep}] HAT bez poprawy od {args.patience} epok — "
                    f"przechodzę do QAT",
                    flush=True,
                )
                hat_epochs = ep + 1
            else:
                print(
                    f"[ep {ep}] QAT bez poprawy od {args.patience} epok — stop",
                    flush=True,
                )
                break

    log_f.close()
    state = torch.load(args.ckpt)
    sd = (
        state["model"] if "model" in state else state
    )  # kompatybilność ze starym best.pt
    if not args.no_quant:
        model.set_quantize(True)
    model.load_state_dict(sd)

    final = evaluate(model, dl_va_full, dev)
    rob = robustness_check(model, dl_va_full, dev)
    print(
        f"\n[best] epoka {state.get('epoch', '?')} ({state.get('phase', '?')}), "
        f"pełna walidacja: rec {final['recall']:.3f} prec {final['precision']:.3f} "
        f"f1 {final['f1']:.3f}"
    )
    print(
        f"[odporność] F1 pod rozrzutem sprzętowym (±½ działki trymera, ±10% τ, "
        f"±2% V_leak): mean {rob['f1_mean']:.3f}, min {rob['f1_min']:.3f}",
        flush=True,
    )
    if final["f1"] - rob["f1_mean"] > 0.05:
        print(
            "[!] duży spadek F1 pod rozrzutem — kalibracja we wtorek musi być "
            "dokładna; rozważ dłuższą fazę HAT (--hat-frac)"
        )

    rob_ens = robustness_check_ensemble(model, dl_va_full, dev)
    print(
        f"[odporność+ensemble] potrójny D, głos {rob_ens['vote']}: "
        f"mean {rob_ens['f1_mean']:.3f}, min {rob_ens['f1_min']:.3f}",
        flush=True,
    )

    print("[zdarzenia] walidacja (poziom klipów):")
    ev_va = evaluate_events(model, va_base, va_indices, dev)
    extra = {
        "val_metrics": {k: round(v, 4) for k, v in final.items()},
        "val_events": ev_va,
        "robustness": rob,
        "robustness_ensemble": rob_ens,
        "best_epoch": state.get("epoch"),
        "wide": bool(args.wide),
    }
    if args.test_data:
        ds_te = SpikeClips(args.test_data, T=args.T, stride=args.stride)
        te = evaluate(model, DataLoader(ds_te, batch_size=256), dev)
        print(
            f"[test] rec {te['recall']:.3f} prec {te['precision']:.3f} "
            f"f1 {te['f1']:.3f}",
            flush=True,
        )
        print("[zdarzenia] test (poziom klipów):")
        ev_te = evaluate_events(model, ds_te, np.arange(len(ds_te)), dev)
        extra["test_metrics"] = {k: round(v, 4) for k, v in te.items()}
        extra["test_events"] = ev_te

    export_config(model, args.out, extra=extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Trening SNN 6->4->3->1 z ograniczeniami sprzętowymi."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="Trenuj model SNN")
    t.add_argument("--data", required=True, help="ścieżka do spikes_csv")
    t.add_argument(
        "--val-data", default=None, help="ścieżka do walidacyjnego spikes_csv"
    )
    t.add_argument("--test-data", default=None, help="ścieżka do testowego spikes_csv")
    t.add_argument(
        "--hw-params", default=None, help="JSON ze zmierzonymi parametrami płytek"
    )
    t.add_argument(
        "--out", default="hw_config.json", help="plik wyjściowy z konfiguracją"
    )
    t.add_argument("--limit", type=int, default=None, help="limit plików (szybki test)")
    t.add_argument("--epochs", type=int, default=100)
    t.add_argument("--bs", type=int, default=64)
    t.add_argument("--lr", type=float, default=0.01)
    t.add_argument("--T", type=int, default=200)
    t.add_argument("--stride", type=int, default=50)
    t.add_argument("--num-samples", type=int, default=10000)
    t.add_argument("--val-cap", type=int, default=2000)
    t.add_argument(
        "--pos-weight",
        type=float,
        default=1.0,
        help="UWAGA: make_sampler() już balansuje klasy 50/50 w epoce "
        "(WeightedRandomSampler) — dodatkowy pos_weight>1 w BCE "
        "podwójnie waży klasę pozytywną. Zmierzone przy pos_weight=3.0: "
        "precyzja 0.22-0.26 @ recall 0.85-0.89 przez cały trening "
        "(sieć zgłasza wszystko). Zwycięskie biegi: pos_weight=1.0.",
    )
    t.add_argument("--margin-w", type=float, default=0.5)
    t.add_argument("--spk-w", type=float, default=0.0)
    t.add_argument("--no-quant", action="store_true", help="wyłącz fazę QAT")
    t.add_argument("--hat-frac", type=float, default=0.5)
    t.add_argument("--patience", type=int, default=20)
    t.add_argument(
        "--aug", action="store_true", help="augmentacja: gubienie spików, jitter"
    )
    t.add_argument("--wide", action="store_true", help="wariant SZEROKI: 8 płytek H")
    t.add_argument("--log", default="train.log")
    t.add_argument("--ckpt", default="best.pt")
    t.set_defaults(func=train)

    c = sub.add_parser(
        "compare",
        help="Porównaj model z symulacją (uruchom z parametrami: --sim, --hw, --layer)",
    )
    c.add_argument("--sim")
    c.add_argument("--hw")
    c.add_argument("--layer")
    # c.set_defaults(func=compare_fn)  # compare_fn logic is assumed implemented elsewhere

    args = ap.parse_args()
    if hasattr(args, "func"):
        args.func(args)
