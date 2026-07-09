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

Użycie:
    python snn_hw_pipeline.py train   --data ./spikes_csv --out hw_config.json
    python snn_hw_pipeline.py train   --data ./spikes_csv --hw-params hw_params.json --out hw_config.json
    python snn_hw_pipeline.py compare --sim sim.npz --hw hw_spikes.csv --layer H
"""
from __future__ import annotations

import argparse, glob, json, math, os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

# ============================================================ stałe sprzętowe

DT = 0.010                      # s, == hop ramki encodera
V_TH = 1.0                      # próg, sprzętowo VDD/2 — nie jest uczony
V_LEAK_MAX = 0.90 * V_TH        # wyżej neuron strzela samoczynnie
TAU_SYN_RANGE = (0.005, 0.220)  # C_syn = 10 uF, g 45 uS..inf
TAU_MEM_RANGE = (0.020, 2.200)  # C_mem = 22 uF, g 10 uS..inf
W_MAX = 1.60                    # waga przy potencjometrze na pełnej skali
W_DEADZONE = 0.05               # poniżej tego nie da się ustawić trymerem
W_LEVELS = 20                   # realna rozdzielczość ręki na trymerze (~5% skali)

CHANNELS = ["peak", "peak_cnt", "crest", "cv", "zcr", "flux"]

# maski łączności: [dla każdego neuronu post] lista indeksów pre, max 3
MASK_H = [[0, 1, 2], [3, 4, 5], [0, 3, 5], [1, 2, 4]]   # 6 -> 4
MASK_G = [[0, 1, 2], [1, 2, 3], [2, 3, 0]]              # 4 -> 3
MASK_O = [[0, 1, 2]]                                    # 3 -> 1  (dokładnie 3 wejścia)

NAMES_H = ["H0", "H1", "H2", "H3"]
NAMES_G = ["G0", "G1", "G2"]
NAMES_O = ["D"]


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

    def __init__(self, n_pre, n_post, mask, names, hw=None, quantize=True):
        super().__init__()
        self.n_pre, self.n_post, self.names = n_pre, n_post, names
        self.quantize = quantize
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
            self.p_tsyn = nn.Parameter(torch.full((n_post,), _inv_map(0.030, *TAU_SYN_RANGE)))
            self.p_tmem = nn.Parameter(torch.full((n_post,), _inv_map(0.150, *TAU_MEM_RANGE)))

        self.p_vleak = nn.Parameter(torch.full((n_post,), _inv_map(0.4, 0.0, V_LEAK_MAX)))
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
        a = torch.exp(torch.tensor(-DT) / self.tau_syn())
        b = torch.exp(torch.tensor(-DT) / self.tau_mem())
        vl, W = self.v_leak(), self.weights()

        I = torch.zeros(B, self.n_post, device=s_in.device)
        V = vl.expand(B, -1).clone()
        out_s, out_v = [], []

        for t in range(T):
            I = a * I + F.linear(s_in[:, t], W)
            V = b * V + (1.0 - b) * vl + I
            out_v.append(V)            # zapis PRZED resetem — stąd płynie gradient straty
            s = spike_fn(V - V_TH)
            V = V * (1.0 - s)          # reset do zera, jak na oscylogramie Lu.i
            out_s.append(s)
        return torch.stack(out_s, 1), torch.stack(out_v, 1)


class LuiNet(nn.Module):
    def __init__(self, hw=None, quantize=True):
        super().__init__()
        self.H = LuiLayer(6, 4, MASK_H, NAMES_H, hw, quantize)
        self.G = LuiLayer(4, 3, MASK_G, NAMES_G, hw, quantize)
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


# ============================================================ dane

class SpikeClips(Dataset):
    """Każdy CSV = jeden klip. Kolumny: frame,s0..s5 (+ opcjonalnie label).
    Bez kolumny label: etykieta 1, jeśli w nazwie pliku jest 'glass'/'szklo'."""

    def __init__(self, root, T=200, stride=50):
        self.win, self.lab = [], []
        files = sorted(glob.glob(os.path.join(root, "**", "*.csv"), recursive=True))
        if not files:
            sys.exit(f"brak CSV w {root}")
        for f in files:
            arr = np.genfromtxt(f, delimiter=",", names=True)
            s = np.stack([arr[f"s{i}"] for i in range(6)], -1).astype(np.float32)
            if "label" in arr.dtype.names:
                y = float(np.max(arr["label"]))
            else:
                nm = os.path.basename(f).lower()
                y = 1.0 if ("glass" in nm or "szklo" in nm) else 0.0
            if len(s) < T:
                s = np.pad(s, ((0, T - len(s)), (0, 0)))
            for i in range(0, len(s) - T + 1, stride):
                self.win.append(s[i:i + T])
                self.lab.append(y)
        self.win = np.stack(self.win)
        self.lab = np.array(self.lab, dtype=np.float32)
        print(f"[data] {len(self.lab)} okien, pozytywnych {int(self.lab.sum())} "
              f"({100*self.lab.mean():.1f}%)")

    def __len__(self):
        return len(self.lab)

    def __getitem__(self, i):
        return torch.from_numpy(self.win[i]), torch.tensor(self.lab[i])


def make_sampler(labels):
    counts = np.bincount(labels.astype(int), minlength=2).astype(float)
    w = (1.0 / np.maximum(counts, 1))[labels.astype(int)]
    return WeightedRandomSampler(torch.from_numpy(w).double(), len(w), replacement=True)


# ============================================================ strata

def loss_fn(out, y, pos_weight, rate_lo=0.02, rate_hi=0.30):
    # logit = jak wysoko membrana wyjściowa doszła w całym oknie.
    # Dokładnie to widzisz na 7. diodzie: "czy kiedykolwiek przekroczyła próg".
    vmax = out["vo"].max(dim=1).values
    logit = 6.0 * (vmax - V_TH)
    bce = F.binary_cross_entropy_with_logits(
        logit, y, pos_weight=torch.tensor(pos_weight, device=y.device))

    # margines na negatywach: tło ma nie podchodzić pod próg
    neg = (y < 0.5).float()
    margin = ((vmax - 0.80).clamp(min=0) ** 2 * neg).sum() / neg.sum().clamp(min=1)

    # neurony ukryte mają żyć: ani martwe, ani zasycone (analog nie strzela w kółko)
    reg = 0.0
    for k in ("sh", "sg"):
        r = out[k].mean(dim=(0, 1))
        reg = reg + ((rate_lo - r).clamp(min=0) ** 2).sum() + ((r - rate_hi).clamp(min=0) ** 2).sum()

    return bce + 0.5 * margin + 0.2 * reg, logit


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


def export_config(model, path):
    cfg = {"dt_s": DT, "v_th": V_TH, "channels": CHANNELS, "boards": {}}
    pre_names = [CHANNELS, NAMES_H, NAMES_G]

    for layer, pres in zip(model.layers(), pre_names):
        W = layer.weights().detach()
        vl, ts, tm = layer.v_leak().detach(), layer.tau_syn().detach(), layer.tau_mem().detach()

        for i, name in enumerate(layer.names):
            w = W[i]
            m = w.abs().max().item()
            if m < W_DEADZONE:
                print(f"[!] {name}: wszystkie wagi w martwej strefie — neuron nieużywany")
                continue

            # Skalujemy neuron tak, by najsilniejsza waga siedziała na pełnej skali
            # trymera, i kompensujemy zapasem do progu: V_leak' = V_th - k*(V_th - V_leak).
            # Niezmienniczość jest ścisła dla pierwszego spiku ze stanu spoczynku
            # (a to jest dokładnie zdarzenie, które wykrywamy). Reset-do-zera łamie ją
            # dla kolejnych spików w serii — stąd dolne ograniczenie na V_leak poniżej.
            V_LEAK_MIN_HW = 0.20 * V_TH   # pasek < 10% jest praktycznie nieustawialny
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
                    print(f"[i] {name}.J{len(syn)+1} ({pre}): pot {pot:.1f}% — poniżej "
                          f"rozdzielczości trymera, zostaw W na zerze (synapsa nieaktywna)")
                syn.append({
                    "port": f"J{len(syn)+1}",
                    "from": pre,
                    "sign": "+" if wij >= 0 else "-",
                    "pot_pct": round(pot, 1),
                    "w_sim": round(wij, 4),
                    "pulses_to_fire_100Hz": pulses_to_fire(abs(wij) * k, ts[i].item(),
                                                           tm[i].item(), v_leak_hw),
                })

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

    print(f"\n[export] {path}")
    print(f"{'płytka':6} {'τ_syn':>8} {'τ_mem':>9} {'pasek LED':>10}  synapsy")
    for n, b in cfg["boards"].items():
        s = "  ".join(f"{x['port']}={x['from']}{x['sign']}{x['pot_pct']:.0f}%"
                      f"(n*={x['pulses_to_fire_100Hz']})" for x in b["synapses"])
        print(f"{n:6} {b['tau_syn_ms']:7.1f}ms {b['tau_mem_ms']:8.1f}ms {b['led_bar_pct']:9.1f}%  {s}")


# ============================================================ trening

def train(args):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    hw = json.load(open(args.hw_params)) if args.hw_params else None
    if hw:
        print(f"[hw] zamrażam zmierzone τ dla {len(hw)} płytek — trenuję wagi i V_leak")

    ds = SpikeClips(args.data, T=args.T, stride=args.stride)
    n_val = max(1, int(0.2 * len(ds)))
    tr, va = torch.utils.data.random_split(ds, [len(ds) - n_val, n_val],
                                           generator=torch.Generator().manual_seed(0))
    tr_lab = ds.lab[tr.indices]
    dl_tr = DataLoader(tr, batch_size=args.bs, sampler=make_sampler(tr_lab))
    dl_va = DataLoader(va, batch_size=args.bs)

    model = LuiNet(hw=hw, quantize=not args.no_quant).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best = -1.0
    for ep in range(args.epochs):
        # znaki wag = przełączniki +/- na płytce. Zamrażamy je pod koniec,
        # żeby ostatnie epoki nie przerzucały polaryzacji synapsy w tę i z powrotem.
        if ep == int(0.8 * args.epochs):
            model.freeze_signs()
            print(f"[ep {ep}] znaki wag zamrożone")

        model.train()
        for x, y in dl_tr:
            x, y = x.to(dev), y.to(dev)
            loss, _ = loss_fn(model(x), y, args.pos_weight)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        m = evaluate(model, dl_va, dev)
        # checkpoint po recallu: fałszywy alarm irytuje, przespane szkło kompromituje
        score = m["recall"] + 0.3 * m["precision"]
        if score > best:
            best = score
            torch.save(model.state_dict(), args.ckpt)
            tag = " *"
        else:
            tag = ""
        if ep % 5 == 0 or tag:
            print(f"ep {ep:3d} loss {loss.item():.4f}  rec {m['recall']:.3f} "
                  f"prec {m['precision']:.3f} f1 {m['f1']:.3f}{tag}")

    model.load_state_dict(torch.load(args.ckpt))
    print("\n[best]", evaluate(model, dl_va, dev))
    export_config(model, args.out)


# ============================================================ sim vs hw

def compare(args):
    """Zgodność spike-trainów symulacji i sprzętu, per neuron (Faza D)."""
    sim = np.load(args.sim)[args.layer]           # [T, n]
    hw = np.genfromtxt(args.hw, delimiter=",", names=True)
    cols = [c for c in hw.dtype.names if c != "frame"]
    real = np.stack([hw[c] for c in cols], -1)
    T = min(len(sim), len(real))
    sim, real = sim[:T], real[:T]

    print(f"{'neuron':8} {'sim Hz':>8} {'hw Hz':>8} {'zgodność':>10} {'vRossum':>9}")
    for i, c in enumerate(cols):
        agree = (sim[:, i] == real[:, i]).mean()
        # van Rossum: wygładzenie eksponentą tau=50ms i L2
        tau = 0.050 / DT
        k = np.exp(-np.arange(0, 5 * tau) / tau)
        d = np.linalg.norm(np.convolve(sim[:, i], k)[:T] - np.convolve(real[:, i], k)[:T])
        print(f"{c:8} {sim[:,i].mean()/DT:7.1f} {real[:,i].mean()/DT:7.1f} "
              f"{100*agree:9.1f}% {d:9.2f}")
        if agree < 0.85:
            print(f"    -> {c} rozjeżdża się. Wróć do Fazy C dla tej płytki.")


# ============================================================ CLI

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", default="hw_config.json")
    t.add_argument("--ckpt", default="best.pt")
    t.add_argument("--hw-params", default=None, help="zmierzone τ per płytka (Faza A)")
    t.add_argument("--epochs", type=int, default=120)
    t.add_argument("--bs", type=int, default=32)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--T", type=int, default=200, help="ramek na okno (200 = 2 s)")
    t.add_argument("--stride", type=int, default=50)
    t.add_argument("--pos-weight", type=float, default=3.0)
    t.add_argument("--no-quant", action="store_true")
    t.set_defaults(func=train)

    c = sub.add_parser("compare")
    c.add_argument("--sim", required=True)
    c.add_argument("--hw", required=True)
    c.add_argument("--layer", default="H", choices=["H", "G", "O"])
    c.set_defaults(func=compare)

    args = ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    args.func(args)


if __name__ == "__main__":
    main()
