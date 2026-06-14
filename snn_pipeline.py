"""
    Pipeline przepuszczony przez czat gpt dla lepszego readability i komentarzy:)


Czyta z encoder/encoder_output/positive.csv i negative.csv skibidi

 Trening sieci snntorch 3→4→7→1
 Mapowanie wag → wartości rezystancji (potencjometry RV1–RV6)
"""

import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
from snntorch import spikegen
from torch.utils.data import DataLoader, TensorDataset, random_split
from pathlib import Path
from torch.utils.data import WeightedRandomSampler

ROOT              = Path(__file__).resolve().parent
POSITIVE_CSV      = ROOT / "encoder" / "encoder_output" / "positive.csv"
NEGATIVE_CSV      = ROOT / "encoder" / "encoder_output" / "negative.csv"
OUT_DIR           = ROOT / "encoder" / "encoder_output"

T_STEPS    = 20       # timestepy rate coding
BATCH_SIZE = 16
EPOCHS     = 150
LR         = 1e-3
SEED       = 42

# Zakresy potencjometrów w ohm
R_WEIGHT_MIN      = 500
R_WEIGHT_MAX      = 22_222
R_TAU_MEM_MAX     = 100_000
R_TAU_SYN_DEFAULT = 10_000

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")



def load_csv(path: Path, label: int) -> tuple[np.ndarray, np.ndarray]:
    """

    Każdy plik WAV → jedna próbka treningowa: agregacja ramek przez max/mean/mean.
      
      - peak =max(Peak)      — najwyższy impuls w całym nagraniu
      - mean = mean(Mean)     — średni poziom energii
      -cv   = mean(CV)       — średnia zmienność (burst detection)
    Zwraca X: (N, 3), y: (N,)
    """
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    required = {"File_Name", "Peak", "Mean", "CV"}
    missing  = required - set(df.columns)
    
    if missing:
        raise ValueError(f"Brak kolumn w {path.name}: {missing}")

    X, y = [], []
    
    
    for fname, group in df.groupby("File_Name"):
        
        peak = float(group["Peak"].max())
        mean = float(group["Mean"].mean())
        cv   = float(group["CV"].mean())
        X.append([peak, mean, cv])
        y.append(label)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def load_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"  Positive: {POSITIVE_CSV}")
    print(f"  Negative: {NEGATIVE_CSV}")

    X_pos, y_pos = load_csv(POSITIVE_CSV, label=1)
    X_neg, y_neg = load_csv(NEGATIVE_CSV, label=0)

    print(f"  glass={len(X_pos)}  other={len(X_neg)}")

    X = np.concatenate([X_pos, X_neg], axis=0)
    y = np.concatenate([y_pos, y_neg], axis=0)

    # Max per kanał z całego zbioru → do normalizacji i Arduino MAX_VALS
    max_vals = X.max(axis=0)
    return X, y, max_vals


def encode_rate(X: np.ndarray, max_vals: np.ndarray, T: int) -> torch.Tensor:
    """
    X: (N, 3) → spikes: (T, N, 3)  Bernoulli rate coding.
    Normalizacja identyczna z Arduino: v_norm = clip(v / max_val, 0, 1)
    """
    X_norm = np.clip(X / (max_vals + 1e-8), 0.0, 1.0)
    X_t    = torch.tensor(X_norm, dtype=torch.float32)
    return spikegen.rate(X_t, num_steps=T)   # (T, N, 3)


class GlassBreakSNN(nn.Module):
    """
    Feedforward SNN: 3 → 4 → 7 → 1  (15 neuronów LIF)
    learn_beta=True  → tau_mem jest uczony
    learn_threshold=True → próg jest uczony
    """
    def __init__(self, beta_init: float = 0.9):
        super().__init__()
        self.fc1  = nn.Linear(3, 4, bias=True)
        self.lif1 = snn.Leaky(beta=beta_init, learn_beta=True,
                               threshold=0.5, learn_threshold=True)
        self.fc2  = nn.Linear(4, 7, bias=True)
        self.lif2 = snn.Leaky(beta=beta_init, learn_beta=True,
                               threshold=0.5, learn_threshold=True)
        self.fc3  = nn.Linear(7, 1, bias=True)
        self.lif3 = snn.Leaky(beta=beta_init, learn_beta=True,
                               threshold=0.5, learn_threshold=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (T, B, 3) → out: (B,) proporcja aktywacji output neuronu"""
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        mem3 = self.lif3.init_leaky()
        acc  = torch.zeros(x.shape[1], 1, device=x.device)
        for t in range(x.shape[0]):
            spk1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            acc += spk3
        return (acc / x.shape[0]).squeeze(-1)   # (B,)


def collate_fn(batch):
    """Transpozycja (B, T, 3) → (T, B, 3) wymagana przez model."""
    xs, ys = zip(*batch)
    return torch.stack(xs).permute(1, 0, 2), torch.stack(ys)


def train_epoch(model, loader, opt, crit):
    model.train()
    loss_sum, correct = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        out  = model(xb)
        if loss_sum == 0:
            print(
                "OUT:",
                out.mean().item(),
                out.min().item(),
                out.max().item()
            )
        loss = crit(out, yb)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )
        opt.step()
        loss_sum += loss.item()
        correct  += ((out > 0.5).float() == yb).sum().item()
    return loss_sum / len(loader), correct / len(loader.dataset)


def eval_epoch(model, loader, crit):
    model.eval()
    loss_sum = correct = tp = fp = fn = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            out   = model(xb)
            loss_sum += crit(out, yb).item()
            preds  = (out > 0.5).float()
            correct += (preds == yb).sum().item()
            tp += ((preds == 1) & (yb == 1)).sum().item()
            fp += ((preds == 1) & (yb == 0)).sum().item()
            fn += ((preds == 0) & (yb == 1)).sum().item()
    prec = tp / (tp + fp + 1e-8)
    rec  = tp / (tp + fn + 1e-8)
    return loss_sum / len(loader), correct / len(loader.dataset), prec, rec



def weights_to_resistances(model: GlassBreakSNN) -> dict:
    """
    Mapowanie:
      waga > 0  → R = R_MIN + w_norm*(R_MAX-R_MIN)   (excitatory, RV1–RV3)
      waga < 0  → R = R_WEIGHT_MAX                    (inhibitory → max opór)
      beta      → R = beta_norm * R_TAU_MEM_MAX        (RV6)
      tau_syn   → R = R_TAU_SYN_DEFAULT (stały)        (RV4)
      V_leak    → kalibracja manualna                  (RV5)
    """
    layers = [
        ("L1_3to4",  model.fc1.weight.data, model.lif1.beta.data),
        ("L2_4to7",  model.fc2.weight.data, model.lif2.beta.data),
        ("Out_7to1", model.fc3.weight.data, model.lif3.beta.data),
    ]
    results = {}
    for name, W, beta in layers:
        W_np   = W.cpu().numpy()
        b_np   = beta.cpu().numpy().flatten()
        w_min, w_max = W_np.min(), W_np.max()
        W_norm = (W_np - w_min) / (w_max - w_min + 1e-8)

        neurons = []
        for i in range(W_np.shape[0]):
            b = float(np.clip(b_np[i] if i < len(b_np) else b_np[0], 0.0, 1.0))
            weights_out = []
            for j in range(W_np.shape[1]):
                w_raw = float(W_np[i, j])
                r = R_WEIGHT_MAX if w_raw < 0 else int(
                    R_WEIGHT_MIN + W_norm[i, j] * (R_WEIGHT_MAX - R_WEIGHT_MIN))
                weights_out.append({
                    "input": j,
                    "w_raw": round(w_raw, 4),
                    "sign": "INH" if w_raw < 0 else "EXC",
                    "RV_ohm": r
                })
            neurons.append({
                "neuron":           i,
                "weights_RV1_RV3":  weights_out,
                "RV4_tau_syn_ohm":  R_TAU_SYN_DEFAULT,
                "RV6_tau_mem_ohm":  int(b * R_TAU_MEM_MAX),
                "RV5_V_leak":       "kalibracja manualna",
                "beta":             round(b, 4),
            })
        results[name] = neurons
    return results


def print_resistance_table(res: dict):
    print("\n" + "=" * 60)
    print("ETAP 5 — TABELA REZYSTANCJI POTENCJOMETRÓW")
    print("=" * 60)
    for layer, neurons in res.items():
        print(f"\n── {layer} ──")
        for n in neurons:
            print(f"  Neuron {n['neuron']}  (beta={n['beta']:.4f})")
            for w in n["weights_RV1_RV3"]:
                rv = w["input"] + 1
                print(f"    RV{rv}: {w['RV_ohm']:>7} Ω  ({w['sign']}, w={w['w_raw']:+.4f})")
            print(f"    RV4 (τ_syn): {n['RV4_tau_syn_ohm']:>7} Ω  (stały)")
            print(f"    RV6 (τ_mem): {n['RV6_tau_mem_ohm']:>7} Ω")
            print(f"    RV5 (V_leak): {n['RV5_V_leak']}")


def main():
    print("\n── ETAP 3: Ładowanie CSV ──")
    X, y, max_vals = load_dataset()
    print(f"  Próbki: {len(X)}   shape: {X.shape}")
    print(f"  MAX_VALS → wpisz do Arduino:")
    print(f"    MAX_PEAK_VAL  {max_vals[0]:.2f}")
    print(f"    MAX_MEAN_VAL  {max_vals[1]:.2f}")
    print(f"    MAX_CV_VAL    {max_vals[2]:.4f}")

    (OUT_DIR).mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "max_vals.json", "w") as f:
        json.dump({"peak": float(max_vals[0]),
                   "mean": float(max_vals[1]),
                   "cv":   float(max_vals[2])}, f, indent=2)
    print("  Zapisano: encoder_output/max_vals.json")

    print("\n── Rate coding ──")
    spikes = encode_rate(X, max_vals, T_STEPS)          # (T, N, 3)
    print(f"  Tensor: {tuple(spikes.shape)}  gęstość={spikes.float().mean():.3f}")

    # DataLoader — przechowujemy (N, T, 3), collate_fn transponuje na (T, B, 3)
    spikes_ds = spikes.permute(1, 0, 2)                  # (N, T, 3)
    labels_t  = torch.tensor(y, dtype=torch.float32)
    dataset   = TensorDataset(spikes_ds, labels_t)

    n_train = int(0.8 * len(dataset))
    n_val   = len(dataset) - n_train
    gen     = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)

    train_labels = labels_t[train_ds.indices]

    class_counts = torch.bincount(train_labels.long())

    class_weights = 1.0 / class_counts.float()

    sample_weights = class_weights[
        train_labels.long()
    ]

    sampler = WeightedRandomSampler(
        sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_fn
    )
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                               shuffle=False, collate_fn=collate_fn)

    print("\n── ETAP 4: Trening SNN 3→4→7→1 ──")
    model     = GlassBreakSNN(beta_init=0.9).to(device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)

            if m.bias is not None:
                nn.init.zeros_(m.bias)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_recall = -1.0
    best_state = {
        k: v.clone()
        for k, v in model.state_dict().items()
    }
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc              = train_epoch(model, train_loader, optimizer, criterion)
        va_loss, va_acc, prec, rec   = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:3d}/{EPOCHS}]  "
                  f"loss {tr_loss:.4f}/{va_loss:.4f}  "
                  f"acc {tr_acc:.3f}/{va_acc:.3f}  "
                  f"P={prec:.3f} R={rec:.3f}")

        if rec > best_recall:
            best_recall = rec
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"\n  Najlepszy recall: {best_recall:.3f}")
    model.load_state_dict(best_state)
    torch.save(best_state, OUT_DIR / "snn_glassbreak_best.pt")
    print("  Zapisano: encoder_output/snn_glassbreak_best.pt")

    print("\n── ETAP 5: Mapowanie → rezystancje ──")
    res = weights_to_resistances(model)
    print_resistance_table(res)
    with open(OUT_DIR / "resistance_table.json", "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print("\n  Zapisano: encoder_output/resistance_table.json")

    print("\nKolejność kalibracji hardware:")
    print("  1. RV4 = 10kΩ dla wszystkich")
    print("  2. RV6 z tabeli (τ_mem per neuron)")
    print("  3. RV1–RV3 z tabeli (wagi per neuron)")
    print("  4. Podaj glass break → obserwuj output")
    print("  5. Dostrój RV5 (V_leak) na neuronie output")


if __name__ == "__main__":
    main()