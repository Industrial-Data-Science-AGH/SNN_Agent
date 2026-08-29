import torch
import numpy as np

def generate_synthetic_sample(T=100):
    """
    Tworzy dane:
    [HF, LF, envelope]
    """

    label = np.random.randint(0, 2)

    hf = np.random.rand(T) * 0.2
    lf = np.random.rand(T) * 0.2
    env = np.random.rand(T) * 0.2

    if label == 1:
        # 🔥 GLASS
        hf += 2.0  # bardzo wysoki HF

        burst = np.zeros(T)
        start = np.random.randint(20, 60)
        burst[start:start+8] = 1.0
        env += burst

        lf *= 0.1

    else:
        # 🔹 NORMAL
        lf += 2.0  # dużo LF
        hf *= 0.05
        env *= 0.1

    hf = hf / (hf.max() + 1e-6)
    lf = lf / (lf.max() + 1e-6)
    env = env / (env.max() + 1e-6)

    features = np.stack([hf, lf, env], axis=1)

    return torch.tensor(features).float(), label