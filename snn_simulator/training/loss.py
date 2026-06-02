import numpy as np


def compute_loss(output, label):
    rate = output.mean(axis=0)

    target = np.zeros(2)
    target[label] = 1

    return ((rate - target) ** 2).mean()
