import torch

def generate_sample_torch(T, input_size, label):
    spikes = torch.rand(T, input_size)

    if label == 1:
        spikes += 0.3 # SYGNAŁ
    else:
        spikes *= 0.3

    spikes = (spikes > 0.8).float() # SZUM

    return spikes, label