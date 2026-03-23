import sys
import torch
import numpy as np

# Dodaj główny folder
sys.path.insert(0, ".")

from snn_pipeline.snn_model import GlassBreakSNN
from snn_pipeline.config import DEVICE

model = GlassBreakSNN(quantize_mode="none").to(DEVICE)
model.eval()

print("Initial thresholds:")
print("vth_n1:", model.vth_n1.item())
print("vth_n2:", model.vth_n2.item())
print("vth_n3:", model.vth_n3.item())
print("vth_inh:", model.vth_inh.item())

print("Initial weights:")
print("w_n1:", model.w_n1.item())
print("w_n2:", model.w_n2.item())
print("w_n3_from_n1:", model.w_n3_from_n1.item())
print("w_n3_from_n2:", model.w_n3_from_n2.item())
print("w_inh:", model.w_inh.item())
print("w_inh_to_n3:", model.w_inh_to_n3.item())

# Create a sequence of 5 spikes on all channels (burst)
# batch=1, channels=1, timesteps=10
x = torch.zeros((1, 1, 10))
x[0, 0, 0:5] = 1.0  # Burst of 5 spikes

x = x.to(DEVICE)

print("\n--- Running forward pass ---")
trigger, all_spikes = model(x)

print(f"\nTrigger: {trigger.item()}")

for name, spikes in all_spikes.items():
    print(f"{name} spikes: {spikes.squeeze().tolist()}")
