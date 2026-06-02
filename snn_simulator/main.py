from models.snn_torch import GlassSNN
from training.train_torch import train_torch

model = GlassSNN()
train_torch(model, epochs=10)