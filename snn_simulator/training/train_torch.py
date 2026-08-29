import torch
import torch.nn as nn
import torch.optim as optim
from utils.synthetic_data import generate_synthetic_sample

def train_torch(model, epochs=10):
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    loss_fn = nn.BCEWithLogitsLoss()

    device = next(model.parameters()).device

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(100):
            # 🔥 generujemy sztuczne dane
            spikes, label = generate_synthetic_sample(T=100)

            # batch
            spikes = spikes.unsqueeze(0).to(device)
            label = torch.tensor([[label]]).float().to(device)

            # forward
            output = model(spikes)

            # loss
            loss = loss_fn(output, label)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.clamp_weights()
            model.quantize_weights()

            total_loss += loss.item()

        print(f"Epoch {epoch}, Loss: {total_loss}")
        print("output:", output.detach().cpu().numpy(), "label:", label.item())