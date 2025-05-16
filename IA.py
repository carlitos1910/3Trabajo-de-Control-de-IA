import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
torch.manual_seed(42)

x = np.linspace(-5, 5, 100)
y_true = 2 * x + 1
y_noisy = y_true + np.random.normal(0, 2, x.shape)

x_tensor = torch.from_numpy(x).float().unsqueeze(1)
y_tensor = torch.from_numpy(y_noisy).float().unsqueeze(1)

class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(1, 1)
    
    def forward(self, x):
        return self.layer(x)

model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

epochs = 200
losses = []

for epoch in range(epochs):
    outputs = model(x_tensor)
    loss = criterion(outputs, y_tensor)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}')

w = model.layer.weight.item()
b = model.layer.bias.item()

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(x, y_noisy, alpha=0.5)
plt.plot(x, y_true, 'r-', linewidth=2)
plt.plot(x, model(x_tensor).detach().numpy(), 'g--', linewidth=2)
plt.title('Linear Regression Fit')
plt.xlabel('x')
plt.ylabel('y')

plt.subplot(1, 2, 2)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE')

plt.tight_layout()
plt.show()

print(f"\nLearned: w = {w:.4f}, b = {b:.4f}")
print(f"Actual:  w = 2.0000, b = 1.0000")
