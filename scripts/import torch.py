import torch

# Перевірка GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Простий тензор
x = torch.tensor([1, 2, 3]).to(device)
print(x)