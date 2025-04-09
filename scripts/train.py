import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from sklearn.utils.class_weight import compute_class_weight
from PIL import Image
import warnings

# 1. Параметри
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 100
TRAIN_DIR = '../data/train'
VAL_DIR = '../data/validation'

# 2. Розрахунок ваг класів (для незбалансованих даних)
def get_class_weights(train_dir):
    dataset = ImageFolder(train_dir)
    classes = dataset.classes
    labels = [sample[1] for sample in dataset.samples]
    
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    
    return torch.tensor(class_weights, dtype=torch.float32)

class_weights = get_class_weights(TRAIN_DIR)
print("Class weights:", class_weights)

# 3. Аугментація даних та завантажувачі
train_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(0, translate=(0.2, 0.2)),
    transforms.RandomAffine(0, shear=20),
    transforms.RandomAffine(0, scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)

val_dataset = ImageFolder(VAL_DIR, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Після цього рядка:
# train_dataset = ImageFolder(TRAIN_DIR, transform=train_transform)

# Додайте:
print("Class names:", train_dataset.classes)
print("Class to index mapping:", train_dataset.class_to_idx)
exit()
# 4. Архітектура моделі
class CNNModel(nn.Module):
    def __init__(self, num_classes=6):
        super(CNNModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 28 * 28, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = CNNModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# 5. Функція втрат та оптимізатор
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = optim.Adam(model.parameters())

# 6. Навчання моделі
best_val_accuracy = 0.8

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels.data)
    
    train_loss = train_loss / len(train_loader)
    train_acc = train_correct.double() / len(train_dataset)
    
    # Валідація
    model.eval()
    val_loss = 0.0
    val_correct = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)
    
    val_loss = val_loss / len(val_loader)
    val_acc = val_correct.double() / len(val_dataset)
    
    print(f'Epoch {epoch+1}/{EPOCHS}')
    print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
    print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')
    
    # Збереження найкращої моделі
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print('Model saved!')

# 7. Оцінка результатів
print("\nTraining completed!")
print(f"Final validation accuracy: {best_val_accuracy:.2f}")