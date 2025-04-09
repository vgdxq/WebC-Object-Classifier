import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import warnings

# 1. Параметри
IMG_SIZE = (224, 224)
NUM_CLASSES = 6
MODEL_PATH = 'best_model.pth'

# 2. Пристрій
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 3. Архітектура моделі
class CNNModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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

# 4. Ініціалізація моделі
model = CNNModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 5. Трансформації
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 6. Назви класів
class_names = {
    0: 'apple',
    1: 'cat',
    2: 'dog',
    3: 'flower',
    4: 'human',
    5: 'spider'
}

# 7. Функція передбачення
def predict_frame(frame):
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(tensor)
        _, predicted_idx = torch.max(outputs, 1)

    return class_names.get(predicted_idx.item(), "Unknown")

# 8. Захоплення з камери
cap = cv2.VideoCapture(0)
print("Натисніть 'q' для виходу.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    resized_frame = cv2.resize(frame, IMG_SIZE)
    label = predict_frame(resized_frame)

    cv2.putText(frame, f'Predicted object: {label}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Webcam Prediction', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
