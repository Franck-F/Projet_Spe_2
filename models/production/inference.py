#!/usr/bin/env python3
"""
Script d'inférence pour le modèle SimpleCNN CAMELYON17
Version: SimpleCNN_v1_20260125_190903
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path
import json

# Architecture du modèle (doit être identique à l'entraînement)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def load_model(model_path):
    """Charger le modèle depuis le checkpoint"""
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def preprocess_image(image_path):
    """Prétraiter une image pour l'inférence"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, threshold=0.3112):
    """Faire une prédiction"""
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > threshold else 0

    return {
        'probability': prob,
        'prediction': prediction,
        'label': 'Tumor' if prediction == 1 else 'Normal'
    }

if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = Path(__file__).parent / 'SimpleCNN_v1_20260125_190903.pth'

    # Charger le modèle
    print(f"Chargement du modèle: {model_path}")
    model, checkpoint = load_model(model_path)
    print(f"Modèle chargé (AUC: {checkpoint['best_val_auc']:.4f})")

    # Prétraiter l'image
    print(f"Traitement de l'image: {image_path}")
    image_tensor = preprocess_image(image_path)

    # Prédiction
    result = predict(model, image_tensor)

    print("\nRésultat:")
    print(f"  Probabilité: {result['probability']:.4f}")
    print(f"  Prédiction: {result['label']}")
