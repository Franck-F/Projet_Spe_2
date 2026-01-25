# Modèle de Production - Détection de Métastases CAMELYON17

## Informations du Modèle
- **Version**: SimpleCNN_v1_20260125_190903
- **Date**: 2026-01-25 19:09:05
- **Architecture**: SimpleCNN
- **Paramètres**: 11,308,097

## Performances

### Patch-Level
- **Accuracy**: 0.8971
- **F1-Score**: 0.8020
- **AUC-ROC**: 0.9049
- **Seuil optimal**: 0.3112

### Patient-Level
- **Accuracy**: 0.8947
- **F1-Score**: 0.9286
- **AUC-ROC**: 0.9429

## Utilisation

### Chargement du modèle
```python
import torch
from pathlib import Path

# Charger le checkpoint
checkpoint = torch.load('SimpleCNN_v1_20260125_190903.pth')
model = SimpleCNN()  # Définir l'architecture
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

### Prétraitement des images
```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Inférence
```python
# Charger et prétraiter l'image
image = Image.open('patch.png').convert('RGB')
image_tensor = transform(image).unsqueeze(0)

# Prédiction
with torch.no_grad():
    output = model(image_tensor)
    prob = torch.sigmoid(output).item()
    prediction = 1 if prob > 0.3112 else 0
```

## Fichiers
- `SimpleCNN_v1_20260125_190903.pth`: Poids du modèle et état de l'entraînement
- `SimpleCNN_v1_20260125_190903_metrics.json`: Métriques détaillées
- `SimpleCNN_v1_20260125_190903_config.json`: Configuration de prétraitement
- `README.md`: Ce fichier

## Notes
- Le modèle a été entraîné sur 436 patchs de 21 patients
- Split stratégique par centre (centres [0 1 2] pour train)
- Utilisation de Focal Loss pour gérer le déséquilibre de classes
- Seuil de décision optimisé sur le set de validation

## Limites
- Dataset de taille limitée (5000 patchs)
- Agrégation patient simpliste (max des probabilités)
- Performances peuvent varier selon le centre hospitalier (domain shift)

## Contact
Pour toute question sur ce modèle, consulter le notebook de modélisation.
