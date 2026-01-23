# %% SAUVEGARDE POUR PRODUCTION
"""
Sauvegarde du modèle pour la mise en production
Inclut le modèle, les métadonnées et un fichier de configuration
"""

import json
from datetime import datetime

# Créer le dossier de production
PRODUCTION_DIR = Path('../models/production')
PRODUCTION_DIR.mkdir(parents=True, exist_ok=True)

# Timestamp pour versioning
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_version = f"SimpleCNN_v1_{timestamp}"

print(f"\n🚀 SAUVEGARDE POUR PRODUCTION")
print(f"Version: {model_version}")
print("="*60)

# 1. Sauvegarder le modèle complet (architecture + poids)
model_path = PRODUCTION_DIR / f"{model_version}.pth"
torch.save({
    'model_state_dict': model.state_dict(),
    'model_architecture': 'SimpleCNN',
    'input_size': (3, 224, 224),
    'num_classes': 1,
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': EPOCHS,
    'best_val_auc': best_val_auc,
    'training_history': history
}, model_path)
print(f"✅ Modèle sauvegardé: {model_path}")

# 2. Sauvegarder les métriques finales
metrics_path = PRODUCTION_DIR / f"{model_version}_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump(final_metrics, f, indent=2)
print(f"✅ Métriques sauvegardées: {metrics_path}")

# 3. Sauvegarder la configuration de prétraitement
preprocessing_config = {
    'image_size': 224,
    'normalization': {
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD
    },
    'transforms': {
        'train': [
            'RandomHorizontalFlip(p=0.5)',
            'RandomVerticalFlip(p=0.5)',
            'RandomRotation(degrees=90)',
            'ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)',
            'ToTensor()',
            'Normalize(ImageNet)'
        ],
        'inference': [
            'ToTensor()',
            'Normalize(ImageNet)'
        ]
    },
    'optimal_threshold': final_metrics['patch_level']['optimal_threshold']
}

config_path = PRODUCTION_DIR / f"{model_version}_config.json"
with open(config_path, 'w') as f:
    json.dump(preprocessing_config, f, indent=2)
print(f"✅ Configuration sauvegardée: {config_path}")

# 4. Créer un README pour la production
readme_content = f"""# Modèle de Production - Détection de Métastases CAMELYON17

## Informations du Modèle
- **Version**: {model_version}
- **Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **Architecture**: SimpleCNN
- **Paramètres**: {trainable_params:,}

## Performances

### Patch-Level
- **Accuracy**: {final_metrics['patch_level']['accuracy']:.4f}
- **F1-Score**: {final_metrics['patch_level']['f1_score']:.4f}
- **AUC-ROC**: {final_metrics['patch_level']['auc_roc']:.4f}
- **Seuil optimal**: {final_metrics['patch_level']['optimal_threshold']:.4f}

### Patient-Level
- **Accuracy**: {final_metrics['patient_level']['accuracy']:.4f}
- **F1-Score**: {final_metrics['patient_level']['f1_score']:.4f}
- **AUC-ROC**: {final_metrics['patient_level']['auc_roc']:.4f}

## Utilisation

### Chargement du modèle
```python
import torch
from pathlib import Path

# Charger le checkpoint
checkpoint = torch.load('{model_path.name}')
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
        mean={IMAGENET_MEAN},
        std={IMAGENET_STD}
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
    prediction = 1 if prob > {final_metrics['patch_level']['optimal_threshold']:.4f} else 0
```

## Fichiers
- `{model_path.name}`: Poids du modèle et état de l'entraînement
- `{metrics_path.name}`: Métriques détaillées
- `{config_path.name}`: Configuration de prétraitement
- `README.md`: Ce fichier

## Notes
- Le modèle a été entraîné sur {len(df_train)} patchs de {df_train['patient'].nunique()} patients
- Split stratégique par centre (centres {df_train['center'].unique()} pour train)
- Utilisation de Focal Loss pour gérer le déséquilibre de classes
- Seuil de décision optimisé sur le set de validation

## Limites
- Dataset de taille limitée (5000 patchs)
- Agrégation patient simpliste (max des probabilités)
- Performances peuvent varier selon le centre hospitalier (domain shift)

## Contact
Pour toute question sur ce modèle, consulter le notebook de modélisation.
"""

readme_path = PRODUCTION_DIR / "README.md"
with open(readme_path, 'w', encoding='utf-8') as f:
    f.write(readme_content)
print(f"✅ README créé: {readme_path}")

# 5. Créer un script d'inférence exemple
inference_script = f"""#!/usr/bin/env python3
\"\"\"
Script d'inférence pour le modèle SimpleCNN CAMELYON17
Version: {model_version}
\"\"\"

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
    \"\"\"Charger le modèle depuis le checkpoint\"\"\"
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleCNN()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def preprocess_image(image_path):
    \"\"\"Prétraiter une image pour l'inférence\"\"\"
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean={IMAGENET_MEAN},
            std={IMAGENET_STD}
        )
    ])
    
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def predict(model, image_tensor, threshold={final_metrics['patch_level']['optimal_threshold']:.4f}):
    \"\"\"Faire une prédiction\"\"\"
    with torch.no_grad():
        output = model(image_tensor)
        prob = torch.sigmoid(output).item()
        prediction = 1 if prob > threshold else 0
    
    return {{
        'probability': prob,
        'prediction': prediction,
        'label': 'Tumor' if prediction == 1 else 'Normal'
    }}

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = Path(__file__).parent / '{model_path.name}'
    
    # Charger le modèle
    print(f"Chargement du modèle: {{model_path}}")
    model, checkpoint = load_model(model_path)
    print(f"Modèle chargé (AUC: {{checkpoint['best_val_auc']:.4f}})")
    
    # Prétraiter l'image
    print(f"Traitement de l'image: {{image_path}}")
    image_tensor = preprocess_image(image_path)
    
    # Prédiction
    result = predict(model, image_tensor)
    
    print("\\nRésultat:")
    print(f"  Probabilité: {{result['probability']:.4f}}")
    print(f"  Prédiction: {{result['label']}}")
"""

inference_path = PRODUCTION_DIR / "inference.py"
with open(inference_path, 'w', encoding='utf-8') as f:
    f.write(inference_script)
print(f"✅ Script d'inférence créé: {inference_path}")

print("\n" + "="*60)
print("✅ SAUVEGARDE TERMINÉE")
print(f"\nFichiers créés dans: {PRODUCTION_DIR}")
print(f"  - {model_path.name}")
print(f"  - {metrics_path.name}")
print(f"  - {config_path.name}")
print(f"  - README.md")
print(f"  - inference.py")
print("\n💡 Pour utiliser le modèle en production:")
print(f"   python {inference_path} <chemin_image>")
