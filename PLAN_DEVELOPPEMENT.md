# Plan de Développement Complet - Projet CAMELYON17

## 📋 Vue d'Ensemble

**Projet** : Détection automatique de métastases ganglionnaires - CAMELYON17  
**Durée** : 9 semaines  
**Équipe** : Groupe 5  
**Objectif** : Développer un système de classification des stades pN (pN0-pN3) à partir d'images histopathologiques

---

## 🎯 PHASE 0 : Cadrage et Organisation (Semaine 1)

### Objectifs

- ✅ Setup du projet et de l'environnement
- ✅ Compréhension du contexte médical
- ✅ Architecture du code mise en place

### Tâches Complétées

- [x] Repository Git créé et poussé sur GitHub
- [x] Environnement UV configuré (155 packages installés)
- [x] Structure du projet créée
- [x] Documentation de base (README, glossaire médical)
- [x] Notebooks templates créés
- [x] Modules Python de base créés

### Tâches Restantes

#### 0.1 Recherche Bibliographique

- [ ] Lire 5-10 articles sur la détection de métastases
- [ ] Étudier le système de classification pN en détail
- [ ] Comprendre les enjeux cliniques des faux négatifs
- [ ] Documenter les findings dans `reports/bibliographie.md`

#### 0.2 Téléchargement du Dataset

- [ ] S'inscrire au challenge CAMELYON17
- [ ] Télécharger les données (WSI + annotations)
- [ ] Organiser dans `data/raw/`
- [ ] Vérifier l'intégrité des fichiers

#### 0.3 Planification d'Équipe

- [ ] Répartir les rôles et responsabilités
- [ ] Définir les jalons hebdomadaires
- [ ] Mettre en place les réunions de suivi

### Livrables

- [x] Repository GitHub fonctionnel
- [x] Environnement de développement prêt
- [ ] Document de bibliographie
- [ ] Dataset téléchargé et organisé

---

## 📊 PHASE 1 : Exploration et Compréhension des Données (Semaines 1-2)

### Objectifs

- Analyser le dataset CAMELYON17
- Comprendre la distribution des données
- Identifier les défis techniques

### Tâches

#### 1.1 Chargement et Inspection (`notebooks/01_EDA.ipynb`)

**Fichiers à implémenter** :

- `src/data/loader.py` : Classes `WSILoader` et `PatchLoader`

**Tâches** :

- [ ] Implémenter `WSILoader.load_wsi()` avec OpenSlide
- [ ] Implémenter `WSILoader.extract_patches()`
- [ ] Créer un script pour lister tous les fichiers WSI
- [ ] Charger les métadonnées (patients, hôpitaux, labels)
- [ ] Créer un DataFrame récapitulatif

**Code à écrire** :

```python
# Dans notebooks/01_EDA.ipynb
from src.data.loader import WSILoader
import pandas as pd

# Charger les métadonnées
metadata = pd.read_csv('../data/raw/metadata.csv')
print(f"Nombre de patients: {metadata['patient_id'].nunique()}")
print(f"Nombre de WSI: {len(metadata)}")
print(f"Hôpitaux: {metadata['hospital'].unique()}")
```

#### 1.2 Statistiques Descriptives

**Tâches** :

- [ ] Calculer le nombre total de patients, slides, patchs
- [ ] Analyser la distribution par hôpital
- [ ] Calculer les statistiques de taille des WSI
- [ ] Créer des tableaux récapitulatifs

**Visualisations Plotly** :

- [ ] Graphique en barres : distribution par hôpital
- [ ] Pie chart : ratio patients par hôpital
- [ ] Histogramme : tailles des WSI

#### 1.3 Distribution des Classes

**Fichiers à utiliser** :

- `src/visualization/eda_plots.py` : `plot_class_distribution()`

**Tâches** :

- [ ] Calculer le ratio normal/tumoral au niveau patch
- [ ] Analyser le déséquilibre des classes
- [ ] Distribution des stades pN au niveau patient
- [ ] Corrélation entre % patchs tumoraux et stade pN

**Visualisations** :

- [ ] Barplot : distribution normal vs tumoral
- [ ] Barplot groupé : distribution par hôpital et classe
- [ ] Barplot : distribution des stades pN (pN0-pN3)
- [ ] Scatter plot : % patchs tumoraux vs stade pN

#### 1.4 Visualisation des WSI

**Tâches** :

- [ ] Afficher 5-10 exemples de WSI complètes
- [ ] Visualiser les annotations (masques tumoraux)
- [ ] Extraire et afficher des patchs représentatifs
- [ ] Comparer patchs normaux vs tumoraux

**Code exemple** :

```python
from src.visualization.eda_plots import plot_patch_samples

# Extraire des patchs
normal_patches = extract_patches(wsi_normal, n=10)
tumor_patches = extract_patches(wsi_tumor, n=10)

# Visualiser
fig = plot_patch_samples(
    images=normal_patches + tumor_patches,
    labels=[0]*10 + [1]*10
)
fig.show()
```

#### 1.5 Quality Check

**Tâches** :

- [ ] Détecter les patchs vides (fond blanc)
- [ ] Identifier les patchs flous
- [ ] Repérer les artefacts de numérisation
- [ ] Calculer le % de patchs à filtrer

### Livrables Phase 1

- [ ] Notebook `01_EDA.ipynb` complété et exécuté
- [ ] Rapport d'analyse statistique (2-3 pages)
- [ ] Visualisations sauvegardées dans `results/figures/eda/`
- [ ] Liste des défis identifiés

---

## 🔧 PHASE 2 : Préparation et Prétraitement (Semaines 2-3)

### Objectifs

- Normaliser la coloration H&E
- Créer un pipeline d'augmentation
- Gérer le déséquilibre des classes
- Préparer les datasets train/val/test

### Tâches

#### 2.1 Normalisation de Coloration (`notebooks/02_preprocessing.ipynb`)

**Fichiers à implémenter** :

- `src/data/preprocessing.py` : Classe `StainNormalizer`

**Tâches** :

- [ ] Implémenter normalisation Macenko
  - [ ] Calcul de la matrice de déconvolution
  - [ ] Extraction des vecteurs de coloration H&E
  - [ ] Normalisation vers image de référence
- [ ] Alternative : implémenter Reinhard
- [ ] Sélectionner une image de référence représentative
- [ ] Tester sur échantillons de chaque hôpital
- [ ] Comparer avant/après normalisation

**Code à écrire** :

```python
from src.data.preprocessing import StainNormalizer

# Initialiser
normalizer = StainNormalizer(method='macenko')

# Fit sur image de référence
ref_image = load_reference_image()
normalizer.fit(ref_image)

# Transformer
normalized = normalizer.transform(test_image)

# Visualiser comparaison
plot_comparison(test_image, normalized)
```

**Métriques** :

- [ ] Calculer la variance de coloration avant/après
- [ ] Mesurer la similarité inter-hôpitaux

#### 2.2 Augmentation de Données

**Fichiers à utiliser** :

- `src/data/preprocessing.py` : `create_augmentation_pipeline()`

**Tâches** :

- [ ] Configurer les transformations dans `configs/config.yaml`
- [ ] Implémenter le pipeline Albumentations
- [ ] Tester chaque transformation individuellement
- [ ] Visualiser les effets de l'augmentation
- [ ] Valider biologiquement les transformations

**Transformations à implémenter** :

- [ ] Flips horizontaux/verticaux (p=0.5)
- [ ] Rotations (±15°)
- [ ] Color jitter léger
- [ ] Gaussian blur (p=0.1)

**Validation** :

- [ ] Afficher 20 versions augmentées d'un même patch
- [ ] Vérifier que les transformations sont réalistes

#### 2.3 Gestion du Déséquilibre

**Tâches** :

- [ ] Calculer les poids de classes

  ```python
  from sklearn.utils.class_weight import compute_class_weight
  weights = compute_class_weight('balanced', classes=[0,1], y=labels)
  ```

- [ ] Implémenter weighted sampling
- [ ] Tester focal loss vs weighted cross-entropy
- [ ] Comparer les stratégies :
  - [ ] Sous-échantillonnage classe majoritaire
  - [ ] Sur-échantillonnage classe minoritaire
  - [ ] Pondération dans la loss
  - [ ] Combinaison des approches

#### 2.4 Split Train/Val/Test

**Fichiers à créer** :

- `scripts/create_splits.py`

**Tâches** :

- [ ] **CRUCIAL** : Stratification au niveau PATIENT
- [ ] Implémenter le split 60/20/20
- [ ] Stratégie par hôpital :
  - [ ] Train : Hôpitaux 1, 2, 3
  - [ ] Val : Hôpital 4
  - [ ] Test : Hôpital 5
- [ ] Vérifier la distribution des classes dans chaque split
- [ ] Sauvegarder les splits dans `data/processed/splits/`

**Code** :

```python
from sklearn.model_selection import train_test_split

# Split au niveau patient
patients = metadata['patient_id'].unique()
train_patients, test_patients = train_test_split(
    patients, test_size=0.2, stratify=patient_labels, random_state=42
)
train_patients, val_patients = train_test_split(
    train_patients, test_size=0.25, stratify=..., random_state=42
)
```

#### 2.5 Quality Filtering

**Fichiers à utiliser** :

- `src/data/preprocessing.py` : `filter_low_quality_patches()`

**Tâches** :

- [ ] Implémenter détection de fond blanc
- [ ] Implémenter détection de flou (Laplacian variance)
- [ ] Filtrer les patchs de mauvaise qualité
- [ ] Documenter le % de patchs filtrés

### Livrables Phase 2

- [ ] Notebook `02_preprocessing.ipynb` complété
- [ ] Pipeline de prétraitement fonctionnel
- [ ] Datasets train/val/test créés et sauvegardés
- [ ] Documentation des choix méthodologiques
- [ ] Rapport de prétraitement (2 pages)

---

## 🧠 PHASE 3 : Modélisation Niveau Patch (Semaines 3-5)

### Objectifs

- Établir une baseline
- Implémenter transfer learning
- Optimiser les hyperparamètres
- Atteindre de bonnes performances au niveau patch

### Tâches

#### 3.1 Baseline Model (`notebooks/03_modeling_patch.ipynb`)

**Fichiers à utiliser** :

- `src/models/cnn_baseline.py` : Classe `BaselineCNN`
- `src/models/train.py` : Classe `Trainer`
- `src/data/dataset.py` : Classe `CAMELYON17Dataset`

**Tâches** :

- [ ] Implémenter `CAMELYON17Dataset.__getitem__()`
- [ ] Créer les DataLoaders
- [ ] Instancier le modèle baseline
- [ ] Configurer l'entraînement :
  - [ ] Loss : CrossEntropyLoss avec poids
  - [ ] Optimizer : Adam (lr=0.001)
  - [ ] Scheduler : ReduceLROnPlateau
- [ ] Entraîner pour 20-30 époques
- [ ] Évaluer sur validation set

**Code** :

```python
from src.models.cnn_baseline import BaselineCNN
from src.models.train import Trainer
from src.data.dataset import create_dataloaders

# Créer les dataloaders
train_loader, val_loader, test_loader = create_dataloaders(
    config, train_dataset, val_dataset, test_dataset
)

# Modèle
model = BaselineCNN(num_classes=2, dropout=0.5)

# Entraînement
trainer = Trainer(model, criterion, optimizer, device='cuda')
trainer.fit(train_loader, val_loader, epochs=30)
```

**Métriques à calculer** :

- [ ] Accuracy, Precision, Recall, F1
- [ ] **F2-score** (pondération recall)
- [ ] AUC-ROC, AUC-PR
- [ ] Matrice de confusion

**Objectif** : Accuracy > 85%, Recall > 90%

#### 3.2 Transfer Learning

**Fichiers à utiliser** :

- `src/models/transfer_learning.py` : `get_pretrained_model()`

**Modèles à tester** :

- [ ] **ResNet50**
  - [ ] Charger avec poids ImageNet
  - [ ] Geler le backbone
  - [ ] Fine-tuner la dernière couche (10 époques)
  - [ ] Dégeler progressivement (10 époques supplémentaires)
- [ ] **ResNet101**
  - [ ] Même procédure que ResNet50
- [ ] **EfficientNet-B3**
  - [ ] Adapter la dernière couche
  - [ ] Fine-tuning progressif
- [ ] **DenseNet121**
  - [ ] Tester comme alternative

**Code** :

```python
from src.models.transfer_learning import get_pretrained_model, unfreeze_layers

# Phase 1 : Backbone gelé
model = get_pretrained_model('resnet50', num_classes=2, freeze_backbone=True)
trainer = Trainer(model, criterion, optimizer)
trainer.fit(train_loader, val_loader, epochs=10)

# Phase 2 : Dégelage progressif
unfreeze_layers(model, num_layers=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
trainer = Trainer(model, criterion, optimizer)
trainer.fit(train_loader, val_loader, epochs=10)
```

**Comparaison** :

- [ ] Créer un tableau comparatif des performances
- [ ] Visualiser avec `src/visualization/results_plots.py`

#### 3.3 Optimisation des Hyperparamètres

**Hyperparamètres à optimiser** :

- [ ] Learning rate : [1e-5, 1e-4, 1e-3]
- [ ] Batch size : [16, 32, 64]
- [ ] Dropout : [0.3, 0.5, 0.7]
- [ ] Weight decay : [1e-5, 1e-4, 1e-3]

**Méthode** :

- [ ] Grid search ou random search
- [ ] Utiliser validation set pour sélection
- [ ] Documenter les résultats

**Code** :

```python
from sklearn.model_selection import ParameterGrid

param_grid = {
    'lr': [1e-4, 1e-3],
    'batch_size': [32, 64],
    'dropout': [0.5, 0.7]
}

results = []
for params in ParameterGrid(param_grid):
    model = get_pretrained_model('resnet50', dropout=params['dropout'])
    # Entraîner et évaluer
    metrics = train_and_evaluate(model, params)
    results.append(metrics)
```

#### 3.4 Évaluation Niveau Patch

**Fichiers à utiliser** :

- `src/evaluation/metrics.py` : `compute_all_metrics()`, `plot_confusion_matrix()`, `plot_roc_curve()`

**Tâches** :

- [ ] Calculer toutes les métriques sur test set
- [ ] Créer la matrice de confusion
- [ ] Tracer la courbe ROC
- [ ] Tracer la courbe Precision-Recall
- [ ] Analyser les faux positifs et faux négatifs
- [ ] Visualiser des exemples d'erreurs

**Visualisations** :

```python
from src.evaluation.metrics import compute_all_metrics, plot_confusion_matrix

# Prédictions
y_pred, y_proba = predict(model, test_loader)

# Métriques
metrics = compute_all_metrics(y_true, y_pred, y_proba)
print(metrics)

# Confusion matrix
fig = plot_confusion_matrix(y_true, y_pred, class_names=['Normal', 'Tumoral'])
fig.write_html('results/figures/confusion_matrix.html')
```

### Livrables Phase 3

- [ ] Notebook `03_modeling_patch.ipynb` complété
- [ ] Meilleur modèle sauvegardé dans `models/final/`
- [ ] Tableau comparatif des modèles
- [ ] Rapport de modélisation (3-4 pages)
- [ ] Visualisations des performances

**Objectif de performance** : Recall > 95%, AUC > 0.95

---

## 🔗 PHASE 4 : Agrégation Patch → Patient (Semaines 5-6)

### Objectifs

- Agréger les prédictions au niveau patient
- Prédire le stade pN
- Comparer différentes stratégies

### Tâches

#### 4.1 Agrégation Statistique (`notebooks/04_aggregation.ipynb`)

**Fichiers à utiliser** :

- `src/models/aggregation.py` : Classe `StatisticalAggregator`

**Tâches** :

- [ ] Implémenter `StatisticalAggregator.aggregate()`
- [ ] Définir les seuils pour classification pN :

  ```python
  thresholds = {
      'pn0': 0.0,    # 0% de patchs tumoraux
      'pn1': 0.05,   # 5% de patchs tumoraux
      'pn2': 0.20    # 20% de patchs tumoraux
  }
  ```

- [ ] Optimiser les seuils sur validation set
- [ ] Tester différentes métriques d'agrégation :
  - [ ] Pourcentage de patchs tumoraux
  - [ ] Probabilité moyenne
  - [ ] Probabilité maximale
  - [ ] Surface tumorale totale

**Code** :

```python
from src.models.aggregation import StatisticalAggregator

aggregator = StatisticalAggregator(thresholds)

# Pour chaque patient
for patient_id in test_patients:
    # Récupérer les prédictions de tous les patchs
    patch_predictions = get_patient_predictions(patient_id)
    
    # Agréger
    pn_stage = aggregator.aggregate(patch_predictions)
    print(f"Patient {patient_id}: pN{pn_stage}")
```

#### 4.2 Modèle ML de Second Niveau

**Fichiers à utiliser** :

- `src/models/aggregation.py` : Classe `MLAggregator`

**Tâches** :

- [ ] Implémenter `MLAggregator.extract_features()`
- [ ] Features à extraire :
  - [ ] % patchs tumoraux
  - [ ] Probabilité moyenne/max/min
  - [ ] Écart-type des probabilités
  - [ ] Nombre total de patchs tumoraux
  - [ ] Percentiles (25, 50, 75, 90)
- [ ] Entraîner XGBoost :

  ```python
  from src.models.aggregation import MLAggregator
  
  aggregator = MLAggregator(model_type='xgboost')
  
  # Extraire features pour tous les patients
  X_train = [extract_features(patient) for patient in train_patients]
  y_train = [get_pn_stage(patient) for patient in train_patients]
  
  # Entraîner
  aggregator.fit(X_train, y_train)
  ```

- [ ] Alternative : Random Forest
- [ ] Comparer les deux approches

#### 4.3 Multiple Instance Learning (Optionnel)

**Tâches** :

- [ ] Implémenter attention-based pooling
- [ ] Traiter chaque patient comme un "bag" de patchs
- [ ] Comparer avec les approches précédentes

#### 4.4 Évaluation Niveau Patient

**Métriques** :

- [ ] Accuracy sur stade pN
- [ ] Cohen's Kappa (accord avec vérité terrain)
- [ ] Matrice de confusion 4x4 (pN0-pN3)
- [ ] Tolérance ±1 stage

**Visualisations** :

```python
from src.evaluation.metrics import plot_confusion_matrix

# Matrice de confusion
fig = plot_confusion_matrix(
    y_true_pn, y_pred_pn,
    class_names=['pN0', 'pN1', 'pN2', 'pN3']
)
fig.show()
```

### Livrables Phase 4

- [ ] Notebook `04_aggregation.ipynb` complété
- [ ] Comparaison des stratégies d'agrégation
- [ ] Meilleure stratégie sélectionnée et documentée
- [ ] Rapport d'agrégation (2-3 pages)

**Objectif** : Accuracy > 80% sur classification pN

---

## 📈 PHASE 5 : Évaluation et Robustesse (Semaines 6-7)

### Objectifs

- Évaluation complète multi-niveaux
- Analyse du domain shift
- Tests de robustesse

### Tâches

#### 5.1 Évaluation Multi-niveaux (`notebooks/05_evaluation.ipynb`)

**Niveau Patch** :

- [ ] Performances globales sur test set
- [ ] Performances par hôpital
- [ ] Analyse de sous-groupes (si métadonnées disponibles)

**Niveau Patient** :

- [ ] Accuracy, Kappa sur stades pN
- [ ] Matrice de confusion détaillée
- [ ] Analyse des erreurs de ±1 stage vs ±2 stages

**Code** :

```python
from src.evaluation.metrics import compute_all_metrics

# Niveau patch
patch_metrics = compute_all_metrics(y_true_patch, y_pred_patch, y_proba_patch)

# Niveau patient
patient_metrics = compute_all_metrics(y_true_patient, y_pred_patient)

# Afficher
print("=== Niveau Patch ===")
for metric, value in patch_metrics.items():
    print(f"{metric}: {value:.4f}")

print("\n=== Niveau Patient ===")
for metric, value in patient_metrics.items():
    print(f"{metric}: {value:.4f}")
```

#### 5.2 Analyse du Domain Shift

**Tâches** :

- [ ] Calculer les performances par hôpital
- [ ] Créer un tableau comparatif
- [ ] Identifier les hôpitaux "difficiles"
- [ ] Analyser les causes :
  - [ ] Différences de coloration
  - [ ] Variabilité des scanners
  - [ ] Caractéristiques des populations

**Visualisations** :

```python
import plotly.express as px

# Performances par hôpital
hospital_metrics = []
for hospital in hospitals:
    mask = test_df['hospital'] == hospital
    metrics = compute_all_metrics(y_true[mask], y_pred[mask])
    hospital_metrics.append({
        'hospital': hospital,
        **metrics
    })

df = pd.DataFrame(hospital_metrics)
fig = px.bar(df, x='hospital', y='recall', title='Recall par Hôpital')
fig.show()
```

**Stratégies d'amélioration** :

- [ ] Normalisation de coloration plus robuste
- [ ] Domain adaptation techniques
- [ ] Entraînement multi-domaine

#### 5.3 Analyse des Erreurs

**Faux Négatifs (CRITIQUE)** :

- [ ] Identifier tous les FN
- [ ] Visualiser les patchs mal classés
- [ ] Caractéristiques communes :
  - [ ] Micro-métastases ?
  - [ ] Zones ambiguës ?
  - [ ] Problèmes de qualité ?
- [ ] Proposer des améliorations

**Faux Positifs** :

- [ ] Identifier les FP
- [ ] Tissus inflammatoires confondus ?
- [ ] Artefacts de coloration ?

**Code** :

```python
from src.evaluation.interpretability import analyze_prediction_errors

# Analyser les erreurs
errors = analyze_prediction_errors(model, test_loader, num_examples=20)

# Visualiser
for error in errors:
    print(f"True: {error['true_label']}, Pred: {error['pred_label']}, "
          f"Confidence: {error['confidence']:.3f}")
    # Afficher l'image
```

#### 5.4 Tests de Robustesse

**Perturbations à tester** :

- [ ] Bruit gaussien
- [ ] Flou
- [ ] Variations de contraste/luminosité
- [ ] Rotations extrêmes

**Code** :

```python
import albumentations as A

# Pipeline de perturbations
perturbations = A.Compose([
    A.GaussianNoise(p=1.0),
    A.GaussianBlur(blur_limit=(5, 5), p=1.0),
])

# Tester
perturbed_metrics = test_robustness(model, test_loader, perturbations)
print(f"Performance avec perturbations: {perturbed_metrics}")
```

**Techniques avancées** :

- [ ] Monte Carlo Dropout pour incertitude
- [ ] Ensemble de modèles
- [ ] Test-time augmentation

### Livrables Phase 5

- [ ] Notebook `05_evaluation.ipynb` complété
- [ ] Rapport d'évaluation complet (4-5 pages)
- [ ] Visualisations des performances
- [ ] Analyse critique des limites
- [ ] Propositions d'amélioration

---

## 🔍 PHASE 6 : Interprétabilité et IA Responsable (Semaines 7-8)

### Objectifs

- Comprendre les décisions du modèle
- Valider médicalement les prédictions
- Discussion éthique

### Tâches

#### 6.1 Grad-CAM (`notebooks/05_evaluation.ipynb`)

**Fichiers à utiliser** :

- `src/evaluation/interpretability.py` : Classe `GradCAM`
- `src/visualization/heatmaps.py` : `plot_gradcam_heatmap()`

**Tâches** :

- [ ] Implémenter `GradCAM.generate_cam()`
- [ ] Sélectionner 20-30 cas représentatifs :
  - [ ] Vrais positifs (métastases bien détectées)
  - [ ] Vrais négatifs (tissus normaux)
  - [ ] Faux positifs (erreurs)
  - [ ] Faux négatifs (métastases manquées)
- [ ] Générer les heatmaps
- [ ] Visualiser avec Plotly

**Code** :

```python
from src.evaluation.interpretability import GradCAM
from src.visualization.heatmaps import plot_gradcam_heatmap

# Initialiser Grad-CAM
model = load_best_model()
target_layer = model.layer4[-1]  # Dernière couche conv
gradcam = GradCAM(model, target_layer)

# Générer CAM
cam = gradcam.generate_cam(input_image, target_class=1)

# Visualiser
fig = plot_gradcam_heatmap(original_image, cam)
fig.write_html('results/figures/gradcam_example.html')
```

**Validation** :

- [ ] Le modèle regarde-t-il les bonnes structures ?
- [ ] Focus sur les cellules tumorales ou artefacts ?
- [ ] Cohérence avec l'expertise pathologiste

#### 6.2 SHAP (Optionnel)

**Tâches** :

- [ ] Installer le groupe `interpretability` :

  ```bash
  uv sync --group interpretability
  ```

- [ ] Utiliser SHAP pour expliquer les prédictions
- [ ] Identifier les features les plus importantes

#### 6.3 Discussion Éthique

**Créer** : `reports/discussion_ethique.md`

**Points à aborder** :

- [ ] **Biais potentiels** :
  - [ ] Déséquilibre racial/géographique ?
  - [ ] Sur-représentation de certains hôpitaux ?
  - [ ] Biais de sélection dans le dataset ?
- [ ] **Limites techniques** :
  - [ ] Sensibilité aux variations de préparation
  - [ ] Généralisabilité à d'autres contextes
  - [ ] Cas où le modèle échoue systématiquement
- [ ] **Positionnement clinique** :
  - [ ] Outil d'aide, pas de remplacement
  - [ ] Workflow proposé (pré-screening, second avis)
  - [ ] Quand demander avis humain ?
- [ ] **Considérations de déploiement** :
  - [ ] Exigences réglementaires (CE, FDA)
  - [ ] Intégration dans le workflow clinique
  - [ ] Maintenance et monitoring
  - [ ] Coûts vs bénéfices

#### 6.4 Cas d'Étude

**Tâches** :

- [ ] Sélectionner 5-10 cas cliniques intéressants
- [ ] Documenter chaque cas :
  - [ ] Image du patch/WSI
  - [ ] Prédiction du modèle
  - [ ] Heatmap Grad-CAM
  - [ ] Interprétation médicale
  - [ ] Validation par expert (si possible)

### Livrables Phase 6

- [ ] Visualisations Grad-CAM (20-30 exemples)
- [ ] Document de discussion éthique (3-4 pages)
- [ ] Cas d'étude documentés
- [ ] Recommandations pour déploiement responsable

---

## 📝 PHASE 7 : Documentation et Livrables Finaux (Semaines 8-9)

### Objectifs

- Finaliser le code et la documentation
- Rédiger le rapport final
- Préparer la présentation

### Tâches

#### 7.1 Nettoyage et Documentation du Code

**Tâches** :

- [ ] Nettoyer tous les notebooks
- [ ] Ajouter docstrings complètes
- [ ] Ajouter type hints
- [ ] Commenter les sections complexes
- [ ] Vérifier la cohérence du code
- [ ] Tester la reproductibilité :

  ```bash
  # Tester sur une machine tierce
  git clone https://github.com/Franck-F/Projet_Spe_2.git
  cd Projet_Spe_2
  uv sync
  uv run jupyter lab
  # Exécuter tous les notebooks
  ```

**README.md** :

- [ ] Mettre à jour avec les résultats finaux
- [ ] Ajouter des exemples d'utilisation
- [ ] Documenter les commandes principales
- [ ] Ajouter des captures d'écran

#### 7.2 Rapport Final (Max 15 pages)

**Créer** : `reports/rapport_final.md`

**Structure** :

**1. Introduction (1 page)** :

- [ ] Contexte médical et enjeux
- [ ] Objectifs du projet
- [ ] Aperçu de l'approche

**2. Données et Prétraitement (2 pages)** :

- [ ] Description CAMELYON17
- [ ] Analyse exploratoire clé
- [ ] Pipeline de preprocessing
- [ ] Gestion du déséquilibre

**3. Méthodologie (4 pages)** :

- [ ] Architectures CNN testées
- [ ] Stratégie d'entraînement
- [ ] Stratégie d'agrégation patch→patient
- [ ] Choix techniques justifiés

**4. Résultats (4 pages)** :

- [ ] Performances niveau patch et patient
- [ ] Comparaison des approches
- [ ] Analyse du domain shift
- [ ] Visualisations clés (tableaux, graphiques)

**5. Interprétabilité et Discussion (3 pages)** :

- [ ] Analyse Grad-CAM
- [ ] Limites et biais
- [ ] Perspectives médicales
- [ ] IA responsable

**6. Conclusion (1 page)** :

- [ ] Synthèse des contributions
- [ ] Recommandations
- [ ] Travaux futurs

**Annexes** :

- [ ] Résultats supplémentaires
- [ ] Hyperparamètres détaillés
- [ ] Code snippets importants

#### 7.3 Présentation Orale (15 min)

**Créer** : `reports/presentation.pptx` ou utiliser Jupyter Slides

**Structure (15 slides)** :

1. **Titre et Équipe** (1 slide)
2. **Contexte Médical** (2 slides)
   - Cancer du sein et métastases
   - Système pN et enjeux cliniques
3. **Dataset et Défis** (2 slides)
   - CAMELYON17
   - Déséquilibre, domain shift
4. **Approche Méthodologique** (4 slides)
   - Architecture CNN (Transfer Learning)
   - Stratégie d'agrégation
   - Pipeline complet
5. **Résultats** (4 slides)
   - Performances niveau patch
   - Performances niveau patient
   - Comparaison des modèles
   - Interprétabilité (Grad-CAM)
6. **Discussion et Conclusion** (2 slides)
   - Limites
   - Perspectives cliniques
   - Recommandations

**Conseils** :

- [ ] Visuels > texte
- [ ] Animations minimales
- [ ] Répétition chronométrée (3-4 fois)
- [ ] Préparation des questions potentielles

#### 7.4 Vérifications Finales

**Checklist** :

- [ ] Tous les notebooks s'exécutent sans erreur
- [ ] Code reproductible testé
- [ ] Citations et références correctes
- [ ] Plagiat vérifié
- [ ] Utilisation IA documentée et transparente
- [ ] Tous les membres maîtrisent le projet
- [ ] Rendus avant deadline

**Git** :

- [ ] Dernier commit avec tag de version :

  ```bash
  git tag -a v1.0 -m "Version finale du projet"
  git push origin v1.0
  ```

- [ ] README.md à jour
- [ ] LICENSE ajoutée si nécessaire

### Livrables Phase 7

- [ ] Code final nettoyé et documenté
- [ ] Rapport final (PDF, 15 pages max)
- [ ] Présentation (PPT/PDF)
- [ ] Repository GitHub complet
- [ ] Vidéo de démonstration (optionnel, 3-5 min)

---

## 📅 CALENDRIER RÉCAPITULATIF

| Semaine | Phase | Objectifs Clés | Livrables |
|---------|-------|----------------|-----------|
| **1** | Phase 0 + Phase 1 | Setup + EDA initial | Env configuré, EDA notebook |
| **2** | Phase 1 + Phase 2 | EDA complet + Preprocessing | Dataset preprocessé |
| **3** | Phase 2 + Phase 3 | Preprocessing + Baseline | Baseline model |
| **4** | Phase 3 | Transfer Learning | Modèles CNN entraînés |
| **5** | Phase 3 + Phase 4 | Optimisation + Agrégation | Meilleur modèle patch |
| **6** | Phase 4 + Phase 5 | Agrégation + Évaluation | Prédictions patient |
| **7** | Phase 5 + Phase 6 | Robustesse + Interprétabilité | Analyse complète |
| **8** | Phase 6 + Phase 7 | Discussion + Début rédaction | Grad-CAM, éthique |
| **9** | Phase 7 | Finalisation | Rapport, présentation |

---

## 🎯 OBJECTIFS DE PERFORMANCE

### Niveau Patch

- **Recall** : > 95% (priorité médicale)
- **Precision** : > 90%
- **AUC-ROC** : > 0.95
- **F2-score** : > 0.93

### Niveau Patient

- **Accuracy stade pN** : > 80%
- **Cohen's Kappa** : > 0.75
- **Tolérance ±1 stage** : > 95%

---

## 🛠️ OUTILS ET RESSOURCES

### Développement

- **IDE** : VS Code avec extensions Python, Jupyter
- **Versioning** : Git + GitHub
- **Package Manager** : UV
- **Notebooks** : Jupyter Lab

### Bibliothèques Principales

- **Deep Learning** : PyTorch, TorchVision
- **Visualisation** : Plotly, Matplotlib
- **ML** : scikit-learn, XGBoost
- **Medical Imaging** : OpenSlide
- **Tracking** : TensorBoard

### Ressources Externes

- **Dataset** : <https://camelyon17.grand-challenge.org/>
- **Documentation PyTorch** : <https://pytorch.org/docs/>
- **Plotly** : <https://plotly.com/python/>
- **Papers** : Google Scholar, arXiv

---

## ⚠️ PIÈGES À ÉVITER

### Data Leakage

- ❌ **NE JAMAIS** mélanger patchs du même patient entre train/test
- ✅ Toujours stratifier au niveau PATIENT

### Overfitting

- ❌ Ne pas sur-optimiser sur validation set
- ✅ Utiliser early stopping et régularisation

### Métriques

- ❌ Ne pas se focaliser uniquement sur accuracy
- ✅ Prioriser le recall (faux négatifs critiques)

### Interprétabilité

- ❌ Ne pas créer une boîte noire
- ✅ Toujours valider avec Grad-CAM

### Déconnexion Clinique

- ❌ Oublier l'objectif médical
- ✅ Toujours penser à l'utilité clinique

---

## 📚 RÉFÉRENCES CLÉS

1. **CAMELYON17 Challenge** : <https://camelyon17.grand-challenge.org/>
2. **Bejnordi et al. (2017)** : Diagnostic Assessment of Deep Learning Algorithms
3. **Liu et al. (2019)** : Detecting Cancer Metastases on Gigapixel Pathology Images
4. **Campanella et al. (2019)** : Clinical-grade computational pathology using weakly supervised deep learning

---

**Ce plan de développement est votre feuille de route complète. Suivez-le étape par étape, documentez votre progression, et n'hésitez pas à adapter selon les découvertes en cours de route. Bon courage ! 🚀**
