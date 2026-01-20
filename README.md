# Projet Spe 2 - Détection de Métastases CAMELYON17

## 🎯 Objectif du Projet

Développement d'un système de détection automatique de métastases ganglionnaires dans le cancer du sein à partir d'images histopathologiques (Whole Slide Images - WSI) du dataset CAMELYON17.

**Enjeu clinique** : Classification automatique des patients selon le système pN (pN0, pN1, pN2, pN3) pour optimiser le diagnostic et le traitement.

## 📊 Dataset

- **Source** : CAMELYON17 Challenge
- **Type** : Whole Slide Images (WSI) de ganglions lymphatiques
- **Coloration** : Hématoxyline et Éosine (H&E)
- **Centres** : 5 hôpitaux différents
- **Niveaux d'annotation** :
  - Niveau patch : normal vs tumoral
  - Niveau patient : stade pN (pN0, pN1, pN2, pN3)

## 🏗️ Architecture du Projet

```
Projet_Spe_2/
├── data/                          # Données (non versionnées)
│   ├── raw/                       # Données brutes CAMELYON17
│   ├── processed/                 # Données prétraitées
│   └── annotations/               # Fichiers d'annotations
│
├── notebooks/                     # Jupyter notebooks pour exploration
│   ├── 01_EDA.ipynb              # Analyse exploratoire
│   ├── 02_preprocessing.ipynb    # Prétraitement
│   ├── 03_modeling_patch.ipynb   # Modélisation niveau patch
│   ├── 04_aggregation.ipynb      # Agrégation patch → patient
│   └── 05_evaluation.ipynb       # Évaluation et interprétabilité
│
├── src/                          # Code source modulaire
│   ├── __init__.py
│   ├── data/                     # Gestion des données
│   │   ├── __init__.py
│   │   ├── loader.py            # Chargement WSI et patchs
│   │   ├── preprocessing.py     # Normalisation, augmentation
│   │   └── dataset.py           # PyTorch/TF datasets
│   │
│   ├── models/                   # Architectures et entraînement
│   │   ├── __init__.py
│   │   ├── cnn_baseline.py      # CNN from scratch
│   │   ├── transfer_learning.py # ResNet, EfficientNet, etc.
│   │   ├── aggregation.py       # Stratégies patch → patient
│   │   └── train.py             # Pipeline d'entraînement
│   │
│   ├── evaluation/               # Métriques et évaluation
│   │   ├── __init__.py
│   │   ├── metrics.py           # Recall, Precision, AUC, etc.
│   │   └── interpretability.py  # Grad-CAM, SHAP
│   │
│   ├── visualization/            # Visualisations Plotly
│   │   ├── __init__.py
│   │   ├── eda_plots.py         # Graphiques EDA
│   │   ├── results_plots.py     # Courbes ROC, confusion matrix
│   │   └── heatmaps.py          # Grad-CAM visualizations
│   │
│   └── utils/                    # Utilitaires
│       ├── __init__.py
│       ├── config.py            # Configuration globale
│       └── logger.py            # Logging
│
├── models/                       # Modèles sauvegardés
│   ├── checkpoints/             # Checkpoints d'entraînement
│   └── final/                   # Modèles finaux
│
├── results/                      # Résultats d'expériences
│   ├── metrics/                 # Métriques JSON/CSV
│   ├── figures/                 # Graphiques générés
│   └── predictions/             # Prédictions sauvegardées
│
├── reports/                      # Documentation et rapports
│   ├── figures/                 # Images pour le rapport
│   ├── glossaire_medical.md     # Terminologie médicale
│   ├── phase_reports/           # Rapports par phase
│   └── final_report.md          # Rapport final
│
├── configs/                      # Fichiers de configuration
│   ├── config.yaml              # Configuration principale
│   ├── model_configs/           # Configs par modèle
│   └── experiment_configs/      # Configs expériences
│
├── scripts/                      # Scripts d'exécution
│   ├── download_data.sh         # Téléchargement CAMELYON17
│   ├── preprocess.py            # Prétraitement batch
│   ├── train_model.py           # Entraînement
│   └── evaluate.py              # Évaluation
│
├── tests/                        # Tests unitaires
│   ├── test_data.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── .gitignore                    # Fichiers à ignorer
├── .dvcignore                    # Fichiers DVC à ignorer
├── requirements.txt              # Dépendances Python
├── environment.yml               # Environnement Conda (optionnel)
├── setup.py                      # Installation du package
├── LICENSE                       # Licence du projet
└── README.md                     # Ce fichier
```

## 🚀 Installation

### Prérequis
- Python 3.8+
- CUDA 11.0+ (pour GPU)
- Git
- DVC (Data Version Control)

### Setup

```bash
# Cloner le repository
git clone https://github.com/[votre-username]/Projet_Spe_2.git
cd Projet_Spe_2

# Créer l'environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer le package en mode développement
pip install -e .
```

## 📦 Dépendances Principales

- **Deep Learning** : PyTorch / TensorFlow
- **Vision** : OpenCV, Pillow, scikit-image
- **Visualisation** : Plotly, Matplotlib, Seaborn
- **ML** : scikit-learn, XGBoost
- **Data** : NumPy, Pandas
- **Interprétabilité** : SHAP, pytorch-grad-cam
- **Tracking** : Weights & Biases / MLflow
- **Versioning** : DVC

## 📋 Roadmap (9 semaines)

### Phase 0 : Cadrage (Semaine 1)
- [x] Setup repository et architecture
- [ ] Recherche bibliographique médicale
- [ ] Documentation glossaire médical

### Phase 1 : EDA (Semaines 1-2)
- [ ] Analyse des WSI
- [ ] Distribution des classes
- [ ] Analyse des labels patients

### Phase 2 : Prétraitement (Semaines 2-3)
- [ ] Normalisation de coloration
- [ ] Augmentation de données
- [ ] Gestion du déséquilibre

### Phase 3 : Modélisation Patch (Semaines 3-5)
- [ ] Baseline CNN
- [ ] Transfer Learning
- [ ] Optimisation hyperparamètres

### Phase 4 : Agrégation (Semaines 5-6)
- [ ] Stratégies d'agrégation
- [ ] Prédiction stade pN

### Phase 5 : Évaluation (Semaines 6-7)
- [ ] Métriques multi-niveaux
- [ ] Analyse domain shift
- [ ] Tests de robustesse

### Phase 6 : Interprétabilité (Semaines 7-8)
- [ ] Grad-CAM
- [ ] Discussion éthique

### Phase 7 : Documentation (Semaines 8-9)
- [ ] Rapport final
- [ ] Présentation

## 🎯 Métriques Clés

**Niveau Patch** :
- Recall (priorité médicale)
- Precision
- AUC-ROC, AUC-PR
- F1-score, F2-score

**Niveau Patient** :
- Accuracy stade pN
- Cohen's Kappa
- Matrice de confusion

## 👥 Équipe

[À compléter]

## 📄 Licence

[À définir]

## 📚 Références

- CAMELYON17 Challenge: https://camelyon17.grand-challenge.org/
- [Autres références à ajouter]

## 🙏 Remerciements

[À compléter]
