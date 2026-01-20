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
│   ├── data/                     # Gestion des données
│   ├── models/                   # Architectures et entraînement
│   ├── evaluation/               # Métriques et évaluation
│   ├── visualization/            # Visualisations Plotly
│   └── utils/                    # Utilitaires
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
│   └── glossaire_medical.md     # Terminologie médicale
│
├── configs/                      # Fichiers de configuration
│   └── config.yaml              # Configuration principale
│
├── scripts/                      # Scripts d'exécution
│
├── .gitignore                    # Fichiers à ignorer
├── requirements.txt              # Dépendances Python
└── README.md                     # Ce fichier
```

## 🚀 Installation

### Prérequis

- Python 3.8+
- CUDA 11.0+ (pour GPU)
- Git

### Setup

```bash
# Cloner le repository
git clone https://github.com/[votre-username]/Projet_Spe_2.git
cd Projet_Spe_2

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## 📤 Pousser sur GitHub

```bash
# Créer le repository sur GitHub, puis :
git remote add origin https://github.com/[votre-username]/Projet_Spe_2.git
git branch -M main
git push -u origin main
```

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

- CAMELYON17 Challenge: <https://camelyon17.grand-challenge.org/>
- [Autres références à ajouter]

## 🙏 Remerciements

[À compléter]
