# Projet Spe 2 - Détection de Métastases CAMELYON17

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![UV](https://img.shields.io/badge/package%20manager-UV-orange)](https://github.com/astral-sh/uv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Next.js](https://img.shields.io/badge/Next.js-14+-black)](https://nextjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: In Development](https://img.shields.io/badge/status-in%20development-orange)](https://github.com/Franck-F/Projet_Spe_2)

## Objectif du Projet

Développement d'un système de détection automatique de métastases ganglionnaires dans le cancer du sein à partir d'images histopathologiques (Whole Slide Images - WSI) du dataset CAMELYON17.

**Enjeu clinique** : Classification automatique des patients selon le système pN (pN0, pN1, pN2, pN3) pour optimiser le diagnostic et le traitement.

## Dataset

- **Source** : CAMELYON17 Challenge & WILDS Benchmark
- **Format** : Patchs 96x96 pré-extraits (Dataset WILDS)
- **Type** : Whole Slide Images (WSI) de ganglions lymphatiques
- **Coloration** : Hématoxyline et Éosine (H&E)
- **Centres** : 5 hôpitaux différents
- **Niveaux d'annotation** :
  - Niveau patch : normal vs tumoral
  - Niveau patient : stade pN (pN0, pN1, pN2, pN3)

## 🌐 Application Web (Next.js)

### Fonctionnalités

- **Upload d'images** : Interface drag-and-drop pour uploader des images
- **Analyse automatique** : Détection du label de cancer (0 = pas de cancer, 1 = cancer)
- **Toggle metadata.csv** : Option pour activer/désactiver l'utilisation du fichier metadata.csv
- **Affichage des résultats** : Métadonnées complètes avec statistiques des pixels
- **Interface moderne** : Design responsive avec Tailwind CSS

### Installation de l'application web

```bash
# Installer les dépendances Node.js
npm install

# Installer les dépendances Python
pip3 install matplotlib numpy Pillow

# Démarrer le serveur de développement
npm run dev
```

Ouvrir [http://localhost:3000](http://localhost:3000) dans votre navigateur.

### Structure de l'application web

```
cancer-image-classifier/
├── app/
│   ├── api/
│   │   └── analyze/
│   │       └── route.ts      # API route pour analyser les images
│   └── page.tsx              # Page principale
├── components/
│   ├── ImageUpload.tsx       # Composant d'upload d'image
│   └── ResultsDisplay.tsx    # Composant d'affichage des résultats
└── README.md
```

## Architecture du Projet

```
Projet_Spe_2/
├── data/                          # Données (non versionnées)
│   ├── raw/
│   │   └── wilds/                 # Dataset WILDS CAMELYON17 (patchs)
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
│   ├── visualization/            # Visualisations 
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
├── .python-version               # Version Python pour UV
├── pyproject.toml                # Configuration et dépendances
└── README.md                     # Ce fichier
```

## Installation

### Prérequis

- Python 3.8+
- Node.js 18+ (pour l'application web)
- CUDA 11.0+ (pour GPU)
- Git
- UV (gestionnaire de paquets ultra-rapide)

### Installation de UV

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup du Projet

```bash
# Cloner le repository
git clone https://github.com/Franck-F/Projet_Spe_2.git
cd Projet_Spe_2

# Créer l'environnement virtuel et installer les dépendances avec UV
uv sync

# Lancer Jupyter Lab directement
uv run jupyter lab
```

### Commandes UV Utiles

```bash
# Ajouter une nouvelle dépendance
uv pip install nom-du-package

# Mettre à jour les dépendances
uv pip install --upgrade -e .

# Synchroniser l'environnement
uv pip sync

# Lancer Jupyter
uv run jupyter lab
```

## Stack Technique

- **Deep Learning** : PyTorch
- **Visualisation** : Plotly
- **Package Manager** : UV
- **Notebooks** : Jupyter Lab
- **Web Application** : Next.js, React, TypeScript, Tailwind CSS

## Métriques Clés

**Niveau Patch** :

- Recall (priorité médicale)
- Precision
- AUC-ROC, AUC-PR
- F1-score, F2-score

**Niveau Patient** :

- Accuracy stade pN
- Cohen's Kappa
- Matrice de confusion

## Équipe

- [Franck Fambou](https://github.com/FranckF)
- [Charlotte Martineau](https://github.com/cmartineau15)
- [Hector Chablis](https://github.com/Hectotor)
- [Valentine Martin](https://github.com/LabigV)

## Licence

MIT License

## Références

- CAMELYON17 Challenge: <https://camelyon17.grand-challenge.org/>
- WILDS Benchmark: <https://wilds.stanford.edu/datasets/#camelyon17>
