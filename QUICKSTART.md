# Projet Spe 2 - CAMELYON17

Système de détection automatique de métastases ganglionnaires dans le cancer du sein.

## 🚀 Démarrage Rapide

```bash
# 1. Installer UV (si pas déjà fait)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# 2. Cloner le projet
git clone https://github.com/[votre-username]/Projet_Spe_2.git
cd Projet_Spe_2

# 3. Créer l'environnement et installer les dépendances
uv venv
uv pip install -e .

# 4. Activer l'environnement
.venv\Scripts\activate  # Windows

# 5. Lancer Jupyter Lab
uv run jupyter lab
```

## 📚 Documentation

- **[README.md](README.md)** - Documentation complète du projet
- **[UV_QUICKSTART.md](UV_QUICKSTART.md)** - Guide UV et commandes utiles
- **[reports/glossaire_medical.md](reports/glossaire_medical.md)** - Terminologie médicale

## 🎯 Roadmap

### Phase 1 : EDA (Semaines 1-2)

- [ ] Télécharger dataset CAMELYON17
- [ ] Créer `notebooks/01_EDA.ipynb`
- [ ] Analyser distribution des classes

### Phase 2 : Prétraitement (Semaines 2-3)

- [ ] Normalisation de coloration
- [ ] Augmentation de données
- [ ] Gestion du déséquilibre

### Phase 3 : Modélisation (Semaines 3-5)

- [ ] Baseline CNN
- [ ] Transfer Learning (ResNet, EfficientNet)
- [ ] Optimisation hyperparamètres

### Phase 4 : Agrégation (Semaines 5-6)

- [ ] Stratégies patch → patient
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

## 🛠️ Stack Technique

- **Deep Learning** : PyTorch
- **Visualisation** : Plotly
- **Package Manager** : UV
- **Notebooks** : Jupyter Lab
