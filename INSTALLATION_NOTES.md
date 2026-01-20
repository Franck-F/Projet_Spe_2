# Notes d'Installation - Projet Spe 2

## ✅ Installation Réussie

**Date** : 2026-01-20  
**Gestionnaire de paquets** : UV 0.9.15  
**Python** : 3.11.14  
**Packages installés** : 155

## 📦 Dépendances Principales Installées

### Deep Learning

- ✅ PyTorch 2.x
- ✅ TorchVision

### Computer Vision

- ✅ OpenCV
- ✅ Pillow
- ✅ scikit-image
- ✅ Albumentations

### Data Science

- ✅ NumPy
- ✅ Pandas
- ✅ SciPy

### Machine Learning

- ✅ scikit-learn
- ✅ XGBoost

### Visualization

- ✅ Plotly
- ✅ Matplotlib
- ✅ Seaborn

### Medical Imaging

- ✅ OpenSlide-Python

### Utilities

- ✅ Jupyter Lab
- ✅ TensorBoard
- ✅ YAML
- ✅ tqdm

## ⚠️ Packages Optionnels (Non Installés)

Les packages suivants nécessitent une compilation et sont disponibles dans le groupe `interpretability` :

- **SHAP** : Interprétabilité avancée (nécessite llvmlite)
- **Grad-CAM** : Visualisation des activations

### Installation Optionnelle

Si vous avez besoin de ces packages plus tard :

```bash
# Installer le groupe interpretability
uv sync --group interpretability

# Ou installer individuellement
uv pip install shap
uv pip install grad-cam
```

**Note** : Ces packages nécessitent un compilateur C/C++ installé sur votre système.

## 🚀 Commandes de Démarrage

```bash
# Activer l'environnement
.venv\Scripts\activate

# Lancer Jupyter Lab
uv run jupyter lab

# Vérifier l'installation
uv run python -c "import torch; import plotly; print('OK')"
```

## 📁 Fichiers Générés

- `.venv/` : Environnement virtuel (1.5 GB+)
- `uv.lock` : Fichier de verrouillage des dépendances (1.5 MB)

## 🔧 Résolution de Problèmes

### Problèmes de Compilation

Si vous rencontrez des erreurs de compilation avec `llvmlite` :

- Ces packages sont optionnels pour commencer le projet
- Vous pouvez les installer plus tard quand nécessaire
- Assurez-vous d'avoir Visual Studio Build Tools (Windows) ou gcc (Linux)

### Réinstallation

```bash
# Supprimer l'environnement
Remove-Item -Recurse -Force .venv

# Réinstaller
uv sync
```

## 📝 Prochaines Étapes

1. **Télécharger le dataset CAMELYON17**
2. **Créer le premier notebook** : `notebooks/01_EDA.ipynb`
3. **Commencer l'analyse exploratoire**

## 🎯 Workflow Recommandé

```bash
# 1. Activer l'environnement
.venv\Scripts\activate

# 2. Lancer Jupyter Lab
uv run jupyter lab

# 3. Créer un nouveau notebook dans notebooks/

# 4. Commencer à coder !
```

---

**Projet prêt pour le développement !** 🎉
