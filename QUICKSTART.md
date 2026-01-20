# Projet Spe 2 - CAMELYON17

Système de détection automatique de métastases ganglionnaires dans le cancer du sein.

## Démarrage Rapide

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

## Documentation

- **[README.md](README.md)** - Documentation complète du projet
- **[UV_QUICKSTART.md](UV_QUICKSTART.md)** - Guide UV et commandes utiles
- **[reports/glossaire_medical.md](reports/glossaire_medical.md)** - Terminologie médicale

## Stack Technique

- **Deep Learning** : PyTorch
- **Visualisation** : Plotly
- **Package Manager** : UV
- **Notebooks** : Jupyter Lab
