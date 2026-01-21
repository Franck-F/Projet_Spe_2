# Guide de Démarrage Rapide avec UV

## Installation de UV

UV est un gestionnaire de paquets Python ultra-rapide (10-100x plus rapide que pip).

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Linux/macOS

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Setup du Projet

```bash
# 1. Cloner le projet
git clone https://github.com/[votre-username]/Projet_Spe_2.git
cd Projet_Spe_2

# 2. Créer l'environnement virtuel avec UV
uv venv

# 3. Activer l'environnement
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 4. Installer le projet et ses dépendances
uv pip install -e .

# 5. (Optionnel) Installer les outils de développement
uv pip install -e ".[dev]"
```

## Commandes UV Essentielles

### Gestion des Dépendances

```bash
# Installer une nouvelle dépendance
uv pip install package-name

# Installer depuis requirements.txt (si besoin)
uv pip install -r requirements.txt

# Mettre à jour toutes les dépendances
uv pip install --upgrade -e .

# Lister les packages installés
uv pip list
```

### Lancer Jupyter

```bash
# Démarrer Jupyter Lab
uv run jupyter lab

# Ou Jupyter Notebook
uv run jupyter notebook
```

### Exécuter des Scripts

```bash
# Lancer un script Python avec UV
uv run python scripts/train_model.py

# Lancer directement sans activer l'environnement
uv run python src/data/preprocessing.py
```

## Workflow de Développement

### 1. Ajouter une Nouvelle Dépendance

```bash
# Installer la dépendance
uv pip install nouvelle-dependance

# Mettre à jour pyproject.toml manuellement
# Ajouter dans la section [project.dependencies]
```

### 2. Travailler sur un Notebook

```bash
# Lancer Jupyter Lab
uv run jupyter lab

# Créer un nouveau notebook dans notebooks/
# Exemple: notebooks/01_EDA.ipynb
```

### 3. Exécuter les Tests (quand implémentés)

```bash
# Installer les dépendances de dev
uv pip install -e ".[dev]"

# Lancer pytest
uv run pytest
```

## Avantages de UV

- [x] **Ultra-rapide** : 10-100x plus rapide que pip  
- [x] **Résolution de dépendances** : Résout les conflits automatiquement  
- [x] **Cache intelligent** : Réutilise les packages déjà téléchargés  
- [x] **Compatible** : Fonctionne avec pip, requirements.txt, pyproject.toml  
- [x] **Moderne** : Suit les standards Python actuels (PEP 621)  

## Troubleshooting

### UV n'est pas reconnu

```bash
# Redémarrer le terminal après installation
# Ou ajouter UV au PATH manuellement
```

### Erreur de version Python

```bash
# Vérifier la version Python
python --version

# UV utilise automatiquement .python-version (3.11)
# Installer Python 3.11 si nécessaire
```

### Problèmes de dépendances

```bash
# Nettoyer et réinstaller
uv pip uninstall -y -r requirements.txt
uv pip install -e .
```

## Migration depuis pip

Si vous avez déjà un environnement pip :

```bash
# 1. Désactiver l'ancien environnement
deactivate

# 2. Créer un nouvel environnement avec UV
uv venv

# 3. Activer le nouvel environnement
.venv\Scripts\activate  # Windows

# 4. Installer avec UV
uv pip install -e .
```

## Ressources

- Documentation UV : <https://github.com/astral-sh/uv>
- PyProject.toml : <https://packaging.python.org/en/latest/specifications/pyproject-toml/>
